import json
import os
import random
import traceback
from functools import partial

import numpy as np
from resemblyzer import VoiceEncoder
from tqdm import tqdm

import wavepainter.runtime.single_thread_env  # NOQA
from wavepainter.audio import librosa_wav2spec
from wavepainter.audio.align import get_mel2ph, mel2token_to_dur
from wavepainter.audio.pitch.utils import f0_to_coarse
from wavepainter.audio.pitch_extractors import extract_pitch
from wavepainter.runtime.hparams import hparams
from wavepainter.runtime.indexed_datasets import IndexedDatasetBuilder
from wavepainter.runtime.multiprocess_utils import multiprocess_run_tqdm
from wavepainter.runtime.os_utils import remove_file, copy_file

np.seterr(divide='ignore', invalid='ignore')

# Split is 98/1/1 as in the FluentEditor2 paper, but seeded and disjoint.
SPLIT_SEED = 1234
VALID_FRAC = 0.01
TEST_FRAC = 0.01


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    # Defaults for anything the config does not pin. These were hardcoded here
    # once; every one of them also exists in the config, so a hardcoded copy is a
    # second source of truth that drifts silently -- the binarized mels would
    # stop matching the mels the vocoder and the evaluator use, with nothing
    # raising. Read from hparams, fall back to these.
    _T2M_DEFAULTS = {'fft_size': 1024, 'hop_size': 256, 'win_size': 1024,
                     'audio_num_mel_bins': 80, 'fmin': 55, 'fmax': 7600,
                     'f0_min': 80, 'f0_max': 600, 'pitch_extractor': 'parselmouth',
                     'audio_sample_rate': 22050, 'loud_norm': False,
                     'mfa_min_sil_duration': 0.1, 'trim_eos_bos': False,
                     'with_align': True, 'text2mel_params': False,
                     'with_f0': True, 'min_mel_length': 64,
                     'mel_type': 'fluenteditor'}

    def __init__(self):
        # Was hardcoded 'vctk'. With VCTK deleted, that made every LibriTTS run
        # read an EMPTY metadata.json and report "0it" three times over a clean
        # exit 0 -- a successful-looking no-op that produced no binary data.
        self.dataset_name = hparams.get('ds_name') or 'vctk'
        self.processed_data_dir = (hparams.get('processed_data_dir')
                                   or f'data/processed/{self.dataset_name}')
        self.binary_data_dir = (hparams.get('binary_data_dir')
                                or f'data/binary/{self.dataset_name}')
        self.items = {}
        self.item_names = []
        self._split_cache = None
        self.shuffle = False
        self.with_spk_embed = True
        self.with_wav = False

        self.text2mel_params = {k: hparams.get(k, v)
                                for k, v in self._T2M_DEFAULTS.items()}
        self.text2mel_params['dataset_name'] = self.dataset_name

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        if not items_list:
            raise BinarizationError(
                f"{processed_data_dir}/metadata.json is empty. Binarizing nothing "
                f"exits 0 and writes no data; run the preprocessor for "
                f"ds_name={self.dataset_name} first.")
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.shuffle:
            random.seed(SPLIT_SEED)
            random.shuffle(self.item_names)

    def _splits(self):
        """Seeded, reproducible, mutually disjoint train/valid/test split.

        Upstream sliced item_names by a hardcoded index ([4182,-1] train,
        [0,4182] for BOTH valid and test) over filesystem glob order, so valid
        and test were identical and the split changed between machines. We sort
        first to kill the glob-order dependence, then shuffle with a fixed seed.
        The resulting lists are written to split.json so the grader can score
        the exact same held-out utterances the model never trained on.
        """
        if self._split_cache is None:
            names = sorted(self.item_names)
            random.Random(SPLIT_SEED).shuffle(names)
            n = len(names)
            n_test = max(1, int(round(n * TEST_FRAC)))
            n_valid = max(1, int(round(n * VALID_FRAC)))
            self._split_cache = {
                'test': sorted(names[:n_test]),
                'valid': sorted(names[n_test:n_test + n_valid]),
                'train': sorted(names[n_test + n_valid:]),
            }
            assert not (set(self._split_cache['valid']) & set(self._split_cache['test']))
            assert not (set(self._split_cache['train']) & set(self._split_cache['test']))
            assert not (set(self._split_cache['train']) & set(self._split_cache['valid']))
        return self._split_cache

    @property
    def train_item_names(self):
        return self._splits()['train']

    @property
    def valid_item_names(self):
        return self._splits()['valid']

    @property
    def test_item_names(self):
        return self._splits()['test']

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process(self):
        self.load_meta_data()
        os.makedirs(self.binary_data_dir, exist_ok=True)
        # Upstream had this commented out, which leaves the binary dir without
        # phone_set.json / word_set.json / spk_map.json -- and speech_base.py:42
        # builds the token encoder from `{binary_data_dir}/phone_set.json`, so
        # training dies at task construction. Restored.
        for fn in ['phone_set.json', 'word_set.json', 'spk_map.json']:
            remove_file(f"{self.binary_data_dir}/{fn}")
            copy_file(f"{self.processed_data_dir}/{fn}", f"{self.binary_data_dir}/{fn}")
        splits = self._splits()
        with open(f'{self.binary_data_dir}/split.json', 'w') as f:
            json.dump({'seed': SPLIT_SEED, 'valid_frac': VALID_FRAC, 'test_frac': TEST_FRAC,
                       **splits}, f, indent=1)
        print(f"| split: train={len(splits['train'])} "
              f"valid={len(splits['valid'])} test={len(splits['test'])} (seed {SPLIT_SEED})")
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = self.binary_data_dir
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item)
        ph_lengths = []
        mel_lengths = []
        total_sec = 0
        items = []
        args = [{'item': item, 'text2mel_params': self.text2mel_params} for item in meta_data]
        for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
            if item is not None:
                items.append(item)
        if self.with_spk_embed:
            args = [{'wav': item['wav']} for item in items]
            for item_id, spk_embed in multiprocess_run_tqdm(
                    self.get_spk_embed, args,
                    init_ctx_func=lambda wid: {'voice_encoder': VoiceEncoder().cuda()}, num_workers=2,
                    desc='Extracting spk embed'):
                items[item_id]['spk_embed'] = spk_embed
                if spk_embed is None:
                    del items[item_id]

        for item in items:
            if not self.with_wav and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            mel_lengths.append(item['len'])
            assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, text2mel_params):
        item['ph_len'] = len(item['ph_token'])
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, text2mel_params)
        if len(mel) < text2mel_params['min_mel_length']:
            return None
        try:
            # alignments
            n_bos_frames, n_eos_frames = 0, 0
            if text2mel_params['with_align']:
                tg_fn = f"data/processed/{text2mel_params['dataset_name']}/mfa_outputs/{item_name}.TextGrid"
                item['tg_fn'] = tg_fn
                cls.process_align(tg_fn, item, text2mel_params)
                if text2mel_params['trim_eos_bos']:
                    n_bos_frames = item['dur'][0]
                    n_eos_frames = item['dur'][-1]
                    T = len(mel)
                    item['mel'] = mel[n_bos_frames:T - n_eos_frames]
                    item['mel2ph'] = item['mel2ph'][n_bos_frames:T - n_eos_frames]
                    item['mel2word'] = item['mel2word'][n_bos_frames:T - n_eos_frames]
                    item['dur'] = item['dur'][1:-1]
                    item['dur_word'] = item['dur_word'][1:-1]
                    item['len'] = item['mel'].shape[0]
                    item['wav'] = wav[n_bos_frames * text2mel_params['hop_size']:len(wav) - n_eos_frames * text2mel_params['hop_size']]
            if text2mel_params['with_f0']:
                cls.process_pitch(item, n_bos_frames, n_eos_frames, text2mel_params)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item

    @classmethod
    def process_audio(cls, wav_fn, res, text2mel_params):
        # The mel convention is part of the DECODER contract, not a free choice.
        # BigVGAN's generator was trained on natural-log, un-normalised,
        # slaney-filterbank, centre=False mel; FluentEditor's is log10 with
        # reference-level and min-level-dB normalisation. Binarising with one and
        # decoding with the other produces noise, not degraded audio, and nothing
        # about the shapes or the loss curve reveals it.
        if text2mel_params.get('mel_type') == 'bigvgan':
            import librosa
            from wavepainter.tasks.vocoder_infer.bigvgan import bigvgan_mel
            wav, _ = librosa.core.load(wav_fn, sr=text2mel_params['audio_sample_rate'])
            mel = bigvgan_mel(wav, text2mel_params)
            # librosa_wav2spec trims the wav to a whole number of hops; mirror
            # that so len(mel) * hop == len(wav) and mel2ph stays aligned.
            wav = wav[:mel.shape[0] * text2mel_params['hop_size']]
            res.update({'mel': mel, 'wav': wav.astype(np.float16),
                        'sec': len(wav) / text2mel_params['audio_sample_rate'],
                        'len': mel.shape[0]})
            return wav, mel
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=text2mel_params['fft_size'],
            hop_size=text2mel_params['hop_size'],
            win_length=text2mel_params['win_size'],
            num_mels=text2mel_params['audio_num_mel_bins'],
            fmin=text2mel_params['fmin'],
            fmax=text2mel_params['fmax'],
            sample_rate=text2mel_params['audio_sample_rate'],
            loud_norm=text2mel_params['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / text2mel_params['audio_sample_rate'], 'len': mel.shape[0]})
        return wav, mel
    

    @staticmethod
    def process_align(tg_fn, item, text2mel_params):
        ph = item['ph']
        mel = item['mel']
        ph_token = item['ph_token']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],
                                     text2mel_params['mfa_min_sil_duration'])
        else:
            raise BinarizationError(f"Align not found")
        if np.array(mel2ph).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}")
        item['mel2ph'] = mel2ph
        item['dur'] = dur

        ph2word = item['ph2word']
        mel2word = [ph2word[p - 1] for p in item['mel2ph']]
        item['mel2word'] = mel2word  # [T_mel]
        dur_word = mel2token_to_dur(mel2word, len(item['word_token']))
        item['dur_word'] = dur_word.tolist()  # [T_word]

    @staticmethod
    def process_pitch(item, n_bos_frames, n_eos_frames, text2mel_params):
        wav, mel = item['wav'], item['mel']
        f0 = extract_pitch(text2mel_params['pitch_extractor'], wav,
                         text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],
                         f0_min=text2mel_params['f0_min'], f0_max=text2mel_params['f0_max'])
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0), (len(mel), len(f0))
        pitch_coarse = f0_to_coarse(f0)
        item['f0'] = f0
        item['pitch'] = pitch_coarse

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx['voice_encoder'].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))


if __name__ == '__main__':
    # set_hparams() is what makes --config take effect; without it hparams is an
    # empty dict and every lookup above falls back to the VCTK-era defaults.
    from wavepainter.runtime.hparams import set_hparams

    set_hparams()
    BaseBinarizer().process()
