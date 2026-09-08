"""Corpus -> phonemised, speaker-mapped items ready for binarisation.

Two passes over the corpus, then MFA inputs:

  1. **first pass** (per item, parallel): phonemise the transcript and link the
     audio into a flat working directory.
  2. **vocabularies**: build the phone set, the word set and the speaker map
     from everything seen in pass 1.
  3. **second pass** (per item, parallel): encode phones, words and speaker to
     integer ids using those vocabularies.
  4. **MFA inputs**: write one `.lab` per item holding the phones grouped by
     word, plus the pronunciation dictionary the aligner needs.

The result is `metadata.json`, which `BaseBinarizer` consumes.

TWO METHODS ARE ALSO ON THE EVALUATION PATH and are not preprocessing-only:
``txt_to_ph`` phonemises at synthesis time, and ``load_dict`` loads the
vocabularies. ``txt_to_ph``'s five-value return is destructured by
``wavepainter.evaluate``, so its shape is a contract.

ph2word IS 1-INDEXED. Word id 0 is reserved for padding, so the n-th word is
id n+1. Alignment code gathers through a mask left-padded by one to match.
"""

import json
import os
import random
import re
import traceback
from collections import Counter
from functools import partial

from tqdm import tqdm

from wavepainter.datasets.txt_processors.en import TxtProcessor
from wavepainter.runtime.multiprocess_utils import multiprocess_run_tqdm
from wavepainter.runtime.os_utils import link_file, move_file, remove_file
from wavepainter.text.text_encoder import build_token_encoder, is_sil_phoneme


class BasePreprocessor:
    """Phonemisation and vocabulary construction for a speech corpus."""

    def __init__(self, dataset_name=None):
        self.txt_processor = TxtProcessor()
        # Taken explicitly rather than hardcoded: a fixed corpus name meant
        # passing --config silently preprocessed a different (absent) corpus
        # and reported "0it" instead of failing.
        self.dataset_name = dataset_name or 'libritts'
        self.raw_data_dir = f'data/raw/{self.dataset_name}'
        self.processed_dir = f'data/processed/{self.dataset_name}'
        self.spk_map_fn = f"{self.processed_dir}/spk_map.json"

        self.reset_phone_dict = True
        self.reset_word_dict = True
        self.word_dict_size = 12500
        # Headroom over LibriTTS train-clean-100's speaker count; the map is
        # asserted against this so an unexpected corpus fails loudly.
        self.num_spk = 1200
        self.use_mfa = True
        self.seed = 1234
        self.nsample_per_mfa_group = 1000
        self.mfa_group_shuffle = False

    # ------------------------------------------------------------ corpus

    def meta_data(self):
        """Yield ``{item_name, wav_fn, txt, spk_name}`` per utterance.

        LibriTTS layout: ``<raw>/<split>/<speaker>/<chapter>/<id>.wav`` beside
        an ``<id>.normalized.txt``. Sorted so the item order -- and therefore
        the MFA grouping and the binarised split -- is reproducible.
        """
        if self.dataset_name != 'libritts':
            raise ValueError(
                f"no meta_data reader for {self.dataset_name!r}; this "
                f"repository preprocesses LibriTTS. Add a branch here for "
                f"another corpus.")

        from glob import glob
        for wav_fn in sorted(glob(f'{self.raw_data_dir}/*/*/*/*.wav')):
            item_name = os.path.basename(wav_fn)[:-4]
            with open(f'{wav_fn[:-4]}.normalized.txt') as f:
                txt = f.read()
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt,
                   'spk_name': item_name.split('_')[0]}

    # -------------------------------------------------------------- text

    @staticmethod
    def txt_to_ph(txt_processor, txt_raw):
        """Transcript -> (phones, normalised text, words, ph2word, ph-by-word).

        ``ph2word`` maps each phone to its word, **1-indexed** because 0 is the
        padding id. ``ph_gb_word`` is the same phones grouped per word with
        underscores, which is the form MFA's dictionary wants.

        The five-value return is destructured by the evaluator; do not reorder.
        """
        txt_struct, txt = txt_processor.process(txt_raw)
        phones = [p for word in txt_struct for p in word[1]]
        words = [word[0] for word in txt_struct]
        ph_gb_word = ["_".join(word[1]) for word in txt_struct]
        ph2word = [w_id + 1
                   for w_id, word in enumerate(txt_struct)
                   for _ in range(len(word[1]))]
        return " ".join(phones), txt, " ".join(words), ph2word, " ".join(ph_gb_word)

    # ------------------------------------------------------------- passes

    @classmethod
    def preprocess_first_pass(cls, item_name, txt_raw, txt_processor,
                              wav_fn, wav_processed_dir, wav_processed_tmp,
                              txt_loader=None, others=None):
        """Phonemise one item and place its audio. None on failure.

        Returning None rather than raising keeps one malformed item from
        killing a multi-hour pass over the corpus; the caller drops it.
        """
        try:
            if txt_loader is not None:
                txt_raw = txt_loader(txt_raw)
            ph, txt, word, ph2word, ph_gb_word = cls.txt_to_ph(txt_processor, txt_raw)

            os.makedirs(wav_processed_dir, exist_ok=True)
            ext = os.path.splitext(wav_fn)[1]
            new_wav_fn = f"{wav_processed_dir}/{item_name}{ext}"
            # Move only what we produced into the temp dir; anything else is
            # the user's corpus and gets linked, never relocated.
            place = move_file if os.path.dirname(wav_fn) == wav_processed_tmp else link_file
            place(wav_fn, new_wav_fn)

            return {'txt': txt, 'txt_raw': txt_raw, 'ph': ph, 'word': word,
                    'ph2word': ph2word, 'ph_gb_word': ph_gb_word,
                    'wav_fn': new_wav_fn, 'wav_align_fn': wav_fn,
                    'others': others}
        except Exception:
            traceback.print_exc()
            print(f"| Error is caught. item_name: {item_name}.")
            return None

    @classmethod
    def preprocess_second_pass(cls, word, ph, spk_name, word_encoder,
                               ph_encoder, spk_map):
        """Encode one item's phones, words and speaker to integer ids."""
        return {'word_token': word_encoder.encode(word),
                'ph_token': ph_encoder.encode(ph),
                'spk_id': spk_map[spk_name]}

    # ------------------------------------------------------- vocabularies

    def _phone_encoder(self, ph_set):
        """Build (or load) the phone set. ORDER IS THE TOKEN ID.

        Sorted so the mapping is reproducible. Rebuilding this against a
        different corpus reorders the ids and silently invalidates any
        checkpoint trained on the old ordering.
        """
        fn = f"{self.processed_dir}/phone_set.json"
        if self.reset_phone_dict or not os.path.exists(fn):
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(fn, 'w'), ensure_ascii=False)
            print("| Build phone set: ", ph_set)
        else:
            print("| Load phone set: ", json.load(open(fn)))
        return build_token_encoder(fn)

    def _word_encoder(self, word_set):
        """Build (or load) the word set: the most frequent ``word_dict_size``.

        Everything rarer becomes the unknown token. ``<BOS>``/``<EOS>`` are
        added before the sort so they occupy stable ids.
        """
        fn = f"{self.processed_dir}/word_set.json"
        if self.reset_word_dict:
            counts = Counter(word_set)
            total = sum(counts.values())
            kept = counts.most_common(self.word_dict_size)
            n_unk = total - sum(n for _, n in kept)
            vocab = sorted(set(['<BOS>', '<EOS>'] + [w for w, _ in kept]))
            json.dump(vocab, open(fn, 'w'), ensure_ascii=False)
            print(f"| Build word set. Size: {len(vocab)}, #total words: {total},"
                  f" #unk_words: {n_unk}, word_set[:10]:, {vocab[:10]}.")
        else:
            vocab = json.load(open(fn))
            print("| Load word set. Size: ", len(vocab), vocab[:10])
        return build_token_encoder(fn)

    def build_spk_map(self, spk_names):
        spk_map = {name: i for i, name in enumerate(sorted(spk_names))}
        assert len(spk_map) == 0 or len(spk_map) <= self.num_spk, len(spk_map)
        print(f"| Number of spks: {len(spk_map)}, spk_map: {spk_map}")
        json.dump(spk_map, open(self.spk_map_fn, 'w'), ensure_ascii=False)
        return spk_map


    def load_dict(self, base_dir):
        """Load the phone and word encoders. Also used by the evaluator."""
        return (build_token_encoder(f'{base_dir}/phone_set.json'),
                build_token_encoder(f'{base_dir}/word_set.json'))

    # ---------------------------------------------------------------- MFA

    @classmethod
    def build_mfa_inputs(cls, item, mfa_input_dir, mfa_group, wav_processed_tmp):
        """Write one item's aligner input: audio plus a `.lab` of its phones.

        Silence phones are stripped from the `.lab` because the aligner infers
        silence itself; leaving them in makes it align against symbols the
        acoustic model has no entry for.
        """
        item_name = item['item_name']
        wav_align_fn = item['wav_align_fn']
        group_dir = f'{mfa_input_dir}/{mfa_group}'
        os.makedirs(group_dir, exist_ok=True)

        ext = os.path.splitext(wav_align_fn)[1]
        new_wav_align_fn = f"{group_dir}/{item_name}{ext}"
        place = move_file if os.path.dirname(wav_align_fn) == wav_processed_tmp else link_file
        place(wav_align_fn, new_wav_align_fn)

        ph_gb_word_nosil = " ".join(
            "_".join(p for p in word.split("_") if not is_sil_phoneme(p))
            for word in item['ph_gb_word'].split(" ") if not is_sil_phoneme(word))
        with open(f'{group_dir}/{item_name}.lab', 'w') as f:
            f.write(ph_gb_word_nosil)
        return ph_gb_word_nosil, new_wav_align_fn

    # ------------------------------------------------------------- driver

    def process(self):
        processed_dir = self.processed_dir
        tmp_dir = f'{processed_dir}/processed_tmp'
        wav_dir = f'{processed_dir}/{self.wav_processed_dirname}'
        for d in (tmp_dir, wav_dir):
            remove_file(d)
            os.makedirs(d, exist_ok=True)

        meta = list(tqdm(self.meta_data(), desc='Load meta data'))
        names = [d['item_name'] for d in meta]
        assert len(names) == len(set(names)), 'Key `item_name` should be Unique.'

        # --- pass 1: phonemise and place audio
        first = partial(self.preprocess_first_pass,
                        txt_processor=self.txt_processor,
                        wav_processed_dir=wav_dir, wav_processed_tmp=tmp_dir)
        args = [{'item_name': m['item_name'], 'txt_raw': m['txt'],
                 'wav_fn': m['wav_fn'], 'txt_loader': m.get('txt_loader'),
                 'others': m.get('others')} for m in meta]

        items, phone_list, word_list, spk_names = [], [], [], set()
        for raw, (item_id, out) in zip(meta, multiprocess_run_tqdm(
                first, args, desc='Preprocess')):
            if out is None:
                continue
            raw.update(out)
            raw.pop('txt_loader', None)
            raw['id'] = item_id
            raw['spk_name'] = raw.get('spk_name', '<SINGLE_SPK>')
            raw['others'] = raw.get('others')
            phone_list += raw['ph'].split(" ")
            word_list += raw['word'].split(" ")
            spk_names.add(raw['spk_name'])
            items.append(raw)

        # --- vocabularies, then pass 2: encode to ids
        ph_encoder = self._phone_encoder(phone_list)
        word_encoder = self._word_encoder(word_list)
        spk_map = self.build_spk_map(spk_names)
        args = [{'ph': it['ph'], 'word': it['word'], 'spk_name': it['spk_name'],
                 'word_encoder': word_encoder, 'ph_encoder': ph_encoder,
                 'spk_map': spk_map} for it in items]
        for idx, encoded in multiprocess_run_tqdm(
                self.preprocess_second_pass, args, desc='Add encoded tokens'):
            items[idx].update(encoded)

        # --- aligner inputs
        if self.use_mfa:
            mfa_input_dir = f'{processed_dir}/mfa_inputs'
            remove_file(mfa_input_dir)
            # Grouped so the aligner can be run in parallel over subdirectories.
            groups = [i // self.nsample_per_mfa_group for i in range(len(items))]
            if self.mfa_group_shuffle:
                random.seed(self.seed)
                random.shuffle(groups)

            args = [{'item': it, 'mfa_input_dir': mfa_input_dir,
                     'mfa_group': g, 'wav_processed_tmp': tmp_dir}
                    for it, g in zip(items, groups)]
            mfa_dict = set()
            for i, (ph_gb_word_nosil, new_wav_align_fn) in multiprocess_run_tqdm(
                    self.build_mfa_inputs, args, desc='Build MFA data'):
                items[i]['wav_align_fn'] = new_wav_align_fn
                # Dictionary entry: the grouped token, then its phones spaced.
                for w in ph_gb_word_nosil.split(" "):
                    mfa_dict.add(f"{w} {w.replace('_', ' ')}")
            with open(f'{processed_dir}/mfa_dict.txt', 'w') as f:
                f.writelines(f'{line}\n' for line in sorted(mfa_dict))

        with open(f"{processed_dir}/{self.meta_csv_filename}.json", 'w') as f:
            # Collapse the indent before list continuations so the file stays
            # readable without ballooning to one line per scalar.
            f.write(re.sub(r'\n\s+([\d+\]])', r'\1',
                           json.dumps(items, ensure_ascii=False,
                                      sort_keys=False, indent=1)))
        remove_file(tmp_dir)

    @property
    def meta_csv_filename(self):
        return 'metadata'

    @property
    def wav_processed_dirname(self):
        return 'wav_processed'


if __name__ == '__main__':
    import sys
    BasePreprocessor(sys.argv[1] if len(sys.argv) > 1 else None).process()
