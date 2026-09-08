from wavepainter.runtime.hparams import hparams


def parse_dataset_configs():
    max_tokens = hparams['max_tokens']
    max_sentences = hparams['max_sentences']
    max_valid_tokens = hparams['max_valid_tokens']
    if max_valid_tokens == -1:
        hparams['max_valid_tokens'] = max_valid_tokens = max_tokens
    max_valid_sentences = hparams['max_valid_sentences']
    if max_valid_sentences == -1:
        hparams['max_valid_sentences'] = max_valid_sentences = max_sentences
    return max_tokens, max_sentences, max_valid_tokens, max_valid_sentences


def parse_mel_losses():
    mel_losses = hparams['mel_losses'].split("|")
    loss_and_lambda = {}
    for i, l in enumerate(mel_losses):
        if l == '':
            continue
        if ':' in l:
            l, lbd = l.split(":")
            lbd = float(lbd)
        else:
            lbd = 1.0
        loss_and_lambda[l] = lbd
    print("| Mel losses:", loss_and_lambda)
    return loss_and_lambda


