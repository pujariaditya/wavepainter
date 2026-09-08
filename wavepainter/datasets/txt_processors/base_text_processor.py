from wavepainter.text.text_encoder import is_sil_phoneme





class BaseTxtProcessor:

    @classmethod
    def process(cls, txt):
        raise NotImplementedError

    @classmethod
    def postprocess(cls, txt_struct):
        # remove sil phoneme in head and tail
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[0][0]):
            txt_struct = txt_struct[1:]
        while len(txt_struct) > 0 and is_sil_phoneme(txt_struct[-1][0]):
            txt_struct = txt_struct[:-1]
        if True:
            txt_struct = cls.add_bdr(txt_struct)
        if True:
            txt_struct = [["<BOS>", ["<BOS>"]]] + txt_struct + [["<EOS>", ["<EOS>"]]]
        return txt_struct

    @classmethod
    def add_bdr(cls, txt_struct):
        txt_struct_ = []
        for i, ts in enumerate(txt_struct):
            txt_struct_.append(ts)
            if i != len(txt_struct) - 1 and \
                    not is_sil_phoneme(txt_struct[i][0]) and not is_sil_phoneme(txt_struct[i + 1][0]):
                txt_struct_.append(['|', ['|']])
        return txt_struct_
