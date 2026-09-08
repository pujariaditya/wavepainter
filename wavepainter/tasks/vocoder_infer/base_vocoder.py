REGISTERED_VOCODERS = {}


def register_vocoder(name):
    def _f(cls):
        REGISTERED_VOCODERS[name] = cls
        return cls

    return _f


def get_vocoder_cls(vocoder_name):
    return REGISTERED_VOCODERS.get(vocoder_name)


class BaseVocoder:
    def spec2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError


