from numba.core.types import scalars
import numpy as np
import os
import librosa
import soundfile as sf
from scipy import signal


class Attack:
    def __init__(self) -> None:
        self.name = 'attack'

    def _filename(self, f):
        dir = os.path.dirname(f)
        fn = os.path.basename(f)
        fna, ext = os.path.splitext(fn)
        return dir, fna, ext

    def _load(self, f):
        sr = librosa.get_samplerate(f)
        x, sr = librosa.core.load(f, sr, False)
        return x, sr

    def _save(self, f, x, sr):
        if self.name[:4] == '_mp3':
            dir, fna, ext = self._filename(f)
            outname = fna + self.name
            out = os.path.join(dir, outname + '.mp3')
            return out
        elif self.name[:4] == '_cro':
            dir, fna, ext = self._filename(f)
            outname = fna + self.name
            out = os.path.join(dir, outname + ext)
            return out
        else:
            dir, fna, ext = self._filename(f)
            outname = fna + self.name
            out = os.path.join(dir, outname + ext)
            sf.write(out, x.T, sr)
            return out


class Closed_loop(Attack):
    def __init__(self) -> None:
        super(Closed_loop, self).__init__()
        self.name = '_none'

    def __call__(self, f):
        return f


class Noise(Attack):
    def __init__(self, snr) -> None:
        super(Noise, self).__init__()
        self.name = '_noise_' + str(snr)
        self.snr = snr

    def __call__(self, f):
        x, sr = self._load(f)
        snr = 10 ** (self.snr / 10)
        xpower = np.sum(x ** 2) / x.size
        npower = xpower / snr

        n = np.random.normal(0, np.sqrt(npower), x.shape)
        y = np.asfarray(x + n)

        return self._save(f, y, sr)


class Amplitude(Attack):
    def __init__(self, scale) -> None:
        super(Amplitude, self).__init__()
        self.name = '_scale_' + str(scale)
        self.scale = scale

    def __call__(self, f):
        x, sr = self._load(f)
        y = x * self.scale

        return self._save(f, y, sr)


class Requanzation(Attack):
    def __init__(self) -> None:
        super(Requanzation, self).__init__()
        self.name = '_quantization'

    def __call__(self, f):
        x, sr = self._load(f)
        y = x.astype(np.float16)
        y = y.astype(np.float32)

        return self._save(f, y, sr)


class Resample(Attack):
    def __init__(self, tsr) -> None:
        super(Resample, self).__init__()
        self.name = '_resample_' + str(tsr)
        self.tsr = tsr

    def __call__(self, f):
        x, sr = self._load(f)
        y = librosa.resample(x, sr, self.tsr)
        xx = librosa.resample(y, self.tsr, sr)
        return self._save(f, xx, sr)


class Mp3_compression(Attack):
    def __init__(self, bps) -> None:
        super(Mp3_compression, self).__init__()
        self.name = '_mp3_' + str(bps)
        self.bps = bps

    def __call__(self, f):
        x, sr = self._load(f)
        out = self._save(f, x, sr)
        cmd = 'ffmpeg -i ' + f + ' -acodec mp3 -ab ' + \
            str(self.bps) + 'k ' + out + ' -y'
        os.system(cmd)
        return out


class Lowpass(Attack):
    def __init__(self, cutfreq) -> None:
        super(Lowpass, self).__init__()
        self.name = '_lowpass_' + str(cutfreq)
        self.cutfreq = cutfreq

    def __call__(self, f):
        x, sr = self._load(f)
        if 0 < self.cutfreq < 1:
            b, a = signal.butter(8, self.cutfreq, 'lowpass')
        else:
            wn = self.cutfreq / (sr / 2)
            b, a = signal.butter(8, wn, 'lowpass')

        if x.ndim > 1:
            pass
        else:
            x = x.reshape((1, -1))

        y = signal.filtfilt(b, a, x)

        return self._save(f, y, sr)


class Cropping(Attack):
    def __init__(self, ss, duration):
        super(Cropping, self).__init__()
        self.name = '_cropping_' + str(ss) + '_' + str(duration)
        self.ss = ss
        self.duration = duration

    def __call__(self, f):
        x, sr = self._load(f)
        out = self._save(f, x, sr)
        cmd = 'ffmpeg -i ' + f + ' -ss ' + \
            str(self.ss) + ' -t ' + str(self.duration) + ' ' + out + ' -y'
        os.system(cmd)
        return out
