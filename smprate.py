from fractions import Fraction
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as sg
import soundfile as sf

if __name__ == '__main__':
    fs_target = 44100
    cutoff_hz = 21000.0
    n_lpf = 4096

    sec = 10

    wav, fs_src = sf.read("/home/data/urasam/sounds/sineWithNoise.wav")
    wav_48kHz = wav[:fs_src * sec]

    frac = Fraction(fs_target, fs_src)

    up = frac.numerator
    down = frac.denominator

    wav_up = np.zeros(np.alen(wav_48kHz) * up)
    wav_up[::up] = up * wav_48kHz
    fs_up = fs_src * up

    cutoff = cutoff_hz / (fs_up / 2.0)
    lpf = sg.firwin(n_lpf, cutoff)

    wav_down = sg.lfilter(lpf, [1], wav_up)[n_lpf // 2::down]

    sf.write("down.wav", wav_down, fs_target)

    w, h = sg.freqz(lpf, a=1, worN=1024)
    f = fs_up * w / (2.0 * np.pi)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.semilogx(f, 20.0 * np.log10(np.abs(h)))
    ax.axvline(fs_target, color="r")
    ax.set_ylim([-80.0, 10.0])
    ax.set_xlim([3000.0, fs_target + 5000.0])
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("power [dB]")
    plt.savefig("hoge.png")
