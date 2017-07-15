import os
import wave
import sys
from tqdm import tqdm
import math
import numpy as np
import array
from scipy import signal, interpolate, fromstring
from scipy.io.wavfile import read
from prepro import save_numpy, save_wav
import wavio


if __name__ == '__main__':
    # w = wavio.read("/home/data/urasam/sounds/hgann.wav")
    # fs = w.rate
    # bit = 8 * w.sampwidth
    # data = w.data.T
    # data = data / float( 2**(bit-1) )
    # x = data[0]
    # print(x.max())
    wave_file = wave.open("/home/data/urasam/sounds/hgann16.wav")
    print "Channel num : ", wave_file.getnchannels()
    print "Sample size : ", wave_file.getsampwidth()
    print "Sampling rate : ", wave_file.getframerate()
    print "Frame num : ", wave_file.getnframes()
    print "Prams : ", wave_file.getparams()
    print "Sec : ", float(wave_file.getnframes()) / wave_file.getframerate()

    samplrate = 24000
    x = wave_file.readframes(wave_file.getnframes())
    x = fromstring(x, dtype="int16")
    x = x[::2]

    save_wav(x[samplrate*3:samplrate*10], "../output/", "hgannvoice.wav", 24000)
