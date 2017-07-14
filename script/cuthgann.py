import os
import wave
import sys
from tqdm import tqdm
import math
import numpy as np
import array
from scipy import signal, interpolate
from prepro import save_wav


if __name__ == '__main__':
    data = "/home/data/urasam/sounds/hgann.wav"
    wavfile = wave.open(data)
    x = wavfile.readframes(wavfile.getframerate())
    x = np.frombuffer(x, dtype="int16")
    print(x.shape)
