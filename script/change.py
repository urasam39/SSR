import os
import wave
import sys
from tqdm import tqdm
import math
import numpy as np
import array
from scipy import signal, interpolate
from prepro import save_wav


if __name__ == "__main__":
    # data = np.load("/home/data/urasam/sounds/VCTKnumpy/xtrain.npy")
    # data = np.load("../output/srhgann.npy")
    data = np.load("../output/cuhgann.npy")
    print(data.shape)
    result = np.zeros((7*48000, 1))
    print(result.shape)
    for i in range(7*48000/6000):
        for j in range(6000):
            result[i*6000+j] = data[i, j]
    save_wav(result, "../output/", "cuhgann.wav", 48000)
