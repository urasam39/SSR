import os
import wave
import sys
from tqdm import tqdm
import math
import numpy as np
import array
from scipy import signal, interpolate

def up_sample_cubic(x):
    # imput:np.array low data, return:np.array hith data
    x_high = []
    for i in tqdm(range(len(x))):
        # print(x[i,:].shape)
        y_interf = interpolate.CubicSpline(np.linspace(0, data_length/2-1, data_length/2),
                                           x[i, :])
        y_inter = y_interf(np.linspace(0, data_length/2-1, data_length)).astype(np.int16)
        x_high.append(y_inter)
    return np.array(x_high)


if __name__ == "__main__":
    filename = "../output/hgannvoice.wav"
    data_length = 6000
    wave_file = wave.open(filename)
    x = wave_file.readframes(wave_file.getnframes())
    x = np.frombuffer(x, dtype="int16")
    audio = []
    for i in range(len(x)/3000):
        audio.append(x[(i*3000):((i+1)*3000)])
    audio = np.array(audio)
    print(audio)
    print(audio.shape)
    audio_up = up_sample_cubic(audio)
    print(audio_up.shape)

    np.save("../output/cuhgann.npy", audio_up)
