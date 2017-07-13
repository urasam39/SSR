#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import sys
matplotlib.use("Agg")


def sinenois():
    # wavfile = "/home/data/urasam/sounds/sineWithNoise.wav"
    if len(sys.argv) != 2:
        print("usage: python audio.py [sound file]")
        sys.exit(1)
    wavfile = sys.argv[1]
    fs, data = read(wavfile)

    print("Sampling rate :", fs)
    if (len(data.shape) == 1):
        plt.plot(data)
        plt.savefig("graph.png")
    if (len(data.shape) == 2 and data.shape[1] == 2):
        left = data[:, 0]
        # right = data[:, 1]
        print(left[0:100, ].shape)
        plt.plot(left[0:100000, ])
        plt.savefig("left.png")


if __name__ == '__main__':
    sinenois()
