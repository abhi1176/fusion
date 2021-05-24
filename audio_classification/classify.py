
import librosa
import numpy as np
import os
import pandas as pd

from glob import glob
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile

nfilt = 26
# Window size is 25 ms.
# 1 sec / 40 points = 25 ms
# Sampling freq = 44,100
# 44,100 / 40 = 1,102.5
nfft =  1103
numcep = 13  # Typically half of nfilt
threshold = 0.0005


def calc_fft(signal, rate):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(signal)/n)
    return (Y, freq)


def envelop(signal, rate, threshold):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


input_dir = "speaker"
clean_files = "speaker_clean"

classes = os.listdir(input_dir)

features = []

for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        rate, signal = wavfile.read(file_path)

        # Get rid of dead space in the audio
        mask = envelop(signal, rate, threshold)
        signal = signal[mask]
        wavfile.write(os.path.join(clean_files, folder, file), rate=rate, data=signal)

        length = signal.shape[0]/rate  # Gives length of the signal in seconds
        fft = calc_fft(signal, rate)
        bank = logfbank(signal[:rate], rate, nfilt=nfilt, nfft=nfft).T
        mel = mfcc(signal[:rate], rate, numcep=numcep, nfilt=nfilt, nfft=nfft)

n_samples = 2 * int(df.length.sum()/0.1)
choice = np.random.choice(classes)
