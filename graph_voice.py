# from msilib import sequence
import sys
from termios import FF1
sys.path.append('/home/elizabeth.gooch/lizthesis')
import needed_functions as needed_functions

from email.policy import default
import torchaudio
from torch.utils import data
import torch
import os
import pickle
import random
import itertools

torchaudio.set_audio_backend("sox_io")

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wave 
import librosa
import scipy.io.wavfile as wavfile

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
# Looking at https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial
import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
# import IPython.display as ipd
import matplotlib.pyplot as plt


# locations needed in file
parser = argparse.ArgumentParser()
parser.add_argument('--vox_directory', 
                    default='/home/elizabeth.gooch/data/VoxCeleb_data/VoxCeleb/',
                    help="Directory containing the dataset")
parser.add_argument('--alsaify_directory', 
                    default='/home/elizabeth.gooch/data/zw4p4p7sdh-1/',
                    help="Directory containing the dataset")
parser.add_argument('--pickle_directory', 
                    default='/home/elizabeth.gooch/lizthesis/data',
                    help='Directory containing the annotations')
parser.add_argument('--alsaify_pickle_directory', 
                    default='/home/elizabeth.gooch/lizthesis/AlsaifyData',
                    help='Directory containing the Alsaify annotations')
parser.add_argument('--params_directory', 
                    default='/home/elizabeth.gooch/lizthesis/experiments/base_model',
                    help='Directory containing the params JSON')
parser.add_argument('--tables_and_figures_dir', default='/home/elizabeth.gooch/lizthesis/output',
                    help='Directory containing results as figures')
args = parser.parse_args()

same_speaker4 = "samePhrase/4/4-10.wav"
# speaker 4 is a 21 year old female, audio 1-10 files
audio_file = os.path.join(args.alsaify_directory, same_speaker4)
sample_rate, audio = wavfile.read(audio_file)

print("Sample rate: {0}Hz".format(sample_rate))
print("Audio duration: {0}s".format(len(audio) / sample_rate))


def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

audio = normalize_audio(audio)
plt.figure(figsize=(15,4))
plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
plt.grid(True)
plt.savefig(os.path.join(args.tables_and_figures_dir, 'normalized_audio_same_speaker4.png'))


# Audio framing
def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=48000):
    # hop_size in ms
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames

hop_size = 15
FFT_size = 4096

audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
print("Framed audio shape: {0}".format(audio_framed.shape))

# Convert to frequency domain
window = get_window("hann", FFT_size, fftbins=True)
plt.figure(figsize=(15,4))
plt.plot(window)
plt.grid(True)
plt.savefig(os.path.join(args.tables_and_figures_dir, 'window_hann.png'))

window_h = get_window("hamming", FFT_size, fftbins=True)
plt.figure(figsize=(15,4))
plt.plot(window_h)
plt.grid(True)
plt.savefig(os.path.join(args.tables_and_figures_dir, 'window_hamming.png'))


audio_win = audio_framed * window

ind = 69
plt.figure(figsize=(15,6))
plt.subplot(2, 1, 1)
plt.plot(audio_framed[ind])
plt.title('Original Frame')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(audio_win[ind])
plt.title('Frame After Windowing')
plt.grid(True)
plt.savefig(os.path.join(args.tables_and_figures_dir, 'windowing_speaker4.png'))


audio_winT = np.transpose(audio_win)

audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

for n in range(audio_fft.shape[1]):
    audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

audio_fft = np.transpose(audio_fft)

# Calculate signal power
audio_power = np.square(np.abs(audio_fft))
print("Signal power shape: {0}".format(audio_power.shape))


# Mel-spaced filterbank
freq_min = 0
freq_high = sample_rate / 2
mel_filter_num = 10

print("Minimum frequency: {0}".format(freq_min))
print("Maximum frequency: {0}".format(freq_high))


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=48000):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = mel_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=48000)
filter_points

# Construct filter bank
def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

filters = get_filters(filter_points, FFT_size)

plt.figure(figsize=(15,4))
for n in range(filters.shape[0]):
    plt.plot(filters[n])
plt.savefig(os.path.join(args.tables_and_figures_dir, 'filters10.png'))

# taken from the librosa library
enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
filters *= enorm[:, np.newaxis]

plt.figure(figsize=(15,4))
for n in range(filters.shape[0]):
    plt.plot(filters[n])
plt.savefig(os.path.join(args.tables_and_figures_dir, 'filters10_norm.png'))

# Filter the signal
audio_filtered = np.dot(filters, np.transpose(audio_power))
audio_log = 10.0 * np.log10(audio_filtered)
audio_log.shape

# Generate the cepstral coefficients
def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

dct_filter_num = 40

dct_filters = dct(dct_filter_num, mel_filter_num)

cepstral_coefficents = np.dot(dct_filters, audio_log)
print("Cepstral coefficient shape: {0}".format(cepstral_coefficents.shape))

print("Cepstral coefficients array: {0}".format(cepstral_coefficents[:, 100]))

plt.figure(figsize=(15,5))
plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
plt.imshow(cepstral_coefficents, aspect='auto', origin='lower');
plt.savefig(os.path.join(args.tables_and_figures_dir, 'cepstral40_speaker4.png'))

x=1