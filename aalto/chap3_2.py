# https://speechprocessingbook.aalto.fi/Representations/Waveform.html

import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import scipy.fft
import numpy as np
import argparse
import os

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
parser.add_argument('--aalto', default='/home/elizabeth.gooch/lizthesis/aalto',
                    help='Directory containing aalot results as figures')
args = parser.parse_args()


same_speaker4 = "samePhrase/4/4-10.wav"
# speaker 4 is a 21 year old female, audio 1-10 files
filename = os.path.join(args.alsaify_directory, same_speaker4)
fs, data = wavfile.read(filename)

data = data.astype(np.int16)

fig = plt.figure(figsize=(12, 4))
ax = fig.subplots(nrows=1,ncols=2)
t = np.arange(0.,len(data))/fs
ax[0].plot(t,data)
ax[0].set_title('Original with 16 bit accuracy and ' + str(fs/1000) + ' kHz sampling rate')
ax[1].plot(t,data)
ax[1].set_title('Original zoomed in (16 bit / ' + str(fs/1000) + ' kHz)')
ax[1].set(xlim=(0.5,0.52),ylim=(-7000,8000))
ax[0].set_xlabel('Time (s)')
ax[1].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
plt.savefig(os.path.join(args.aalto, '3_3 sound sample.png'))
plt.clf()

