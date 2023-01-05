# https://speechprocessingbook.aalto.fi/Representations/Short-time_analysis.html

import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import scipy.fft
import numpy as np
import argparse
import os
import pandas as pd
import torch, torchaudio
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
parser.add_argument('--tables_and_figures_dir', default='/home/elizabeth.gooch/lizthesis/output',
                    help='Directory containing results as figures')

args = parser.parse_args()


def get_feature_extractor(data):
    """
    Args:
        method: (str) audio feature extraction
    Returns:
        transform method
    """



    return transform

def get_deltas(mfccs):
    delta = torchaudio.transforms.ComputeDeltas(mfcc)
    delta2 = torchaudio.transforms.ComputeDeltas(delta)
    
    return delta, delta2

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    same_speaker = "samePhrase/4/4-10.wav"
    # speaker 4 is a 21 year old female, audio 1-10 files
    # same_speaker = "samePhrase/21/21-10.wav"
    # speaker 21 is a 21 year old male, audio 1-10 files

    filename = os.path.join(args.alsaify_directory, same_speaker)
    data,fs = torchaudio.load(filename)


    sample_rate = fs
    # n_fft = 2048
    win_length = 30
    hop_length = 20
    n_mels = 20
    n_mfcc = 3 

    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            # "n_fft": n_fft,
            "n_mels": n_mels,
            "win_length": win_length,
            "hop_length": hop_length,
            "mel_scale": "htk",
            "window_fn": torch.hamming_window
        },
    )

    mfcc = transform(data)
    mfcc = torch.squeeze(mfcc)
    mfcc_np = torch.Tensor.numpy(mfcc)
    mfcc_first = mfcc_np[0, :]
    mfcc_second = mfcc_np[1, :]
    mfcc_third = mfcc_np[2, :]
    total_time = mfcc_np.shape[1]
    time = list(range(total_time))

    plt.figure(figsize=[12,8])
    plt.subplot(311)
    plt.plot(time, mfcc_first)
    # Speaker 21
    # plt.text(1500,0,'``Machine learning"')
    # plt.text(5100,0,'``1"')
    # plt.text(6450,0,'``2"')
    # plt.text(7900,0,'``3"')
    # plt.text(9300,0,'``4"')
    # plt.text(10900,0,'``5"')
    # plt.text(12500,0,'``6"')
    # plt.text(13900,0,'``7"')
    # plt.text(15400,0,'``8"')
    # plt.text(17200,0,'``9"')
    # plt.text(18150,0,'``10"')
    # Speaker 4
    plt.text(1200,-275,'``Machine learning"')
    plt.text(4400,-275,'``1"')
    plt.text(6250,-275,'``2"')
    plt.text(7700, -275,'``3"')
    plt.text(9100,-275,'``4"')
    plt.text(10600,-275,'``5"')
    plt.text(12300,-275,'``6"')
    plt.text(13700,-275,'``7"')
    plt.text(15100,-275,'``8"')
    plt.text(16400,-275,'``9"')
    plt.text(17650,-275,'``10"')
    plt.title('MFCC #1')
    plt.ylabel('Coefficient magnitude')

    plt.subplot(312)
    plt.plot(time, mfcc_second)
    plt.title('MFCC #2')
    plt.ylabel('Coefficient magnitude')

    plt.subplot(313)
    plt.plot(time, mfcc_third)
    plt.title('MFCC #3')
    plt.xlabel('Frames')
    plt.ylabel('Coefficient magnitude')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(args.tables_and_figures_dir, 'mfcc_speaker4.png'))
    plt.clf()
    x=1

# What would be a suitable window size?
 # Choose a window length in milliseconds.
 # Try to find the longest window where segments still look stationary (waveform character does not change much during segment).
# data_length = data.shape[0]

# starting_position = 0
# window_length_ms = 30
# window_length = int(window_length_ms*fs/1000)
# print('Window length in samples ' + str(window_length))
# # 19200
# # time_vector = np.linspace(0,window_length_ms,window_length)

# # Windowing functions
# zero_length = int(window_length/4)
# zero_length_ms = window_length_ms/4
# zero_vector = np.zeros([zero_length,])
# data_vector = np.concatenate((zero_vector,data[starting_position:(starting_position+window_length)],zero_vector))
# windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2
# windowing_function_extended = np.concatenate((zero_vector,windowing_function,zero_vector))

# ######################
# # Spectrogram
# ######################
# window_step = int(window_length/8) # step between analysis windows
# window_count = int(np.floor((len(data)-window_length)/window_step)+1)
# spectrum_length = int((window_length+1)/2)+1
# spectrogram = np.zeros((window_count,spectrum_length))
# time_vector = np.linspace(0,window_length_ms,window_length)
# frequency_vector = np.linspace(0,fs/2000,spectrum_length)

# for k in range(window_count):
#     starting_position = k*window_step
#     data_vector = data[starting_position:(starting_position+window_length),]
#     window_spectrum = np.abs(scipy.fft.rfft(data_vector*windowing_function))
#     spectrogram[k,:] = window_spectrum

# black_eps = 1e-1 # minimum value for the log-value in the spectrogram - helps making the background really black
    
# import matplotlib as mpl
# default_figsize = mpl.rcParamsDefault['figure.figsize']
# mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]
# plt.imshow(20*np.log10(np.abs(np.transpose(spectrogram))+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, fs/2000])
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (kHz)')
# plt.axis([0, len(data)/fs, 0, 8])
# plt.title('Spectrogram zoomed to 8 kHz')
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'speaker4_spectrogram.png'))
# plt.clf()

# #######################
# # Cepstrum preparation
# window = data_vector*windowing_function
# time_vector = np.linspace(0,window_length_ms,window_length)
# spectrum = scipy.fft.rfft(window,n=spectrum_length)
# frequency_vector = np.linspace(0,fs/2000,len(spectrum))

# # downsample to 16kHz (that is, Nyquist frequency is 8kHz, that is, everything about 8kHz can be removed)
# idx = np.nonzero(frequency_vector <= 8)
# frequency_vector = frequency_vector[idx]
# spectrum = spectrum[idx]

# logspectrum = 20*np.log10(np.abs(spectrum))
# cepstrum = scipy.fft.rfft(logspectrum)

# ctime = np.linspace(0 , 0.5*1000*spectrum_length/fs, len(cepstrum))
# cepstrogram = np.zeros((len(cepstrum),window_count))

# for k in range(window_count): 
#     starting_position = k*window_step
#     data_vector = data[starting_position:(starting_position+window_length),]
#     window = data_vector*windowing_function
#     time_vector = np.linspace(0,window_length_ms,window_length)
#     spectrum = scipy.fft.rfft(window,n=spectrum_length)
#     frequency_vector = np.linspace(0,fs/2000,len(spectrum))
#     # downsample to 16kHz (that is, Nyquist frequency is 8kHz, that is, everything about 8kHz can be removed)
#     idx = np.nonzero(frequency_vector <= 8)
#     frequency_vector = frequency_vector[idx]
#     spectrum = spectrum[idx]
#     logspectrum = 20*np.log10(np.abs(spectrum))
#     cepstrum = scipy.fft.rfft(logspectrum)
#     cepstrogram[:,k] = np.abs(cepstrum)
    
# plt.imshow(np.log(np.abs(cepstrogram)+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, ctime[-1]])
# #plt.imshow(np.log(np.abs(cepstrogram)+black_eps),aspect='auto',origin='lower')
# plt.xlabel('Time (s)')
# plt.ylabel('Quefrency (ms)')
# plt.axis([0, len(data)/fs, 0, 7.5])
# #plt.title('Spectrogram zoomed to 8 kHz')
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'speaker4_cepstrogram.png'))
# plt.clf()

# ##########################
# # Amplitude

# t = np.arange(0,len(data))/fs
# plt.figure(figsize=[12,3])
# plt.plot(t,data)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Speech sample')
# plt.show()
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'speaker4_amplitude.png'))


    


# ############
# # Log-Spectrum and Cepstrum of a window

# # choose segment from random position in sample
# # starting_position = np.random.randint(len(data) - window_length)
# starting_position = 96000
# data_vector = data[starting_position:(starting_position+window_length)]

# window = data_vector*windowing_function
# time_vector = np.linspace(0,window_length_ms,window_length)

# spectrum = scipy.fft.rfft(window,n=spectrum_length)
# frequency_vector = np.linspace(0,fs/2000,len(spectrum))

# # downsample to 16kHz (that is, Nyquist frequency is 8kHz, that is, everything about 8kHz can be removed)
# idx = np.nonzero(frequency_vector <= 8)
# frequency_vector = frequency_vector[idx]
# spectrum = spectrum[idx]

# logspectrum = 20*np.log10(np.abs(spectrum))
# cepstrum = scipy.fft.rfft(logspectrum)
# a = cepstrum.real 
# std_cepstrum =(a - np.mean(a)) / np.std(a)
# ctime = np.linspace(0,0.5*1000*spectrum_length/fs,len(cepstrum))


# plt.figure(figsize=[12,8])
# plt.subplot(321)
# plt.plot(time_vector,data_vector)
# plt.title('Original window')
# plt.xlabel('Time (ms)')
# plt.ylabel('Amplitude')

# plt.subplot(322)
# plt.plot(time_vector,window)
# plt.title('Windowed signal')
# plt.xlabel('Time (ms)')
# plt.ylabel('Amplitude')

# plt.subplot(323)
# plt.plot(frequency_vector,logspectrum)
# plt.title('Log-magnitude spectrum')
# plt.xlabel('Frequency (kHz)')
# plt.ylabel('Magnitude $20\log_{10}|X_k|$')

# plt.subplot(324)
# plt.plot(frequency_vector,logspectrum)
# plt.title('Log-magnitude spectrum (zoomed to low frequencies)')
# plt.xlabel('Frequency (kHz)')
# plt.ylabel('Magnitude $20\log_{10}|X_k|$')
# ax = plt.axis()
# ax = [0, 4, ax[2], ax[3]]
# plt.axis(ax)

# plt.subplot(325)
# plt.plot(ctime,np.abs(cepstrum))
# plt.title('Cepstrum')
# plt.xlabel('Quefrency (ms)')
# plt.ylabel('Standardized Magnitude $|C_k|$')
# ax = plt.axis()
# ax = [0, 7.5, ax[2], ax[3]]
# plt.axis(ax)

# plt.subplot(326)
# plt.plot(ctime,np.log(np.abs(cepstrum)))
# plt.title('Log-Cepstrum')
# plt.xlabel('Quefrency (ms)')
# plt.ylabel('Log-Magnitude $\log|C_k|$')
# ax = plt.axis()
# ax = [0, 7.5, ax[2], ax[3]]
# plt.axis(ax)
# plt.tight_layout()
# plt.show()
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'speaker4_spec_cep.png'))
# plt.clf()


# #####################
# # Mel scale

# def freq2mel(f): 
#     return 2595*np.log10(1 + (f/700))

# def mel2freq(m): 
#     return 700*(10**(m/2595) - 1)

# f = np.linspace(0,8000,1000)
# plt.plot(f/1000,freq2mel(f))
# plt.xlabel('Frequency (kHz)')
# plt.ylabel('Mel-scale')
# plt.title('The mel-scale as a function of frequency')
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_mel.png'))
# plt.clf()

# melbands = 10
# maxmel = freq2mel(8000)
# mel_idx = np.array(np.arange(.5,melbands,1)/melbands)*maxmel
# freq_idx = mel2freq(mel_idx)

# melfilterbank = np.zeros((len(spectrum),melbands))
# freqvec = np.arange(0,len(spectrum),1)*8000/len(spectrum)
# for k in range(melbands-2):    
#     if k>0:
#         upslope = (freqvec-freq_idx[k])/(freq_idx[k+1]-freq_idx[k])
#     else:
#         upslope = 1 + 0*freqvec
#     if k<melbands-3:
#         downslope = 1 - (freqvec-freq_idx[k+1])/(freq_idx[k+2]-freq_idx[k+1])
#     else:
#         downslope = 1 + 0*freqvec
#     triangle = np.max([0*freqvec,np.min([upslope,downslope],axis=0)],axis=0)
#     melfilterbank[:,k] = triangle
    
# melreconstruct = np.matmul(np.diag(np.sum(melfilterbank**2+1e-12,axis=0)**-1),np.transpose(melfilterbank))
    
# plt.plot(freqvec/1000,melfilterbank)
# plt.xlabel('Frequency (kHz)')
# plt.ylabel('Amplitude')
# plt.title('The mel-filterbank')
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'thesis_melbank.png'))
# plt.clf()
                           
# logmelspectrum = 10*np.log10(np.matmul(np.transpose(melfilterbank),np.abs(spectrum)**2)+1e-12)
# logenvelopespectrum = 10*np.log10(np.matmul(np.transpose(melreconstruct),10**(logmelspectrum/10)))

# plt.plot(freqvec/1000,logspectrum,label='Spectrum')
# plt.plot(freqvec/1000,logenvelopespectrum,label='Mel-envelope')
# plt.legend()
# plt.xlabel('Frequency (kHz)')
# plt.ylabel('Magnitude (dB)')
# plt.title('The mel-envelope')
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'speaker4_melenvelope.png'))
# plt.clf()

# melcepstrum = scipy.fft.rfft(logmelspectrum)
# ctime = np.linspace(0,0.5*1000*spectrum_length/fs,len(melcepstrum))

# plt.plot(ctime,np.abs(melcepstrum))
# plt.title('Cepstrum')
# plt.xlabel('Quefrency (ms)')
# plt.ylabel('Standardized Magnitude $|C_k|$')
# ax = plt.axis()
# ax = [0, 7.5, ax[2], ax[3]]
# plt.axis(ax)
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'speaker4_melcep.png'))
# plt.clf()



# mel_idx = np.array(np.arange(.5,melbands,1)/melbands)*maxmel
# freq_idx = mel2freq(mel_idx)

# melfilterbank = np.zeros((spectrogram.shape[1],melbands))
# freqvec = np.arange(0,spectrogram.shape[1],1)*8000/spectrogram.shape[1]
# for k in range(melbands-2):    
#     if k>0:
#         upslope = (freqvec-freq_idx[k])/(freq_idx[k+1]-freq_idx[k])
#     else:
#         upslope = 1 + 0*freqvec
#     if k<melbands-3:
#         downslope = 1 - (freqvec-freq_idx[k+1])/(freq_idx[k+2]-freq_idx[k+1])
#     else:
#         downslope = 1 + 0*freqvec
#     triangle = np.max([0*freqvec,np.min([upslope,downslope],axis=0)],axis=0)
#     melfilterbank[:,k] = triangle
    
# melreconstruct = np.matmul(np.diag(np.sum(melfilterbank**2+1e-12,axis=0)**-1),np.transpose(melfilterbank))

# logmelspectrogram = 10*np.log10(np.matmul(np.abs(spectrogram)**2,melfilterbank)+1e-12)
# mfcc = scipy.fft.dct(logmelspectrogram)

# import matplotlib as mpl
# default_figsize = mpl.rcParamsDefault['figure.figsize']
# mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]
# plt.imshow(np.transpose(logmelspectrogram),aspect='auto',origin='lower')
# plt.xlabel('Window frame (out of 2114)')
# #plt.ylabel('Quefrency (ms)')
# #plt.axis([0, len(data)/fs, 0, 20])
# plt.title('Log-mel spectrogram using 10 mel bands')
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'speaker4_spectLOG.png'))
# plt.clf()


# plt.imshow(np.transpose(mfcc),aspect='auto',origin='lower')
# plt.xlabel('Window frame (out of 2114)')
# #plt.ylabel('Quefrency (ms)')
# #plt.axis([0, len(data)/fs, 0, 20])
# plt.title('10 MFCCs by Window using 10 mel bands')
# plt.savefig(os.path.join(args.tables_and_figures_dir, 'speaker4_mfcc.png'))
# plt.clf()

# # df = pd.DataFrame(columns=['mfcc1', 'mfcc2', 'mfcc3','mfcc4', 'mfcc5', 'mfcc6','mfcc7', 'mfcc8', 'mfcc9', 'mfcc10'])
# df = pd.DataFrame(columns=['mfcc1', 'mfcc2', 'mfcc3'])

# print("Shape: ", mfcc.shape[1])
# # for i in range(mfcc.shape[0]):
# rows_list = []
# for arr in mfcc:
#     dict1 = {}
#     dict1.update(arr)
#     rows_list.append(dict1)


# x=1