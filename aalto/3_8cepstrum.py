# Initialization
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import scipy.fft
import numpy as np
import argparse
import os


# import helper_functions # local file helper_functions.py

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
filename = os.path.join(args.alsaify_directory, same_speaker4)

# read from storage
fs, data = wavfile.read(filename)
# data = data[:,0]

window_length_ms = 30
window_step_ms = 20
spectrum_length = 5000
window_length = int(window_length_ms*fs/1000)
window_step = int(window_step_ms*fs/1000)
windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2
total_length = len(data)

t = np.arange(0,total_length)/fs
plt.figure(figsize=[12,3])
plt.plot(t,data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Speech sample')
plt.show()
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_amplitude_speaker4.png'))

######
# https://speechprocessingbook.aalto.fi/Representations/Melcepstrum.html

# choose segment from random position in sample
# starting_position = np.random.randint(total_length - window_length)
starting_position = 48000*2-30
data_vector = data[starting_position:(starting_position+window_length)]
# data_vector = data[starting_position:(starting_position+window_length),]
window = data_vector*windowing_function
time_vector = np.linspace(0,window_length_ms,window_length)

spectrum = scipy.fft.rfft(window,n=spectrum_length)
frequency_vector = np.linspace(0,fs/2000,len(spectrum))

# downsample to 16kHz (that is, Nyquist frequency is 8kHz, that is, everything about 8kHz can be removed)
idx = np.nonzero(frequency_vector <= 8)
frequency_vector = frequency_vector[idx]
spectrum = spectrum[idx]

logspectrum = 20*np.log10(np.abs(spectrum))
cepstrum = scipy.fft.rfft(logspectrum)

ctime = np.linspace(0,0.5*1000*spectrum_length/fs,len(cepstrum))

plt.figure(figsize=[12,8])
plt.subplot(321)
plt.plot(time_vector,data_vector)
plt.title('Original window')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.subplot(322)
plt.plot(time_vector,window)
plt.title('Windowed signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.subplot(323)
plt.plot(frequency_vector,logspectrum)
plt.title('Log-magnitude spectrum')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$')

plt.subplot(324)
plt.plot(frequency_vector,logspectrum)
plt.title('Log-magnitude spectrum (zoomed to low frequencies)')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$')
ax = plt.axis()
ax = [0, 4, ax[2], ax[3]]
plt.axis(ax)

plt.subplot(325)
plt.plot(ctime,np.abs(cepstrum))
plt.title('Cepstrum')
plt.xlabel('Quefrency (ms)')
plt.ylabel('Magnitude $|C_k|$')
ax = plt.axis()
ax = [0, 20, ax[2], ax[3]]
plt.axis(ax)

plt.subplot(326)
plt.plot(ctime,np.log(np.abs(cepstrum)))
plt.title('Log-Cepstrum')
plt.xlabel('Quefrency (ms)')
plt.ylabel('Log-Magnitude $\log|C_k|$')
ax = plt.axis()
ax = [0, 20, ax[2], ax[3]]
plt.axis(ax)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_sixfigures_speaker4.png'))
plt.clf()


######################
# Features in the Cepstrum

# spectrogram = helper_functions.stft(data,fs)
# window_count = spectrogram.shape[0]
window_count = int(np.floor((len(data)-window_length)/window_step)+1)
cepstrogram = np.zeros((len(cepstrum),window_count))

for k in range(window_count): 
    starting_position = k*window_step
    data_vector = data[starting_position:(starting_position+window_length),]
    window = data_vector*windowing_function
    time_vector = np.linspace(0,window_length_ms,window_length)

    spectrum = scipy.fft.rfft(window,n=spectrum_length)
    frequency_vector = np.linspace(0,fs/2000,len(spectrum))

    # downsample to 16kHz (that is, Nyquist frequency is 8kHz, that is, everything about 8kHz can be removed)
    idx = np.nonzero(frequency_vector <= 8)
    frequency_vector = frequency_vector[idx]
    spectrum = spectrum[idx]

    logspectrum = 20*np.log10(np.abs(spectrum))
    cepstrum = scipy.fft.rfft(logspectrum)
    cepstrogram[:,k] = np.abs(cepstrum)
    
#    cepstrogram[:,k] = np.abs(scipy.fft.rfft(20*np.log10(np.abs(spectrogram[k,idx])),axis=1))
#cepstrogram = np.abs(scipy.fft.rfft(20*np.log10(np.abs(spectrogram[:,idx])),axis=1))


black_eps = 1e-1 # minimum value for the log-value in the spectrogram - helps making the background really black
    
import matplotlib as mpl
default_figsize = mpl.rcParamsDefault['figure.figsize']
mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]

# mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]
# plt.imshow(20*np.log10(np.abs(np.transpose(spectrogram))+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, fs/2000])
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (kHz)')
# plt.axis([0, len(data)/fs, 0, 8])
# plt.title('Spectrogram zoomed to 8 kHz')
# plt.show()


plt.imshow(np.log(np.abs(cepstrogram)+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, ctime[-1]])
#plt.imshow(np.log(np.abs(cepstrogram)+black_eps),aspect='auto',origin='lower')
plt.xlabel('Time (s)')
plt.ylabel('Quefrency (ms)')
plt.axis([0, len(data)/fs, 0, 20])
#plt.title('Spectrogram zoomed to 8 kHz')
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_cepstrum_speaker4.png'))
plt.clf()




#####################
# Mel scale

def freq2mel(f): 
    return 2595*np.log10(1 + (f/700))

def mel2freq(m): 
    return 700*(10**(m/2595) - 1)

f = np.linspace(0,8000,1000)
plt.plot(f/1000,freq2mel(f))
plt.xlabel('Frequency (kHz)')
plt.ylabel('Mel-scale')
plt.title('The mel-scale as a function of frequency')
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_mel.png'))
plt.clf()

melbands = 20
maxmel = freq2mel(8000)
mel_idx = np.array(np.arange(.5,melbands,1)/melbands)*maxmel
freq_idx = mel2freq(mel_idx)

melfilterbank = np.zeros((len(spectrum),melbands))
freqvec = np.arange(0,len(spectrum),1)*8000/len(spectrum)
for k in range(melbands-2):    
    if k>0:
        upslope = (freqvec-freq_idx[k])/(freq_idx[k+1]-freq_idx[k])
    else:
        upslope = 1 + 0*freqvec
    if k<melbands-3:
        downslope = 1 - (freqvec-freq_idx[k+1])/(freq_idx[k+2]-freq_idx[k+1])
    else:
        downslope = 1 + 0*freqvec
    triangle = np.max([0*freqvec,np.min([upslope,downslope],axis=0)],axis=0)
    melfilterbank[:,k] = triangle
    
melreconstruct = np.matmul(np.diag(np.sum(melfilterbank**2+1e-12,axis=0)**-1),np.transpose(melfilterbank))
    
plt.plot(freqvec/1000,melfilterbank)
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amplitude')
plt.title('The mel-filterbank')
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_melbank.png'))
plt.clf()
                           
#plt.plot(freqvec/1000,np.transpose(melreconstruct))
#plt.xlabel('Frequency (kHz)')
#plt.ylabel('Amplitude')
#plt.title('The mel-filterbank')
#plt.show()




# choose segment from random position in sample
# starting_position = np.random.randint(total_length - window_length)

data_vector = data[starting_position:(starting_position+window_length),]
window = data_vector*windowing_function
time_vector = np.linspace(0,window_length_ms,window_length)

spectrum = scipy.fft.rfft(window,n=spectrum_length)
frequency_vector = np.linspace(0,fs/2000,len(spectrum))

# downsample to 16kHz (that is, Nyquist frequency is 8kHz, that is, everything about 8kHz can be removed)
idx = np.nonzero(frequency_vector <= 8)
frequency_vector = frequency_vector[idx]
spectrum = spectrum[idx]
logspectrum = 20*np.log10(np.abs(spectrum))

logmelspectrum = 10*np.log10(np.matmul(np.transpose(melfilterbank),np.abs(spectrum)**2)+1e-12)
logenvelopespectrum = 10*np.log10(np.matmul(np.transpose(melreconstruct),10**(logmelspectrum/10)))

plt.plot(freqvec/1000,logspectrum,label='Spectrum')
plt.plot(freqvec/1000,logenvelopespectrum,label='Mel-envelope')
plt.legend()
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)')
plt.title('The mel-envelope')
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_melenvelope.png'))
plt.clf()



#######################################
# MFCC

######################
# Spectrogram from 3.1
window_step = int(window_length/8) # step between analysis windows
window_count = int(np.floor((len(data)-window_length)/window_step)+1)
spectrum_length = int((window_length+1)/2)+1

spectrogram = np.zeros((window_count,spectrum_length))
time_vector = np.linspace(0,window_length_ms,window_length)
frequency_vector = np.linspace(0,fs/2000,spectrum_length)


for k in range(window_count):
    starting_position = k*window_step
    
    data_vector = data[starting_position:(starting_position+window_length),]
    window_spectrum = np.abs(scipy.fft.rfft(data_vector*windowing_function))
    
    spectrogram[k,:] = window_spectrum


black_eps = 1e-1 # minimum value for the log-value in the spectrogram - helps making the background really black
    
import matplotlib as mpl
default_figsize = mpl.rcParamsDefault['figure.figsize']
# mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]
# plt.imshow(20*np.log10(np.abs(np.transpose(spectrogram))+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, fs/2000])
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (kHz)')
# plt.title('Spectrogram of signal')
# plt.savefig(os.path.join(args.aalto, '3_1 Spectrogram 1.png'))
# plt.clf()

# mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]
# plt.imshow(20*np.log10(np.abs(np.transpose(spectrogram))+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, fs/2000])
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (kHz)')
# plt.axis([0, len(data)/fs, 0, 8])
# plt.title('Spectrogram zoomed to 8 kHz')
# plt.savefig(os.path.join(args.aalto, '3_1 Spectrogram 2.png'))
# plt.clf()


melbands = 20
maxmel = freq2mel(8000)
mel_idx = np.array(np.arange(.5,melbands,1)/melbands)*maxmel
freq_idx = mel2freq(mel_idx)

melfilterbank = np.zeros((spectrogram.shape[1],melbands))
freqvec = np.arange(0,spectrogram.shape[1],1)*8000/spectrogram.shape[1]
for k in range(melbands-2):    
    if k>0:
        upslope = (freqvec-freq_idx[k])/(freq_idx[k+1]-freq_idx[k])
    else:
        upslope = 1 + 0*freqvec
    if k<melbands-3:
        downslope = 1 - (freqvec-freq_idx[k+1])/(freq_idx[k+2]-freq_idx[k+1])
    else:
        downslope = 1 + 0*freqvec
    triangle = np.max([0*freqvec,np.min([upslope,downslope],axis=0)],axis=0)
    melfilterbank[:,k] = triangle
    
melreconstruct = np.matmul(np.diag(np.sum(melfilterbank**2+1e-12,axis=0)**-1),np.transpose(melfilterbank))


logmelspectrogram = 10*np.log10(np.matmul(np.abs(spectrogram)**2,melfilterbank)+1e-12)

mfcc = scipy.fft.dct(logmelspectrogram)

import matplotlib as mpl
default_figsize = mpl.rcParamsDefault['figure.figsize']
mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]
plt.imshow(20*np.log10(np.abs(np.transpose(spectrogram))+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, fs/2000])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')
plt.axis([0, len(data)/fs, 0, 8])
plt.title('Spectrogram zoomed to 8 kHz')
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_spect.png'))
plt.clf()

plt.imshow(np.transpose(logmelspectrogram),aspect='auto',origin='lower')
plt.xlabel('Time (s)')
#plt.ylabel('Quefrency (ms)')
#plt.axis([0, len(data)/fs, 0, 20])
plt.title('Log-mel spectrogram')
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_spectLOG.png'))
plt.clf()


plt.imshow(np.transpose(mfcc),aspect='auto',origin='lower')
plt.xlabel('Time (s)')
#plt.ylabel('Quefrency (ms)')
#plt.axis([0, len(data)/fs, 0, 20])
plt.title('MFCCs over time')
plt.savefig(os.path.join(args.tables_and_figures_dir, 'aalto_mfcc.png'))
plt.clf()