# https://speechprocessingbook.aalto.fi/Representations/Short-time_analysis.html

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

# What would be a suitable window size?
 # Choose a window length in milliseconds.
 # Try to find the longest window where segments still look stationary (waveform character does not change much during segment).
data_length = data.shape[0]

# choose segment that emcompasses 'one' 
starting_position = 84000
window_length_ms = 400
window_length = int(window_length_ms*fs/1000)
print('Window length in samples ' + str(window_length))
# 19200
time_vector = np.linspace(0,window_length_ms,window_length)

plt.plot(time_vector,data[starting_position:(starting_position+window_length)])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()
plt.savefig(os.path.join(args.aalto, '3_1 Window of One.png'))
plt.clf()

# choose segment that emcompasses 'one' 
starting_position = 94000
window_length_ms = 30
window_length = int(window_length_ms*fs/1000)
print('Window length in samples ' + str(window_length))
# 19200
time_vector = np.linspace(0,window_length_ms,window_length)

plt.plot(time_vector,data[starting_position:(starting_position+window_length)])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()
plt.savefig(os.path.join(args.aalto, '3_1 Stationary Window of One.png'))
plt.clf()

# Windowing functions
zero_length = int(window_length/4)
zero_length_ms = window_length_ms/4

# choose segment from random position in sample
# starting_position = zero_length + np.random.randint(data_length - window_length - 2*zero_length)

time_vector = np.linspace(-zero_length_ms,window_length_ms+zero_length_ms,window_length+2*zero_length)

zero_vector = np.zeros([zero_length,])
data_vector = np.concatenate((zero_vector,data[starting_position:(starting_position+window_length)],zero_vector))


plt.plot(time_vector,data_vector)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.savefig(os.path.join(args.aalto, '3_1 Windowing 1.png'))
plt.clf()




windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2

windowing_function_extended = np.concatenate((zero_vector,windowing_function,zero_vector))

# choose segment from random position in sample
# starting_position = zero_length + np.random.randint(data_length - window_length - 2*zero_length)

# zero_vector = np.zeros([zero_length,])
# data_vector = np.concatenate((zero_vector,data[starting_position:(starting_position+window_length)],zero_vector))

plt.figure(figsize=[12,6])

plt.subplot(221)
plt.plot(time_vector,data_vector)
plt.title('Original window (with zero extension)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.subplot(222)
plt.plot(time_vector,windowing_function_extended)
plt.title('Hann windowing function')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.subplot(223)
plt.plot(time_vector,data_vector*windowing_function_extended)
plt.title('Windowed signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig(os.path.join(args.aalto, '3_1 Windowing 2.png'))
plt.clf()

# Spectral analysis
# windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2

# choose segment from random position in sample
# starting_position = zero_length + np.random.randint(data_length - window_length)

# Now without the zero-extension, because that was just for illustration
data_vector = data[starting_position:(starting_position+window_length),]
time_vector = np.linspace(0,window_length_ms,window_length)
frequency_vector = np.linspace(0,fs/2000,int(window_length/2+1))

plt.figure(figsize=[12,8])

plt.subplot(321)
plt.plot(time_vector,data_vector)
plt.title('Original window')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.subplot(322)
plt.plot(time_vector,data_vector*windowing_function)
plt.title('Windowed signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.subplot(323)
plt.plot(frequency_vector,np.abs(scipy.fft.rfft(data_vector*windowing_function)))
plt.title('Magnitude spectrum')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $|X_k|$')

plt.subplot(324)
plt.plot(frequency_vector,np.abs(scipy.fft.rfft(data_vector*windowing_function))**2)
plt.title('Power spectrum')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $|X_k|^2$')

plt.subplot(325)
plt.plot(frequency_vector,20*np.log10(np.abs(scipy.fft.rfft(data_vector*windowing_function))))
plt.title('Log-magnitude spectrum')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$')
ax = plt.axis()

plt.subplot(326)
plt.plot(frequency_vector,20*np.log10(np.abs(scipy.fft.rfft(data_vector*windowing_function))))
plt.title('Log-magnitude spectrum zoomed to [0,8kHz]')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$ (dB)')
ax = [0, 8, ax[2], ax[3]]
plt.axis(ax)

plt.tight_layout()
plt.savefig(os.path.join(args.aalto, '3_1 Spectral 1.png'))
plt.clf()

# Speech features in the spectrum
windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2

# choose segment from random position in sample
# starting_position = zero_length + np.random.randint(data_length - window_length)

# Now without the zero-extension, because that was just for illustration
data_vector = data[starting_position:(starting_position+window_length),]
time_vector = np.linspace(0,window_length_ms,window_length)
frequency_vector = np.linspace(0,fs/2000,int(window_length/2+1))

plt.figure(figsize=[12,6])


plt.subplot(221)
plt.plot(time_vector,data_vector)
plt.title('Original window')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

plt.subplot(222)
plt.plot(time_vector,data_vector*windowing_function)
plt.title('Windowed signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')


# Envelope calculation - the theory of the mathematical methods is not covered here
from scipy.linalg import solve_toeplitz, toeplitz
autocorrelation = scipy.fft.irfft(np.abs(scipy.fft.rfft(data_vector*windowing_function))**2)
lpc_order = int(fs/1000 + 2)
u = np.zeros([lpc_order+1,1])
u[0]=1
lpc_model = solve_toeplitz(autocorrelation[0:lpc_order+1],u)
lpc_model /= lpc_model[0]
envelope_spectrum = np.abs(scipy.fft.rfft(lpc_model,window_length,axis=0))**-1
signal_spectrum = np.abs(scipy.fft.rfft(data_vector*windowing_function))
envelope_spectrum *= np.max(signal_spectrum)/np.max(envelope_spectrum)

plt.subplot(223)
plt.plot(frequency_vector,20*np.log10(signal_spectrum),label='Spectrum')
plt.plot(frequency_vector,20*np.log10(envelope_spectrum),linewidth=3,label='Envelope')
plt.legend()
plt.title('Log-magnitude spectrum and its envelope')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$')
ax = plt.axis()

plt.subplot(224)
plt.plot(frequency_vector,20*np.log10(signal_spectrum),label='Spectrum')
plt.plot(frequency_vector,20*np.log10(envelope_spectrum),linewidth=3,label='Envelope')
plt.legend()
plt.title('Log-magnitude spectrum zoomed to [0,8kHz]')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$ (dB)')
ax = [0, 8, ax[2], ax[3]]
plt.axis(ax)

plt.tight_layout()
plt.savefig(os.path.join(args.aalto, '3_1 Speech features.png'))
plt.clf()

# Formants
windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2

# choose segment from random position in sample
# starting_position = zero_length + np.random.randint(data_length - window_length)

# Now without the zero-extension, because that was just for illustration
data_vector = data[starting_position:(starting_position+window_length),]
time_vector = np.linspace(0,window_length_ms,window_length)
frequency_vector = np.linspace(0,fs/2000,int(window_length/2+1))

# Envelope calculation - the theory of the mathematical methods is not covered here
from scipy.linalg import solve_toeplitz, toeplitz
autocorrelation = scipy.fft.irfft(np.abs(scipy.fft.rfft(data_vector*windowing_function))**2)
lpc_order = int(fs/1000 + 2)
u = np.zeros([lpc_order+1,1])
u[0]=1
lpc_model = solve_toeplitz(autocorrelation[0:lpc_order+1],u)
lpc_model /= lpc_model[0]
envelope_spectrum = np.abs(scipy.fft.rfft(lpc_model,window_length,axis=0))**-1
signal_spectrum = np.abs(scipy.fft.rfft(data_vector*windowing_function))
envelope_spectrum *= np.max(signal_spectrum)/np.max(envelope_spectrum)

# Find peaks of envelope
## Derivative is zero at peaks and valleys; similarly, the first difference changes sign.
## At peaks, the first difference changes sign from positive negative
diff_envelope = np.diff(envelope_spectrum,axis=0)
sign_diff_envelope = np.sign(diff_envelope)
diff_sign_diff_envelope = np.diff(sign_diff_envelope,axis=0)
peak_indices = np.argwhere(diff_sign_diff_envelope[:,0] < 0)+1
peak_indices = peak_indices[0:5]

plt.figure(figsize=[12,4])

plt.subplot(121)
plt.plot(frequency_vector,20*np.log10(signal_spectrum),label='Spectrum')
plt.plot(frequency_vector,20*np.log10(envelope_spectrum),linewidth=3,label='Envelope')
plt.plot(frequency_vector[peak_indices],20*np.log10(envelope_spectrum[peak_indices,0]),marker='^',linestyle='',label='Peaks')
for k in range(len(peak_indices)): 
    x = frequency_vector[peak_indices[k]]
    y = 20*np.log10(envelope_spectrum[peak_indices[k],0]) + 5
    plt.text(x,y,'F' + str(k+1))    
plt.legend()
plt.title('Log-magnitude spectrum and its envelope')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$')
ax = plt.axis()
ax = [ax[0], ax[1], ax[2], ax[3]+5]
plt.axis(ax)

plt.subplot(122)
plt.plot(frequency_vector,20*np.log10(signal_spectrum),label='Spectrum')
plt.plot(frequency_vector,20*np.log10(envelope_spectrum),linewidth=3,label='Envelope')
plt.plot(frequency_vector[peak_indices],20*np.log10(envelope_spectrum[peak_indices,0]),marker='^',linestyle='',label='Peaks')
for k in range(len(peak_indices)): 
    x = frequency_vector[peak_indices[k]]
    y = 20*np.log10(envelope_spectrum[peak_indices[k],0]) + 5
    if x < 8:  plt.text(x,y,'F' + str(k+1))    
plt.legend()
plt.title('Log-magnitude spectrum zoomed to [0,8kHz]')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$ (dB)')
ax = [0, 8, ax[2], ax[3]+5]
plt.axis(ax)

plt.tight_layout()
plt.savefig(os.path.join(args.aalto, '3_1 Formants.png'))
plt.clf()

###########################
# Fundamental frequency 
# windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2

# choose segment from random position in sample
# starting_position = zero_length + np.random.randint(data_length - window_length)

# Now without the zero-extension, because that was just for illustration
data_vector = data[starting_position:(starting_position+window_length),]
time_vector = np.linspace(0,window_length_ms,window_length)
frequency_vector = np.linspace(0,fs/2000,int(window_length/2+1))

signal_spectrum = np.abs(scipy.fft.rfft(data_vector*windowing_function))

# Find fundamental frequency peak
## Derivative is zero at peaks and valleys; similarly, the first difference changes sign.
## At peaks, the first difference changes sign from positive negative
diff_envelope = np.diff(signal_spectrum,axis=0)
sign_diff_envelope = np.sign(diff_envelope)
diff_sign_diff_envelope = np.diff(sign_diff_envelope,axis=0)
peak_indices = np.argwhere(diff_sign_diff_envelope[:] < 0)[:,0]+1
peak_frequencies = frequency_vector[peak_indices]*1000

# Suppose F0 is in the range 80 to 400, then the highest peak in that range belongs to the comb structure
in_range_indices = np.argwhere((peak_frequencies >= 80) & (peak_frequencies <= 400))[:,0]
if in_range_indices.size > 0:
    largest_index = np.argmax(signal_spectrum[peak_indices[in_range_indices]])
    within_6dB = np.argwhere(signal_spectrum[peak_indices[in_range_indices]] > signal_spectrum[peak_indices[largest_index]]*0.5)
    F0_index = peak_indices[in_range_indices[within_6dB[0]] ]
    

plt.figure(figsize=[12,6])

plt.subplot(221)
plt.plot(time_vector,data_vector)
plt.title('Original window')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')


plt.subplot(222)
plt.plot(frequency_vector,20*np.log10(signal_spectrum),label='Spectrum')
#plt.plot(frequency_vector[peak_indices],20*np.log10(envelope_spectrum[peak_indices,0]),marker='^',linestyle='',label='Peaks')
#for k in range(len(peak_indices)): 
#    x = frequency_vector[peak_indices[k]]
#    y = 20*np.log10(envelope_spectrum[peak_indices[k],0]) + 5
#    plt.text(x,y,'F' + str(k+1))    
plt.legend()
plt.title('Log-magnitude spectrum and its envelope')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$')
ax = plt.axis()
ax = [ax[0], ax[1], ax[2], ax[3]+5]
plt.axis(ax)

plt.subplot(223)
plt.plot(frequency_vector*1000,20*np.log10(signal_spectrum),label='Spectrum')
plt.plot(frequency_vector[F0_index]*1000,20*np.log10(signal_spectrum[F0_index]),marker='v',linestyle='',label='Fundamental')
plt.text(frequency_vector[F0_index]*1000,20*np.log10(signal_spectrum[F0_index]) + 5,'F0='+str(frequency_vector[F0_index[0]]*1000)+'Hz')
plt.legend()
plt.title('Log-magnitude spectrum zoomed to [0,2kHz]')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$ (dB)')
ax = [0, 2000, ax[2], ax[3]+5]
plt.axis(ax)

plt.tight_layout()
plt.savefig(os.path.join(args.aalto, '3_1 Fundamental frequency.png'))
plt.clf()

###############
# No fundamental frequency

# Click to hidewindowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2

# choose segment from random position in sample
# starting_position = zero_length + np.random.randint(data_length - window_length)

# Now without the zero-extension, because that was just for illustration
data_vector = data[starting_position:(starting_position+window_length),]
time_vector = np.linspace(0,window_length_ms,window_length)

signal_spectrum = np.abs(scipy.fft.rfft(data_vector*windowing_function,n=window_length*4))
frequency_vector = np.linspace(0,fs/2000,len(signal_spectrum))

# Find fundamental frequency peak
## Derivative is zero at peaks and valleys; similarly, the first difference changes sign.
## At peaks, the first difference changes sign from positive negative
diff_envelope = np.diff(signal_spectrum,axis=0)
sign_diff_envelope = np.sign(diff_envelope)
diff_sign_diff_envelope = np.diff(sign_diff_envelope,axis=0)
peak_indices = np.argwhere(diff_sign_diff_envelope[:] < 0)[:,0]+1
peak_frequencies = frequency_vector[peak_indices]*1000

# Suppose F0 is in the range 80 to 400, then the highest peak in that range belongs to the comb structure
in_range_indices = np.argwhere((peak_frequencies >= 80) & (peak_frequencies <= 400))[:,0]
if in_range_indices.size > 0:
    largest_index = np.argmax(signal_spectrum[peak_indices[in_range_indices]])
    within_6dB = np.argwhere(signal_spectrum[peak_indices[in_range_indices]] > signal_spectrum[peak_indices[largest_index]]*0.5)
    F0_index = peak_indices[in_range_indices[within_6dB[0]] ]
    
    

plt.figure(figsize=[12,6])
    
plt.subplot(221)
plt.plot(time_vector,data_vector)
plt.title('Original window')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')


plt.subplot(222)
plt.plot(frequency_vector,20*np.log10(signal_spectrum),label='Spectrum')
plt.legend()
plt.title('Log-magnitude spectrum and its envelope')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$')
ax = plt.axis()
ax = [ax[0], ax[1], ax[2], ax[3]+5]
plt.axis(ax)


plt.subplot(223)
plt.plot(frequency_vector*1000,20*np.log10(signal_spectrum),label='Spectrum')
F0vector = np.arange(frequency_vector[F0_index],2,frequency_vector[F0_index])
for h in F0vector:
    plt.plot(h*np.ones(2)*1000,[-100,100],linestyle='--',color='gray')
plt.legend()
plt.title('Log-magnitude spectrum zoomed to [0,2kHz]')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude $20\log_{10}|X_k|$ (dB)')
ax = [0, 2000, ax[2], ax[3]+5]
plt.axis(ax)

plt.tight_layout()
plt.savefig(os.path.join(args.aalto, '3_1 Fundamental frequency NONE.png'))
plt.clf()


######################
# Spectrogram
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
mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]
plt.imshow(20*np.log10(np.abs(np.transpose(spectrogram))+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, fs/2000])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')
plt.title('Spectrogram of signal')
plt.savefig(os.path.join(args.aalto, '3_1 Spectrogram 1.png'))
plt.clf()

mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]
plt.imshow(20*np.log10(np.abs(np.transpose(spectrogram))+black_eps),aspect='auto',origin='lower',extent=[0, len(data)/fs,0, fs/2000])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')
plt.axis([0, len(data)/fs, 0, 8])
plt.title('Spectrogram zoomed to 8 kHz')
plt.savefig(os.path.join(args.aalto, '3_1 Spectrogram 2.png'))
plt.clf()

x=1