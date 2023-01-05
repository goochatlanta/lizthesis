""" Basic wavfile manipulation and feature extraction using
    - scipy ()
    - pyWavelet (Haar)
    - pyAudioProcessing (GFCC)
"""
from os.path import dirname, join as pjoin
from pyexpat import features
from scipy.io import wavfile
import scipy.io
# from pyAudioProcessing.extract_features import get_features
from spafe.features import gfcc, lfcc, mfcc
import pywt
from math import sqrt
from gatspy.periodic import LombScargleFast
import pickle

test_data_dir = "/home/elizabeth.gooch/data/VoxCeleb_data/VoxCeleb/vox2_test/aac"
dev_data_dir = "/home/elizabeth.gooch/data/VoxCeleb_data/VoxCeleb/vox2_dev/aac"
annotation_dir = '/home/elizabeth.gooch/lizthesis/data'

wav_fname = pjoin(test_data_dir, 'id00017/01dfn2spqyE/00001.wav')

samplerate, data = wavfile.read(wav_fname)
# print(f"number of channels = {data.shape[1]}")
# number of channels = 2
length = data.shape[0] / samplerate
print(f"length = {length}s")

# mfccs = mfcc.mfcc(data, fs=samplerate, num_ceps=39, nfilts=128) # default number of filters in pytorch mel spectogram
# gfccs =  gfcc.gfcc(data, fs=samplerate, num_ceps=39, nfilts=128)
# lfccs = lfcc.lfcc(data, fs=samplerate, num_ceps=39, nfilts=128)

"""
Using my annotations, I can cycle through the files and create a train, val, and test dataset pickle for each feature type
for sample rate 16000. 

Reshaped to fit on one line, each observation will be 94 x 39 or so if I stick with 3 seconds at 16000 samle rate. 

I should save these in my data section because they aren't that big. 
"""

def cut_if_necessary(signal):
        if signal.shape[0] > 56000:
            signal = signal[8000:56000]
        return signal

def transform_save(filename):
    almost_full_path = filename.split(".")[0] + filename.split(".")[1]
    samplerate, data = wavfile.read(filename)
    # Trim
    clipped = cut_if_necessary(data)

    # LFCC Transform and save - why so wide? 298 x 40?
    lfccs = lfcc.lfcc(clipped, fs=16000, num_ceps=40, nfilts=128, normalize=1)
    # Rename item    
    name = almost_full_path + '_lfcc_16000.pickle'
    with open(name, 'wb') as g:
        pickle.dump(lfccs, g)

    # MFCC Transform and save
    mfccs = mfcc.mfcc(clipped, fs=16000, num_ceps=40, nfilts=128, normalize=1)
    # Rename item    
    name = almost_full_path + '_mfcc_16000.pickle'
    with open(name, 'wb') as g:
        pickle.dump(mfccs, g)

    # GFCC Transform and save
    gfccs = gfcc.gfcc(clipped, fs=16000, num_ceps=40, nfilts=128, normalize=1)
    # Rename item    
    name = almost_full_path + '_gfcc_16000.pickle'
    with open(name, 'wb') as g:
        pickle.dump(gfccs, g)


# Call in a file
# bring in dictionary_pickle - dev or test
annotation_dictionary = "/annotation_dict_" + 'dev' + ".pickle"
filepath = pjoin(annotation_dir + annotation_dictionary)
with open(filepath, "rb") as f:
    my_dict = pickle.load(f)
    # use the map function. https://stackoverflow.com/questions/18453566/dictionary-get-list-of-values-for-list-of-keys

x=1

            

# Trim the file

# Transform in num_frames x num_features

# Reshape to 1D

# Add to dataset by id 
# Or do i save the new features along the same file path to use the already written dictionary? Can I create dictionaries for each file type?
# Right now, I just have the dictionary calling the wavfiles?


# compute features
mfccs = mfcc.mfcc(data, fs=samplerate, num_ceps=39, nfilts=128, normalize="mvn") # default number of filters in pytorch mel spectogram
gfccs =  gfcc.gfcc(data, fs=samplerate, num_ceps=39, nfilts=128, normalize="mvn")
lfccs = lfcc.lfcc(data, fs=samplerate, num_ceps=39, nfilts=128, normalize="mvn")

# https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html
class MyHaarFilterBank(object):
     @property
     def filter_bank(self):
         
        return ([sqrt(2)/2, sqrt(2)/2], [-sqrt(2)/2, sqrt(2)/2],
           [sqrt(2)/2, sqrt(2)/2], [sqrt(2)/2, -sqrt(2)/2])

my_wavelet = pywt.Wavelet('My Haar Wavelet', filter_bank=MyHaarFilterBank())

my_filter_bank = ([sqrt(2)/2, sqrt(2)/2], [-sqrt(2)/2, sqrt(2)/2],
                   [sqrt(2)/2, sqrt(2)/2], [sqrt(2)/2, -sqrt(2)/2])

my_wavelet = pywt.Wavelet('My Haar Wavelet', filter_bank=my_filter_bank)
my_wavelet.orthogonal = True
my_wavelet.biorthogonal = True

# https://jakevdp.github.io/blog/2015/06/13/lomb-scargle-in-python/
# model = LombScargleFast().fit(t, mag, dmag)
# power = model.score_frequency_grid(fmin, df, N)



z=1