""" DataClass and dataloader
    Class: VoxDataset(data.Datset)
        Args:
            data_dir: (string) directory containing the dataset
            feature_extraction_method: (torchvision.transforms) transformation to apply on files
    Functions:
        1. fetch_dataloader(types(str), data_dir, feature_extraction_method(str), params
        2. get_feature_extractor(method(str))
    When fetch_dataloader is called from the train.py file, it will get the params from experiments/base_model/params.json
"""
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


# locations needed in file
parser = argparse.ArgumentParser()
parser.add_argument('--vox_directory', 
                    default='/home/elizabeth.gooch/data/VoxCeleb_data/VoxCeleb/',
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
args = parser.parse_args()


def get_feature_extractor(method, sr):
    """
    Args:
        method: (str) audio feature extraction
    Returns:
        transform method
    """
    sample_rate = sr
    n_fft = int(sample_rate*0.03) #30 ms window
    n_mels = 128
    

    if method == 'mfcc3':
        n_mfcc = 3 

        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "mel_scale": "htk",
                "window_fn": torch.hamming_window
            },
        )


    if method == 'mfcc12':
        n_mfcc = 12 

        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "mel_scale": "htk",
                "window_fn": torch.hamming_window
            },
        )

    if method == 'mfcc40':
        n_mfcc = 40 

        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "mel_scale": "htk",
                "window_fn": torch.hamming_window
            },
        )

    return transform
    


class Standardize():
    """Frame-wise MFCC standardization."""
    def __call__(self):
        # Shape: D x Tmax
        sequence -= sequence.mean(axis=0)
        sequence /= sequence.std(axis=0)
        return sequence






class VoxDataset(data.Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, pickle_name, frontend):
        # def __init__(self, data_dir, transform):
        """
        Store the filenames of the wav to use. Specifies transforms to apply on files.
        Args:
            pickle (string): Path to the pickle file with annotations
            root_dir (string): Directory with all the .wav files 
            feature_extraction_method (torchvision.transforms): Transform to be applied to sample
        """
        filepath = os.path.join(args.pickle_directory + '/' + pickle_name + '.pickle')
        with open(filepath, "rb") as f:
            list_of_list = pickle.load(f)
        self.vox = pd.DataFrame(list_of_list)
       
        self.frontend = get_feature_extractor(frontend, sr=16000)
        # https://discuss.pytorch.org/t/how-to-generate-a-dataloader-with-file-paths-in-pytorch/147928/2

    def __len__(self):
        # return size of dataset
        return len(self.vox)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., number_of_files-1]  <<- I should make this an attribute
        Returns:
            waveform, sample rate: (Tensor) audio
            label: (int) corresponding label of audio
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        wavefiles = self._get_audio_sample_paths(idx)
        label = self._get_audio_sample_label(idx)
        
        t1, t2 = [self._process_file(file) for file in wavefiles]
        
        assert t1[1] == t2[1], "The pairs of tensors differ in shape"

        return t1[0], t2[0], label    


    def _get_audio_sample_paths(self, idx):
        wavefiles = self.vox.iloc[idx, 0:2]
        return wavefiles

    def _get_audio_sample_label(self, idx):
        label_boolean =  self.vox.iloc[idx, 2]
        return label_boolean.astype(float)

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > 48000:
            offset = 8000
            signal = signal[:, offset:(offset+48000)]
        return signal

    def _process_file(self, filepath):
        signal, sr = torchaudio.load(filepath)
        signal = self._cut_if_necessary(signal)
        signal = self.frontend(signal)
        signal = torch.squeeze(signal)
        signal = self._standardize(signal)
        signal = torch.transpose(signal, 0, 1)
        # https://stackoverflow.com/questions/59397558/transform-the-input-of-the-mfccs-spectogram-for-a-cnn-audio-recognition
        dim = signal.size()
        return signal, dim

    def _standardize(self, signal):
        # Shape: D x Tmax
        signal -= signal.mean(axis=0)
        signal /= signal.std(axis=0)
        return signal








class AlsaifyDataset(data.Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, pickle_name, frontend):
        # def __init__(self, data_dir, transform):
        """
        Store the filenames of the wav to use. Specifies transforms to apply on files.
        Args:
            pickle (string): Path to the pickle file with annotations
            root_dir (string): Directory with all the .wav files 
            feature_extraction_method (torchvision.transforms): Transform to be applied to sample
        """
        filepath = os.path.join(args.alsaify_pickle_directory + '/' + pickle_name + '.pickle')
        with open(filepath, "rb") as f:
            list_of_list = pickle.load(f)
        self.vox = pd.DataFrame(list_of_list)
       
        self.frontend = get_feature_extractor(frontend, sr=48000)
        # https://discuss.pytorch.org/t/how-to-generate-a-dataloader-with-file-paths-in-pytorch/147928/2

    def __len__(self):
        # return size of dataset
        return len(self.vox)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., number_of_files-1]  <<- I should make this an attribute
        Returns:
            waveform, sample rate: (Tensor) audio
            label: (int) corresponding label of audio
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        wavefiles = self._get_audio_sample_paths(idx)
        label = self._get_audio_sample_label(idx)
        
        t1, t2 = [self._process_file(file) for file in wavefiles]
        
        assert t1[1] == t2[1], "The pairs of tensors differ in shape"

        return t1[0], t2[0], label    


    def _get_audio_sample_paths(self, idx):
        wavefiles = self.vox.iloc[idx, 0:2]
        return wavefiles

    def _get_audio_sample_label(self, idx):
        label_boolean =  self.vox.iloc[idx, 2]
        return label_boolean.astype(float)

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > 48000:
            offset = 8000
            signal = signal[:, offset:(offset+48000)]
        return signal

    def _process_file(self, filepath):
        signal, sr = torchaudio.load(filepath)
        signal = self._cut_if_necessary(signal)
        signal = self.frontend(signal)
        signal = torch.squeeze(signal)
        signal = self._standardize(signal)
        signal = torch.transpose(signal, 0, 1)
        # https://stackoverflow.com/questions/59397558/transform-the-input-of-the-mfccs-spectogram-for-a-cnn-audio-recognition
        dim = signal.size()
        return signal, dim

    def _standardize(self, signal):
        # Shape: D x Tmax
        signal -= signal.mean(axis=0)
        signal /= signal.std(axis=0)
        return signal
   


def fetch_dataloader(pickle_name, feature_extraction_method, batch_size, params):
    dl = data.DataLoader(VoxDataset(pickle_name, feature_extraction_method),
                                    batch_size,
                                    shuffle=True,
                                    num_workers=params.num_workers,
                                    pin_memory=True)

    return dl

def fetch_AlsaifyDataloader(pickle_name, feature_extraction_method, batch_size, params):
    dl = data.DataLoader(AlsaifyDataset(pickle_name, feature_extraction_method),
                                    batch_size,
                                    shuffle=True,
                                    num_workers=params.num_workers,
                                    pin_memory=True)

    return dl

 
if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.params_directory, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = needed_functions.Params(json_path)
