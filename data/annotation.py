"""Extracting, Converting, and Organizing analysis of VoxCeleb2

Files start off in LARGE .zip files --> TODO unzip
Files start off as .m4a files --> TODO convert to .wav

VoxCeleb dataset file structure is: 
    vox_test/
        aac/
            id's/
                uri's/
                    files

    vox_dev/
        dev/
            aac/
                id's/
                    uri's/
                        files

Best to leave extracted folder strurcture entacted but with converted .wav files.

Create lists of tuples of filenames and id's to use in train/val split and test.

In DatasetClass creation, limit the number of filenames used in dataset creation to scale the training.

For audio import: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

# To start: 
1. srun --pty --partition=beards --gres=gpu:1 --nodelist=compute-8-13 bash
2. module load lang/miniconda3/4.10.3
3. conda activate pytorch_audio

However, if I enter Hamming via: srun --pty bash, then I get 62 cpu's!

"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import argparse
import random
import os
import io
import tarfile
import tempfile
import time
import zipfile
import torch
import numba.cuda
import pathlib
from pydub import AudioSegment
import pickle
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--dev_folder', default='/home/elizabeth.gooch/data/VoxCeleb_data/VoxCeleb/vox2_dev', 
                    help="Dev directory")
parser.add_argument('--test_folder', default='/home/elizabeth.gooch/data/VoxCeleb_data/VoxCeleb/vox2_test', 
                    help="Test directory")


def find_all_files(directory):
    all_files = list(pathlib.Path(directory).rglob("*.[mw][4a][av]"))

    pure_paths = []

    for line in all_files:
        pure_paths.append(line.as_posix())

    return pure_paths


def annotate_directory(directory, type, annotation_audio_list = []):

    filenames = find_all_files(directory)

    for file in filenames:

        if file.endswith('.wav'):

            if type=='test':
                speaker_id = file.split("/")[8]
                annotation_audio_list.append((speaker_id, file))

            elif type=='dev':
                speaker_id = file.split("/")[9]
                annotation_audio_list.append((speaker_id, file))

        else:
            continue
    
    file_dict = defaultdict(list)

    for speaker_id, file in annotation_audio_list:
        file_dict[speaker_id].append(file)    

    saving_directory = '/home/elizabeth.gooch/lizthesis/data'
    annotation_filename = 'annotation_dict_' + type + '.pickle'

    with open(os.path.join(saving_directory, annotation_filename), 'wb') as f:
        pickle.dump(file_dict, f)
    
    return file_dict


if __name__ == '__main__':

    args = parser.parse_args()

    assert os.path.isdir(args.test_folder), "Couldn't find the dataset at {}".format(args.test_folder)
    
    test_dict = annotate_directory(args.test_folder, 'test')
    print(len(test_dict.get('id04276')))
    dev_dict = annotate_directory(args.dev_folder, 'dev')