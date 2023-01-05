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
from multiprocess import Pool
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', default='/home/elizabeth.gooch/lizthesis/pytorch_code/data', 
                    help="My directory with VoxCeleb2 zip files and folders")
parser.add_argument('--test_data', default='/home/elizabeth.gooch/lizthesis/pytorch_code/data/vox2_test', 
                    help="Test directory")
parser.add_argument('--dev_data', default='/home/elizabeth.gooch/lizthesis/pytorch_code/data/vox2_dev', 
                    help="Dev directory")


def find_all_files(directory):
    # Recursive function to create all the paths in a directory
    # https://stackoverflow.com/questions/18394147/how-to-do-a-recursive-sub-folder-search-and-return-files-in-a-list
    all_files = list(pathlib.Path(directory).rglob("*.[mw][4a][av]"))

    pure_paths = []

    for line in all_files:
        pure_paths.append(line.as_posix())
        # https://stackoverflow.com/questions/54671385/convert-windowspath-to-posixpath
    return pure_paths



def convert_m4a_2_wav(directory):
    """
    Args:
        directory: the directory of .m4a

    Returns:
        written over files in the same directory

    Important: the environmental variable for the path to ffmpeg/bin needs to be set.
    """
    formats_to_convert = ['.m4a']

    filepaths = find_all_files(directory)
    for filepath in filepaths:
        if filepath.endswith(tuple(formats_to_convert)):
            (path, file_extension) = os.path.splitext(filepath)
            file_extension_final = file_extension.replace('.', '')
            try:
                track = AudioSegment.from_file(filepath, format=file_extension_final)
                wav_filename = filepath.replace(file_extension_final, 'wav')
                wav_path = wav_filename
                # print('CONVERTING: ' + str(filepath))
                file_handle = track.export(wav_path, format='wav')
                os.remove(filepath)
            except:
                continue
                print("ERROR CONVERTING " + str(filepath))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.dev_data), "Couldn't find the dataset at {}".format(args.dev_data)
    tic = time.time()
    convert_m4a_2_wav(args.dev_data)
    toc = time.time()
    print('Done in {:.4f} seconds'.format(toc - tic))