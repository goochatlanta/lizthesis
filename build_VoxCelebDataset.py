"""Organizing data from VoxCeleb2

In file-centric manipulations: 
    Files start off in LARGE .zip files --> TODO unzip using bash
    Files start off as .m4a files --> TODO convert to .wav in place using pydub.AudioSegment

The .wav files live in a folder outside of this repo in the structure:
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

Considerations in organizing the annotations for input, I want to:
1. split the dev set into training and validation that is speaker disjointed. 
2. have an equal number of male and female speakers in each. (What is test?)
3. create indication of gender-specific sample based on the VoxCeleb meta data
3. pair speakers into combinations and create label of same/different speaker. 
4. have these pairs be specific to sample type: female only, male only, mixed. Mixed is a different size. 
    Should I limit it to be equal to Female and Male?
5. know about the distributions of speakers by uri and by gender

Dataset:
- (((x1_filepath, x2_filepath), Boolean label), (x1_label, x2_label), split_type, sample_type)

Steps:
1. Split_type for dev
2. sample_type of what gender of speaker are included in combination creation
3. create enough combinations to get balance on label
4. (In DatasetClass) draw 400,000 observations balanced by label for training

For audio import: https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

# To start: 
1. srun --pty --partition=beards --gres=gpu:1 --nodelist=compute-8-13 bash
2. module load lang/miniconda3/4.10.3
3. conda activate pytorch_audio

However, if I enter Hamming via: srun --pty bash, then I get 62 cpu's!

"""
from cgi import test
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
import needed_functions as needed_functions
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import itertools
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from itertools import islice
import model.data_loader as data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='/home/elizabeth.gooch/data/VoxCeleb_data/', 
                    help="My directory with VoxCeleb2 files")
parser.add_argument('--annotation_directory', 
                    default='/home/elizabeth.gooch/lizthesis/data',
                    help='Directory containing the annotations')
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
args = parser.parse_args()


def get_gendered_lists(all_list, male_list = [], female_list =[]):

    for obs in all_list:
        if obs[1]=="f":
            female_list.append(obs[0])
        else:
            male_list.append(obs[0])

    return female_list, male_list


def include_keys(dictionary, keys):
    """Filters a dict by only including certain keys."""
    key_set = set(keys) & set(dictionary.keys())
    return {key: dictionary[key] for key in key_set}



def random_different(dictionary, size, different_list = []):
    """Create list of random pairs from different speakers"""
    for _ in range(0,int(size)):
        # randomly select two id's
        keys = [random.choice(list(dictionary)) for i in range(2)]
        if keys[0]!=keys[1]:
            pair = []
            for key in keys:
                files = dictionary.get(key)
                random_file = random.choice(list(files))
                pair.append(random_file)
            pair.append(False)
            different_list.append(pair)
            
        else:
            continue
    
    return different_list



def random_same(dictionary, size, same_list = []):
    """Create list of random pairs from different speakers"""
    for _ in range(0,int(size)):
        # randomly select two id's
        key = random.choice(list(dictionary))
        files = dictionary.get(key)
        random_files = [random.choice(list(files)) for i in range(2)]
        random_files.append(True)
        same_list.append(random_files)

    return same_list


def create_sample(id_type, id_type_name, type, total_size, sample = []):
    """Create list of random pairs from same and different speakers"""
    
    # bring in dictionary_pickle - dev or test
    annotation_dictionary = "/annotation_dict_" + type + ".pickle"
    filepath = os.path.join(args.annotation_directory + annotation_dictionary)
    with open(filepath, "rb") as f:
        my_dict = pickle.load(f)

    # refine dictionary to split typee
    new_dict = include_keys(my_dict, id_type)
    print("Old dictionary length: ", len(my_dict))
    print("New dictionary length: ", len(new_dict))

    same_list = random_same(new_dict, total_size/2)
    print("Same list length: ", len(same_list))

    diff_list = random_different(new_dict, total_size/2)
    print("Diff list length: ", len(diff_list))

    sample = list(islice(reversed(same_list), 0, int(total_size/2))) + list(islice(reversed(diff_list), 0, int(total_size/2)))
    sample.reverse()    
    print(len(sample))
    sample_file = '/sample_' + id_type_name + '.pickle'
    savepath = os.path.join(args.annotation_directory + sample_file)
    with open(savepath, 'wb') as g:
        pickle.dump(sample, g)



if __name__ == '__main__':
        # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = needed_functions.Params(json_path)


    # location of all VoxCeleb2 data at highest level
    assert os.path.isdir(args.data_folder), "Couldn't find the dataset at {}".format(args.data_folder)

    """For test data
    1. Using speaker ID:gender from mets file, create female-only, male-only, mixed samples of equal size 
        and allow full mixed sample (a natural benefit of mixed) of speakers.
    2. For each sample, make combinations and labels and ensure that there are at least 250,000 observations 
        of same speaker and different speaker.

    ??? When do I annotate? When do I extract the speaker ID to identify the gender? Do I split the annotation? 
        When do I do the train/val split?
    """

    vox2_meta = '/home/elizabeth.gooch/lizthesis/data/vox2_meta.csv'

    dev_meta = []
    test_meta = []
    with open(vox2_meta, newline='') as csvfile:
        meta = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(meta)
        for line in meta:
            split_type = line[3].strip()
            if split_type=="dev":
                speaker = line[0].strip()
                gender = line[2].strip()
                dev_meta.append((speaker, gender))
            elif split_type=="test":
                speaker = line[0].strip()
                gender = line[2].strip()
                test_meta.append((speaker, gender))

    dev_gender = list(zip(*dev_meta))[1]

    train_meta, val_meta = train_test_split(dev_meta, 
                                            test_size=0.20, 
                                            stratify=dev_gender,
                                            random_state=42)

    train_female_ids, train_male_ids = get_gendered_lists(train_meta)
    val_female_ids, val_male_ids = get_gendered_lists(val_meta)
    test_female_ids, test_male_ids = get_gendered_lists(test_meta)

    random.seed(42)



    ######################################################################################################################
    # For creating ONE combination of audio pairs. 

    train_name = 'train_coed'
    val_name = 'val_coed'
    test_name = 'test_coed'

    tic = time.time()
    train_coed_ids = list(zip(*train_meta))[0]
    create_sample(train_coed_ids, train_name, 'dev', params.training_length)
    toc = time.time()
    print('Training sample done in {:.4f} seconds'.format(toc - tic))
    
    tic = time.time()
    val_coed_ids = list(zip(*val_meta))[0]
    create_sample(val_coed_ids, val_name, 'dev', params.eval_length)
    toc = time.time()
    print('Validation sample done in {:.4f} seconds'.format(toc - tic))

    tic = time.time()
    test_coed_ids = list(zip(*test_meta))[0]
    create_sample(test_coed_ids, test_name, 'test', params.eval_length)
    toc = time.time()
    print('Test sample done in {:.4f} seconds'.format(toc - tic))




    ######################################################################################################################
    # For creating many combinations of audio pairs. 

    # randlist = random.sample(range(100000), 2)

    # for seed in randlist:
    #     train_name = 'train_coed_seed' + str(seed)
    #     val_name = 'val_coed_seed' + str(seed)
    #     test_name = 'test_coed_seed' + str(seed)

    #     tic = time.time()
    #     train_coed_ids = list(zip(*train_meta))[0]
    #     create_sample(train_coed_ids, train_name, 'dev', params.training_length)
    #     toc = time.time()
    #     print('Training sample done in {:.4f} seconds'.format(toc - tic))
        
    #     tic = time.time()
    #     val_coed_ids = list(zip(*val_meta))[0]
    #     create_sample(val_coed_ids, val_name, 'dev', params.eval_length)
    #     toc = time.time()
    #     print('Validation sample done in {:.4f} seconds'.format(toc - tic))
    
    #     tic = time.time()
    #     test_coed_ids = list(zip(*test_meta))[0]
    #     create_sample(test_coed_ids, test_name, 'test', params.eval_length)
    #     toc = time.time()
    #     print('Test sample done in {:.4f} seconds'.format(toc - tic))






    #############################################################################################################
    # For SKlearn models
    # Create static datasets

    # training_all = params.training_length
    # eval_all = params.eval_length

    # for dataseed in randlist: 
    #         train_pickle = 'sample_train_coed_seed' + str(dataseed)
    #         val_pickle = 'sample_val_coed_seed' + str(dataseed)

    #         for seed in randlist:                  

    #             for feature_label in ['mfcc13','mfcc26','mfcc40']:
                    
    #                 tic0 = time.time()
    #                 train_name = "FULLtrain_" + feature_label + "_dataSeed" + str(dataseed)
    #                 train_dl = data_loader.fetch_dataloader(train_pickle, feature_label, training_all, params)
    #                 train_full = next(iter(train_dl))
                    
    #                 file_path = os.path.join(args.annotation_directory, train_name)
    #                 with open(file_path, 'wb') as file:
    #                     pickle.dump(train_full, file)


    #                 val_name = "FULLval_" + feature_label + "_dataSeed" + str(dataseed)
    #                 val_dl = data_loader.fetch_dataloader(val_pickle, feature_label, eval_all, params)
    #                 val_full = next(iter(val_dl))

    #                 file_path = os.path.join(args.annotation_directory, val_name)
    #                 with open(file_path, 'wb') as file:
    #                     pickle.dump(val_full, file)

    #                 toc0 = time.time()
    #                 print('Single set of training and val data readied in {:.4f} seconds'.format(toc0 - tic0))