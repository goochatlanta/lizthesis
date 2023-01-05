"""Evaluates the model"""
import glob
import pickle
from pyexpat import model
import random
import argparse
import logging
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import needed_functions as needed_functions
import model.net as net
import model.data_loader as data_loader
import model.distribution as distribution


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', 
                    default='/home/elizabeth.gooch/data/VoxCeleb_data/VoxCeleb/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--distributional_model_dir', default='experiments/distribution_models',
                    help="Directory containing saved model outputs")
parser.add_argument('--logistic_dir', default='experiments/distribution_models/bestModel_logistic',
                    help="Directory of best models from logistic regressions")
parser.add_argument('--svm_dir', default='experiments/distribution_models/bestModel_svm',
                    help="Directory of best models from SVM")
parser.add_argument('--rf_dir', default='experiments/distribution_models/bestModel_rf',
                    help="Directory of best models from RF")

def evaluate(best_model_file, test_data, record_dictionary, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # best_model = os.path.join(args.distributional_model_dir, best_model_file)
    with open(best_model_file, "rb") as f:
        loaded_model = pickle.load(f)

    return record_dictionary(test_data, loaded_model)
        

def eval(directory, feature_label, record_dictionary, dataseed, model_dictionary):
    for best_model_file in glob.iglob(f'{directory}/*'):
                
                if best_model_file.find(feature_label) != -1:
                    
                    with open(best_model_file, "rb") as f:
                        loaded_model = pickle.load(f)
                        dictionary = record_dictionary(test_full, loaded_model)
                        dictionary['testseed'] = dataseed
                        model_file = best_model_file.split("/")[3]
                        model_name = model_file.split(".")[0]
                        if model_name in model_dictionary:
                            model_dictionary[model_name].append(dictionary.copy())
                        else:
                            model_dictionary[model_name] = [dictionary.copy()]

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = needed_functions.Params(json_path)

    eval_all = params.eval_length
    record_dictionary = distribution.record_model_performance

    random.seed(42)
    randlist = random.sample(range(100000), 4)
    print(randlist)

    needed_functions.set_logger(os.path.join(args.model_dir, 'evaluate.log'))


    evalVox_log_dict = {}
    evalVox_svm_dict = {}
    evalVox_rf_dict = {}


    def save_dictionary(dict_file, save_name):
        file_path = os.path.join(args.distributional_model_dir,save_name)
        with open(file_path, 'wb') as file:
            pickle.dump(dict_file, file)


    for dataseed in randlist: 
        # evalVox_log_dict.setdefault(dataseed, [])
        test_pickle = 'sample_test_coed_seed' + str(dataseed)

        for feature_label in ['mfcc13', 'mfcc26', 'mfcc40']:
            # model_eval = []          
            test_dl = data_loader.fetch_dataloader(test_pickle, feature_label, eval_all, params)
            test_full = next(iter(test_dl))

            eval(args.logistic_dir, feature_label, record_dictionary, dataseed, evalVox_log_dict)
            eval(args.svm_dir, feature_label, record_dictionary, dataseed, evalVox_svm_dict)
            eval(args.rf_dir, feature_label, record_dictionary, dataseed, evalVox_rf_dict)

    
    save_dictionary(evalVox_log_dict, 'VoxEvaluation_logistic_dictionary.pickle')
    save_dictionary(evalVox_svm_dict, 'VoxEvaluation_svm_dictionary.pickle')
    save_dictionary(evalVox_rf_dict, 'VoxEvaluation_rf_dictionary.pickle')



    # evalAL_log_dict = {}
    # evalAL_svm_dict = {}
    # evalAL_rf_dict = {}

    # for dataseed in randlist: 
    #     # evalVox_log_dict.setdefault(dataseed, [])
    #     test_pickle = 'sample_test_Alsaify_seed' + str(dataseed)

    #     for feature_label in ['mfcc13', 'mfcc26', 'mfcc40']:
    #         # model_eval = []          
    #         test_dl = data_loader.fetch_AlsaifyDataloader(test_pickle, feature_label, eval_all, params)
    #         test_full = next(iter(test_dl))

    #         eval(args.logistic_dir, feature_label, record_dictionary, dataseed, evalVox_log_dict)
    #         eval(args.svm_dir, feature_label, record_dictionary, dataseed, evalVox_svm_dict)
    #         eval(args.rf_dir, feature_label, record_dictionary, dataseed, evalVox_rf_dict)

    
    # save_dictionary(evalAL_log_dict, 'AlsaifyEvaluation_logistic_dictionary.pickle')
    # save_dictionary(evalAL_svm_dict, 'AlsaifyEvaluation_svm_dictionary.pickle')
    # save_dictionary(evalAL_rf_dict, 'AlsaifyEvaluation_rf_dictionary.pickle')

