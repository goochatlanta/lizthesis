"""Train the model"""

import argparse
from ast import mod
import logging
import os
import pickle
import re
import numpy as np
import random
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import time 
import needed_functions as needed_functions
import model.net as net
import model.distribution as distribution
import model.data_loader as data_loader
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', 
                    default='/home/elizabeth.gooch/data/VoxCeleb_data/VoxCeleb/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--distributional_model_dir', default='experiments/distribution_models',
                    help="Directory containing saved model outputs")

  

def train_and_validate(training_model, training_data, feature_label, model_label, dataseed, seed, params):
    """Train the model and evaluate across full dataset.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        model_dir: (string) directory containing config, weights and log
    """    
    best_score, best_params, best_estimator = training_model(training_data, seed, params)
    model_name = feature_label + "_datasetSeed" + str(dataseed)  + "_seed" + str(seed) + "_bestModel"

    pkl_filename = "bestModel_" + model_label + "/" + model_name + ".pickle"
    file_path = os.path.join(args.distributional_model_dir, pkl_filename)
    with open(file_path, 'wb') as file:
        pickle.dump(best_estimator, file)

    return best_score, best_params, model_name




if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = needed_functions.Params(json_path)

    training_all = params.training_length
    eval_all = params.eval_length
    record_dictionary = distribution.record_model_performance

    random.seed(42)
    randlist = random.sample(range(100000), 1)
    print(randlist)

    needed_functions.set_logger(os.path.join(args.model_dir, 'train.log'))

    trainVox_log_dict = {}
    trainVox_svm_dict = {}
    trainVox_rf_dict = {}
    # trainVox_ab_dict = {}
    

    def save_dictionary(dict_file, save_name):
        file_path = os.path.join(args.distributional_model_dir,save_name)
        with open(file_path, 'wb') as file:
            pickle.dump(dict_file, file)


    def update_dictionary(model_name, best_parameters, dict_name):
        if model_name in dict_name:
            dict_name[model_name].append(best_parameters.copy())
        else:
            dict_name[model_name] = [best_parameters.copy()]

    tic_all = time.time()

    for dataseed in randlist: 
        train_pickle = 'sample_train_coed_seed' + str(dataseed)
        val_pickle = 'sample_val_coed_seed' + str(dataseed)

        for feature_label in ['mfcc13']:
            
            logging.info("BEGIN data loading and training for..." + train_pickle + " " + feature_label)
            tic0 = time.time()
            train_dl = data_loader.fetch_dataloader(train_pickle, feature_label, training_all, params)
            train_full = next(iter(train_dl))
            # val_dl = data_loader.fetch_dataloader(val_pickle, feature_label, eval_all, params)
            # val_full = next(iter(val_dl))
            toc0 = time.time()
            logging.info('Data loaded in {:.4f} seconds'.format(toc0 - tic0))

            tic1 = time.time()

            for seed in randlist:                  
                tic2 = time.time()

                for model_label in ['logistic', 'svm', 'rf', 'ab']:

                    # if model_label == 'logistic':
                    #     training_model = distribution.logistic_regression_training
                    #     best_params, model_name = train_and_validate(training_model, train_full, feature_label, model_label, dataseed, seed, params)                 
                    #     # best_params["score"] = best_score
                    #     logging.info('\n')
                    #     # logging.info("Best score for " + model_label + ": " + str(best_score))
                    #     update_dictionary(model_name, best_params, trainVox_log_dict)
                       
                    if model_label == 'svm':
                        training_model = distribution.svm_training
                        best_params, model_name = train_and_validate(training_model, train_full, feature_label, model_label, dataseed, seed, params)                 
                        # best_params["score"] = best_score
                        logging.info('\n')
                        # logging.info("Best score for " + model_label + ": " + str(best_score))
                        update_dictionary(model_name, best_params, trainVox_svm_dict)

                    elif model_label == 'rf':
                        training_model = distribution.rf_training
                        best_params, model_name = train_and_validate(training_model, train_full, feature_label, model_label, dataseed, seed, params)                 
                        # best_params["score"] = best_score
                        logging.info('\n')
                        # logging.info("Best score for " + model_label + ": " + str(best_score))   
                        logging.info('\n')                     
                        update_dictionary(model_name, best_params, trainVox_rf_dict)

                    # if model_label == 'ab':
                    #     training_model = distribution.ab_training
                    #     best_score, best_params, model_name = train_and_validate(training_model, train_full, feature_label, model_label, dataseed, seed, params)                 
                    #     best_params["score"] = best_score
                    #     logging.info('\n')
                    #     logging.info("Best score for " + model_label + ": " + str(best_score))   
                    #     logging.info('\n')                     
                    #     update_dictionary(model_name, best_params, trainVox_ab_dict)
                
                toc2 = time.time()
                logging.info('Training for SEED' + str(seed) + ' on one dataset trained in {:.4f} seconds'.format(toc2 - tic2))
                logging.info('\n')

            toc1 = time.time()
            logging.info('All model SEEDS on one dataset trained in {:.4f} seconds'.format(toc1 - tic1))
            logging.info('\n')



    save_dictionary(trainVox_log_dict, 'VoxTrained_logistic_dictionary.pickle')
    save_dictionary(trainVox_svm_dict, 'VoxTrained_svm_dictionary.pickle')
    save_dictionary(trainVox_rf_dict, 'VoxTrained_rf_dictionary.pickle')
    # save_dictionary(trainVox_ab_dict, 'VoxTrained_rf_dictionary.pickle')

    toc_all = time.time()
    logging.info('All training DONE in {:.4f} seconds'.format(toc_all - tic_all))














    # for model_label in ['logistic', 'svm', 'rf']:
    #     # Train and evaluate the models

    #     if model_label == 'logistic':
    #         tic = time.time()
    #         training_model = distribution.logistic_regression_training

    #         model_dictionary = train_and_return(randlist, training_model)

    #         pkl_filename = "bestParameters_logistic_dictionary.pickle"
    #         file_path = os.path.join(args.distributional_model_dir, pkl_filename)
    #         with open(file_path, 'wb') as file:
    #             pickle.dump(model_dictionary, file)
    #         toc = time.time()
    #         logging.info('\n')
    #         logging.info('#####################################################################################')
    #         logging.info('Logistic regression training all 100 samples done in {:.4f} seconds'.format(toc - tic))
    #         logging.info('\n')
    #         log_dict = model_dictionary
        
    #     elif model_label == 'svm':
    #         tic = time.time()
    #         training_model = distribution.svm_training

    #         model_dictionary = train_and_return(randlist, training_model)

    #         pkl_filename = "bestParameters_svm_dictionary.pickle"
    #         file_path = os.path.join(args.distributional_model_dir, pkl_filename)
    #         with open(file_path, 'wb') as file:
    #             pickle.dump(model_dictionary, file)
    #         toc = time.time()
    #         logging.info('\n')
    #         logging.info('################################################################################')
    #         logging.info('SVM regression training all 100 samples done in {:.4f} seconds'.format(toc - tic))
    #         logging.info('\n')
    #         svm_dict = model_dictionary

    #     elif model_label == 'rf':
    #         tic = time.time()
    #         training_model = distribution.rf_training

    #         model_dictionary = train_and_return(randlist, training_model)

    #         pkl_filename = "bestParameters_rf_dictionary.pickle"
    #         file_path = os.path.join(args.distributional_model_dir, pkl_filename)
    #         with open(file_path, 'wb') as file:
    #             pickle.dump(model_dictionary, file)
    #         toc = time.time()
    #         logging.info('\n')
    #         logging.info('###############################################################################')
    #         logging.info('RF regression training all 100 samples done in {:.4f} seconds'.format(toc - tic))
    #         logging.info('\n')

                   
    #     toc_all = time.time()
    #     logging.info('\n')
    #     logging.info('#############################################################')
    #     logging.info('#############################################################')
    #     logging.info('900 samples done in {:.4f} seconds'.format(toc_all - tic_all))
    #     logging.info('\n')

                
   