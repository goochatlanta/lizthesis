"""Aggregates results from best_model in a parent folder"""

import argparse
import json
import os
from pathlib import Path
from this import d
from tabulate import tabulate
import pandas as pd
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--performance_dir', default='/home/elizabeth.gooch/lizthesis/experiments/distribution_models/performance_dictionaries',
                    help='Directory containing results of experiments')
parser.add_argument('--tables_and_figures_dir', default='/home/elizabeth.gooch/lizthesis/output',
                    help='Directory containing results as figures')

def create_df(folder, df_dict = {}):
    dictionaries_list = list(Path(folder).glob('*'))


    for posixfilepath in dictionaries_list: 
        # cut name to make labels
        dict_name = posixfilepath.parts[7]

        model_type = dict_name.split("_")[0]
        if "model_type" in df_dict:
            df_dict["model_type"].append(model_type)
        else:
            df_dict["model_type"]=[model_type]

        feature_type = dict_name.split("_")[1]
        if "feature_type" in df_dict:
            df_dict["feature_type"].append(feature_type)
        else:
            df_dict["feature_type"]=[feature_type]
        
        dsSeed = dict_name.split("_")[2]
        if "dsSeed" in df_dict:
            df_dict["dsSeed"].append(dsSeed)
        else:
            df_dict["dsSeed"]=[dsSeed]

        rSeed = dict_name.split("_")[3]
        if "rSeed" in df_dict:
            df_dict["rSeed"].append(rSeed)
        else:
            df_dict["rSeed"]=[rSeed]

        
        with open(posixfilepath.as_posix(), "rb") as f:
            loaded_dictionary = pickle.load(f)

        eer = loaded_dictionary.get('eer')
        if "eer" in df_dict: 
            df_dict["eer"].append(eer)
        else: 
            df_dict["eer"]=[eer]

        eer_threshold = loaded_dictionary.get('eer_threshold')
        if "eer_threshold" in df_dict: 
            df_dict["eer_threshold"].append(eer_threshold)
        else: 
            df_dict["eer_threshold"]=[eer_threshold]

    df = pd.DataFrame(data=df_dict) 
    return df.sort_values(by=['eer']) 



def add_roc_curves(folder):
    dictionaries_list = list(Path(folder).glob('*'))


    for posixfilepath in dictionaries_list: 
        # cut name to make labels
        dict_name = posixfilepath.parts[7]
        model_type = dict_name.split("_")[0]
        feature_type = dict_name.split("_")[1]
        dsSeed = dict_name.split("_")[2]
        rSeed = dict_name.split("_")[3]
                
        with open(posixfilepath.as_posix(), "rb") as f:
            loaded_dictionary = pickle.load(f)
        
        fpr, tpr, thresholds =loaded_dictionary.get('roc_curve')
        label = dsSeed + ", " + rSeed
        plt.plot(fpr, tpr, label=label)



if __name__ == "__main__":
    args = parser.parse_args()

print("Creating results..")
df = create_df(args.performance_dir)

latex_table = df.to_latex(index=False)
with open(os.path.join(args.tables_and_figures_dir, 'eer_baseline.tex'), 'wb') as file:
        pickle.dump(latex_table, file)


#set up plotting area
plt.figure(0).clf()

add_roc_curves(args.performance_dir)

#add legend
plt.legend()
plt.savefig(os.path.join(args.tables_and_figures_dir, 'roc.png'))
plt.show()


# load pickles of performance dictionaries
# create labels from file title

# extract the eer test
# create pandas dataframe
# export to latex
# save latex

# extract the roc data
# create graphic 
# save graphic