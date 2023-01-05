import torch
import numpy as np
import pickle
import os
import argparse
# example of grid searching key hyperparametres for logistic regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
# from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn import metrics
# naive grid search implementation
from sklearn.svm import SVC
# from sklearn_train_evaluate import record_model_performance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--distributional_model_dir', default='experiments/distribution_models',
                    help="Directory containing saved model outputs")
args = parser.parse_args()
"""Defines the distribution based models"""

import pandas as pd


def print_dataframe(filtered_cv_results):
    """Pretty print for filtered dataframe"""
    for mean_precision, std_precision, mean_recall, std_recall, params in zip(
        filtered_cv_results["mean_test_precision"],
        filtered_cv_results["std_test_precision"],
        filtered_cv_results["mean_test_recall"],
        filtered_cv_results["std_test_recall"],
        filtered_cv_results["params"],
    ):
        print(
            f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
            f" for {params}"
        )
    print()


def refit_strategy(cv_results):
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter-out all results below a precision threshold
    of 0.98, rank the remaining by recall and keep all models with one standard
    deviation of the best by recall. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    # print the info about the grid-search for the different scores
    precision_threshold = 0.98

    cv_results_ = pd.DataFrame(cv_results)
    print("All grid-search results:")
    print_dataframe(cv_results_)

    # Filter-out all results below the threshold
    high_precision_cv_results = cv_results_[
        cv_results_["mean_test_precision"] > precision_threshold
    ]

    print(f"Models with a precision higher than {precision_threshold}:")
    print_dataframe(high_precision_cv_results)

    high_precision_cv_results = high_precision_cv_results[
        [
            "mean_score_time",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "rank_test_recall",
            "rank_test_precision",
            "params",
        ]
    ]

    # Select the most performant models in terms of recall
    # (within 1 sigma from the best)
    best_recall_std = high_precision_cv_results["mean_test_recall"].std()
    best_recall = high_precision_cv_results["mean_test_recall"].max()
    best_recall_threshold = best_recall - best_recall_std

    high_recall_cv_results = high_precision_cv_results[
        high_precision_cv_results["mean_test_recall"] > best_recall_threshold
    ]
    print(
        "Out of the previously selected high precision models, we keep all the\n"
        "the models within one standard deviation of the highest recall model:"
    )
    print_dataframe(high_recall_cv_results)

    # From the best candidates, select the fastest model to predict
    fastest_top_recall_high_precision_index = high_recall_cv_results[
        "mean_score_time"
    ].idxmin()

    print(
        "\nThe selected final model is the fastest to predict out of the previously\n"
        "selected subset of best models based on precision and recall.\n"
        "Its scoring time is:\n\n"
        f"{high_recall_cv_results.loc[fastest_top_recall_high_precision_index]}"
    )

    return fastest_top_recall_high_precision_index




def prepare_data(dataloader):
    t1, t2, labels = dataloader
    # full tensor
    X = torch.cat((t1, t2), dim=1)
    X = torch.reshape(X, (t1.shape[0], 2*t1.shape[1]*t1.shape[2]))
    return X.numpy(), labels.numpy()
 
scaler = StandardScaler()
scores = ["precision", "recall"]
 




def logistic_regression_training(training_data, seed, params):
    
    X_train, y_train = prepare_data(training_data)

    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=seed)
    # liblinear is recommended when you have a high dimension dataset - solving large-scale classification problems.
    pipe = Pipeline(steps=[("scaler", scaler), ("logistic", model)])

    # Parameters of pipelines can be set using '_ _' separated parameter names:
    param_grid = {
        "logistic__C": np.logspace(-4, 4, 4), 
        "logistic__max_iter": list(range(500, 2000, 100))
    }

    search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=3, refit=refit_strategy, scoring=scores)
    search.fit(X_train, y_train)

    return search.best_params_, search.best_estimator_





def svm_training(training_data, seed, params):

    X_train, y_train = prepare_data(training_data)

    model = SVC(random_state=seed)
    pipe = Pipeline(steps=[("scaler", scaler), ("SCV", model)])
    
    param_grid = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]
    
    search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=3, scoring='f1')
    search.fit(X_train, y_train)
    x=1
    return search.best_params_, search.best_estimator_






def rf_training(training_data, seed, params):

    X_train, y_train = prepare_data(training_data)
    model = RandomForestClassifier(n_jobs=params.num_workers, random_state=seed)
    pipe = Pipeline(steps=[("scaler", scaler), ("rf", model)])

    param_grid = {
        "rf__max_depth": list(range(2, 42, 10)),
        "rf__min_samples_split": list(range(10, 30, 10)),
        'rf__min_samples_leaf': list(range(10, 30, 10)),
        'rf__n_estimators': list(range(70, 350, 70))
    }

    search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=3, refit=refit_strategy, scoring=scores)
    search.fit(X_train, y_train)

    return search.best_params_, search.best_estimator_


# def ab_training(training_data, seed, params):

#     X_train, y_train = prepare_data(training_data)
#     model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=seed))
#     pipe = Pipeline(steps=[("scaler", scaler), ("ab", model)])

#     param_grid = {
#         'base_estimator__max_depth':[i for i in range(2,11,2)],
#         'base_estimator__min_samples_leaf':[5,10],
#         'n_estimators':[10,50,250,1000],
#         'learning_rate':[0.01,0.1]
#     }

#     search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=3, scoring='f1')
#     search.fit(X_train, y_train)

#     return search.best_score_, search.best_params_, search.best_estimator_






###############################################################################################################
# Model record

def model_highlights_dictionary(best_model, X, y, model_dict = {}):
    # Record some information
 
    prediction = best_model.predict(X)

    score = best_model.score(X, y)
    model_dict['accuracy'] = score


    def record_evaluation(function, fn_label):
        
        value = function
        key = fn_label
        model_dict[key] = value
        return value


    record_evaluation(metrics.confusion_matrix(y, prediction), 'confusion_matrix', )
    record_evaluation(metrics.f1_score(y, prediction), 'f1_score')
    fprs, tprs, thresholds = record_evaluation(metrics.roc_curve(y, prediction), 'roc_curve')


    def compute_eer(fpr, tpr, thresholds):
        """ Returns equal error rate (EER) and the corresponding threshold. """
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        return eer, thresholds[min_index]

    
    eer, eer_threshold = compute_eer(fprs, tprs, thresholds)
    model_dict['eer'] = eer
    model_dict['eer_threshold'] = eer_threshold

    return model_dict


def record_model_performance(data, best_model):
    X, y = prepare_data(data)

    record = model_highlights_dictionary(best_model, X, y)
    return record


