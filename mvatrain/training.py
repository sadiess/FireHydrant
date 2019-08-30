#!/usr/bin/env python

import os
from os.path import join
import pickle
import time

import awkward
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from mvatrain.xgbo import XgboClassifier

#TIME_STR = time.strftime("%y%m%d")  # 190530
TIME_STR = "190815"
DATAFILE_NAME = join(os.environ["FFANA_BASE"], f"mvatrain/data/combo_{TIME_STR}_2.awkd") #data for training
OUTPUT_DIR = join(os.environ["FFANA_BASE"], f"mvatrain/outputs/irish_breakfast_redux") #this is where the completed model will be saved

def main():

    # -------------------------------------------------------------------------

    train_size = 0.75

    # The space of hyperparameters for the Bayesian optimization
    hyperparams_ranges = {
        "min_child_weight": (1, 30),
        "colsample_bytree": (0.1, 1),
        "max_depth": (2, 14),
        "subsample": (0.5, 1),
        "eta": (0.0, 0.5),
        "gamma": (0, 20),
        "reg_alpha": (0, 10),
        "reg_lambda": (0, 20),
    }

    # The default xgboost parameters
    xgb_default = {
        "min_child_weight": 1,
        "colsample_bytree": 1,
        "max_depth": 6,
        "subsample": 1,
        "eta": 0.3,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
    }

    # -------------------------------------------------------------------------

    dataset_ = awkward.load(DATAFILE_NAME)
    df = pd.DataFrame(dict(dataset_))
    df.fillna(0)

    # -------------------------------------------------------------------------

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

        output = open(OUTPUT_DIR + "/param_range.pkl", "wb")
        pickle.dump(hyperparams_ranges, output)
        output.close()
        output = open(OUTPUT_DIR + "/param_default.pkl", "wb")
        pickle.dump(xgb_default, output)
        output.close()

    feature_cols = [n for n in dataset_.keys() if n != "target"]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["target"], random_state=42, test_size=1 - train_size
    )

    # Entries from the class with more entries are discarded. This is because
    # classifier performance is usually bottlenecked by the size of the
    # dataset for the class with fewer entries. Having one class with extra
    # statistics usually just adds computing time.
    n_per_class = min(y_train.value_counts())

    # The number of entries per class might also be limited by a parameter
    # in case the dataset is just too large for this algorithm to run in a
    # reasonable time.

    selection = np.concatenate(
        [
            y_train[y_train == 0].head(n_per_class).index.values,
            y_train[y_train == 1].head(n_per_class).index.values,
        ]
    )

    X_train = X_train.loc[selection]
    y_train = y_train.loc[selection]

    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtest = xgb.DMatrix(X_test, label=y_test)

    # -------------------------------------------------------------------------

    print("Running bayesian optimized training...")
    xgbo_classifier = XgboClassifier(out_dir=OUTPUT_DIR, early_stop_rounds=25)

    xgbo_classifier.optimize(xgtrain, init_points=5, n_iter=10, acq="ei")

    xgbo_classifier.fit(xgtrain, model="default")
    xgbo_classifier.fit(xgtrain, model="optimized")

    xgbo_classifier.save_model(feature_cols, model="default")
    xgbo_classifier.save_model(feature_cols, model="optimized")

    preds_default = xgbo_classifier.predict(xgtest, model="default")
    preds_optimized = xgbo_classifier.predict(xgtest, model="optimized")

    # -------------------------------------------------------------------------

    print("Saving reduced data frame...")
    # Create a data frame with bdt outputs and kinematics to calculate the working points
    df_reduced = df.loc[y_test.index, ["pt", "eta", "target"]]
    df_reduced["bdt_score_default"] = preds_default
    df_reduced["bdt_score_optimized"] = preds_optimized
    df_reduced.to_hdf(os.path.join(OUTPUT_DIR, "pt_eta_score.h5"), key="pt_eta_score")


if __name__ == "__main__":
    main()
