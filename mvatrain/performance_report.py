#!/usr/bin/env python
import os
from os.path import join
import time

import awkward
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import utils.histoHelpers as uhh
from mvatrain.ROCPlot import ROCPlot
from mvatrain.hist_errorbars import hist_errorbars

plt.style.use("default")
plt.rcParams.update(uhh.ffPltStyleSettings)

#TIME_STR = time.strftime("%y%m%d")  # 190530
TIME_STR = "190815"
TIME_STR_CURRENT = "190815"
COMBO_PATH = join(os.environ["FFANA_BASE"], f"mvatrain/data/combo_190812.awkd")
MU_DATA_PATH = join(os.environ["FFANA_BASE"], f"mvatrain/data/combo_{TIME_STR}_1.awkd")
ELEC_DATA_PATH = join(os.environ["FFANA_BASE"], f"mvatrain/data/combo_{TIME_STR}_2.awkd")
OUTPUT_DIR = join(os.environ["FFANA_BASE"], f"mvatrain/outputs/irish_breakfast_redux")  #model
REPORT_DIR = join(os.environ["FFANA_BASE"], f"mvatrain/reports/{TIME_STR_CURRENT}_5")


def main():
    """
    Evaluate optimization result by:
    - prediction distribution
    - feature importance
    - ROC
    - accuracy score
    - AUC score
    - classification report
    """

    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    # load model
    print("loading model...")
    xgbm_default = xgb.Booster({"nthread": 16})
    xgbm_default.load_model(join(OUTPUT_DIR, "model_default/model.bin"))
    if xgbm_default.attributes().get('SAVED_PARAM_predictor', None)=='gpu_predictor':
        xgbm_default.set_attr(SAVED_PARAM_predictor=None)
    xgbm_optimized = xgb.Booster({"nthread": 16})
    xgbm_optimized.load_model(join(OUTPUT_DIR, "model_optimized/model.bin"))
    if xgbm_optimized.attributes().get('SAVED_PARAM_predictor', None)=='gpu_predictor':
        xgbm_optimized.set_attr(SAVED_PARAM_predictor=None)

    # load data
    print("loading data...")
    dataset_ = awkward.load(ELEC_DATA_PATH)
    df = pd.DataFrame(dict(dataset_))
    df.fillna(0)
    feature_cols = [n for n in dataset_.keys() if n != "target"]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df["target"], random_state=42, test_size=0.25
    )
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtest = xgb.DMatrix(X_test, label=y_test)

    train_preds_default = xgbm_default.predict(xgtrain)
    train_preds_optimized = xgbm_optimized.predict(xgtrain)
    test_preds_default = xgbm_default.predict(xgtest)
    test_preds_optimized = xgbm_optimized.predict(xgtest)

    # prediction disttribution
    print(f"Making prediction distribution plots under {REPORT_DIR}")
    commonkw = {
        "range": [-12, 12],
        "bins": 50,
        "histtype": "stepfilled",
        "alpha": 0.75,
        "density": True,
        "linewidth": 2,
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(train_preds_default[y_train], label="signal (train)", **commonkw)
    ax.hist(train_preds_default[~y_train], label="background (train)", **commonkw)
    commonkw.pop("histtype")
    commonkw.pop("alpha")
    hist_errorbars(test_preds_default[y_test], ax=ax, **commonkw).set_label(
        "signal (test)"
    )
    hist_errorbars(test_preds_default[~y_test], ax=ax, **commonkw).set_label(
        "background (test)"
    )

    ax.legend(title="default")
    ax.set_xlabel("BDT score", ha="right", x=1)
    ax.set_ylabel("A.U.", ha="right", y=1)
    ax.set_title("leptonJet BDT prediction value", ha="left", x=0)
    plt.savefig(join(REPORT_DIR, "prediction_dist_default.pdf"), bbox_inches='tight')
    plt.close()

    commonkw.update({"range": [-10, 10], "histtype": "stepfilled", "alpha": 0.8})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(train_preds_optimized[y_train], label="signal (train)", **commonkw)
    ax.hist(train_preds_optimized[~y_train], label="background (train)", **commonkw)
    commonkw.pop("histtype")
    commonkw.pop("alpha")
    hist_errorbars(test_preds_optimized[y_test], ax=ax, **commonkw).set_label(
        "signal (test)"
    )
    hist_errorbars(test_preds_optimized[~y_test], ax=ax, **commonkw).set_label(
        "background (test)"
    )

    ax.legend(title="optimized")
    ax.set_xlabel("BDT score", ha="right", x=1)
    ax.set_ylabel("A.U.", ha="right", y=1)
    ax.set_title("leptonJet BDT prediction value", ha="left", x=0)
    plt.savefig(join(REPORT_DIR, "prediction_dist_optimized.pdf"), bbox_inches='tight')
    plt.close()

    # feature importce
    print(f"Making feature importance plots under {REPORT_DIR}")
    plt.figure(figsize=(8, 6))
    xgb.plot_importance(xgbm_default, height=0.8)
    plt.savefig(join(REPORT_DIR, "feature_imp_default.pdf"), bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8, 6))
    xgb.plot_importance(xgbm_optimized, height=0.8)
    plt.savefig(join(REPORT_DIR, "feature_imp_optimized.pdf"), bbox_inches='tight')
    plt.close()

    to_print = []

    # ROC plot
    print(f"Making ROC plot under {REPORT_DIR}")
    plt.figure(figsize=(8, 6))
    roc = ROCPlot(
        xlim=(0.6, 1),
        ylim=(0.00011, 1),
        logscale=True,
        grid=True,
        percent=True,
        height_ratios=[3, 1],
        ncol=2,
        rlim=(0.95, 1.05),
    )
    roc.plot(y_test, test_preds_default, label="default")
    roc.plot(y_test, test_preds_optimized, label="optimized")
    plt.savefig(join(REPORT_DIR, "roc.pdf"))
    plt.close()

    to_print.append("Working points extracted from ROC")
    to_print.append("=================================")
    head_ = "\t".join(
        ["target", "Fake positive rate", "True positive rate", "Threshold"]
    )
    fprTarget_ = [1e-4, 1e-3, 1e-2]

    rocInfoDefault = roc_curve(y_test, test_preds_default)
    to_print.append("==> Default")
    to_print.append(head_)
    for t in fprTarget_:
        fpr = rocInfoDefault[0]
        fpr_ = rocInfoDefault[0][fpr > t][0]
        tpr_ = rocInfoDefault[1][fpr > t][0]
        thres_ = rocInfoDefault[2][fpr > t][0]
        to_print.append("{}\t{}\t{}\t{}".format(t, fpr_, tpr_, thres_))

    rocInfoOpt = roc_curve(y_test, test_preds_optimized)
    to_print.append("==> Optimized")
    to_print.append(head_)
    for t in fprTarget_:
        fpr = rocInfoOpt[0]
        fpr_ = rocInfoOpt[0][fpr > t][0]
        tpr_ = rocInfoOpt[1][fpr > t][0]
        thres_ = rocInfoOpt[2][fpr > t][0]
        to_print.append("{}\t{}\t{}\t{}".format(t, fpr_, tpr_, thres_))

    # accuracy score
    to_print.append("\n")
    to_print.append("Accuracy score")
    to_print.append("==============")
    to_print.append(
        "{:>10} {:.4f}".format(
            "Default", accuracy_score(y_test, test_preds_default > 0)
        )
    )
    to_print.append(
        "{:>10} {:.4f}".format(
            "Optimized", accuracy_score(y_test, test_preds_optimized > 0)
        )
    )
    # AUC score
    to_print.append("\n")
    to_print.append("AUC score")
    to_print.append("=========")
    to_print.append(
        "{:>10} {:.4f}".format("Default", roc_auc_score(y_test, test_preds_default))
    )
    to_print.append(
        "{:>10} {:.4f}".format("Optimized", roc_auc_score(y_test, test_preds_optimized))
    )
    # classification report
    to_print.append("\n")
    to_print.append("Classification Report")
    to_print.append("=====================")
    to_print.append(">> Default")
    to_print.append(classification_report(y_test, test_preds_default > 0, digits=4))
    to_print.append(">> Optimized")
    to_print.append(classification_report(y_test, test_preds_optimized > 0, digits=4))

    with open(join(REPORT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(to_print))
    print("\n".join(to_print))


if __name__ == "__main__":
    main()
