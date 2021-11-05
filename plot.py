from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np


def make_important_plot(xgb):
    fix, ax = plt.subplots(figsize=(10, 12))
    plot_importance(xgb, ax=ax)
    plt.show()


def roc_plot(y_test, pred):
    fprs, tprs, thresholds = roc_curve(y_test, pred)

    plt.plot(fprs, tprs, label='Roc')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()
