from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import plot


def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)

    print("오차행렬")
    print(confusion)
    print(
        "정확도: {0:.4f}\n정밀도: {1:.4f}\n재현율: {2:.4f}\nF1: {3:.4f}\nROC AUC 값 :{4:.4f}".format(accuracy, precision, recall,
                                                                                           f1, roc_score))


def scoring(model, x_val, y_val):
    pred = model.predict(x_val)
    pred_prob = model.predict_proba(x_val)[:, 1]

    get_clf_eval(y_val, pred)
    plot.make_important_plot(model)
    plot.roc_plot(y_val, pred_prob)


def train_scoring(model, x_train, x_val, y_train, y_val):
    score = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=5)
    print("cross_val_score: {0:.4f}".format(score.mean()))

    scoring(model, x_val, y_val)


def search_param(model):
    score_df = pd.DataFrame(model.cv_results_)
    print(score_df[['params', 'mean_test_score', 'rank_test_score']])
    print(model.best_params_)
