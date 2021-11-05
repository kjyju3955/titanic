import data_pro
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
from sklearn.model_selection import cross_val_score
import plot
import clf_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # train_path = './data/septic_shock_dataset.csv'
    # test_path = './data/saint_shock_dataset.csv'
    train_path = './data/train_shock.csv'
    test_path = './data/test_shock.csv'

    train_data = pd.read_csv(train_path)
    train_y = train_data.loc[:, ['dlabel']]
    train_x = train_data.iloc[:, 3:35]  # 31

    test_data = pd.read_csv(test_path)
    test_y = test_data.loc[:, ['dlabel']]
    test_x = test_data.iloc[:, 3:35]  # 30

    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=156)

    xgb = XGBClassifier(learning_rate=0.05, n_estimators=800, max_depth=5, use_label_encoder=False, scale_pos_weight=4)

    params = {'n_estimators': range(500, 2500, 100),
              'learning_rate': [i/100.0 for i in range(1, 20, 2)],
              'max_depth': range(3, 6, 1),
              # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
              # 'subsample': [i/10.0 for i in range(6, 10)],
              # 'gamma': [i/10.0 for i in range(0, 5)],
              # 'min_child_weight': range(1, 6, 2),
              # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
              'objective': ['binary:logistic'],
              'eval_metric': ['error']
              }
    fit_params = {"early_stopping_rounds": 100,
                  "eval_set": [[X_val, y_val]]
                  }

    # grid_xgb = GridSearchCV(xgb, param_grid=params, cv=5, refit=True)
    grid_xgb = RandomizedSearchCV(xgb, param_distributions=params, cv=5, refit=True)
    grid_xgb.fit(X_train, y_train, **fit_params)

    score_df = pd.DataFrame(grid_xgb.cv_results_)
    print(score_df[['params', 'mean_test_score', 'rank_test_score']])

    print(grid_xgb.best_params_)

    estimator = grid_xgb.best_estimator_

    print("[train]")
    train_pred = estimator.predict(X_val)
    train_pred_prob = estimator.predict_proba(X_val)[:, 1]

    train_score = cross_val_score(estimator, X_train, y_train, cv=5)
    print("cross_val_score: {0:.4f}".format(train_score.mean()))

    clf_eval.get_clf_eval(y_val, train_pred)
    plot.make_important_plot(estimator)
    plot.roc_plot(y_val, train_pred_prob)

    print("[test]")
    test_pred = estimator.predict(test_x)
    test_pred_prob = estimator.predict_proba(test_x)[:, 1]

    accuracy = accuracy_score(test_y, test_pred)
    print("Test accuracy : %.3f%%" % (accuracy * 100.0))
    clf_eval.get_clf_eval(test_y, test_pred)
    plot.roc_plot(test_y, test_pred_prob)

'''

def make_model(x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=156)

    params = {'n_estimators': range(50, 1000, 100),
              'learning_rate': [i/100.0 for i in range(1, 50)],
              'max_depth': [4, 5, 6],
              # 'max_depth': range(3, 10, 1),
              # 'colsample_bytree': [i / 10.0 for i in range(6, 10)]
              # 'subsample': [i/10.0 for i in range(6, 10)],
              # 'gamma': [i/10.0 for i in range(0, 5)],
              # 'min_child_weight': range(1, 6, 2),
              # 'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
              'objective': ['binary:logistic'],
              'eval_metric': ['error']
              }
    fit_params = {"early_stopping_rounds": 100,
                  "eval_set": [[x_val, y_val]]
                  }

    xgb = XGBClassifier(use_label_encoder=False, num_class=2)

    grid_xgb = GridSearchCV(xgb, param_grid=params, cv=5, refit=True)
    model = grid_xgb.fit(x_train, y_train, **fit_params)

    # model = xgb.fit(x_val, y_train, **fit_params)

    scoring(model, x_train, x_val, y_train, y_val)


def scoring(model, x_train, x_val, y_train, y_val):
    score_df = pd.DataFrame(model.cv_results_)
    print(score_df[['params', 'mean_test_score', 'rank_test_score']])

    print(model.best_params_)

    estimator = model.best_estimator_
    pred = estimator.predict(x_val)
    pred_prob = estimator.predict_proba(x_val)[:, 1]

    score = cross_val_score(estimator, x_train, y_train, cv=5)
    print("cross_val_score: {0:.4f}".format(score.mean()))

    clf_eval.get_clf_eval(y_val, pred)
    plot.make_important_plot(estimator)
    plot.roc_plot(y_val, pred_prob)


if __name__ == "__main__":
    train_path = './data/septic_shock_dataset.csv'
    train_data = pd.read_csv(train_path)
    train_y = train_data.loc[:, ['dlabel']]
    train_x = train_data.iloc[:, 3:31]

    make_model(train_x, train_y)

'''