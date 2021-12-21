from data_pro import *
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
import argments
from clf_eval import *
import joblib
from sklearn.model_selection import train_test_split


class XGB:
    def __init__(self, mode):
        self.mode = mode
        self.param_fig = argments.param_fig
        self.param_grid = argments.param_grid
        self.select_model()

    def select_model(self):
        if self.mode == "train":
            self.model_train()
        else:
            self.model_test()

    def save_model(self, model):  # train한 모델 저장
        joblib.dump(model, argments.env['model_path'])

    def load_model(self):  # test에 쓸 모델 불러오기
        return joblib.load(argments.env['model_path'])

    def model_train(self):
        x, y = data_processing(self.mode)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=argments.env['train_test_size'], random_state=156)

        xgb = XGBClassifier(use_label_encoder=False, objective='binary:logistic')  # 타이타닉은 이진 분류 이기에 해당 objective 사용

        cv = KFold(n_splits=argments.env['cv'], random_state=42, shuffle=True)
        random_xgb = RandomizedSearchCV(xgb, param_distributions=self.param_grid, cv=cv, scoring='accuracy', n_jobs=8,
                                        random_state=156)

        random_xgb.fit(x_train, y_train, **self.param_fig, eval_set=[(x_val, y_val)])

        best_xgb = random_xgb.best_estimator_
        best_xgb.fit(x_train, y_train, **self.param_fig, eval_set=[(x_val, y_val)])

        search_param(random_xgb)
        train_scoring(best_xgb, x_train, x_val, y_train, y_val)

        self.save_model(best_xgb)

    def model_test(self):
        x = data_processing(self.mode)

        model = self.load_model()

        pred_y = model.predict(x)

        pred_y = pd.DataFrame(pred_y, columns=['Survived_predict'])
        result = pd.concat([x, pred_y], axis=1)
        result = result.reset_index()

        result.to_csv(argments.env['result_save_path'])


if __name__ == "__main__":
    mode = argments.env['mode']
    run = XGB(mode)
