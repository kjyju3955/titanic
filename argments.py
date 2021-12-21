param_grid = {'n_estimators': range(50, 1000, 100),
              'max_depth': range(4, 9),
              'learning_rate': [i / 100.0 for i in range(1, 50)],
              'colsample_bytree': [0.5, 0.8],
              'gamma': [1, 2, 3],
              'min_child_weight': [1, 2, 3],
              'subsample': [0.8, 1]}

param_fig = {'eval_metric': 'error',  # 회귀의 경우 rmse, 분류의 경우 error 주로 사용 / 반복 수행 시 사용하는 비용 평가 지표
             'early_stopping_rounds': 100,  # 더 이상 비용 평가 지표가 감소하지 않는 최대 반복횟수
             'verbose': True}

env = {'mode': 'train',
       'model_path': './model/train.pkl',
       'result_save_path': './result.csv',
       'cv': 5,
       'train_test_size': 0.2}
