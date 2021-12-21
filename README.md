Titanic 🛥️
---
### 1. Subject
##### 타이타닉 데이터와 xgboost의 다양한 기법을 이용해서 성능을 높여보쟈!!
xgboost 공부하는중..!

---
### 2. Data

---
### 3. 과정
#### 1차. 
 - 'PassengerId', 'Name', 'Ticket', 'Cabin' 컬럼을 빼고 진행
 - 'Sex' 컬럼은 여성, 남성을 one-hot encoding으로 만들어줌
 - 'Embarked' 컬럼도 one-hot encoding으로 만들어줌
 - train 과정
   - train 데이터를 train_test_split을 통해서 validataion으로 확인해줌 (test_size = 0.2)
   - 타이타닉은 이진분류이기 때문에 objective='binary:logistic' 사용
   - 하이퍼파라미터튜닝을 위해서 RandomizedSearchCV 사용 (cv = 5)
   - 불필요한 학습을 막기 위해서 early_stopping_rounds 설정 / 이때, classification이기 때문에 평가 지표로 'error'사용

   <결과> <br>
    : cross_val_score: 0.8118(scoring='accuracy'로 cv=5)
    : 정확도 = 0.8547
    : 정밀도 = 0.8710
    : 재현율 = 0.7500
    : F1 = 0.8060
    : ROC AUC 값 = 0.8376
    
  ![image](https://user-images.githubusercontent.com/55525705/146897969-f0efcab0-09e9-438c-ad88-78fc6ade7531.png)

    : feature_importance = Age, Fare, Pclass, Sibsp, Parch, Sex_female, Embarked_C, Embarked_S, Sex_male, Embarked_Q 순으로 영향을 많이 줌
    


### 4. 기능
 - 최대한 argments.py 만 변경하면 코드 조절이 가능하게 만듦
 - train의 경우 모델을 저장해서 test하면 예측한 결과를 결합해서 엑셀파일로 내보내줌
