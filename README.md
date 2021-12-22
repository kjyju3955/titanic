Titanic 🛥️
---
### 1. Subject
##### 타이타닉 데이터와 xgboost의 다양한 기법을 이용해서 성능을 높여보쟈!!
xgboost 공부하는중..!

---
### 2. Data
 - 컬럼 설명
   - PassengerID : 고객 번호
   - Pclass : 티켓 등급 (1=1st, 2=2nd, 3=3rd)
   - Name : 승객 이름
   - Sex : 성별
   - Age : 연령
   - SibSp : 타이타닉호에 탑승한 형제/배우자의 수
   - Parch : 타이타닉호에 탑승한 부모/자녀의 수
   - Ticket : 티켓 번호
   - Fare : 승객 요금
   - Cabin : 객실 번호
   - Embarked : 기항지 위치 (C:Cherbourg, Q:Queenstown, S:Southampton)

---
### 3. 과정
##### accuracy 값을 기준으로 평가
  - train 과정
   - train 데이터를 train_test_split을 통해서 validataion으로 확인해줌 (test_size = 0.2)
   - 타이타닉은 이진분류이기 때문에 objective='binary:logistic' 사용
   - 하이퍼파라미터튜닝을 위해서 RandomizedSearchCV 사용 (cv = 5)
   - 불필요한 학습을 막기 위해서 early_stopping_rounds 설정 / 이때, classification이기 때문에 평가 지표로 'error'사용

#### 1차.
 - 'PassengerId', 'Name', 'Ticket', 'Cabin' 컬럼을 빼고 진행
 - 'Sex' 컬럼은 여성, 남성을 one-hot encoding으로 만들어줌
 - 'Embarked' 컬럼도 one-hot encoding으로 만들어줌
 - 결측값을 처리해주는 방법에 대한 실험 ('Age')
  <br>
  
   Score|제거|평균으로 대체|0으로 대체|빈도가 큰 숫자로 대체|아무 처리 안해줌
   :---:|:---:|:---:|:---:|:---:|:---:
   cross_val_score|0.8039|0.8273|0.8160|0.8315|0.8118
   ROC_AUC|0.8431|0.8237|0.8352|0.8376|0.8376
   
      빈도가 큰 숫자로 대체했을 때 성능이 제일 좋게 나옴
      
![image](https://user-images.githubusercontent.com/55525705/147059211-1941e8ae-4e6d-4f92-abed-b762d3e41aab.png)
    
#### 2차.
 1. 직업이 생존율에 영향을 줄 수 있을 것이라 가정
 - 1차 + 이름을 별칭만 뽑아내서 one-hot encoding (ex-Mr, Miss, Mrs 등등)
    -> 성능의 변화
       : cross_val_score = 0.8315 <br>
       : ROC AUC 값 = 0.8236
       
    ![image](https://user-images.githubusercontent.com/55525705/147039710-23eb13ec-d396-47fe-a962-fd127eac3a83.png)
    
 2. 1차 + 1번 + 가족이 있는 손님이 없는 손님보다 가족들의 도움을 받아서 생존율에 영향을 줄 수 있을 것이라 가정
 - 가족의 수가 영향이 있나 보기 위해서 SibSp + Parch를 이용해서 가족 수를 나타내는 컬럼 'Family_size'추가
 - 혼자 탄 승객과 동행인이 있는 승객을 나누기 위해서 'Alone' 컬럼 추가 (0:혼자 탑승 / 1:동행인 존재)
    -> 성능의 변화
       : cross_val_score: 0.9312 <br>
       : ROC AUC 값 = 0.9151
       
     ![image](https://user-images.githubusercontent.com/55525705/147043711-193a5647-9571-423a-b569-61f88e74eaff.png)
      
     -> overfitting..?
 
 3. 1차 + 1번 + 연속형 데이터 보다는 범주형 데이터가 classification에서는 좋음 <br>
   (사람을 보고 성별같은 범주형 데이터로의 예측이 나이같은 연속형 데이터로 예측하는 것보다 쉬워서)
 - 변경한 연속형 데이터 : 'Age', 'Fare' <br><br>
 
    Score|'Age'->범주형|'Age'->범주형 + one-hot|'Fare'->범주형|'Fare'->범주형 + one-hot|'Fare','Age'->범주형|'Fare','Age'->범주형 + one-hot
    :---:|:---:|:---:|:---:|:---:|:---:|:---:
    cross_val_score|0.8329|0.8385|0.8230|0.8076|0.8175|0.8287
    ROC_AUC|0.8260|0.8283|0.8237|0.8144|0.8145|0.8120
    
    <br>
       : 'Age'를 범주형으로 바꾸고 one-hot encoding을 한 경우의 성능이 가장 좋음
         
   ##### 1차 + 1번 + 2번의 "age" 범주형 변경 + one-hot의 성능이 제일 좋음

### 4. 기능
 - 최대한 argments.py 만 변경하면 코드 조절이 가능하게 만듦
 - train의 경우 모델을 저장해서 test하면 예측한 결과를 결합해서 엑셀파일로 내보내줌
