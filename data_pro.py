import pandas as pd


def data_processing(mode):
    data = pd.read_csv('./data/' + mode + '.csv')
    print(mode)

    data = pd.get_dummies(data=data, columns=['Embarked'], prefix='Embarked')
    data = pd.get_dummies(data=data, columns=['Sex'], prefix='Sex')
    # data = data[data['Age'] > 0] # 결측치 제거
    # data['Age'] = data['Age'].fillna(0) # 0으로 대체
    # data['Age] = data['Age'].fillna(data['Age'].mean()] # 평균으로 대체
    data['Age'] = data['Age'].fillna(24)  # 가장 빈도가 높은 숫자로 대체
    data['Name'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)  # Name에서 주요 특징만 추출
    data = pd.get_dummies(data=data, columns=['Name'], prefix='Name')

    data['Family_size'] = data['SibSp'] + data['Parch']  # 가족 수
    # data['Alone'] = 0
    # data.loc[data['Family_size'] == 0] = 1

    data['Age_range'] = 0
    data.loc[data['Age'] <= 16, 'Age_range'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age_range'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age_range'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age_range'] = 3
    data.loc[data['Age'] > 64, 'Age_range'] = 4
    data = pd.get_dummies(data=data, columns=['Age_range'], prefix='Age')

    print(pd.qcut(data['Fare'], 4).value_counts())  # 'Fare'를 4개의 범위로 나눔
    data['Fare_div'] = 0
    data.loc[data['Fare'] <= 7.91, 'Fare_div'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_div'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_div'] = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 513), 'Fare_div'] = 3
    data = pd.get_dummies(data=data, columns=['Fare_div'], prefix='Fare')

    x = data.drop(['PassengerId', 'Survived', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare', 'Age'], axis=1)
    y = data.loc[:, ['Survived']]

    print(x)

    return x, y


if __name__ == "__main__":
    mode = "train"
    data_processing(mode)
