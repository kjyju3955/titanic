import pandas as pd
from sklearn.model_selection import train_test_split


def data_processing(file_path):
    '''
    # titanic
    data = pd.read_csv('./data/test.csv')

    #data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}, na_action=None)
    data = pd.get_dummies(data=data, columns=['Embarked'], prefix='Embarked')
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}, na_action=None)

    y = data.loc[:, ['Survived']]
    X = data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156)'''
    data = pd.read_csv(file_path)
    y = data.loc[:, ['dlbael']]
    x = data.iloc[:, []]

    return x, y