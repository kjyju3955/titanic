import pandas as pd


def data_processing(mode):
    data = pd.read_csv('./data/' + mode + '.csv')
    print(mode)
    data = pd.get_dummies(data=data, columns=['Embarked'], prefix='Embarked')  # one-hot encoding
    data = pd.get_dummies(data=data, columns=['Sex'], prefix='Sex')
    # data['Name'] = pd.Series([i.split()[1][:-1] if i.split()[1][0] == "M" else i.split()[2][:-1] for i in data['Name']])
    # print(data['Name'].value_counts())
    # data = pd.get_dummies(data=data, columns=['Name'], prefix='Name')
    if mode == "test":
        return data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    x = data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = data.loc[:, ['Survived']]

    print(data)

    return x, y


if __name__ == "__main__":
    mode = "train"
    data_processing(mode)
