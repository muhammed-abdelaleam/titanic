import pandas as pd
from sklearn.ensemble import RandomForestClassifier

feature_cols = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked',
]


def load_data(file_name):
    df = pd.read_csv(file_name)

    dfX = df[feature_cols].copy()

    dfX['Age'].fillna(dfX['Age'].mean(), inplace=True)

    embarked = dfX.pop('Embarked')
    embarked.fillna(embarked.mode()[0], inplace=True)

    embarked_dummies = pd.get_dummies(embarked)

    gender = dfX.pop('Sex')
    gender_dummies = pd.get_dummies(gender)

    dfX = pd.concat([dfX, embarked_dummies, gender_dummies], axis=1)

    X = dfX.values
    y = df['Survived'].values

    return X, y


def create_model(X, y):
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf


if __name__ == '__main__':
    import pickle

    from sklearn.model_selection import train_test_split

    X, y = load_data('titanic.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = create_model(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f'score: {score}')
    with open('model.pkl', 'wb') as out:
        pickle.dump(clf, out)
