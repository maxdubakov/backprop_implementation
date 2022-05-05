import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from nn import Sequential, mse, mae, cross_entropy


def simple_data():
    df = pd.read_csv('./data/dane2.csv')
    x, y = np.expand_dims(df['x'].to_numpy(), axis=1), np.expand_dims(df['y'].to_numpy(), axis=1)
    return train_test_split(x, y, test_size=0.10)


def complex_data():
    train = pd.read_csv('./data/train.csv')
    X = train.iloc[:, :20].values
    y = train.iloc[:, 20:21].values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()
    return train_test_split(X, y, test_size=0.1)


def get_model(x_train, y_train, verbose=False):
    model = Sequential(input_dim=1)
    model.add_layer(8, activation='sigmoid')
    model.add_layer(4, activation='sigmoid')
    model.add_layer(2, activation='sigmoid')
    model.add_layer(1)
    if verbose:
        model.summary()
    model.compile(lr=0.001, loss='mae')
    history = model.fit(x_train, y_train, epochs=1000, batch_size=4, verbose=verbose)
    return model, history


def more_complex(x_train, y_train, verbose=False):
    model = Sequential(input_dim=20)
    model.add_layer(64, activation='sigmoid')
    model.add_layer(32, activation='sigmoid')
    model.add_layer(16, activation='sigmoid')
    model.add_layer(8, activation='sigmoid')
    model.add_layer(4, activation='softmax')
    if verbose:
        model.summary()
    model.compile(lr=0.001, loss='categorical_crossentropy')
    history = model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=verbose)
    return model, history


def model1():
    x_train, x_test, y_train, y_test = simple_data()
    model, history = get_model(x_train, y_train, verbose=True)
    y_pred = model.predict(x_test)
    print_predict(model, x_test, y_test)
    print(f'Model MAE: {mae(y_pred, y_test)}')


def model2():
    x_train, x_test, y_train, y_test = complex_data()
    model, history = more_complex(x_train, y_train, verbose=True)
    y_pred = model.predict(x_test)
    print(f'Model Cross-Entropy: {cross_entropy(y_pred, y_test)}')


def print_predict(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(f'{"x_test": >5} {"y_test": >10} {"y_pred": >10}')
    for x, y_t, y_p in zip(x_test, y_test, y_pred):
        print(f'{round(x[0], 2): >5} {round(y_t[0], 2): >10} {round(y_p[0], 2): >10}')
    print('\n')


def main():
    model1()
    # model2()


if __name__ == '__main__':
    main()
