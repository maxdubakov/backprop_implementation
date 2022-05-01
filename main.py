import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential

from nn import Sequential, mse, mae, cross_entropy


def simple_data():
    df = pd.read_csv('./data/dane2.csv')
    x, y = np.expand_dims(df['x'].to_numpy(), axis=1), np.expand_dims(df['y'].to_numpy(), axis=1)
    # return x, y
    return train_test_split(x, y, test_size=0.15)


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


def print_predict(model, x_test, y_test):
    y_pred = model.predict(x_test)

    print('x test\t\t\ty_true\t\ty_pred')
    for x, y_t, y_p in zip(x_test, y_test, y_pred):
        print(f'{x[0]} \t\t{y_t[0]}\t\t{y_p[0]}')


def main():
    # x_train, x_test, y_train, y_test = simple_data()
    # for i in range(100):
    #     model, history = get_model(x_train, y_train)
    #     y_pred = model.predict(x_test)
    #     print(f'Model {i+1} MSE: {mse(y_pred, y_test)}')
    more_complex()


def more_complex():
    x_train, x_test, y_train, y_test = complex_data()

    model = Sequential(input_dim=20)
    model.add_layer(64, activation='sigmoid')
    model.add_layer(32, activation='sigmoid')
    model.add_layer(16, activation='sigmoid')
    model.add_layer(8, activation='sigmoid')
    model.add_layer(4, activation='softmax')
    model.summary()
    model.compile(lr=0.001, loss='categorical_crossentropy')
    model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=True)


if __name__ == '__main__':
    main()
