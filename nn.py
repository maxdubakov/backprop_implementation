import numpy as np


def to_numpy(_list):
    if type(_list) != np.ndarray:
        _list = np.array(_list, dtype=object)
    return _list


def random(shape, _from, _to):
    return (_to - _from) * np.random.random_sample(shape) + _from


def cross_entropy(y_pred, y_true):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    y_pred = y_pred.astype(float)
    n = y_pred.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / n


def mae(y_pred, y_true, axis=None):
    y_pred = to_numpy(y_pred)
    y_true = to_numpy(y_true)
    return (np.abs(y_true - y_pred)).mean(axis=axis)


def mse(y_pred, y_true, axis=None):
    y_pred = to_numpy(y_pred)
    y_true = to_numpy(y_true)
    return (np.square(y_true - y_pred)).mean(axis=axis)


def softmax(x, axis=-1):
    kw = dict(axis=axis, keepdims=True)
    xrel = x - x.max(**kw)

    exp_xrel = np.exp(xrel)
    return exp_xrel / exp_xrel.sum(**kw)


_activation = {
    'None': lambda x: x,
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'relu': lambda x: np.maximum(x, 0),
    'softmax': softmax
    # lambda x: np.exp(x) / sum(np.exp(x))
}

derivative = {
    'None': lambda a: np.ones(shape=a.shape),
    'sigmoid': lambda a: np.array(a * (1 - a)),
    'relu': lambda a: np.array((a > 0) * 1)
}

cost_function = {
    'mae': mae,
    'mse': mse,
    'categorical_crossentropy': cross_entropy
}


class Sequential(object):

    def __init__(self, input_dim):

        # a single weight matrix is sized k*j where j -- number of neurons in l-1 layer and k -- # of neurons in l layer
        self.__weights = []
        self.__biases = []
        self.__layers = [{
            'neurons': input_dim,
            'activation': 'None'
        }]

    def compile(self, lr=0.001, loss='mae'):
        self.__lr = lr
        self.__loss = cost_function[loss]

    def fit(self, x, y, batch_size=32, epochs=5, verbose=False):
        x = to_numpy(x)
        y = to_numpy(y)

        self.__check_dims(x)
        history = []
        for epoch in range(epochs):
            i0 = 0
            i1 = batch_size
            while i0 < x.shape[0]:
                x_batched, y_batched = x[i0:i1], y[i0:i1]
                self.__step(x_batched, y_batched, i1-i0)
                i0 += batch_size
                i1 += batch_size
                if i1 >= x.shape[0]:
                    i1 = x.shape[0]

            epoch_predict = self.predict(x)
            current_error = round(self.__loss(epoch_predict, y), 3)
            history.append(current_error)
            if verbose:
                print(f'\rEpoch {epoch + 1}\t\tloss: {current_error}', end='', flush=True)
        print('\n')
        return history

    def __step(self, x, y, batch_size):
        all_activations = self.__fit_predict(x)
        y_pred = np.array([_a for _a in all_activations[:, -1]])
        delta_l_1 = np.subtract(y_pred, y).T
        for i in range(len(self.__weights) - 1, -1, -1):
            activations = self.__get_current_activations(all_activations, i)
            activation_derivative = derivative[self.__layers[i]['activation']](activations)
            delta_l = np.multiply(np.matmul(self.__weights[i].T, delta_l_1), activation_derivative)
            self.__weights[i] = np.subtract(self.__weights[i], self.__lr * np.matmul(delta_l_1, activations.T))
            upd = np.squeeze(np.matmul(delta_l_1, np.ones(shape=(batch_size, 1))))
            self.__biases[i] = np.subtract(self.__biases[i], np.multiply(self.__lr, upd))
            delta_l_1 = delta_l

    def __get_current_activations(self, all_activations, i):
        activations = np.array([_a for _a in all_activations[:, i]])
        return activations.T

    def predict(self, x):
        x = to_numpy(x)
        self.__check_dims(x)

        y_pred = []
        for _x in x:
            a = _x.T
            for layer, weights, bias in zip(self.__layers[1:], self.__weights, self.__biases):
                z = np.add(np.matmul(weights, a), bias)
                a = _activation[layer['activation']](z)
            y_pred.append(a)
        return to_numpy(y_pred)

    def __fit_predict(self, x):
        x = to_numpy(x)
        self.__check_dims(x)
        all_activations = []
        for _x in x:
            a = _x.T
            current_activation = [a]
            for layer, weights, bias in zip(self.__layers[1:], self.__weights, self.__biases):
                z = np.add(np.matmul(weights, a), bias)
                a = _activation[layer['activation']](z)
                current_activation.append(a)
            all_activations.append(np.array(current_activation, dtype=object))
        return np.array(all_activations, dtype=object)

    def __check_dims(self, x):
        input_neurons = self.__layers[0]['neurons']
        if x.shape[1] != input_neurons:
            raise ValueError(
                f'Dimensional mismatch: x with dim {x.shape[0]} does not fit provided input dim ({input_neurons})'
            )

    def add_layer(self, neurons: int, activation=None):
        new_weight_matrix = random((neurons, self.__layers[-1]['neurons']), -1, 1)
        new_bias_vector = random(neurons, -1, 1)

        if activation is None:
            activation = 'None'
        self.__layers.append({
            'neurons': neurons,
            'activation': activation
        })

        self.__weights.append(new_weight_matrix)
        self.__biases.append(new_bias_vector)

    def summary(self):
        total_params = 0

        print('Model: "Deep Neural Network"\n')
        print('-----------------------------------------------------------')
        print('Layer number\t\t\tOutput Neurons\t\t\t\tParam #')
        print('===========================================================')
        for i, (layer, weights, bias) in enumerate(zip(self.__layers[1:], self.__weights, self.__biases)):
            params = weights.shape[0] * weights.shape[1] + bias.shape[0]
            total_params += params
            print(f'Layer {i}\t\t\t\t\t{layer["neurons"]}\t\t\t\t\t\t\t{params}\n')
        print('===========================================================')
        print(f'Total params: {total_params}')
        print('-----------------------------------------------------------')
