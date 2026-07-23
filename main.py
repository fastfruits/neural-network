import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv")

data.head()

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T  # Transpose
y_dev = data_dev[0]
x_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n] / 255.0

print(x_train[:, 0].shape)


def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2


def RelU(z):
    return np.maximum(0, z)  # x if > 0 | 0 if <= 0


def SoftMax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)  # Sum for each column


def forward_propagate(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = RelU(z1)
    z2 = w2.dot(a1) + b2
    a2 = SoftMax(z2)

    return z1, a1, z2, a2


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T

    return one_hot_y


def deriv_RelU(z):
    return z > 0


def backward_propagation(z1, a1, z2, a2, w2, x, y):
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2)

    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)

    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1

    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    return w1, b1, w2, b2


def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def make_predictions(x, w1, b1, w2, b2):
    _, _, _, a2 = forward_propagate(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    return predictions

def test_predictions(index, w1, b1, w2, b2):
    current_image = x_train[:, index, None]

    prediction = make_predictions(x_train[:, index, None], w1, b1, w2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()

    for i in range(iterations + 1):
        z1, a1, z2, a2 = forward_propagate(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a2), y))

    return w1, b1, w2, b2


def main():
    w1, b1, w2, b2 = gradient_descent(x_train, y_train, 250, 0.20)
    test_predictions(0 , w1, b1, w2, b2)

    test_predictions(78 , w1, b1, w2, b2)


main()