from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

import consts


def learn_polynomial_regression(X, Y, degree):
    return np.poly1d(np.polyfit(X, Y, degree))


def print_result(model, X, Y):

    XP = np.linspace(np.amin(X), np.amax(X))

    plt.xlabel("2 coordinate abcissa")
    plt.ylabel("1 coordinate abcissa")

    plt.plot(XP, model(XP), color='red', linewidth=3)
    plt.scatter(X, Y, alpha=0.5)

    return plt.show()


def build_polynomial_regression(X, Y):
    print('polynomial')
    model = learn_polynomial_regression(X, Y, 100)

    return print_result(model, X, Y)
