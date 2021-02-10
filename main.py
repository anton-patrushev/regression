import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

result_column = "FIRST"
dependency_one_column = "THIRD"


def read_data():
    return pd.read_csv('./data/movement_libras.data',
                       usecols=[result_column, dependency_one_column])


def learn_linear_regression(X, Y):
    model = LinearRegression()
    model.fit(X, Y)

    return model


def get_equation(model):
    return "f(x) = {K}*x + {B}".format(
        K=model.coef_[0][0], B=model.intercept_[0])


def get_correlation_coef(data):
    return "{C}".format(C=data[result_column].corr(
        data[dependency_one_column]))


def print_result(model, data, X, Y):

    print("Equation: " + get_equation(model))
    print("Correlation coefficient: " + get_correlation_coef(data))

    plt.xlabel("2 coordinate abcissa")
    plt.ylabel("1 coordinate abcissa")

    plt.plot(X, model.predict(X), color='red', linewidth=3)
    plt.scatter(X, Y, alpha=0.5)

    plt.show()

    return 0


def main():
    data = read_data()
    result_column_value = data[result_column]
    dependency_one_column_value = data[dependency_one_column]

    X = pd.DataFrame(dependency_one_column_value)
    Y = pd.DataFrame(result_column_value)

    model = learn_linear_regression(X, Y)

    print_result(model, data, X, Y)

    return 0


main()
