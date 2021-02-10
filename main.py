import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import consts
import linear_regression
import polynomial_regression


def read_data():
    return pd.read_csv('./data/movement_libras.data',
                       usecols=[consts.result_column, consts.dependency_one_column, consts.dependency_two_column])


def main():
    data = read_data()
    result_column_value = data[consts.result_column]
    dependency_one_column_value = data[consts.dependency_one_column]
    # dependency_two_column_value = data[consts.dependency_two_column]

    X1 = pd.DataFrame(dependency_one_column_value)
    # X2 = pd.DataFrame(dependency_two_column_value)
    Y = pd.DataFrame(result_column_value)

    linear_regression.build_linear_regression(data, X1, Y)

    polynomial_regression.build_polynomial_regression(
        dependency_one_column_value.to_numpy(), result_column_value.to_numpy())

    return 0


main()
