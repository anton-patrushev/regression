import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import consts
import linear_regression
# import polynomial_regression


def read_data():
    return pd.read_csv('./data/movement_libras.data',
                       usecols=[consts.Y_COLUMN, consts.X_COLUMN])


def main():
    data = read_data()

    X = pd.DataFrame(data[consts.X_COLUMN])
    Y = pd.DataFrame(data[consts.Y_COLUMN])

    linear_regression.build_linear_regression(data, X, Y)

    # polynomial_regression.build_polynomial_regression(
    #     dependency_one_column_value.to_numpy(), result_column_value.to_numpy())

    return 0


main()
