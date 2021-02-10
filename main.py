import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

result_column = "FIRST"
dependency_one_column = "THIRD"

raw_data = pd.read_csv('./data/movement_libras.data',
                       usecols=[result_column, dependency_one_column])

result_column_value = raw_data[result_column]
dependency_one_column_value = raw_data[dependency_one_column]

X = pd.DataFrame(dependency_one_column_value)
Y = pd.DataFrame(result_column_value)

plt.xlabel("2 coordinate abcissa")
plt.ylabel("1 coordinate abcissa")

model = LinearRegression()

model.fit(X, Y)

plt.plot(X, model.predict(X), color='red', linewidth=3)
plt.scatter(dependency_one_column_value, result_column_value, alpha=0.5)

plt.show()
