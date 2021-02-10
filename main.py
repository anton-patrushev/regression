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

equation = "f(x) = {K}*x + {B}".format(
    K=model.coef_[0][0], B=model.intercept_[0])

correlation_coef = raw_data[result_column].corr(
    raw_data[dependency_one_column])

print("Equation:" + equation)
print("Correlation coefficient: {C}".format(C=correlation_coef))


plt.plot(X, model.predict(X), color='red', linewidth=3)
plt.scatter(dependency_one_column_value, result_column_value, alpha=0.5)

plt.show()
