"""
Justin Rebollo
Class: CS 677 - Spring 2
Date: 4/12/2021
Homework Problem #4 Questions 1-3
See Answers PDF for questions and answers
"""
import numpy
import pandas as pd
import numpy as np
from array import *
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print("1_1")
# 1_1
# ______________________________________________________________________________
# load csv
health_data = pd.read_csv(
    'heart_failure_clinical_records_dataset.csv', sep=",")

print(health_data)
# split on death events
df_0 = health_data[health_data['DEATH_EVENT'] == 0]

df_1 = health_data[health_data['DEATH_EVENT'] == 1]

# 1_2
# ______________________________________________________________________________

# create correlation matrix
df_0.corr()
df_1.corr()

corrMatrix0 = df_0.corr()
corrMatrix1 = df_1.corr()

# plot correlation matrix
sns.heatmap(corrMatrix0, annot=True)

plt.show()
sns.heatmap(corrMatrix1, annot=True)
plt.show()

# 2_1
# ______________________________________________________________________________
# split into x and y
# Group 4: X: platelets, Y : serum creatinine , 6/7


X_0 = df_0['platelets']
Y_0 = df_0['serum_creatinine']

X_1 = df_1['platelets']
Y_1 = df_1['serum_creatinine']

# train test split
X_train0, X_test0, y_train0, y_test0 = train_test_split(X_0, Y_0,
                                                        test_size=0.50)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_1, Y_1,
                                                        test_size=0.50)


# implement models
# 1. y = ax + b (simple linear regression)
# a. fit the model on Xtrain
# b. print the weights (a, b, . . .)
# c. compute predicted values using Xtest
# d. plot (if possible) predicted and actual values in Xtrain
# e. compute (and print) the corresponding loss function


def poly_model(x_train, y_train, x_test, y_test, degree):
    weights = np.polyfit(x_train, y_train, degree)
    print("Weights:")
    print(weights)
    model = np.poly1d(weights)

    predicted = model(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)
    print("RMSE")
    print(rmse)
    print("R^2")

    # plot

    x_points = np.linspace(min(x_test), max(x_test), len(x_test))
    y_points = model(x_points)
    plt.plot(x_points, y_points, color='blue')
    plt.scatter(x_points, y_test, color='red')
    plt.show()
    return (r2)


print("---------------------------------------------")
print("simple linear regression- 0")
e_m1_0 = poly_model(X_train0, y_train0, X_test0, y_test0, 1)
print(e_m1_0)
print("---------------------------------------------")
print("simple linear regression- 1")
e_m1_1 = poly_model(X_train1, y_train1, X_test1, y_test1, 1)
print(e_m1_1)
# 2. y = ax2 + bx + c (quadratic)
print("---------------------------------------------")
print("quadratic- 0")
e_m2_0 = poly_model(X_train0, y_train0, X_test0, y_test0, 2)
print(e_m2_0)
print("---------------------------------------------")
print("quadratic- 1")
e_m2_1 = poly_model(X_train1, y_train1, X_test1, y_test1, 2)
print(e_m2_1)
# 3. y = ax3 + bx2 + cx + d (cubic spline)
print("---------------------------------------------")
print("cubic spline- 0")
e_m3_0 = poly_model(X_train0, y_train0, X_test0, y_test0, 3)
print(e_m3_0)
print("---------------------------------------------")
print("cubic spline- 1")
e_m3_1 = poly_model(X_train1, y_train1, X_test1, y_test1, 3)
print(e_m3_1)

# 4. y = a log x + b (GLM - generalized linear model)
X_train0 = X_train0.values.reshape(-1, 1)
X_train1 = X_train1.values.reshape(-1, 1)
X_test0 = X_test0.values.reshape(-1, 1)
X_test1 = X_test1.values.reshape(-1, 1)

X_train0 = numpy.log10(X_train0)
X_train1 = numpy.log10(X_train0)

y_train0_log = numpy.log10(y_train0)
y_train1_log = numpy.log10(y_train0)


standard_scaler = sklearn.preprocessing.StandardScaler()

X_train0 = standard_scaler.fit_transform(X_train0)
X_test0 = standard_scaler.fit_transform(X_test0)

X_train1 = standard_scaler.fit_transform(X_train1)
X_test1 = standard_scaler.fit_transform(X_test1)

X_train0 = X_train0.flatten()
X_train1 = X_train1.flatten()

X_test0 = X_train0.flatten()
X_test1 = X_train1.flatten()

y_test0 = y_test0[:101]
y_test1 = y_test1[:101]

X_test1 = X_test1[:48]
X_train1 = X_train1[:48]
print(len(X_train1))
print(len(X_test1))
print(len(y_train1))
print(len(y_test1))

"""
X_train0 = X_train0.div(80000)
X_test0 = X_test0.div(80000)

X_train1 = X_train1.div(80000)
X_test1 = X_test1.div(80000)

X_train0= X_train0.values.reshape(-1, 1)
y_train0= y_train0.values.reshape(-1, 1)
X_test0 = X_test0.values.reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(X_train0)

X_train = scaler.transform(X_train0)
X_test = scaler.transform(X_test0)

X_train0 = X_train0.flatten()
X_test0 = X_test0.flatten()
"""


def logistic(x_train, y_train, y_test, x_test):
    degree = 1

    weights = np.polyfit(x_train, y_train, degree)
    print("Weights:")
    print(weights)
    model = np.poly1d(weights)

    predicted = model(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)
    print("RMSE")
    print(rmse)
    print("R^2")

    # plot

    x_points = np.linspace(min(x_test), max(x_test), len(x_test))
    y_points = model(x_points)
    plt.plot(x_points, y_points, color='blue')
    plt.scatter(x_points, y_test, color='red')
    plt.show()
    return(r2)

#
print("Logistic : y = a log x + b")

e_log_0 = logistic(X_train0, y_train0, X_test0, y_test0)
print(e_log_0)

e_log_1 = logistic(X_train1, y_train1, X_test1, y_test1)
print(e_log_1)


# 5. log y = a log x + b (GLM - generalized linear model)

def logistic_2(x_train, y_train, y_test, x_test):
    degree = 1

    weights = np.polyfit(x_train, y_train, degree)
    print("Weights:")
    print(weights)
    model = np.poly1d(weights)

    predicted = model(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)
    print("RMSE")
    print(rmse)
    print("R^2")

    # plot

    x_points = np.linspace(min(x_test), max(x_test), len(x_test))
    y_points = model(x_points)
    plt.plot(x_points, y_points, color='blue')
    plt.scatter(x_points, y_test, color='red')
    plt.show()
    return (r2)


# scale the data

print("Logistic :  log y = a log x + b")
print("Logistic : 0")
e_log2_0 = logistic_2(X_train0, y_train0_log, X_test0, y_test0)
print(e_log2_0)
print("Logistic :  1")
y_train1_log = y_train1_log[:48]
e_log2_1 = logistic_2(X_train1, y_train1_log, X_test1, y_test1)
print(e_log2_1)
# 3_1
# ______________________________________________________________________________
print()
print("Question 3 Table")
print()
print("Model               SSE(death_event = 0     SSE(death_event = 1")
print("y = ax + b", "                    ", round(e_m1_0, 2), "          ",
      round(e_m1_1, 2), "  ")
print("y = ax2 + bx + c", "              ", round(e_m2_0, 2), "          ",
      round(e_m2_1, 2), "  ")
print("y = ax3 + bx2 + cx + d", "        ", round(e_m3_0, 2), "          ",
      round(e_m3_1, 2), "  ")
print("y = a log x + b", "               ", round(e_log_0, 2), "          ",
      round(e_log_1), "  ")
print("log y = a log x + b", "           ", round(e_log2_0, 2), "           ",
      round(e_log2_1, 2), "  ")


