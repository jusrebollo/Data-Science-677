"""
Justin Rebollo
Class: CS 677 - Spring 2
Date: 4/16/2021
Homework Problem #
See Answers PDF for questions and answers
"""

import numpy
import xlrd
import pandas as pd
import numpy as np
from array import *
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix


# 1_1
# ______________________________________________________________________________
# load data
raw = pd.read_excel (r'/Users/jreb/PycharmProjects/rebollowhw_5/CTG.xls',
                     sheet_name='Raw Data')


# 1_2
# ______________________________________________________________________________
# Combine NSP Labels

raw.loc[raw['NSP'] == 1, 'NSP_C'] = 1
raw.loc[raw['NSP'] != 1, 'NSP_C'] = 0

# 2_1-3
# ______________________________________________________________________________
# Drop NaaN
raw.dropna()
raw = raw.drop(raw.index[0])
raw = raw.drop(raw.index[[2127,2128]])
raw = raw.drop(raw.index[[2126]])

print(raw)
# Split data
X = raw[['LB', 'MLTV', 'Width', 'Variance']].values
y = raw['NSP_C'].values

# Drop NaaN

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.50)

NB_classifier = GaussianNB().fit(X_train, y_train)

prediction_nb = NB_classifier.predict(X_test)

accuracy_nb = round((accuracy_score(y_test, prediction_nb)* 100), 2)

confusion_nb = (confusion_matrix(y_test, prediction_nb))

print("Naive Bayesian Classifier")
print("Accuracy")
print(accuracy_nb)
print("Confusion Matrix")
print(confusion_nb)

TN_nb = confusion_nb[0][0]
FN_nb = confusion_nb[1][0]
TP_nb = confusion_nb[1][1]
FP_nb = confusion_nb[0][1]
true_pos_nb = round(((TP_nb / (TP_nb + FN_nb)) * 100), 2)
true_neg_nb = round(((TN_nb / (TN_nb + FP_nb)) * 100), 2)


# 3_1-3
# ______________________________________________________________________________
tree_classifier = sklearn.tree.DecisionTreeClassifier(criterion = 'entropy')
tree_classifier = tree_classifier.fit(X, y)
prediction_tree = tree_classifier.predict(X_test)

accuracy_tree = round((accuracy_score(y_test, prediction_tree)* 100), 2)
confusion_tree = (confusion_matrix(y_test, prediction_tree))

print("Decision Tree Classifier")
print("Accuracy")
print(accuracy_tree)
print("Confusion Matrix")
print(confusion_tree)

TN_ct = confusion_tree[0][0]
FN_ct = confusion_tree[1][0]
TP_ct = confusion_tree[1][1]
FP_ct = confusion_tree[0][1]

true_pos_ct = round(((TP_ct / (TP_ct + FN_ct)) * 100), 2)
true_neg_ct = round(((TN_ct / (TN_ct + FP_ct)) * 100), 2)
# 4_1-4
# ______________________________________________________________________________
# loop through all possible n and d values to find most accurate
nlen = 11
dlen = 6
n = 1
d = 1
error_rates = []

for i in range(1,dlen):
    for i in range(1,nlen):
        model = RandomForestClassifier(n_estimators=n,max_depth=d, criterion=
            'entropy')
        model.fit(X_train , y_train)
        predictionrf = model.predict(X_test)
        error_rate = np.mean(predictionrf != y_test)
        accuracy = accuracy_score(y_test, predictionrf)
        error_rates.append(error_rate)
        n = n+1
        d = d+1


plt.plot(error_rates)
plt.ylabel("Error Rate")
plt.xlabel("N & D Values: Ex.12 = D=1 & N=2")
plt.show()


best= np.argmin(error_rates)
# retrieve best n and d from index value of lowest error rate from error rates
# N & D Values: Ex.12 = D=1 & N=2"
best = str(best)
print(best)
if(len(best)) == 1:
    d_best = 1
    n_best = best[0]
else:
    if best[1] == '0':
        d_best = best[0]
        n_best = 10
    else:
        n_best = best[1]
        d_best = int(best[0]) + 0

print("Best N:", n_best, "Best D:", d_best)

n_best = int(n_best)
d_best = int(d_best)
model_best = RandomForestClassifier(n_estimators=n_best,max_depth=d_best,
                                    criterion='entropy')
model_best.fit(X_train, y_train)
prediction_rf_best = model_best.predict(X_test)
confusion_rf_best = (confusion_matrix(y_test, prediction_rf_best))
accuracy_rf_best = round((accuracy_score(y_test, prediction_rf_best)* 100), 2)
print("Accuracy")
print(accuracy_rf_best)
print("Confusion Matrix")
print(confusion_rf_best)

TN_rf_best = confusion_rf_best[0][0]
FN_rf_best= confusion_rf_best[1][0]
TP_rf_best = confusion_rf_best[1][1]
FP_rf_best = confusion_rf_best[0][1]

true_pos_rf_best = round(((TP_rf_best / (TP_rf_best + FN_rf_best)) * 100), 2)
true_neg_rf_best = round(((TN_rf_best / (TN_rf_best + FP_rf_best)) * 100), 2)

# 5
# ______________________________________________________________________________


print("Question 5 Table")
print("Model     TP     FP     TN     FN     Accuracy    TPR     TNR")
print("NB", "     ", TP_nb, "  ", FP_nb, "  ", TN_nb, "  ", FN_nb, "   ",
      accuracy_nb,"     ", true_pos_nb, " ", true_neg_nb)
print("DT", "      ", TP_ct, "  ", FP_ct, "  ", TN_ct, "  ", FN_ct, "   ",
      accuracy_tree,"     ", true_pos_ct, " ",true_neg_ct)
print("RF", "     ", TP_rf_best, "  ", FP_rf_best, "  ", TN_rf_best, "  ",
      FN_rf_best,"   ", accuracy_rf_best, "     ", true_pos_rf_best, " ",
      true_neg_rf_best)
