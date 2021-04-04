import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# 1_1
# ______________________________________________________________________________
banknotes = pd.read_csv(
    'data_banknote_authentication.txt', sep=",")

banknotes.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

print(banknotes)
class_list = banknotes['Class'].to_list()
class_list = [int(s) for s in class_list]
color_list = []

for i in range(len(class_list)):
    if class_list[i] == 0:
        color_list.append("Green")
    else:
        color_list.append("Red")

banknotes['Color'] = color_list

# 1_2
# ______________________________________________________________________________
all = (banknotes.std(axis=0, skipna=True))
print(banknotes.mean(axis=0, skipna=True))
print(all)
class_0 = banknotes[banknotes['Color'] == "Green"]
class_1 = banknotes[banknotes['Color'] == "Red"]
print(class_0.std(axis=0, skipna=True))
print(class_0.mean(axis=0, skipna=True))
print(class_1.std(axis=0, skipna=True))
print(class_1.mean(axis=0, skipna=True))
# finish table


# 2_1
# ______________________________________________________________________________
# split train and test

X = banknotes.iloc[:, :-1].values
y = banknotes.iloc[:, ].values

X_train, X_test, = train_test_split(banknotes, test_size=0.53, random_state=42)

# split into test_0 test_1
train_0 = X_train[X_train['Color'] == "Green"]
train_1 = X_train[X_train['Color'] == "Red"]

# plot in seaborn
sns.pairplot(train_0)
sns.pairplot(train_1)


# plt.show()


# 2_2-3
# ______________________________________________________________________________
# come up with strategy fake = curtosis <5 , skewness > 0 , variance <2.5

def predict(df):
    if (df['Curtosis'] < 5) and (df['Skewness'] >= 0) and (df['Variance'] <= 1):
        return 'Red'
    else:
        return 'Green'


X_test['Predict'] = X_test.apply(predict, axis=1)

print(X_test)

# 2_4
# ______________________________________________________________________________
print("My strategy: ")
print(confusion_matrix(X_test['Color'], X_test['Predict']))
tp, fn, fp, tn = confusion_matrix(X_test['Color'], X_test['Predict'],
                                  labels=["Green", "Red"]).reshape(-1)
print(tp, fn, fp, tn)
accuracy = round((((tp + tn) / (fn + fp + tp + tn)) * 100), 2)
true_pos = round(((tp / (tp + fn)) * 100), 2)
true_neg = round(((tn / (tn + fp)) * 100), 2)

print("TP     FP     TN     FN     Accuracy    TPR     TNR")
print(tp, "  ", fp, "  ", tn, "  ", fn, "   ", accuracy, "     ", true_pos, " ",
      true_neg)

# 3_1-3
# ______________________________________________________________________________
# scale data
del banknotes['Color']

banknotes = banknotes.astype(float)
X = banknotes.iloc[:, :-1].values
y = banknotes.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# k_models and plotting
k_range = [3, 5, 7, 9, 11]
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

print("Accuracy Scores for K: ", scores)

plt.plot(k_range, scores_list)
plt.xlabel("K")
plt.ylabel("Testing Accuracy")
plt.show()

best_k = max(scores, key=scores.get)
print(best_k)

knn1 = KNeighborsClassifier(n_neighbors=best_k)
knn1.fit(X_train, y_train)

y_pred_best = knn1.predict(X_test)

print(confusion_matrix(y_test, y_pred_best))
confusion = (confusion_matrix(y_test, y_pred_best))
print("Accuracy for knn:", accuracy_score(y_test, y_pred_best))

TN = confusion[0][0]
FN = confusion[1][0]
TP = confusion[1][1]
FP = confusion[0][1]

accuracy = round((((TP + TN) / (FN + FP + TP + TN)) * 100), 2)
true_pos = round(((TP / (TP + FN)) * 100), 2)
true_neg = round(((TN / (TN + TP)) * 100), 2)
print("Best knn")
print("TP     FP     TN     FN     Accuracy    TPR     TNR")
print(tp, "  ", fp, "  ", tn, "  ", fn, "   ", accuracy, "     ", true_pos, " ",
      true_neg)


# 3_4
# ______________________________________________________________________________
# BU ID 2156


# 4_1
# ______________________________________________________________________________

def knn(xtrain, ytrain, ytest, xtest):
    knn_loop = KNeighborsClassifier(n_neighbors=best_k)
    knn_loop.fit(xtrain, ytrain)

    y_pred_best = knn_loop.predict(xtest)

    confusion_matrix(ytest, y_pred_best)
    confusion = (confusion_matrix(ytest, y_pred_best))
    return (
    "Accuracy for knn:", "{:.2f}".format((accuracy_score(ytest, y_pred_best) * 100)))


banknotes = banknotes.astype(float)
x = banknotes.iloc[:, :-1].values
Y = banknotes.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.50)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# drop f1

X_train1 = np.delete(X_train, 0, 1)

X_test1 = np.delete(X_test, 0, 1)

print("Drop F1")
knn1 = knn(X_train1, y_train, y_test, X_test1)
print(knn1)

# drop f2
X_train2 = np.delete(X_train, 1, 1)

scaler.fit(X_train2)
X_test2 = np.delete(X_test, 1, 1)

print("Drop F2")
knn2 = knn(X_train2, y_train, y_test, X_test2)
print(knn2)


# drop f3
X_train3 = np.delete(X_train, 2, 1)

X_test3 = np.delete(X_test, 2, 1)

print("Drop F3")
print(knn(X_train3, y_train, y_test, X_test3))

# drop f4
X_train4 = np.delete(X_train, 3, 1)

X_test4 = np.delete(X_test, 3, 1)

print("Drop F4")
print(knn(X_train4, y_train, y_test, X_test4))

# 5_1
# ______________________________________________________________________________

banknotes = banknotes.astype(float)
x = banknotes.iloc[:, :-1].values
Y = banknotes.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.50)

logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

y_pred_log = logistic_regression.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred_log)
accuracy_percentage = 100 * accuracy
print("Logistic Regression Accuracy: ")
print(accuracy_percentage)

# 5_2
# ______________________________________________________________________________
print(confusion_matrix(y_test, y_pred_log))
confusion = (confusion_matrix(y_test, y_pred_log))

TN = confusion[0][0]
FN = confusion[1][0]
TP = confusion[1][1]
FP = confusion[0][1]

accuracy = round((((TP + TN) / (FN + FP + TP + TN)) * 100), 2)
true_pos = round(((TP / (TP + FN)) * 100), 2)
true_neg = round(((TN / (TN + TP)) * 100), 2)
print("Logistic")
print("TP     FP     TN     FN     Accuracy    TPR     TNR")
print(tp, "  ", fp, "  ", tn, "  ", fn, "   ", accuracy, "     ", true_pos, " ",
      true_neg)


# 6
# ______________________________________________________________________________

def logistic(xtrain, ytrain, ytest, xtest):
    logistic_regression = LogisticRegression()

    logistic_regression.fit(xtrain, ytrain)

    y_pred_log = logistic_regression.predict(xtrain)

    accuracy_loop = metrics.accuracy_score(ytest, y_pred_log)
    accuracy_percentage_loop = 100 * accuracy_loop
    print("Logistic Regression Accuracy: ")
    print(accuracy_percentage_loop)
    print(confusion_matrix(ytest, y_pred_log))
    confusion = (confusion_matrix(ytest, y_pred_log))

    accuracy = round((((TP + TN) / (FN + FP + TP + TN)) * 100), 2)
    true_pos = round(((TP / (TP + FN)) * 100), 2)
    true_neg = round(((TN / (TN + TP)) * 100), 2)

    return("")

# drop f1

X_train1 = np.delete(X_train, 0, 1)

X_test1 = np.delete(X_test, 0, 1)

y_test = np.delete(y_test, -1)
X_test1 = np.delete(X_test1, -1)

print("Drop F1")
print(logistic(X_train1, y_train, y_test, X_test1))
# drop f2
X_train2 = np.delete(X_train, 1, 1)

scaler.fit(X_train2)
X_test2 = np.delete(X_test, 1, 1)

print("Drop F2")
print(logistic(X_train2, y_train, y_test, X_test2))

# drop f3
X_train3 = np.delete(X_train, 2, 1)

X_test3 = np.delete(X_test, 2, 1)

print("Drop F3")
print(logistic(X_train3, y_train, y_test, X_test3))

# drop f4
X_train4 = np.delete(X_train, 3, 1)

X_test4 = np.delete(X_test, 3, 1)

print("Drop F4")
print(logistic(X_train4, y_train, y_test, X_test4))
