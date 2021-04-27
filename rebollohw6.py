# BU ID U06002156
#  L = 1 (negative) and L = 2 (positive)
from random import randrange

import numpy
import xlrd
import openpyxl
import pandas as pd
import numpy as np
from array import *
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans
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

# 1
# ______________________________________________________________________________
# load data
raw = pd.read_excel(
    r'/Users/jreb/PycharmProjects/rebollohw6/seeds_dataset.xlsx')
# Group Remainder 0, using L = 1 (negative) and L = 2 (positive)
seeds = raw.loc[raw[8] != 3]
print(seeds)
# 1_1
# implement a linear kernel SVM
# ______________________________________________________________________________
X = seeds[[1, 2, 3, 4, 5, 6, 7]].values
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
y = seeds[8].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.50)

svm_classifier = sklearn.svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
predict_x = scaler.transform(X_test)
predicted = svm_classifier.predict(predict_x)


confusion_svml = (confusion_matrix(y_test, predicted, labels= [1,2]))
accuracy = accuracy_score(y_test, predicted)
accuracy = round((accuracy * 100), 2)

print("SVM Linear")
print("The Accuracy is: ", accuracy)
print(confusion_svml)

TN_svml = confusion_svml[0][0]
FN_svml = confusion_svml[1][0]
TP_svml = confusion_svml[1][1]
FP_svml = confusion_svml[0][1]
true_pos_svml = round(((TP_svml / (TP_svml + FN_svml)) * 100), 2)
true_neg_svml = round(((TN_svml / (TN_svml + FP_svml)) * 100), 2)


# 1_2
# ______________________________________________________________________________
# implement a Gaussian kernel SVM

X_traing, X_testg, y_traing, y_testg = train_test_split(X, y,
                                                        test_size=0.50)
svm_classifier_g = sklearn.svm.SVC(kernel='rbf')
svm_classifier_g.fit(X_traing, y_traing)
predict_g = scaler.transform(X_testg)
predicted_g = svm_classifier.predict(predict_g)

accuracy_svmg = accuracy_score(y_testg, predicted_g)
confusion_svmg = (confusion_matrix(y_testg, predicted_g, labels= [1,2]))

accuracy_svmg = round((accuracy_svmg * 100), 2)
print("SVM Gaussian")
print("The Accuracy is: ",accuracy_svmg)
print(confusion_svmg)

TN_svmg = confusion_svmg[0][0]
FN_svmg = confusion_svmg[1][0]
TP_svmg = confusion_svmg[1][1]
FP_svmg = confusion_svmg[0][1]
true_pos_svmg = round(((TP_svmg / (TP_svmg + FN_svmg)) * 100), 2)
true_neg_svmg = round(((TN_svmg / (TN_svmg + FP_svmg)) * 100), 2)

# 1_3
# ______________________________________________________________________________
#  implement a polynomial kernel SVM of degree 3
X_trainp, X_testp, y_trainp, y_testp = train_test_split(X, y,
                                                        test_size=0.50)
svm_classifier_p = sklearn.svm.SVC(kernel='poly', degree=3)
svm_classifier_p.fit(X_trainp, y_trainp)
predict_p = scaler.transform(X_testp)
predicted_p = svm_classifier.predict(predict_p)

accuracy_svmp = accuracy_score(y_testp, predicted_p)
confusion_svmp = (confusion_matrix(y_testp, predicted_p , labels= [1,2]))

accuracy_svmp = round((accuracy_svmp * 100), 2)

print("SVM Polynomial Degree 3")
print("The Accuracy is: ",accuracy_svmp)
print(confusion_svmp)

TN_svmp = confusion_svmp[0][0]
FN_svmp = confusion_svmp[1][0]
TP_svmp = confusion_svmp[1][1]
FP_svmp = confusion_svmp[0][1]
true_pos_svmp = round(((TP_svmp / (TP_svmp + FN_svmp)) * 100), 2)
true_neg_svmp = round(((TN_svmp / (TN_svmp + FP_svmp)) * 100), 2)

# 2_1
# ______________________________________________________________________________
#  implement a model of choice: Naive Bayesian
X_trainb, X_testb, y_trainb, y_testb = train_test_split(X, y,
                                                        test_size=0.50)

NB_classifier = GaussianNB().fit(X_trainb, y_trainb)
prediction_b = NB_classifier.predict(X_testb)

accuracy_b = accuracy_score(y_testb, prediction_b)
confusion_b = (confusion_matrix(y_testb, prediction_b, labels= [1,2]))

accuracy_b = round((accuracy_b * 100), 2)
print("Naive Bayesian")
print("The Accuracy is: ",accuracy_b)
print(confusion_b)

TN_b = confusion_b[0][0]
FN_b = confusion_b[1][0]
TP_b = confusion_b[1][1]
FP_b = confusion_b[0][1]
true_pos_b = round(((TP_b / (TP_b + FN_b)) * 100), 2)
true_neg_b = round(((TN_b / (TN_b + FP_b)) * 100), 2)

# 2_2
# ______________________________________________________________________________
print("Question 5 Table")
print("Model     TP     FP     TN     FN     Accuracy    TPR     TNR")
print("SVML", "   ", TP_svml, "  ", FP_svml, "    ", TN_svml, "  ", FN_svml,
      "   ",
      accuracy, "     ", true_pos_svml, " ", true_neg_svml)
print("SVMG", "    ", TP_svmg, "  ", FP_svmg, "    ", TN_svmg, "  ", FN_svmg,
      "   ",
      accuracy_svmg, "     ", true_pos_svmg, " ", true_neg_svmg)
print("SVMP3", "   ", TP_svmp, "  ", FP_svmp, "    ", TN_svmp, "  ",
      FN_svmp, "   ", accuracy_svmp, "     ", true_pos_svmp, " ",
      true_neg_svmp)
print("NB ", "     ", TP_b, "  ", FP_b, "     ", TN_b, "  ",
      FN_b, "   ", accuracy_b, "     ", true_pos_b, " ",
      true_neg_b)

# 3_1
# ______________________________________________________________________________
i_list = []
for i in range(1, 9):
    kmeans_classifier = KMeans(n_clusters=i)
    y_kmeans = kmeans_classifier.fit_predict(raw)
    inertia = kmeans_classifier.inertia_
    i_list.append(inertia)

fig, ax = plt.subplots(1, figsize=(7, 5))
plt.plot(range(1, 9), i_list, marker='x', color='red')
plt.xlabel('Number of clusters: k')
plt.ylabel('Inertia')
plt.tight_layout()
plt.show()

# 3_2
# ______________________________________________________________________________
best_k = 3

kmeans_classifier = KMeans(n_clusters=best_k)
y_kmeans = kmeans_classifier.fit_predict(X)
centroids = kmeans_classifier.cluster_centers_

labels = kmeans_classifier.labels_

# pick the x and y values , random of number 1-7
x = randrange(0, 7)
y = randrange(0, 7)
if x == y:
    y = randrange(0, 7)

fig, ax = plt.subplots(1, figsize=(7, 5))

plt.scatter(X[y_kmeans == 0, x], X[y_kmeans == 0, y]
            , s=75, c='red', label='Class 1')

plt.scatter(X[y_kmeans == 1, x], X[y_kmeans == 1, y],
            s=75, c='blue', label='Class 2')

plt.scatter(X[y_kmeans == 2, x], X[y_kmeans == 2, y],
            s=75, c='green', label='Class 3')

plt.scatter(centroids[:, x], centroids[:, y], s=200, c='black',
            label='Cluster Centroids')

plt.legend()
# plot
plt.tight_layout()
x_axis = str(x)
y_axis = str(y)
plt.xlabel('f{}'.format(x_axis))
plt.ylabel('f{}'.format(y_axis))
plt.show()

original_labels = seeds[8].values

# 3_3
# ______________________________________________________________________________
# see what the actual class is for each point and assign the labels to the
# cluster
# (Kama: 1, Rosa: 2)
cluster0 = []
cluster1 = []
cluster2 = []

while i <140:
    if y_kmeans[i] == 0:
        cluster0.append(original_labels[i])
    i = i+1

i = 0
while i <140:
    if y_kmeans[i] == 1:
        cluster1.append(original_labels[i])
    i = i+1

i = 0
while i <140:
    if y_kmeans[i] == 2:
        cluster2.append(original_labels[i])
    i = i+1


count1 = cluster0.count(1)
count2 = cluster0.count(2)

print("Centroid 0" )
if count1 > count2:
    cluster0_label = "Kama"
    cluster0_cen = centroids[0]
    print(cluster0_label)
    print(centroids[0])
elif count1 < count2:
    cluster0_label = "Rosa"
    cluster0_cen = centroids[0]
    print(cluster0_label)
    print(centroids[0])


print("Centroid 1" )
count1_1 = cluster1.count(1)
count2_1 = cluster1.count(2)
if count1_1 > count2_1:
    cluster1_label = "Kama"
    cluster1_cen = centroids[1]
    print(cluster1_label)
    print(centroids[1])
elif count1_1 < count2_1:
    cluster1_label = "Rosa"
    cluster1_cen = centroids[1]
    print(cluster1_label)
    print(centroids[1])

print("Centroid 2" )
count1_2 = cluster2.count(1)
count2_2 = cluster2.count(2)
if count1_2 > count2_2:
    cluster2_label = "Kama"
    cluster2_cen = centroids[2]
    print(cluster2_label)
    print(centroids[2])
elif count1_2 < count2_2:
    cluster2_label = "Rosa"
    cluster2_cen = centroids[2]
    print(cluster2_label)
    print(centroids[2])

# 3_4
# ______________________________________________________________________________
# implement based on distance from centroid
# find two largest clusters



cluster0_len = len(cluster0)
cluster1_len = len(cluster1)
cluster2_len = len(cluster2)


lengths = [(cluster0_len, cluster0_cen), (cluster1_len,cluster1_cen),
           (cluster2_len, cluster2_cen)]
# find the 2 largest centroids since I am using the 2 label version

lengths = sorted(lengths)

largest_cen1 = lengths[-1][1]  # 39
largest_cen2 = lengths[-2][1]  # 26

# calculate the distance between points and the two centroids to classify


def eucledian_distance(points, centroida, centroidb):

    distances = []
    it = 0
    while it < len(points):
        dist1 = numpy.linalg.norm(points[it] - centroida)
        dist2 = numpy.linalg.norm(points[it] - centroidb)

        if abs(dist1) < abs(dist2):
            distances.append(1)
        elif abs(dist1) > abs(dist2):
            distances.append(2)

        dist1 = 0
        dist2 - 0
        it = it+1

    return(distances)

# make given data a list

X = X.tolist()

# output the predictions
predicted_label = eucledian_distance(X, largest_cen1 , largest_cen2)

# test set to validate against
test_y = seeds[8].values
test_y
# compare accuracy

accuracy_34= accuracy_score(test_y, predicted_label)
accuracy_34 = round((accuracy_34 * 100), 2)

print("Question 3-4 Model")
print("The Accuracy is: ", accuracy_34)




# 3_5
# ______________________________________________________________________________
# determine confusion matrix
confusion_34 = (confusion_matrix(y_test, predicted, labels= [1,2]))
print(confusion_34)