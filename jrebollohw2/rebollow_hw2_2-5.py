import pandas as pd
import numpy as np
import datetime as dt
import sklearn as sk
from sklearn.metrics import confusion_matrix
import os
import matplotlib as matplotlib
import matplotlib.pyplot as plt

# Question 2_1
################################################################################
from sklearn.metrics import confusion_matrix

SPY = pd.read_csv("SPY.csv")

RTX = pd.read_csv("RTX.csv")

SPY.loc[SPY['Return'] >= 0.0, 'True Label'] = '+'
SPY.loc[SPY['Return'] < 0.0, 'True Label'] = '-'

RTX.loc[RTX['Return'] >= 0.0, 'True Label'] = '+'
RTX.loc[RTX['Return'] < 0.0, 'True Label'] = '-'

# serperate into training and test

training_data_spy = SPY.iloc[:753, :]
test_data_spy = SPY.iloc[754:, :]
print(len(test_data_spy))
training_data_rtx = RTX.iloc[:753, :]
test_data_rtx = RTX.iloc[754:, :]

# k = 1
training_spy = training_data_spy['True Label'].to_numpy()
test_spy = test_data_spy['True Label'].to_numpy()

training_rtx = training_data_rtx['True Label'].to_numpy()
test_rtx = test_data_rtx['True Label'].to_numpy()


# for previous string x, x+1 find value of x+2
# search for all (x, x+1) find the x+2 and find prob

def predict_for_2(training, test):
    k = 2
    k_2 = 0
    not_k_2 = 0
    holder = []
    window = []
    length = len(test)
    prediction_label = []

    for i in range(505):
        if i < 504:

            holder.append(test[i])
            holder.append(test[i+1])

            if training[i] == holder[0] and training[i + 1] == holder[1]:
                sign = training[i + 2]
                if sign[0] == '+':
                    k_2 = k_2 + 1
                if sign[0] == '-':
                    not_k_2 = not_k_2 + 1
            if k_2 > not_k_2:
                prediction_label.append("+")
            if not_k_2 > k_2:
                prediction_label.append("-")
            if not_k_2 == k_2:
                label = (k_2 / (k_2 + not_k_2+1))
                if label >= .50:
                    prediction_label.append('+')
                else:
                    prediction_label.append('-')

        holder.clear()

        continue

    return prediction_label

print("***********************************************************************")
print("W = 2")
spy_2_predicted = (predict_for_2(training_spy, test_spy))
rtx_2_predicted = (predict_for_2(training_rtx, test_rtx))
spy_2_predicted.append("-")
rtx_2_predicted.append("-")
print("SPY k= 2 Predicted")
print(spy_2_predicted)
print("RTX k=  Predicted")
print(rtx_2_predicted)
test_data_spy['Predicted Label 2'] = spy_2_predicted
test_data_rtx['Predicted Label 2'] = rtx_2_predicted

def predict_for_3(training, test):
    k = 3
    k_3 = 0
    not_k_3 = 0
    holder = []
    window = []
    length = len(test)
    prediction_label = []

    for i in range(505):
        if i < 503:

            holder.append(test[i])
            holder.append(test[i+1])
            holder.append(test[i+2])

            if training[i] == holder[0] and training[i + 1] == holder[1] and \
                    training[i+2] == holder[2]:
                sign = training[i + 3]
                if sign[0] == '+':
                    k_3 = k_3 + 1
                if sign[0] == '-':
                    not_k_3 = not_k_3 + 1
            if k_3 > not_k_3:
                prediction_label.append("+")
            if not_k_3 > k_3:
                prediction_label.append("-")
            if not_k_3 == k_3:
                label = (k_3 / (k_3 + not_k_3+1))
                if label >= .50:
                    prediction_label.append('+')
                else:
                    prediction_label.append('-')

        holder.clear()

        continue

    return prediction_label
print("***********************************************************************")
print("W = 3 ")

spy_3_predicted = (predict_for_3(training_spy, test_spy))
rtx_3_predicted = (predict_for_3(training_rtx, test_rtx))
spy_3_predicted.append("-")
rtx_3_predicted.append("-")

spy_3_predicted.append("+")
rtx_3_predicted.append("+")

print("SPY k= 3 Predicted")
print(spy_3_predicted)
print("RTX k= 3 Predicted")
print(rtx_3_predicted)

test_data_spy['Predicted Label 3'] = spy_3_predicted
test_data_rtx['Predicted Label 3'] = rtx_3_predicted



def predict_for_4(training, test):
    k = 4
    k_4 = 0
    not_k_4 = 0
    holder = []
    window = []
    length = len(test)
    prediction_label = []

    for i in range(505):
        if i < 502:

            holder.append(test[i])
            holder.append(test[i+1])
            holder.append(test[i+2])
            holder.append(test[i+3])

            if training[i] == holder[0] and training[i + 1] == holder[1] and \
                    training[i+2] == holder[2] and training[i+3] == holder[3]:
                sign = training[i + 4]
                if sign[0] == '+':
                    k_4 = k_4 + 1
                if sign[0] == '-':
                    not_k_4 = not_k_4 + 1
            if k_4 > not_k_4:
                prediction_label.append("+")
            if not_k_4 > k_4:
                prediction_label.append("-")
            if not_k_4 == k_4:
                label = (k_4 / (k_4 + not_k_4+1))
                if label >= .50:
                    prediction_label.append('+')
                else:
                    prediction_label.append('-')

        holder.clear()

        continue

    return prediction_label
print("***********************************************************************")
print("W = 4 ")
spy_4_predicted = (predict_for_4(training_spy, test_spy))
rtx_4_predicted = (predict_for_4(training_rtx, test_rtx))
spy_4_predicted.append("-")
rtx_4_predicted.append("-")
spy_4_predicted.append("+")
rtx_4_predicted.append("+")
spy_4_predicted.append("-")
rtx_4_predicted.append("-")

print("SPY k= 4 Predicted")
print(spy_4_predicted)
print("RTX k= 4 Predicted")
print(rtx_4_predicted)

test_data_spy['Predicted Label 4'] = spy_4_predicted
test_data_rtx['Predicted Label 4'] = rtx_4_predicted

print(test_data_spy)
print(test_data_rtx)
print("***********************************************************************")
print("SPY")
print()
print("2")
print(confusion_matrix(test_data_spy['True Label'], spy_2_predicted ))
tp_spy2, fn_spy2, fp_spy2, tn_spy2 = confusion_matrix(test_data_spy['True Label'],spy_2_predicted,
                                  labels=["+", "-"]).reshape(-1)
print(tp_spy2, fn_spy2, fp_spy2, tn_spy2)
accuracy_spy2 = (tp_spy2+tn_spy2) /((fn_spy2 +fp_spy2+ tp_spy2+tn_spy2)*100)
true_pos_spy2 = (tp_spy2/(tp_spy2+fn_spy2))
true_neg_spy2 = (tn_spy2 / (tn_spy2 + fp_spy2))
print()


print("3")
print(confusion_matrix(test_data_spy['True Label'], spy_3_predicted ))
tp_spy3, fn_spy3, fp_spy3, tn_spy3= confusion_matrix(test_data_spy['True Label'],spy_3_predicted,
                                  labels=["+", "-"]).reshape(-1)
print(tp_spy3, fn_spy3, fp_spy3, tn_spy3)
accuracy_spy3 = (tp_spy3+tn_spy3) /((fn_spy3 +fp_spy3+ tp_spy3+tn_spy3)*100)
true_pos_spy3 = (tp_spy3/(tp_spy3+fn_spy3))
true_neg_spy3 = (tn_spy3 / (tn_spy3 + fp_spy3))
print()
print()


print("4")
print(confusion_matrix(test_data_spy['True Label'], spy_3_predicted ))
tp_spy4, fn_spy4, fp_spy4, tn_spy4= confusion_matrix(test_data_spy['True Label'],spy_4_predicted,
                                  labels=["+", "-"]).reshape(-1)
print(tp_spy3, fn_spy4, fp_spy4, tn_spy4)
accuracy_spy4 = (tp_spy4+tn_spy4) /((fn_spy4 +fp_spy4+ tp_spy4+tn_spy4)*100)
true_pos_spy4 = (tp_spy4/(tp_spy4+fn_spy4))
true_neg_spy4 = (tn_spy4 / (tn_spy4 + fp_spy4))
print()
print()
print("***********************************************************************")
print("RTX")
print()
print("2")
print(confusion_matrix(test_data_rtx['True Label'], rtx_2_predicted ))
tp_rtx2, fn_rtx2, fp_rtx2, tn_rtx2 = confusion_matrix(test_data_rtx['True Label'],rtx_2_predicted,
                                  labels=["+", "-"]).reshape(-1)
print(tp_rtx2, fn_rtx2, fp_rtx2, tn_rtx2)
accuracy_rtx2 = (tp_rtx2+tn_rtx2) /((fn_rtx2 +fp_rtx2+ tp_rtx2+tn_rtx2)*100)
true_pos_rtx2 = (tp_rtx2/(tp_rtx2+fn_rtx2))
true_neg_rtx2 = (tn_rtx2 / (tn_rtx2 + fp_rtx2))
print()


print("3")
print(confusion_matrix(test_data_rtx['True Label'], rtx_3_predicted ))
tp_rtx3, fn_rtx3, fp_rtx3, tn_rtx3= confusion_matrix(test_data_rtx['True Label'],rtx_3_predicted,
                                  labels=["+", "-"]).reshape(-1)
print(tp_rtx3, fn_rtx3, fp_rtx3, tn_rtx3)
accuracy_rtx3 = (tp_rtx3+tn_rtx3) /((fn_rtx3 +fp_rtx3+ tp_rtx3+tn_rtx3)*100)
true_pos_rtx3 = (tp_rtx3/(tp_rtx3+fn_rtx3))
true_neg_rtx3 = (tn_rtx3 / (tn_rtx3 + fp_rtx3))
print()
print()


print("4")
print(confusion_matrix(test_data_rtx['True Label'], rtx_3_predicted ))
tp_rtx4, fn_rtx4, fp_rtx4, tn_rtx4= confusion_matrix(test_data_rtx['True Label'],rtx_4_predicted,
                                  labels=["+", "-"]).reshape(-1)
print(tp_rtx3, fn_rtx4, fp_rtx4, tn_rtx4)
accuracy_rtx4 = (tp_rtx4+tn_rtx4) /((fn_rtx4 +fp_rtx4+ tp_rtx4+tn_rtx4)*100)
true_pos_rtx4 = (tp_rtx4/(tp_rtx4+fn_rtx4))
true_neg_rtx4 = (tn_rtx4 / (tn_rtx4 + fp_rtx4))

# Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN.
#  TPR = TP/(TP + FN)
# TNR = TN/(TN + FP)


# count the number of +, - for each column
# if + > - label = +  else -
pos_spy = 0
neg_spy = 0
ensemble_prediction_spy = []

for i in range(len(spy_2_predicted)):
    if spy_2_predicted[i] == '+':
        pos_spy = pos_spy + 1
    if spy_2_predicted[i] == '-':
        neg_spy = neg_spy + 1
    if spy_3_predicted[i] == '+':
        pos_spy = pos_spy + 1
    if spy_3_predicted[i] == '-':
        neg_spy = neg_spy + 1
    if spy_4_predicted[i] == '+':
        pos_spy = pos_spy + 1
    if spy_4_predicted[i] == '-':
        neg_spy = neg_spy + 1

    if pos_spy > neg_spy:
        ensemble_prediction_spy.append('+')
    if neg_spy > pos_spy:
        ensemble_prediction_spy.append('-')
    elif pos_spy == neg_spy:
        result = pos_spy/(pos_spy +neg_spy)
        if result >= .50:
            ensemble_prediction_spy.append('+')
        else:
            ensemble_prediction_spy.append('-')

print(ensemble_prediction_spy)
print()
print("Ensemble Method SPY")
print(confusion_matrix(test_data_spy['True Label'], ensemble_prediction_spy ))
tp_e_spy, fn_e_spy, fp_e_spy, tn_e_spy = confusion_matrix(test_data_spy['True Label'],ensemble_prediction_spy,
                                  labels=["+", "-"]).reshape(-1)
print(tp_e_spy, fn_e_spy, fp_e_spy, tn_e_spy)
accuracy_e_spy = (tp_e_spy+tn_e_spy) /((fn_e_spy +fp_e_spy+ tp_e_spy+tn_e_spy)*100)
true_pos_e_spy = (tp_e_spy/(tp_e_spy+fn_e_spy))
true_neg_e_spy = (tn_e_spy / (tn_e_spy + fp_e_spy))


pos = 0
neg = 0
ensemble_prediction = []

for i in range(len(rtx_2_predicted)):
    if rtx_2_predicted[i] == '+':
        pos = pos + 1
    if rtx_2_predicted[i] == '-':
        neg = neg + 1
    if rtx_3_predicted[i] == '+':
        pos = pos + 1
    if rtx_3_predicted[i] == '-':
        neg = neg + 1
    if rtx_4_predicted[i] == '+':
        pos = pos + 1
    if rtx_4_predicted[i] == '-':
        neg = neg + 1

    if pos > neg:
        ensemble_prediction.append('+')
    if neg > pos:
        ensemble_prediction.append('-')
    elif pos == neg:
        result = pos/(pos +neg)
        if result >= .50:
            ensemble_prediction.append('+')
        else:
            ensemble_prediction.append('-')

print(ensemble_prediction)
print()
print("Ensemble Method RTX")
print(confusion_matrix(test_data_rtx['True Label'], ensemble_prediction ))
tp_e_rtx, fn_e_rtx, fp_e_rtx, tn_e_rtx = confusion_matrix(test_data_rtx['True Label'],ensemble_prediction,
                                  labels=["+", "-"]).reshape(-1)
print(tp_e_rtx, fn_e_rtx, fp_e_rtx, tn_e_rtx)
accuracy_e_rtx = (tp_e_rtx+tn_e_rtx) /((fn_e_rtx +fp_e_rtx+ tp_e_rtx+tn_e_rtx)*100)
true_pos_e_rtx = (tp_e_rtx/(tp_e_rtx+fn_e_rtx))
true_neg_e_rtx = (tn_e_rtx / (tn_e_rtx + fp_e_rtx))


print('W    Ticker  TP  FP  TN FN  accuracy    TPR     TNR')
print('2    SPY     ',tp_spy2, fp_spy2, tn_spy2, fn_spy2, "    %.2f      " % (accuracy_spy2*100),
      "%.2f " % (true_pos_spy2*100), "%.2f      " % (true_neg_spy2*100))

print('3    SPY     ',tp_spy3, fp_spy3, tn_spy3, fn_spy3, "    %.2f      " % (accuracy_spy3*100),
      "%.2f " % (true_pos_spy3*100), "%.2f      " % (true_neg_spy3*100))

print('4    SPY     ',tp_spy4, fp_spy4, tn_spy4, fn_spy4, "    %.2f      " % (accuracy_spy4*100),
      "%.2f " % (true_pos_spy4*100), "%.2f      " % (true_neg_spy4*100))

print('E    SPY     ',tp_e_spy, fp_e_spy, tn_e_spy, fn_e_spy, "    %.2f      " % (accuracy_e_spy*100),
      "%.2f " % (true_pos_e_spy*100), "%.2f      " % (true_neg_e_spy*100))

print('2    RTX     ',tp_rtx2, fp_rtx2, tn_rtx2, fn_rtx2, "    %.2f      " % (accuracy_rtx2*100),
      "%.2f " % (true_pos_rtx2*100), "%.2f      " % (true_neg_rtx2*100))

print('3    RTX     ',tp_rtx3, fp_rtx3, tn_rtx3, fn_rtx3, "    %.2f      " % (accuracy_rtx3*100),
      "%.2f " % (true_pos_rtx3*100), "%.2f      " % (true_neg_rtx3*100))

print('4    RTX     ',tp_rtx4, fp_rtx4, tn_rtx4, fn_rtx4, "    %.2f      " % (accuracy_rtx4*100),
      "%.2f " % (true_pos_rtx4*100), "%.2f      " % (true_neg_rtx4*100))

print('E    RTX     ',tp_e_rtx, fp_e_rtx, tn_e_rtx, fn_e_rtx, "    %.2f      " % (accuracy_e_rtx*100),
      "%.2f " % (true_pos_e_rtx*100), "%.2f      " % (true_neg_e_rtx*100))

print("***********************************************************************")
print("PLOT")

# W4 RTX
print(rtx_4_predicted)


ticker = 'RTX'
input_dir = r'/Users/jreb/PycharmProjects/ds_week_1'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)

    """    your code for assignment 1 goes here
    """


except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

# make the list of lines usable by making 2D list

separator = ','
ntable = []
for row in lines:
    if (separator in row):
        row = row.rstrip()
        row = row.split(separator)
        ntable += [row]


# needed variables and lists
closing_price_list = []
start = float(100)
money_on_hand_list = [100]
total_shares = []
decision_list = []
years = ['2019', '2020']


# load all data for closing
c = 0
try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] in years:
            closing_price_list.append(ntable[c][10])

except IndexError as IE:
    print()
    float_closing = [float(i) for i in closing_price_list]



# if rtx_predicts + buy , if rtx_predicts sell

total_shares = (money_on_hand_list[0])/(float_closing[0])
counter = 0
c = 0
sell = 0.0
try:
    for j in ntable:
        counter = counter +1
        if rtx_4_predicted[c+1] == '-':
            # sell at closing_price_list[c]
            sell = total_shares * float(float_closing[c])
            # buy at closing_price_list[c+1]
            buy = sell/(float_closing[c+1])
            # update stock
            total_shares = buy
            # update money
            money_on_hand_list.append(total_shares*(float_closing[c+1]))
            decision_list.append("Sell")
            c = c+1

        elif rtx_4_predicted[c+1] == '+':
            decision_list.append("Hold")
            c = c+1
            # update money
            money_on_hand_list.append((total_shares)*(float_closing[c+1]))

except IndexError as IE:
    test_data_rtx['W Return'] = money_on_hand_list



# Ensemble
print("ENSEMBLE")
print(ensemble_prediction)


# needed variables and lists
closing_price_list = []
start = float(100)
money_on_hand_ensemble = [100]
total_shares = []
decision_list = []
years = ['2019', '2020']


# load all data for closing
c = 0
try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] in years:
            closing_price_list.append(ntable[c][10])

except IndexError as IE:
    print()
    float_closing = [float(i) for i in closing_price_list]



# if rtx_predicts + buy , if rtx_predicts sell

total_shares_ensemble = (money_on_hand_ensemble[0])/(float_closing[0])
counter = 0
c = 0
sell = 0.0
try:
    for j in ntable:
        counter = counter + 1
        if ensemble_prediction[c+1] == '-':
            # sell at closing_price_list[c]
            sell = total_shares_ensemble * float(float_closing[c])
            # buy at closing_price_list[c+1]
            buy = sell/(float_closing[c+1])
            # update stock
            total_shares_ensemble = buy
            # update money
            money_on_hand_ensemble.append(total_shares_ensemble*(float_closing[c+1]))
            decision_list.append("Sell")
            c = c+1

        elif ensemble_prediction[c+1] == '+':
            decision_list.append("Hold")
            c = c+1
            # update money
            money_on_hand_ensemble.append((total_shares_ensemble)*(float_closing[c+1]))


except IndexError as IE:
    print()
    test_data_rtx['E Return'] = money_on_hand_ensemble


# Buy and Hold
# Buy $100 worth on first day and sell stock at last day price
stock_number = 100/float(68.16)
stock_number = float(stock_number)
sell = float(71.51)
sold = stock_number*float(71.51)
final_profit = sold - 100
print("The profit from buy and hold is:", round(final_profit,2))



ax = plt.gca()

test_data_rtx.plot(kind='line',x='Date',y='E Return', color= 'blue', ax=ax)
test_data_rtx.plot(kind='line',x='Date',y='W Return', color='red', ax=ax)

plt.show()