import pandas as pd
import numpy as np

# Question 1_1
################################################################################
SPY = pd.read_csv("SPY.csv")

RTX = pd.read_csv("RTX.csv")

print(SPY.head())
print(RTX.head())

SPY.loc[SPY['Return'] >= 0.0, 'True Label'] = '+'
SPY.loc[SPY['Return'] < 0.0, 'True Label'] = '-'

RTX.loc[RTX['Return'] >= 0.0, 'True Label'] = '+'
RTX.loc[RTX['Return'] < 0.0, 'True Label'] = '-'

print(SPY.head())
print(RTX.head())

# Question 1_2
################################################################################

probs = SPY.groupby('True Label').size().div(len(SPY))
print(probs)

probs = RTX.groupby('True Label').size().div(len(RTX))
print(probs)

# Question 1_3
################################################################################

# k = 1
# calculate that - , +
day_spy = SPY['True Label'].to_numpy()
day_rtx = RTX['True Label'].to_numpy()


def k_days_1(days):
    k_1 = 0
    not_k_1 = 0
    for index, element in enumerate(days):
        if days[index] == '-':
            if days[index + 1] == '+':
                k_1 = k_1 + 1
            else:
                not_k_1 = not_k_1 + 1
    return (k_1 / (k_1 + not_k_1))


print("K Values for - days repeating")
print()
print("SPY: K = 1 " + "{:.2%}".format(k_days_1(day_spy)))
print("RTX: K = 1 " + "{:.2%}".format(k_days_1(day_rtx)))
print()


# k = 2
# calculate that -, -,  +

def k_days_2(days):
    k_2 = 0
    not_k_2 = 0
    for index, element in enumerate(days):
        if days[index] == '-':
            if days[index + 1] == '-':
                if days[index + 2] == '+':
                    k_2 = k_2 + 1
                else:
                    not_k_2 = not_k_2 + 1
    return (k_2 / (k_2 + not_k_2))


print("SPY: K = 2 " + "{:.2%}".format(k_days_2(day_spy)))
print("RTX: K = 2 " + "{:.2%}".format(k_days_2(day_rtx)))
print()


# k = 3
# calculate that - , - , - , +
def k_days_3(days):
    k_3 = 0
    not_k_3 = 0
    for index, element in enumerate(days):
        if days[index] == '-':
            if days[index + 1] == '-':
                if days[index + 2] == '-':
                    if days[index + 3] == '+':
                        k_3 = k_3 + 1
                    else:
                        not_k_3 = not_k_3 + 1
    return k_3 / (k_3 + not_k_3)


print("SPY: K = 3 " + "{:.2%}".format(k_days_3(day_spy)))
print("RTX: K = 3 " + "{:.2%}".format(k_days_3(day_rtx)))

# Question 1_4
################################################################################


day_spy_pos = SPY['True Label'].to_numpy()
day_rtx_pos = RTX['True Label'].to_numpy()


def k_days_1_pos(days):
    k_1_pos = 0
    not_k_1_pos = 0
    for index, element in enumerate(days):
        if index <1258:
            if days[index] != '-':
                if days[index + 1] != '-':
                    k_1_pos = k_1_pos + 1
                else:
                    not_k_1_pos = not_k_1_pos + 1
    return (k_1_pos / (k_1_pos + not_k_1_pos))

print()
print("K Values for + days repeating")
print("SPY: K = 1 " + "{:.2%}".format(k_days_1_pos(day_spy_pos)))
print("RTX: K = 1 " + "{:.2%}".format(k_days_1_pos(day_rtx_pos)))
print()

# k = 2


def k_days_2_pos(days):
    k_2_pos = 0
    not_k_2_pos = 0
    for index, element in enumerate(days):
        if index < 1257:
            if days[index] != '-':
                if days[index + 1] != '-':
                    if days[index + 2] != '+':
                        k_2_pos = k_2_pos + 1
                    else:
                        not_k_2_pos = not_k_2_pos + 1
    return (k_2_pos / (k_2_pos + not_k_2_pos))


print("SPY: K = 2 " + "{:.2%}".format(k_days_2_pos(day_spy_pos)))
print("RTX: K = 2 " + "{:.2%}".format(k_days_2_pos(day_rtx_pos)))
print()


# k = 3
# calculate that - , - , - , +
def k_days_3_pos(days):
    k_3_pos = 0
    not_k_3_pos = 0
    for index, element in enumerate(days):
        if index < 1256:
            if days[index] != '-':
                if days[index + 1] != '-':
                    if days[index + 2] != '-':
                        if days[index + 3] != '+':
                            k_3_pos = k_3_pos + 1
                        else:
                            not_k_3_pos = not_k_3_pos + 1
    return k_3_pos / (k_3_pos + not_k_3_pos)


print("SPY: K = 3 " + "{:.2%}".format(k_days_3_pos(day_spy)))
print("RTX: K = 3 " + "{:.2%}".format(k_days_3_pos(day_rtx)))