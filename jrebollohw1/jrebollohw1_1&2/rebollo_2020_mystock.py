"""
Justin Rebollo
Class: CS 677 - Spring 2
Date: 3/23/2021
Homework Problem # 1 & 2
Description of Problem (just a 1-2 line summary!): Question  1 & 2 year 2017
for each of the 5 years, compute the mean and standard deviation for the sets R,
R− and R+ of daily returns for your stock for each day of the week

"""

#given code
import os
import math

ticker = 'RTX'
input_dir = r'/Users/jreb/PycharmProjects/ds_week_1'
ticker_file = os.path.join(input_dir, ticker + '.csv')

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

#create 2d list
separator = ','
ntable = []
for row in lines:
    if (separator in row):
        row = row.rstrip()
        row = row.split(separator)
        ntable += [row]

try:
    with open(ticker_file) as f:
        lines = f.read().splitlines()
    print('opened file for ticker: ', ticker)

    """    your code for assignment 1 goes here
    """


except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

#to calculate values
def variance(data):
    n = len(data)
    mean = sum(data) / n
    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / n
    return variance


def stdev(data):
    var = variance(data)
    std_dev = math.sqrt(var)
    return std_dev

## 2020

# access only the return value for each entry and create new list
c = 0
# all returns for 2020
# 2020 broken into days
r_total = []
r_total_2020_Monday = []
r_total_2020_Tuesday = []
r_total_2020_Wednesday = []
r_total_2020_Thursday = []
r_total_2020_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] == '2020':
            if ntable[c][4] == 'Monday':
                r_total_2020_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_total_2020_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_total_2020_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_total_2020_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_total_2020_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_total_2020_Monday)):
    r_total_2020_Monday[i] = float(r_total_2020_Monday[i])
for i in range(0, len(r_total_2020_Tuesday)):
    r_total_2020_Tuesday[i] = float(r_total_2020_Tuesday[i])
for i in range(0, len(r_total_2020_Wednesday)):
    r_total_2020_Wednesday[i] = float(r_total_2020_Wednesday[i])
for i in range(0, len(r_total_2020_Thursday)):
    r_total_2020_Thursday[i] = float(r_total_2020_Thursday[i])
for i in range(0, len(r_total_2020_Friday)):
    r_total_2020_Friday[i] = float(r_total_2020_Friday[i])
print("2020 Monday Average:",
      (sum(r_total_2020_Monday)) / len(r_total_2020_Monday))
variance(r_total_2020_Monday)
print("2020 Monday Standard Deviation:", stdev(r_total_2020_Monday))
print("2020 Tuesday Average:",
      (sum(r_total_2020_Tuesday)) / len(r_total_2020_Tuesday))
variance(r_total_2020_Tuesday)
print("2020 Monday Standard Deviation:", stdev(r_total_2020_Tuesday))
print("2020 Wednesday Average:",
      (sum(r_total_2020_Wednesday)) / len(r_total_2020_Wednesday))
variance(r_total_2020_Wednesday)
print("2020 Monday Standard Deviation:", stdev(r_total_2020_Wednesday))
print("2020 Thursday Average:",
      (sum(r_total_2020_Thursday)) / len(r_total_2020_Thursday))
variance(r_total_2020_Thursday)
print("2020 Monday Standard Deviation:", stdev(r_total_2020_Thursday))
print("2020 Friday Average:",
      (sum(r_total_2020_Friday)) / len(r_total_2020_Friday))
variance(r_total_2020_Friday)
print("2020 Monday Standard Deviation:", stdev(r_total_2020_Friday))

# positive returns
# access only the return value for each entry and create new list
c = 0
# all positive returns for 2020
# 2020 positive broken into days
r_pos = []
r_pos_2020_Monday = []
r_pos_2020_Tuesday = []
r_pos_2020_Wednesday = []
r_pos_2020_Thursday = []
r_pos_2020_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] == '2020' and ntable[c][13] > '0':
            if ntable[c][4] == 'Monday':
                r_pos_2020_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_pos_2020_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_pos_2020_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_pos_2020_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_pos_2020_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_pos_2020_Monday)):
    r_pos_2020_Monday[i] = float(r_pos_2020_Monday[i])
for i in range(0, len(r_pos_2020_Tuesday)):
    r_pos_2020_Tuesday[i] = float(r_pos_2020_Tuesday[i])
for i in range(0, len(r_pos_2020_Wednesday)):
    r_pos_2020_Wednesday[i] = float(r_pos_2020_Wednesday[i])
for i in range(0, len(r_pos_2020_Thursday)):
    r_pos_2020_Thursday[i] = float(r_pos_2020_Thursday[i])
for i in range(0, len(r_pos_2020_Friday)):
    r_pos_2020_Friday[i] = float(r_pos_2020_Friday[i])
print("2020 Monday Positive Average:",
      (sum(r_pos_2020_Monday)) / len(r_pos_2020_Monday))
variance(r_pos_2020_Monday)
print("2020 Monday Standard Deviation:", stdev(r_pos_2020_Monday))
print("2020 Tuesday Positive Average:",
      (sum(r_pos_2020_Tuesday)) / len(r_pos_2020_Tuesday))
variance(r_pos_2020_Tuesday)
print("2020 Monday Standard Deviation:", stdev(r_pos_2020_Tuesday))
print("2020 Wednesday Positive Average:",
      (sum(r_pos_2020_Wednesday)) / len(r_pos_2020_Wednesday))
variance(r_pos_2020_Wednesday)
print("2020 Monday Standard Deviation:", stdev(r_pos_2020_Wednesday))
print("2020 Thursday Positive Average:",
      (sum(r_pos_2020_Thursday)) / len(r_pos_2020_Thursday))
variance(r_pos_2020_Thursday)
print("2020 Monday Standard Deviation:", stdev(r_pos_2020_Thursday))
print("2020 Friday Positive Average:",
      (sum(r_pos_2020_Friday)) / len(r_pos_2020_Friday))
variance(r_pos_2020_Friday)
print("2020 Monday Standard Deviation:", stdev(r_pos_2020_Friday))

# negative returns
r_neg = []

# access only the return value for each entry and create new list
c = 0
# all negative returns for 2020
# 2020 negative broken into days
r_neg = []
r_neg_2020_Monday = []
r_neg_2020_Tuesday = []
r_neg_2020_Wednesday = []
r_neg_2020_Thursday = []
r_neg_2020_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] == '2020' and ntable[c][13] < '0':
            if ntable[c][4] == 'Monday':
                r_neg_2020_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_neg_2020_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_neg_2020_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_neg_2020_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_neg_2020_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_neg_2020_Monday)):
    r_neg_2020_Monday[i] = float(r_neg_2020_Monday[i])
for i in range(0, len(r_neg_2020_Tuesday)):
    r_neg_2020_Tuesday[i] = float(r_neg_2020_Tuesday[i])
for i in range(0, len(r_neg_2020_Wednesday)):
    r_neg_2020_Wednesday[i] = float(r_neg_2020_Wednesday[i])
for i in range(0, len(r_neg_2020_Thursday)):
    r_neg_2020_Thursday[i] = float(r_neg_2020_Thursday[i])
for i in range(0, len(r_neg_2020_Friday)):
    r_neg_2020_Friday[i] = float(r_neg_2020_Friday[i])
print("2020 Monday Negative Average:",
      (sum(r_neg_2020_Monday)) / len(r_neg_2020_Monday))
variance(r_neg_2020_Monday)
print("2020 Monday Standard Deviation:", stdev(r_neg_2020_Monday))

print("2020 Tuesday Negative Average:",
      (sum(r_neg_2020_Tuesday)) / len(r_neg_2020_Tuesday))
variance(r_neg_2020_Tuesday)
print("2020 Monday Standard Deviation:", stdev(r_neg_2020_Tuesday))

print("2020 Wednesday Negative Average:",
      (sum(r_neg_2020_Wednesday)) / len(r_neg_2020_Wednesday))
variance(r_neg_2020_Wednesday)
print("2020 Monday Standard Deviation:", stdev(r_neg_2020_Wednesday))

print("2020 Thursday Negative Average:",
      (sum(r_neg_2020_Thursday)) / len(r_neg_2020_Thursday))
variance(r_neg_2020_Thursday)
print("2020 Monday Standard Deviation:", stdev(r_neg_2020_Thursday))

print("2020 Friday Negative Average:",
      (sum(r_neg_2020_Friday)) / len(r_neg_2020_Friday))
variance(r_neg_2020_Friday)
print("2020 Monday Standard Deviation:", stdev(r_neg_2020_Friday))