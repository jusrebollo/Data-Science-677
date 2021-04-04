"""
Justin Rebollo
Class: CS 677 - Spring 2
Date: 3/23/2021
Homework Problem #  1 & 2
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

# create 2d list
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

#for value calculations
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

## 2019

# access only the return value for each entry and create new list
c = 0
# all returns for 2019
# 2019 broken into days
r_total = []
r_total_2019_Monday = []
r_total_2019_Tuesday = []
r_total_2019_Wednesday = []
r_total_2019_Thursday = []
r_total_2019_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] == '2019':
            if ntable[c][4] == 'Monday':
                r_total_2019_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_total_2019_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_total_2019_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_total_2019_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_total_2019_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_total_2019_Monday)):
    r_total_2019_Monday[i] = float(r_total_2019_Monday[i])
for i in range(0, len(r_total_2019_Tuesday)):
    r_total_2019_Tuesday[i] = float(r_total_2019_Tuesday[i])
for i in range(0, len(r_total_2019_Wednesday)):
    r_total_2019_Wednesday[i] = float(r_total_2019_Wednesday[i])
for i in range(0, len(r_total_2019_Thursday)):
    r_total_2019_Thursday[i] = float(r_total_2019_Thursday[i])
for i in range(0, len(r_total_2019_Friday)):
    r_total_2019_Friday[i] = float(r_total_2019_Friday[i])
print("2019 Monday Average:",
      (sum(r_total_2019_Monday)) / len(r_total_2019_Monday))
variance(r_total_2019_Monday)
print("2019 Monday Standard Deviation:", stdev(r_total_2019_Monday))
print("2019 Tuesday Average:",
      (sum(r_total_2019_Tuesday)) / len(r_total_2019_Tuesday))
variance(r_total_2019_Tuesday)
print("2019 Monday Standard Deviation:", stdev(r_total_2019_Tuesday))
print("2019 Wednesday Average:",
      (sum(r_total_2019_Wednesday)) / len(r_total_2019_Wednesday))
variance(r_total_2019_Wednesday)
print("2019 Monday Standard Deviation:", stdev(r_total_2019_Wednesday))
print("2019 Thursday Average:",
      (sum(r_total_2019_Thursday)) / len(r_total_2019_Thursday))
variance(r_total_2019_Thursday)
print("2019 Monday Standard Deviation:", stdev(r_total_2019_Thursday))
print("2019 Friday Average:",
      (sum(r_total_2019_Friday)) / len(r_total_2019_Friday))
variance(r_total_2019_Friday)
print("2019 Monday Standard Deviation:", stdev(r_total_2019_Friday))

# positive returns
# access only the return value for each entry and create new list
c = 0
# all positive returns for 2019
# 2019 positive broken into days
r_pos = []
r_pos_2019_Monday = []
r_pos_2019_Tuesday = []
r_pos_2019_Wednesday = []
r_pos_2019_Thursday = []
r_pos_2019_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] == '2019' and ntable[c][13] > '0':
            if ntable[c][4] == 'Monday':
                r_pos_2019_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_pos_2019_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_pos_2019_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_pos_2019_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_pos_2019_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_pos_2019_Monday)):
    r_pos_2019_Monday[i] = float(r_pos_2019_Monday[i])
for i in range(0, len(r_pos_2019_Tuesday)):
    r_pos_2019_Tuesday[i] = float(r_pos_2019_Tuesday[i])
for i in range(0, len(r_pos_2019_Wednesday)):
    r_pos_2019_Wednesday[i] = float(r_pos_2019_Wednesday[i])
for i in range(0, len(r_pos_2019_Thursday)):
    r_pos_2019_Thursday[i] = float(r_pos_2019_Thursday[i])
for i in range(0, len(r_pos_2019_Friday)):
    r_pos_2019_Friday[i] = float(r_pos_2019_Friday[i])
print("2019 Monday Positive Average:",
      (sum(r_pos_2019_Monday)) / len(r_pos_2019_Monday))
variance(r_pos_2019_Monday)
print("2019 Monday Standard Deviation:", stdev(r_pos_2019_Monday))
print("2019 Tuesday Positive Average:",
      (sum(r_pos_2019_Tuesday)) / len(r_pos_2019_Tuesday))
variance(r_pos_2019_Tuesday)
print("2019 Monday Standard Deviation:", stdev(r_pos_2019_Tuesday))
print("2019 Wednesday Positive Average:",
      (sum(r_pos_2019_Wednesday)) / len(r_pos_2019_Wednesday))
variance(r_pos_2019_Wednesday)
print("2019 Monday Standard Deviation:", stdev(r_pos_2019_Wednesday))
print("2019 Thursday Positive Average:",
      (sum(r_pos_2019_Thursday)) / len(r_pos_2019_Thursday))
variance(r_pos_2019_Thursday)
print("2019 Monday Standard Deviation:", stdev(r_pos_2019_Thursday))
print("2019 Friday Positive Average:",
      (sum(r_pos_2019_Friday)) / len(r_pos_2019_Friday))
variance(r_pos_2019_Friday)
print("2019 Monday Standard Deviation:", stdev(r_pos_2019_Friday))

# negative returns
r_neg = []

# access only the return value for each entry and create new list
c = 0
# all negative returns for 2019
# 2019 negative broken into days
r_neg = []
r_neg_2019_Monday = []
r_neg_2019_Tuesday = []
r_neg_2019_Wednesday = []
r_neg_2019_Thursday = []
r_neg_2019_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] == '2019' and ntable[c][13] < '0':
            if ntable[c][4] == 'Monday':
                r_neg_2019_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_neg_2019_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_neg_2019_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_neg_2019_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_neg_2019_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_neg_2019_Monday)):
    r_neg_2019_Monday[i] = float(r_neg_2019_Monday[i])
for i in range(0, len(r_neg_2019_Tuesday)):
    r_neg_2019_Tuesday[i] = float(r_neg_2019_Tuesday[i])
for i in range(0, len(r_neg_2019_Wednesday)):
    r_neg_2019_Wednesday[i] = float(r_neg_2019_Wednesday[i])
for i in range(0, len(r_neg_2019_Thursday)):
    r_neg_2019_Thursday[i] = float(r_neg_2019_Thursday[i])
for i in range(0, len(r_neg_2019_Friday)):
    r_neg_2019_Friday[i] = float(r_neg_2019_Friday[i])
print("2019 Monday Negative Average:",
      (sum(r_neg_2019_Monday)) / len(r_neg_2019_Monday))
variance(r_neg_2019_Monday)
print("2019 Monday Standard Deviation:", stdev(r_neg_2019_Monday))

print("2019 Tuesday Negative Average:",
      (sum(r_neg_2019_Tuesday)) / len(r_neg_2019_Tuesday))
variance(r_neg_2019_Tuesday)
print("2019 Monday Standard Deviation:", stdev(r_neg_2019_Tuesday))

print("2019 Wednesday Negative Average:",
      (sum(r_neg_2019_Wednesday)) / len(r_neg_2019_Wednesday))
variance(r_neg_2019_Wednesday)
print("2019 Monday Standard Deviation:", stdev(r_neg_2019_Wednesday))

print("2019 Thursday Negative Average:",
      (sum(r_neg_2019_Thursday)) / len(r_neg_2019_Thursday))
variance(r_neg_2019_Thursday)
print("2019 Monday Standard Deviation:", stdev(r_neg_2019_Thursday))

print("2019 Friday Negative Average:",
      (sum(r_neg_2019_Friday)) / len(r_neg_2019_Friday))
variance(r_neg_2019_Friday)
print("2019 Monday Standard Deviation:", stdev(r_neg_2019_Friday))