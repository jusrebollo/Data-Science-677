"""
Justin Rebollo
Class: CS 677 - Spring 2
Date: 3/23/2021
Homework Problem #  3 My stock
Description of Problem (just a 1-2 line summary!): Compute the aggregate table
 across all 5 years, one table for both your stock and one table for S&P-500
 (using data for ”spy”).
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

    """    your code for assignment 1 goes here
    """


except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)


# for standard deviation calculation
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


# make the list of lines usable by making 2D list

separator = ','
ntable = []
for row in lines:
    if (separator in row):
        row = row.rstrip()
        row = row.split(separator)
        ntable += [row]

###five_year
# access only the return value for each entry and create new list
c = 0
# all returns for five_year
# five_year broken into days
r_total = []
r_total_five_year_Monday = []
r_total_five_year_Tuesday = []
r_total_five_year_Wednesday = []
r_total_five_year_Thursday = []
r_total_five_year_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] != '2021':
            if ntable[c][4] == 'Monday':
                r_total_five_year_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_total_five_year_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_total_five_year_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_total_five_year_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_total_five_year_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_total_five_year_Monday)):
    r_total_five_year_Monday[i] = float(r_total_five_year_Monday[i])
for i in range(0, len(r_total_five_year_Tuesday)):
    r_total_five_year_Tuesday[i] = float(r_total_five_year_Tuesday[i])
for i in range(0, len(r_total_five_year_Wednesday)):
    r_total_five_year_Wednesday[i] = float(r_total_five_year_Wednesday[i])
for i in range(0, len(r_total_five_year_Thursday)):
    r_total_five_year_Thursday[i] = float(r_total_five_year_Thursday[i])
for i in range(0, len(r_total_five_year_Friday)):
    r_total_five_year_Friday[i] = float(r_total_five_year_Friday[i])

print("five_year Monday Average:",
      (sum(r_total_five_year_Monday)) / len(r_total_five_year_Monday))
variance(r_total_five_year_Monday)
print("five_year Monday Standard Deviation:", stdev(r_total_five_year_Monday))
print("five_year Tuesday Average:",
      (sum(r_total_five_year_Tuesday)) / len(r_total_five_year_Tuesday))
variance(r_total_five_year_Tuesday)
print("five_year Monday Standard Deviation:", stdev(r_total_five_year_Tuesday))
print("five_year Wednesday Average:",
      (sum(r_total_five_year_Wednesday)) / len(r_total_five_year_Wednesday))
variance(r_total_five_year_Wednesday)
print("five_year Monday Standard Deviation:", stdev(r_total_five_year_Wednesday))
print("five_year Thursday Average:",
      (sum(r_total_five_year_Thursday)) / len(r_total_five_year_Thursday))
variance(r_total_five_year_Thursday)
print("five_year Monday Standard Deviation:", stdev(r_total_five_year_Thursday))
print("five_year Friday Average:",
      (sum(r_total_five_year_Friday)) / len(r_total_five_year_Friday))
variance(r_total_five_year_Friday)
print("five_year Monday Standard Deviation:", stdev(r_total_five_year_Friday))

# positive returns
# access only the return value for each entry and create new list
c = 0
# all positive returns for five_year
# five_year positive broken into days
r_pos = []
r_pos_five_year_Monday = []
r_pos_five_year_Tuesday = []
r_pos_five_year_Wednesday = []
r_pos_five_year_Thursday = []
r_pos_five_year_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] != '2021' and ntable[c][13] > '0':
            if ntable[c][4] == 'Monday':
                r_pos_five_year_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_pos_five_year_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_pos_five_year_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_pos_five_year_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_pos_five_year_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_pos_five_year_Monday)):
    r_pos_five_year_Monday[i] = float(r_pos_five_year_Monday[i])
for i in range(0, len(r_pos_five_year_Tuesday)):
    r_pos_five_year_Tuesday[i] = float(r_pos_five_year_Tuesday[i])
for i in range(0, len(r_pos_five_year_Wednesday)):
    r_pos_five_year_Wednesday[i] = float(r_pos_five_year_Wednesday[i])
for i in range(0, len(r_pos_five_year_Thursday)):
    r_pos_five_year_Thursday[i] = float(r_pos_five_year_Thursday[i])
for i in range(0, len(r_pos_five_year_Friday)):
    r_pos_five_year_Friday[i] = float(r_pos_five_year_Friday[i])
print(len(r_pos_five_year_Monday + r_pos_five_year_Tuesday + r_pos_five_year_Wednesday
          +r_pos_five_year_Thursday+ r_pos_five_year_Friday))
print("five_year Monday Positive Average:",
      (sum(r_pos_five_year_Monday)) / len(r_pos_five_year_Monday))
variance(r_pos_five_year_Monday)
print("five_year Monday Standard Deviation:", stdev(r_pos_five_year_Monday))
print("five_year Tuesday Positive Average:",
      (sum(r_pos_five_year_Tuesday)) / len(r_pos_five_year_Tuesday))
variance(r_pos_five_year_Tuesday)
print("five_year Monday Standard Deviation:", stdev(r_pos_five_year_Tuesday))
print("five_year Wednesday Positive Average:",
      (sum(r_pos_five_year_Wednesday)) / len(r_pos_five_year_Wednesday))
variance(r_pos_five_year_Wednesday)
print("five_year Monday Standard Deviation:", stdev(r_pos_five_year_Wednesday))
print("five_year Thursday Positive Average:",
      (sum(r_pos_five_year_Thursday)) / len(r_pos_five_year_Thursday))
variance(r_pos_five_year_Thursday)
print("five_year Monday Standard Deviation:", stdev(r_pos_five_year_Thursday))
print("five_year Friday Positive Average:",
      (sum(r_pos_five_year_Friday)) / len(r_pos_five_year_Friday))
variance(r_pos_five_year_Friday)
print("five_year Monday Standard Deviation:", stdev(r_pos_five_year_Friday))

# negative returns
r_neg = []

# access only the return value for each entry and create new list
c = 0
# all negative returns for five_year
# five_year negative broken into days
r_neg = []
r_neg_five_year_Monday = []
r_neg_five_year_Tuesday = []
r_neg_five_year_Wednesday = []
r_neg_five_year_Thursday = []
r_neg_five_year_Friday = []

try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] != '2021' and ntable[c][13] < '0':
            if ntable[c][4] == 'Monday':
                r_neg_five_year_Monday.append(ntable[c][13])
            if ntable[c][4] == 'Tuesday':
                r_neg_five_year_Tuesday.append(ntable[c][13])
            if ntable[c][4] == 'Wednesday':
                r_neg_five_year_Wednesday.append(ntable[c][13])
            if ntable[c][4] == 'Thursday':
                r_neg_five_year_Thursday.append(ntable[c][13])
            if ntable[c][4] == 'Friday':
                r_neg_five_year_Friday.append(ntable[c][13])
except IndexError as IE:
    print()

for i in range(0, len(r_neg_five_year_Monday)):
    r_neg_five_year_Monday[i] = float(r_neg_five_year_Monday[i])
for i in range(0, len(r_neg_five_year_Tuesday)):
    r_neg_five_year_Tuesday[i] = float(r_neg_five_year_Tuesday[i])
for i in range(0, len(r_neg_five_year_Wednesday)):
    r_neg_five_year_Wednesday[i] = float(r_neg_five_year_Wednesday[i])
for i in range(0, len(r_neg_five_year_Thursday)):
    r_neg_five_year_Thursday[i] = float(r_neg_five_year_Thursday[i])
for i in range(0, len(r_neg_five_year_Friday)):
    r_neg_five_year_Friday[i] = float(r_neg_five_year_Friday[i])
print(len(r_neg_five_year_Monday + r_neg_five_year_Tuesday + r_neg_five_year_Wednesday
          +r_neg_five_year_Thursday+ r_neg_five_year_Friday))
print("five_year Monday Negative Average:",
      (sum(r_neg_five_year_Monday)) / len(r_neg_five_year_Monday))
variance(r_neg_five_year_Monday)
print("five_year Monday Standard Deviation:", stdev(r_neg_five_year_Monday))

print("five_year Tuesday Negative Average:",
      (sum(r_neg_five_year_Tuesday)) / len(r_neg_five_year_Tuesday))
variance(r_neg_five_year_Tuesday)
print("five_year Monday Standard Deviation:", stdev(r_neg_five_year_Tuesday))

print("five_year Wednesday Negative Average:",
      (sum(r_neg_five_year_Wednesday)) / len(r_neg_five_year_Wednesday))
variance(r_neg_five_year_Wednesday)
print("five_year Monday Standard Deviation:", stdev(r_neg_five_year_Wednesday))

print("five_year Thursday Negative Average:",
      (sum(r_neg_five_year_Thursday)) / len(r_neg_five_year_Thursday))
variance(r_neg_five_year_Thursday)
print("five_year Monday Standard Deviation:", stdev(r_neg_five_year_Thursday))

print("five_year Friday Negative Average:",
      (sum(r_neg_five_year_Friday)) / len(r_neg_five_year_Friday))
variance(r_neg_five_year_Friday)
print("five_year Monday Standard Deviation:", stdev(r_neg_five_year_Friday))


