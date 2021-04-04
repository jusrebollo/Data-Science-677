"""
Justin Rebollo
Class: CS 677 - Spring 2
Date: 3/23/2021
Homework Problem #  5 SPY
Description of Problem (just a 1-2 line summary!):
Question 5: Consider ”buy-and-hold” strategy: you buy on the first trading day
and sell on the last day. So you do not listen to your oracle at all. As before,
 assume that you start with $100 for both your stock and ”spy”.
"""

#given code
import os
import math

ticker = 'SPY'
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
    c = 0

# buy 100 worth of stock on day 1
# sell stock on last day
# calculates profit

try:
    stock_number = 100/float(ntable[1][12])
    stock_number = float(stock_number)
    sell = float(ntable[1259][12])
    sold = stock_number*float(ntable[1259][12])
    final_profit = sold - 100
    print("The profit from buy and hold is:", round(final_profit,2))

except IndexError as IE:
    print()