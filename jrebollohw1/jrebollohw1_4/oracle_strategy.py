"""
Justin Rebollo
Class: CS 677 - Spring 2
Date: 3/23/2021
Homework Problem #  4 My stock
Description of Problem (just a 1-2 line summary!):Question 4: You listen to the
 oracle and follow its advice. How much much money will you have on the last
 trading day of 2019:
"""

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
years = ['2016' , '2017' , '2018' ,'2019', '2020']


# load all data for closing
c = 0
try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] in years:
            closing_price_list.append(ntable[c][10])

except IndexError as IE:
    print()

[float(i) for i in closing_price_list]


#buy on the first day
total_shares = float(money_on_hand_list[0])/float(closing_price_list[0])

counter = 0
c = 0
try:
    for j in ntable:
        counter = counter +1
        if closing_price_list[c+1] <= closing_price_list[c]:
            # sell at closing_price_list[c]
            sell = total_shares * float(closing_price_list[c])
            # buy at closing_price_list[c+1]
            buy = sell/float(closing_price_list[c+1])
            # update stock
            total_shares = buy
            # update money
            money_on_hand_list.append(total_shares*float(closing_price_list[c+1]))
            decision_list.append("Sell")
            c = c+1

        elif closing_price_list[c + 1] > closing_price_list[c]:
            decision_list.append("Hold")
            c = c+1
            # update money
            money_on_hand_list.append(float(total_shares)*float(closing_price_list[c+1]))

except IndexError as IE:

    print("Total Return $", round(money_on_hand_list[-1],2))



