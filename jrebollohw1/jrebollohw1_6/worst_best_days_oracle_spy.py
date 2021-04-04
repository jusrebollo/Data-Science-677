"""
Justin Rebollo
Class: CS 677 - Spring 2
Date: 3/23/2021
Homework Problem # 6 SPY
Description of Problem (just a 1-2 line summary!):Question 6: Your oracle got
very upset that you did not follow its advice. It decided to take revenge by
giving you wrong advice from time to time.
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



closing_price_list = []
return_list = []
money_on_hand_list = [100]
total_shares = []
decision_list = []
years = ['2016' , '2017' , '2018' ,'2019', '2020']


c = 0
try:
    for j in ntable:
        c = c + 1
        if ntable[c][1] in years:
            closing_price_list.append(ntable[c][10])
            return_list.append(ntable[c][13])

except IndexError as IE:
    print()
    [float(i) for i in return_list ]



[float(i) for i in closing_price_list]

# zip close and return
return_and_closing_list = zip(return_list, closing_price_list)
final_return_and_closing_list = list(return_and_closing_list)


# remove top 10 BEST return

raw_top_10 = sorted(final_return_and_closing_list,key=lambda x: x[0], reverse = True)
#ignore scientific notation values
top_10 = raw_top_10[4:14]

missed_top_10 = list([x for x in final_return_and_closing_list if x not in top_10])


# unzip and run strat
unzip= [[ i for i, j in missed_top_10 ],
       [ j for i, j in missed_top_10 ]]

clean_closing_price = unzip[1]
[float(i) for i in clean_closing_price]



#buy on the first day
total_shares = float(money_on_hand_list[0])/float(clean_closing_price[0])

counter = 0
c = 0
try:
    for j in ntable:
        counter = counter +1
        if clean_closing_price[c+1] <= clean_closing_price[c]:
            # sell at closing_price_list[c]
            sell = total_shares * float(clean_closing_price[c])
            # buy at closing_price_list[c+1]
            buy = sell/float(clean_closing_price[c+1])
            # update stock
            total_shares = buy
            # update money
            money_on_hand_list.append(total_shares*float(clean_closing_price[c+1]))
            decision_list.append("Sell")
            c = c+1

        elif clean_closing_price[c + 1] > clean_closing_price[c]:
            decision_list.append("Hold")
            c = c+1
            # update money
            money_on_hand_list.append(float(total_shares)*float(clean_closing_price[c+1]))

except IndexError as IE:
    print()
    print("Total return after missing 10 best days $", round(money_on_hand_list[-1],2))



###############################################################################
#TEN WORST DAYS


[float(i) for i in closing_price_list]

# zip close and return
return_and_closing_list = zip(return_list, closing_price_list)
final_return_and_closing_list = list(return_and_closing_list)


# remove top 10 return

raw_top_10 = sorted(final_return_and_closing_list,key=lambda x: x[0])
top_10 = raw_top_10[0:10]


missed_top_10 = list([x for x in final_return_and_closing_list if x not in top_10])

# unzip and run strat
unzip= [[ i for i, j in missed_top_10 ],
       [ j for i, j in missed_top_10 ]]

clean_closing_price = unzip[1]
[float(i) for i in clean_closing_price]


#buy on the first day
total_shares = float(money_on_hand_list[0])/float(clean_closing_price[0])

counter = 0
c = 0
try:
    for j in ntable:
        counter = counter +1
        if clean_closing_price[c+1] <= clean_closing_price[c]:
            # sell at closing_price_list[c]
            sell = total_shares * float(clean_closing_price[c])
            # buy at closing_price_list[c+1]
            buy = sell/float(clean_closing_price[c+1])
            # update stock
            total_shares = buy
            # update money
            money_on_hand_list.append(total_shares*float(clean_closing_price[c+1]))
            decision_list.append("Sell")
            c = c+1

        elif clean_closing_price[c + 1] > clean_closing_price[c]:
            decision_list.append("Hold")
            c = c+1
            # update money
            money_on_hand_list.append(float(total_shares)*float(clean_closing_price[c+1]))

except IndexError as IE:
    print()
    print("Total return after missing 10 worst days $", round(money_on_hand_list[-1],2))

###############################################################################
#5 BEST and WORST

[float(i) for i in closing_price_list]


# zip close and return
return_and_closing_list = zip(return_list, closing_price_list)
final_return_and_closing_list = list(return_and_closing_list)


# remove top 5 and bottom 5 return

raw_top_10 = sorted(final_return_and_closing_list,key=lambda x: x[0])
worst_best_5 = raw_top_10[5:-5]

missed_5= list([x for x in final_return_and_closing_list if x in worst_best_5])

# unzip and run strat
unzip= [[ i for i, j in missed_5 ],
       [ j for i, j in missed_5 ]]

clean_closing_price = unzip[1]
[float(i) for i in clean_closing_price]


#buy on the first day
total_shares = float(money_on_hand_list[0])/float(clean_closing_price[0])

counter = 0
c = 0
try:
    for j in ntable:
        counter = counter +1
        if clean_closing_price[c+1] <= clean_closing_price[c]:
            # sell at closing_price_list[c]
            sell = total_shares * float(clean_closing_price[c])
            # buy at closing_price_list[c+1]
            buy = sell/float(clean_closing_price[c+1])
            # update stock
            total_shares = buy
            # update money
            money_on_hand_list.append(total_shares*float(clean_closing_price[c+1]))
            decision_list.append("Sell")
            c = c+1

        elif clean_closing_price[c + 1] > clean_closing_price[c]:
            decision_list.append("Hold")
            c = c+1
            # update money
            money_on_hand_list.append(float(total_shares)*float(clean_closing_price[c+1]))

except IndexError as IE:
    print()
    print("Total return after missing best 5 and worst 5 days $", round(money_on_hand_list[-1],2))