#coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


'''
Find the maximum difference between the price within a given time period 
This can then be used for time periods over the entire dataset
'''
def MaxDiff(Market1Sec, Market2Sec, SplitLen):
    # define MaxDiff very negative so it is always redefined
    MaxDiff = -99999
    x = 0
    while x < len(Market1Sec) - 4:
        # print(x)
        y = 0
        # Find the minimum price at time step x
        if Market1Sec[x] < Market2Sec[x]:
            MinPrice = Market1Sec[x]
            BuyMarket = 1
        else:
            MinPrice = Market2Sec[x]
            BuyMarket = 2
        for y in range(x + 1, len(Market1Sec) - 1):
            if Market1Sec[y] > Market2Sec[y]:
                MaxPrice = Market1Sec[y]
                SellMarket = 1
            else:
                MaxPrice = Market2Sec[y]
                SellMarket = 2
            Diff = MaxPrice - MinPrice
            if Diff > MaxDiff:
                MaxDiff = Diff
                BestXValue = x
                BestYValue = y
                BestBuyMarket = BuyMarket
                BestSellMarket = SellMarket
                # print('New MaxDiff = ' + str(Diff))
                # print(BestXValue)
            y += 1
        x += 1
    # Add the best value of x to account for the shortening of the vector
    # print('Final BestXValue = ' + str(BestXValue))
    return BestXValue, BestYValue, BestBuyMarket, BestSellMarket


def CalcEarnings(Market1Sec, Market2Sec, SplitLen):
    BestX, BestY, BuyMarket, SellMarket = MaxDiff(Market1Sec, Market2Sec, SplitLen)
    # print('BuyMarket = ' + str(BuyMarket))
    # Ensure the X and Y value chosen are within the range of the current market section
    if BestY == len(Market1Sec) - 2:
        # print('Y chosen 1 too high')
        BestY -= 1
    if BestY == len(Market1Sec) - 1:
        # print('Y chosen 2 too high')
        BestY -= 2
    if BestX == len(Market1Sec) - 2:
        # print('X chosen 1 too high')
        BestX -= 1
    if BestX == len(Market1Sec) - 1:
        # print('X chosen 2 too high')
        BestX -= 2
    if BuyMarket == 1:
        TotalBuyCost = Market1Sec[BestX-1]+Market1Sec[BestX]+Market1Sec[BestX+1]+Market1Sec[BestX+2]
    else:
        TotalBuyCost = Market2Sec[BestX-1]+Market2Sec[BestX]+Market2Sec[BestX+1]+Market2Sec[BestX+2]
    if SellMarket == 1:
        TotalSellCost = Market1Sec[BestY-1]+Market1Sec[BestY]+Market1Sec[BestY+1]+Market1Sec[BestY+2]
    else:
        TotalSellCost = Market2Sec[BestY-1]+Market2Sec[BestY] + Market2Sec[BestY+1]+Market2Sec[BestY+2]

    Earnings = (0.95**2)*(TotalSellCost) - TotalBuyCost
    return Earnings


df = pd.read_excel('Market Data.xlsx')
Market1, Market2 = df['Market 2 Price [£/MWh]'], df['Market 2 Price [£/MWh]']
Market1 = pd.Series.tolist(Market1)
Market2 = pd.Series.tolist(Market2)
AllEarnings, AllSplitLens, AllNumCycles = [], [], []
SplitLen = 10

while SplitLen < 100:
    AllSplitLens.append(SplitLen)
    TotalEarnings = 0
    SplitNum = 0
    NumCycles = 0
    NumSplits = math.floor(len(Market1) / SplitLen)
    if NumCycles == 0:
        MaxStorage = 4
    while SplitNum < NumSplits:
        MinBound = (SplitLen * SplitNum)
        MaxBound = (SplitLen - 1) + MinBound
        Market1Sec = Market1[MinBound:MaxBound]
        Market2Sec = Market2[MinBound:MaxBound]
        # print('Min Bound is ' + str(MinBound))
        # print('Max Bound is ' + str(MaxBound))
        Earnings = CalcEarnings(Market1Sec, Market2Sec, SplitLen)
        if Earnings > 0:
            # Only buy if it is a profitable trade
            TotalEarnings += Earnings
            NumCycles += (3.8 / MaxStorage)
            # Reduce max storage each charge/discharge
            MaxStorage = MaxStorage - ((0.00001) * (3.8 / MaxStorage))
        #else:
            #print('Time period ' + str(SplitNum + 1) + ' made a loss')
        SplitNum += 1
    AllNumCycles.append(NumCycles)
    AllEarnings.append(TotalEarnings)
    SplitLen += 1

print(AllNumCycles)
print(AllSplitLens)
print(AllEarnings)

dict = {'Earnings':AllEarnings, 'Cycles':AllNumCycles, 'Length of Split':AllSplitLens}
df2 = pd.DataFrame(dict)
df2.to_csv('DailyDiffData.csv')
