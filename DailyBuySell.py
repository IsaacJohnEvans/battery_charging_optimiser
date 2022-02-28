#coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def DailyMinimum(TotalMarketData1, TotalMarketData2, x, Range):  # Find the minimum value
    MarketUsed = 0
    z = 0
    ChargeRange = 0
    MinBound = Range * x
    MaxBound = (Range - 1) + MinBound
    MarketData1 = TotalMarketData1[MinBound:MaxBound]
    MarketData1 = pd.Series.tolist(MarketData1)
    MarketData2 = TotalMarketData2[MinBound:MaxBound]
    MarketData2 = pd.Series.tolist(MarketData2)
    MinValue1 = min(MarketData1)
    MinValue2 = min(MarketData2)
    if MinValue1 > MinValue2:
        MinValue = MinValue2
        MarketData = MarketData2
        MarketUsed = 2
    else:
        MinValue = MinValue1
        MarketData = MarketData1
        MarketUsed = 1
    MinIndex = MarketData.index(MinValue)
    LowIndex = MinIndex - 1
    HighIndex = MinIndex + 1
    if LowIndex < MinBound:
        LowIndex = HighIndex
    if HighIndex > MaxBound:
        HighIndex = LowIndex
    if LowIndex == HighIndex & LowIndex == (Range - 1):
        z = 1
        return z, ChargeRange, MarketUsed
    LocalMax = max(MarketData[LowIndex], MarketData[HighIndex])
    LocalIndex = MarketData.index(LocalMax)
    if LocalIndex == LowIndex:
        ChargeRange = [MinIndex + MinBound, HighIndex + MinBound]
    if LocalIndex == HighIndex:
        ChargeRange = [LowIndex + MinBound, MinIndex + MinBound]
    return z, ChargeRange, MarketUsed


def DailyMaximum(TotalMarketData1, TotalMarketData2, ChargeRange, x, Range):  # Find the maximum value
    z = 0
    MarketUsed = 0
    DischargeRange = 0
    MinBound = Range * x
    MaxBound = (Range -1) + MinBound
    MarketData1 = TotalMarketData1[max(ChargeRange):MaxBound]
    MarketData1 = pd.Series.tolist(MarketData1)
    MarketData2 = TotalMarketData2[MinBound:MaxBound]
    MarketData2 = pd.Series.tolist(MarketData2)
    MaxValue1 = max(MarketData1)
    MaxValue2 = max(MarketData2)
    if MaxValue1 < MaxValue2:
        MaxValue = MaxValue2
        MarketData = MarketData2
        MarketUsed = 2
    else:
        MaxValue = MaxValue1
        MarketData = MarketData1
        MarketUsed = 1
    MaxIndex = MarketData.index(MaxValue)
    LowIndex = MaxIndex - 1
    HighIndex = MaxIndex + 1
    if LowIndex < 0:
        LowIndex = HighIndex
    if HighIndex > MaxBound:
        HighIndex = LowIndex
    if LowIndex == HighIndex & LowIndex == 1:
        z = 1
        return z, DischargeRange, MarketUsed
    if HighIndex == len(MarketData):
        z = 1
        return z, DischargeRange, MarketUsed
    LocalMin = min(MarketData[LowIndex], MarketData[HighIndex])
    LocalIndex = MarketData.index(LocalMin)
    if LocalIndex == LowIndex:
        DischargeRange = [MaxIndex + max(ChargeRange), HighIndex + max(ChargeRange)]
    if LocalIndex == HighIndex:
        DischargeRange = [LowIndex + max(ChargeRange), MaxIndex + max(ChargeRange)]

    return z, DischargeRange, MarketUsed


def DailyBuySellRun(TotalMarketData1, TotalMarketData2, x, NumCycles, Range):
    MinBound = Range * x
    MaxBound = (Range - 1) + MinBound
    z, ChargeRange, MarketUsedBuy = DailyMinimum(TotalMarketData1, TotalMarketData2, x, Range)
    if z == 1 or ChargeRange == 0:
        DailyProfit = 0
        return DailyProfit, NumCycles
    z, DischargeRange, MarketUsedSell = DailyMaximum(TotalMarketData1, TotalMarketData2, ChargeRange, x, Range)
    if z == 1 or DischargeRange == 0:
        DailyProfit = 0
        return DailyProfit, NumCycles
    if MarketUsedSell == 1:
        MarketDataSell = TotalMarketData1
    else:
        MarketDataSell = TotalMarketData2
    if MarketUsedBuy == 1:
        MarketDataBuy = TotalMarketData1
    else:
        MarketDataBuy = TotalMarketData2
    DailyProfit = (MarketDataSell[DischargeRange[0]] + MarketDataSell[DischargeRange[1]])*(0.95**2) - MarketDataBuy[ChargeRange[0]] - MarketDataBuy[ChargeRange[1]]
    NumCycles += 0.5
    return DailyProfit, NumCycles


DailyProfits, x, z, NumCycles = 0, 0, 0, 0
df = pd.read_excel('Market Data.xlsx')
TotalMarketData1 = df['Market 2 Price [£/MWh]']
TotalMarketData2 = df['Market 2 Price [£/MWh]']
AllDailyProfits, AllNumCycles = [], []
Range = 20

while Range < 100:
    DailyProfits = 0
    x = 0
    NumCycles = 0
    # Need to ensure we have an integer number of days
    IntNumDays = math.floor(len(df['Market 1 Price [£/MWh]']) / Range)
    if IntNumDays > 49:
        IntNumDays -= 1
    while x < IntNumDays:
        DailyProfit, TotalCycles = DailyBuySellRun(TotalMarketData1, TotalMarketData2, x, NumCycles, Range)
        if DailyProfit > 0:
            DailyProfits += DailyProfit
            NumCycles = TotalCycles
        x += 1
    AllDailyProfits.append(DailyProfits)
    AllNumCycles.append(TotalCycles)
    Range += 1


MaxDailyProfit = max(AllDailyProfits)
MaxDailyProfitIndex = AllDailyProfits.index(MaxDailyProfit)
print(MaxDailyProfit)

