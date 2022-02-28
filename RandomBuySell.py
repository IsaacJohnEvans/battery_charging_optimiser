#coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BatteryClass import Battery
from random import choices

'''
Baseline function 
Buys and sells electricity randomly
'''

def RandomBuySell(Cell, MarketPrice, ChargeTime, ChargingRate):
    if ChargingRate == 2 and Cell.Charge + Cell.ChargingEfficiency * ChargingRate * ChargeTime  < Cell.MaxStorage * (1 - Cell.MaxStorageLoss * abs(Cell.ChargingEfficiency * ChargingRate * ChargeTime * 0.5)):
        Cell.Charging(ChargingRate, MarketPrice)
    elif ChargingRate == -2 and Cell.Charge + Cell.ChargingEfficiency * ChargingRate * ChargeTime >= 0:
        Cell.Charging(ChargingRate, MarketPrice)
    else:
        Cell.UpdateBattery(0)

def RandomBuySellRun(OptionList, NumChoices):
    BankTotal = []
    Cell = Battery()
    BankData = []
    ChargeData = []
    UpTime = []
    AllChargingRates = choices(OptionList, k=NumChoices)
    for i, Price in enumerate(df['Market 2 Price [£/MWh]']):
        ChargingRate = AllChargingRates[i]
        RandomBuySell(Cell, Price, 0.5, ChargingRate)
        BankData.append(Cell.Bank)
        ChargeData.append(Cell.Charge)
        UpTime.append(Cell.UpTime)
    BankTotal.append(Cell.Bank)
    return BankTotal, Cell.Cycles

df = pd.read_excel('Market Data.xlsx')
N = 5
NList = []
AllMeanBank = []
OptionList = [0] * N
OptionList.append(-2)
OptionList.append(2)
NumChoices = len(df['Market 2 Price [£/MWh]'])

while N < 6:
    NList.append(N / (N + 2))  # Add the probability of holding
    # Create the list of charging options
    i = 0
    BankTotal = []
    TotalCycles = []
    while i < 10000:  # Define the number of runs at each probability
        BankAmount, NumCycles = RandomBuySellRun(OptionList, NumChoices)
        BankAmount = BankAmount[0]
        BankTotal.append(BankAmount)
        TotalCycles.append(NumCycles)
        i += 1
    # Calculate the average profit each cycle for each probability
    if NumCycles < 100:
        MeanBank = (sum(BankTotal) / len(BankTotal)) * (10 / 3)
    else:
        MeanBank = ((sum(BankTotal) / len(BankTotal)) / NumCycles) * 5000
    AllMeanBank.append(MeanBank)
    OptionList.append(0)
    N += 1


dict = {'Earnings': BankTotal, 'Cycles': TotalCycles}
df2 = pd.DataFrame(dict)
df2.to_csv('RandomData.csv')


# plt.plot(NList, AllMeanBank, label='Average Earnings')
# plt.title('Graph showing the performance of a model which randomly buys and sells energy as the probability of holding changes')
# plt.xlabel('Probability of Holding [%]')
# plt.ylabel('Average Earnings per cycle [£]')
# plt.legend()
# plt.grid()
# plt.show()
