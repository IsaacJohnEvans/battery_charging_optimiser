#coding:utf8
#%%
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BatteryClass import Battery

'''
Baseline function 
Buys electricity when it is cheaper than the mean cost
Sells electricity when it is more expensive than the mean cost
'''

def BuySellMean(cell, market_price, mean_cost, charge_time, x, y):
    min_sell = mean_cost + x
    max_sell = mean_cost + x + y
    min_buy = mean_cost - x
    max_buy = mean_cost - x - y
    charging_rate = cell.MaxChargingRate
    if market_price < min_buy:
        if market_price < max_buy:
            charging_rate = cell.MaxChargingRate
        else:
            charging_rate = cell.MaxChargingRate * (mean_cost - x - market_price)/y
    elif market_price > min_sell:
        if market_price > max_sell:
            charging_rate = -cell.MaxChargingRate
        else:
            charging_rate = cell.MaxChargingRate * (mean_cost + x - market_price)/y
    if (0 < cell.Charge + cell.ChargingEfficiency * charging_rate * charge_time  < cell.MaxStorage * (1 - cell.MaxStorageLoss * abs(cell.ChargingEfficiency * charging_rate * charge_time * 0.5))):
        cell.Charging(charging_rate, market_price)
    else:
        cell.UpdateBattery(0)

#%%
# max rows = 52608
nrows = 52608
df = pd.read_excel('Market Data.xlsx', nrows= nrows)
mean_cost_1 =  df['Market 1 Price [£/MWh]'].mean()
mean_cost_2 = df['Market 2 Price [£/MWh]'].mean()

charge_time = 0.5
bank_data = np.zeros((10, 10))
charge_data = np.zeros((10, 10))
up_time = []

prices = df[['Market 1 Price [£/MWh]','Market 2 Price [£/MWh]']].to_numpy()
best_price = np.zeros(nrows)
mean_cost = np.mean(prices)
# mean_cost - prices[k, :].min() > prices[k, :].max() - mean_cost
for i in range(nrows):
    if mean_cost - prices[i, :].min() > prices[i, :].max() - mean_cost:
        best_price[i] = prices[i, :].min()
    else:
        best_price[i] = prices[i, :].max()

#%%

distance_from_mean = 0
linear_scale_length = 20
memory_data = 47
cell = Battery()
for i in range(nrows):
    BuySellMean(cell, best_price[i], mean_cost, charge_time, distance_from_mean, linear_scale_length)
cell.PrintAttributes()
if cell.Cycles / cell.MaxCycles > cell.UpTime / cell.Lifetime:
    total_revenue = cell.Bank/cell.Cycles * cell.MaxCycles
else:
    total_revenue = cell.Bank/cell.UpTime * cell.Lifetime
print(total_revenue)
#%%
weight_value = 66
distance_from_mean = 0
linear_scale_length = 20
memory_data = 42
'''
weight_value = 66
distance_from_mean = 0
linear_scale_length = 20
memory_data = 43

2 MaxChargingRate 
 3.804560935528812 MaxStorage 
 0.95 ChargingEfficiency 
 87600 Lifetime 
 5000 MaxCycles 
 1e-05 MaxStorageLoss 
 0.015749115638763467 Charge 
 1284.238067676104 Cycles 
 26304.0 Uptime 
 118757.35657890589 Bank 

395496.6710885096
'''
cell = Battery()
rolled_best_price = np.concatenate((np.zeros(memory_data), best_price))
weights = np.linspace(0, memory_data, memory_data, endpoint=False)
weights = np.zeros(memory_data)
for i in range(memory_data):
    weights[-i] = weight_value** (i/10)
print(weights)
weights = np.ones(memory_data)
sr = 1000
X = df.to_numpy()
X = np.fft.fft(X[:, 1])
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 
weights = np.abs(X[:memory_data])
#weights[1:3] = 1000
for i in range(nrows):
    rolling_mean = np.average(rolled_best_price[i: i + memory_data], weights = weights)
    BuySellMean(cell, rolled_best_price[i + memory_data], rolling_mean, charge_time, distance_from_mean, linear_scale_length)
cell.PrintAttributes()
if cell.Cycles / cell.MaxCycles > cell.UpTime / cell.Lifetime:
    total_revenue = cell.Bank/cell.Cycles * cell.MaxCycles
else:
    total_revenue = cell.Bank/cell.UpTime * cell.Lifetime
print(total_revenue)

#%%
for weight_value in range(8, 100):
    for distance_from_mean in range(0,1):
        for linear_scale_length in range(20, 21, 1):
            for memory_data in range(43, 44, 1):
                weights = np.zeros(memory_data)
                for i in range(memory_data):
                    weights[-i] = weight_value** (i/10)
                cell = Battery()
                market_price = np.concatenate((np.zeros(memory_data), best_price))
                for i in range(nrows):
                    rolling_mean = np.average(market_price[i: i + memory_data], weights= weights)
                    BuySellMean(cell, market_price[i + memory_data], rolling_mean, charge_time, distance_from_mean, linear_scale_length)
                #bank_data[i, j//5] = (cell.Bank)
                #charge_data[i, j//5] = (cell.Charge)
                up_time.append(cell.UpTime)
                if cell.Cycles / cell.MaxCycles > cell.UpTime / cell.Lifetime:
                    total_revenue = cell.Bank/cell.Cycles * cell.MaxCycles
                else:
                    total_revenue = cell.Bank/cell.UpTime * cell.Lifetime
                print('weight value: ', weight_value,
                      '\ndistance from mean', distance_from_mean, 
                      '\n linear scale length', linear_scale_length, 
                      '\nmemory_for_mean', memory_data, 
                      '\nTotal revenue: ', total_revenue)

#%%
'''
#print(np.max(bank_data))

distance_from_mean = 0
linear_scale_length = 21
memory_for_mean = 46

'''
# i = 3 j = 20 is best for revenue of 218593
# i = 0 j = 20 z = 48 is best at 335187
# i = 0 j = 22 z = 45 is best at 336920.18

#%%
plt.figure('BankData')
plt.plot(Time, BankTotal, 'r')
plt.show()

bank_data.tofile('BankData.txt', ',')


