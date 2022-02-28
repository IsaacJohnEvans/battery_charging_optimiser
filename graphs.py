# coding : utf8
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime as dt

df_half = pd.read_excel('Market Data.xlsx', sheet_name='Half-hourly data')
df_daily = pd.read_excel('Market Data.xlsx', sheet_name='Daily data')

# Extract the values of each market and the time
Market_1 = df_half.loc[:, 'Market 1 Price [£/MWh]']
Market_2 = df_half.loc[:, 'Market 2 Price [£/MWh]']
Market_3_small = df_daily.loc[:, 'Market 3 Price [£/MWh]']
Time = df_half.loc[:, 'Time']


# Extract the values of the generation
Coal_Gen = df_half.loc[:, 'Coal Generation [MW]']
Gas_Gen = df_half.loc[:, 'Gas Generation [MW]']
Wind_Gen = df_half.loc[:, 'Wind Generation [MW]']
Sol_Gen = df_half.loc[:, 'Solar Generation [MW]']
Demand = df_half.loc[:, 'Transmission System Electricity Demand [MW]']


# Make the data for market 3 equally sized
Market_3 = []
for n in Market_3_small:
    i = 0
    while i < 48:
        Market_3.append(n)
        i += 1

# Make a graph of the market prices
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=70))
# plt.plot(Time, Market_1, linewidth=0.5, label='Market 1')
# plt.plot(Time, Market_2, linewidth=0.5, label='Market 2')
# plt.plot(Time, Market_3, linewidth=0.5, label='Market 3')
# plt.title('Graph showing the price of each market over a 3 year period (2018-2020)')
# plt.xlabel('Time')
# plt.ylabel('Market Price [£/MWh]')
# plt.legend()
# plt.grid()
# plt.gcf().autofmt_xdate()
# plt.show()


# Make a graph of the generation data
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=70))
plt.plot(Time, Coal_Gen, linewidth=0.5, label='Coal Generation')
plt.plot(Time, Sol_Gen, linewidth=0.5, label='Solar Generation')
plt.plot(Time, Gas_Gen, linewidth=0.5, label='Gas Generation')
plt.plot(Time, Wind_Gen, linewidth=0.5, label='Wind Generation')
plt.title('Graph showing the price of each market over a 3 year period (2018-2020)')
plt.xlabel('Time')
plt.ylabel('Power Generation [MW]')
plt.legend()
plt.grid()
plt.gcf().autofmt_xdate()
plt.show()





