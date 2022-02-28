#coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

# Define the lowest value of SplitLen used in MaxDailyDiff

# For 10 - 48
# df = pd.read_csv('Data/DailyDiffData.csv')

# For 10 - 99 I think
df = pd.read_csv('DailyDiffData.csv')

NumCycles = df['Cycles'].tolist()
TotalEarnings = df['Earnings'].tolist()
SplitLengths = df['Length of Split'].tolist()
# Convert SplitLengths into hours
#SpitLengths = SplitLengths / 2

x = 0
while x < len(TotalEarnings):
    # Convert earnings into earnings per battery and
    if NumCycles[x] > 1500:
        TotalEarnings[x] = TotalEarnings[x] * (5000 / NumCycles[x])
        # Calculate how much battery is used each year
        YearlyBatLife = (NumCycles[x] / 5000) * (1 / 3)
        # Calculate the number of years the battery would last to subtract operating cost
        SplitLengths[x] = SplitLengths[x] / 2
    else:
        TotalEarnings[x] = TotalEarnings[x] * (10 / 3)
        # remove cost of battery and operational cost
        SplitLengths[x] = SplitLengths[x] / 2
    x += 1

MaxEarning = str(round(max(TotalEarnings)))
print(MaxEarning)
MaxPos = TotalEarnings.index(max(TotalEarnings))
print(SplitLengths[MaxPos])
print(NumCycles[MaxPos])


plt.plot(SplitLengths, TotalEarnings, 'b-')
plt.ylabel('Predicted earnings per battery [£]')
plt.xlabel('Length of time period [hours]')
axes = plt.gca()
axes.yaxis.label.set_size(17)
axes.xaxis.label.set_size(17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
axes.set_ylim([100000, 450000])
axes.set_xlim([0, 50])
plt.grid()
plt.show()
