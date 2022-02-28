#coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import statsmodels.api as sm
import pylab
import scipy.stats as stats



df = pd.read_csv('Data/RandomData.csv')
NumCycles = df['Cycles']
TotalEarnings = df['Earnings']
AverageCycles = sum(NumCycles) / len(NumCycles)
MeanEarning = sum(TotalEarnings) / len(TotalEarnings)
SDEarning = statistics.stdev(TotalEarnings)
print(SDEarning)
print(MeanEarning)

plt.hist(TotalEarnings, bins=90, facecolor='r', linewidth=0.5, rwidth=0.92, alpha=0.75)
plt.xlabel('Earnings [£]')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
axes = plt.gca()
axes.yaxis.label.set_size(17)
axes.xaxis.label.set_size(17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


TotalEarnings = TotalEarnings.values
n = 100
values = np.linspace(0, len(TotalEarnings), len(TotalEarnings) + 1)
index = np.random.choice(values, n, replace=False)
index = index.astype(int)
RandTotalEarnings = TotalEarnings[index]

sm.qqplot(RandTotalEarnings, loc=MeanEarning, scale=SDEarning,  line='45')
axes = plt.gca()
axes.yaxis.label.set_size(14)
axes.xaxis.label.set_size(14)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
