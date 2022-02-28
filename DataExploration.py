#%%
# coding : utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def LoadData(NRows):
    '''
    A function to load in the Market Data 
    '''
    df = pd.read_excel('Market Data.xlsx', nrows=NRows)
    df['Total Generation'] = df['Wind Generation [MW]'] + df['Solar Generation [MW]'] + df['Coal Generation [MW]'] + df['Gas Generation [MW]'] 
    df['Other Generation [MW]'] = df['Transmission System Electricity Demand [MW]'] - df['Total Generation']
    return df

def Corr(Save = True):
    df_corr = df.corr().round(decimals=3)
    if Save:
        (df_corr).to_csv('Corr.csv')
    return df_corr

def SmoothData(GroupSize, SmoothingFactor):
    dfSmooth = df.groupby(np.arange(len(df))//(GroupSize * SmoothingFactor)).mean()
    dfSmooth['Time'] = df['Time'].groupby(np.arange(len(df))//GroupSize).mean()
    return dfSmooth

#%%
MaxChargingRate = 2
MaxStorage = 4
ChargingEfficiency = 0.95
Lifetime = 10
MaxCycles = 5000
MaxStorageLoss = 0.00001
nrows = 52608

df = LoadData(nrows)

#%%
group_size = 48
df_var = pd.DataFrame(np.zeros((nrows//48, 2)), columns= ['Time', 'Price'])
df_var['Var'] = df['Market 1 Price [£/MWh]'].rolling(group_size).var().groupby(np.arange(len(df))//(group_size)).mean()
print(np.max(df_var.to_numpy()))
df_var['Time'] = df['Time'].groupby(np.arange(len(df))//group_size).mean()
df_var.to_csv('rolling_var.csv')
#%%
df_var['Var'].plot.hist(bins = 50)

#%%


sr = 2000
X = df.to_numpy()
X = np.fft.fft(X[:, 1])
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 


plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency [Hz]', fontsize=17)
plt.ylabel('Amplitude', fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim(0, 50)
plt.ylim(0, 250000)
#%%
df.plot(x = 'Time', y = {'Transmission System Electricity Demand [MW]', 'Total Generation'}, kind = 'line')
dfSmooth = SmoothData(48, 1)
dfSmooth.plot(x = 'Time', y = {'Transmission System Electricity Demand [MW]', 'Total Generation', 'Other Generation [MW]'}, kind = 'line')
#
df.plot(x = 'Time', y = {'Market 1 Price [£/MWh]', 'Market 2 Price [£/MWh]'})
print(df.mean())
print(df['Market 1 Price [£/MWh]'].loc[df['Market 1 Price [£/MWh]'] > df['Market 1 Price [£/MWh]'].mean()])
print(df['Market 1 Price [£/MWh]'].shape)

plt.show()


# %%
