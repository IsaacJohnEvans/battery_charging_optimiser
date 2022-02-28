#%%
import numpy as np 
import pandas as pd
from BatteryClass import Battery
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import matplotlib.pyplot as plt
from run_battey import run_battery, total_revenue, charge_battery
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

def create_model(layers, input_shape):
    # Create and compile model
    model = keras.Sequential(layers)
    model.build(input_shape=input_shape)
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    return model

def optimize_charging(M, size=48):
    x = np.zeros_like(M, dtype='float')

    def func(dec):
        return sum(dec * M)

    bnds = [[-max_charge_rate*discharge_efficiency, max_charge_rate*charge_efficiency] for _ in x]

    # define battery capacity constraint
    linear_constraint = LinearConstraint(np.tri(size, size, 0), 0 * np.ones_like(M), max_storage_vol * np.ones_like(M))

    # define cycle limit constraint
    def cycle_cons(x):
        return sum(np.absolute(x))

    non_linear_constraint = NonlinearConstraint(cycle_cons, 0, daily_cycles * 8)

    # run optimisation
    res = minimize(func, x0=x, bounds=bnds, constraints=[linear_constraint, non_linear_constraint])

    capacity = np.cumsum(res.x)
    balance = np.cumsum(-res.x * M)
    return res.x, capacity, balance

def conf_mat(err, pred_decisions, static_decisions):
    # Confusion matrix
    con_mat = np.zeros((3, 3))
    for k in range(len(pred_decisions)):
        if static_decisions[k] > err:
            i = 0
        elif abs(static_decisions[k]) <= err:
            i = 1
        else:
            i =2
        if pred_decisions[k] > err:
            j = 0
        elif abs(pred_decisions[k]) <= err:
            j = 1
        else:
            j = 2
        con_mat[i, j] += 1
    return con_mat
#%%
# Constants and load data
nrows = 52608
data_memory = 48
time_step = 1
test_size = 0.25
samples = 2 # Number of x variables so like generation data is greater than two
split_point = 800 * 48
market_df = pd.read_excel('Market Data.xlsx')
prices = market_df.iloc[:, 1:3].to_numpy()

market_data = market_df.to_numpy(dtype=float)[:, 1: 1 + samples]
#%%
# Normalise data
y = np.roll(market_data[:, 1], time_step)
static_df = pd.read_csv('static_optimiser_data.csv')
#y = np.roll(static_df['Decisions'], -1)
X = np.repeat(market_data, data_memory, 0).reshape(nrows, data_memory, samples)
y_attrs = [np.mean(y), np.var(y)]
y = np.roll((y - np.mean(y))/np.var(y)**0.5, time_step)
X_attrs = []
for i in range(X.shape[-1]):
    X_attrs += [np.mean(X[:, :, i]), np.var(X[:, :, i])]
    X = (X - np.mean(X[:, :, i]))/np.var(X[:,:, i])**0.5

x_train = X[:split_point, :, :]
x_test = X[split_point:, :, :]
y_train = y[:split_point]
y_test = y[split_point:]

#%%
#layers.GRU(64, dropout = 0.1, recurrent_dropout = 0.1,return_sequences = True)
input_shape = (x_train.shape[0], data_memory, samples)
model_layer = [layers.GRU(64),
               layers.Dense(1)]

model = create_model(model_layer, input_shape)

#%%
# Fit model
time_step = 48
history = model.fit(x_train, y_train, validation_split = 0.1, epochs=3  , batch_size=10)
y_pred_output = np.roll(np.concatenate((model.predict(x_train[:, :, :]), model.predict(x_test[:, :, :]))), -time_step)
#%%
# Predict the future decisions
static_decisions = static_df['Decisions'].to_numpy()/0.475
y_pred_output = np.roll(np.concatenate((model.predict(x_train[:, :, :]), model.predict(x_test[:, :, :]))), -time_step)
#%%
'''
#%%
y_pred = y_pred_output - np.mean(y_pred_output)
print(np.mean(y_pred))
y_pred /= np.abs(np.var(y_pred)) 
y_pred = np.tanh(y_pred/4)*2
y_pred = -2 + (y_pred - np.min(y_pred))/(np.max(y_pred) - np.min(y_pred))*4

y_pred = y_pred_output *y_attrs[1]**0.5 + y_attrs[0] 
print(np.min(y_pred), np.max(y_pred))
y_pred = np.tanh(y_pred * 2)*2
y_pred = -2 + (y_pred - np.min(y_pred))/(np.max(y_pred) - np.min(y_pred))*4

print(np.mean(y_pred), np.var(y_pred), np.min(y_pred), np.max(y_pred))


pred_charging_rate = np.zeros(nrows)
# Run predicted decisions on training data
print('Training data')
cell, rev, pred_charging_rate[:split_point] = run_battery(split_point, y_pred[:split_point], prices[:split_point, 1])
print(rev)
cell.PrintAttributes()
# Run predicted decisions on test data
print('Test data')
cell, rev, pred_charging_rate[split_point:] = run_battery(nrows - split_point, y_pred[split_point:], prices[split_point:, 1])
print(rev)
#%%
# Training confusion matrix
err = 0.1

pred_charging_rate = np.tanh(pred_charging_rate)
pred_charging_rate= -2 + (pred_charging_rate - np.min(pred_charging_rate))/(np.max(pred_charging_rate) - np.min(pred_charging_rate))*4
cell, rev, pred_charging_rate[split_point:] = run_battery(nrows - split_point, pred_charging_rate[split_point:], prices[split_point:, 1])
print(rev)
train_con_mat = conf_mat(err, pred_charging_rate[:split_point], static_decisions[:split_point])
# Test confusion matrix
test_con_mat = conf_mat(err, pred_charging_rate[split_point:], static_decisions[split_point:])
print('Train: \n', train_con_mat, '\n Test: \n', test_con_mat)
a, b = 40000, 40048
t = np.linspace(0, 48, 48, endpoint=False)
plt.plot(t, static_decisions[a:b], 'g', t, pred_charging_rate[a:b], 'r')
plt.xlabel('Time step [half hour]')
plt.ylabel('Charging rate [MW]')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(['Static Optimiser decisions', 'Predicted decisions'], fontsize= 10)
plt.show()
'''
#%%
time_step = 48
history = model.fit(x_train, y_train, validation_split = 0.1, epochs=3  , batch_size=10)
y_pred_output = np.roll(np.concatenate((model.predict(x_train[:, :, :]), model.predict(x_test[:, :, :]))), -time_step)
#%%
time_step = 1
history = model.fit(x_train, y_train, validation_split = 0.1, epochs=3  , batch_size=10)
y_pred_output_2 = np.roll(np.concatenate((model.predict(x_train[:, :, :]), model.predict(x_test[:, :, :]))), -time_step)
#%%
# Plot the future data
plot_len = 96
plot_start = 40000
y_pred_plot = y_pred_output * y_attrs[1]**0.5 + y_attrs[0]
y_pred_plot = y_pred_plot[plot_start: plot_start + plot_len]
y_pred_plot_2 = y_pred_output_2 * y_attrs[1]**0.5 + y_attrs[0]
y_pred_plot_2 = y_pred_plot_2[plot_start: plot_start + plot_len]
t = np.linspace(0, y_pred_plot.size, y_pred_plot.size, endpoint=False)
plt.plot(t, prices[40000:40096][:, 0], 'g', t, prices[40000:40096][:, 1], 'r', t, y_pred_plot, 'b', t, y_pred_plot_2, 'c')
plt.xlabel('Time step [half hour]', fontsize = 11)
plt.ylabel('Charging Rate [MW]', fontsize = 11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(['Market 1 price', 'Market 2 price','Predicted market price for 48 time steps', 'Predicted market price for one time step'], fontsize = 11)
#plt.legend('Predicted price', 'Combined market price')
plt.show()
plt.savefig('Pred_prices.png')

#%%
'''
#%%
# Save predictions
pred_df = pd.DataFrame(y_pred_output)
pred_df.columns = ['Market Price']
pred_df.to_csv('pred_price.csv')

# %%
# Optimiser constants
max_charge_rate = 1
max_discharge_rate = 1
max_storage_vol = 4
charge_efficiency = 0.95
discharge_efficiency = 0.95
lifetime = 5000
storage_col_deg = 0.00001 * max_storage_vol
yearly_cost = 5000
num_days = 365 *3
daily_cycles = 1500 / num_days
day_end_balance = np.zeros(num_days)
output_data = np.zeros((4 ,48*num_days))
#%%
# Run optimiser
for day in range(num_days):
    #M = M2_price[day*size:(day+1)*size]
    #M = best_price[day * size:(day + 1) * size]
    M = y_pred[day * time_step:(day + 1) * time_step, 0]
    decisions, capacity, balance = optimize_charging(M,time_step)
    day_end_balance[day] = balance[-1]
    if day % 10 == 0:
       print("block "+str(day))
    output_data[:, 48*(day): 48*(day + 1)] = np.concatenate((np.array([M]), np.array([decisions]), np.array([capacity]), np.array([balance])), axis = 0)
    
#%%
# run predicted prices
output_df = pd.DataFrame(np.transpose(output_data))
output_df.columns = ['Market Price', 'Decisions', 'Battery Capacity', 'Balance']
output_df.to_csv('static_optimiser_pred_data.csv')
#%%
print(max(day_end_balance))
final_balance = sum(day_end_balance)
final_balance *= 10/3
print(final_balance)
#%%
print(output_data.shape)
print(y.size)
pred_prices = (y_pred * y_attrs[1]**0.5) + y_attrs[0]
#run_prices = y[40000 :40000 +time_step* num_days]* y_attrs[1]**0.5 + y_attrs[0]
cell, rev = run_battery(pred_prices[split_point:, 0].size, output_data[1, :], prices[split_point:, 1])
#cell, rev = run_battery(output_data.shape[1], output_data[1, :], pred_prices[split_point:, 0])

cell.PrintAttributes()
print(rev)
t = np.linspace(0,time_step * num_days, time_step * num_days , endpoint=False)
#plt.plot(t, y_pred[:time_step * num_days, 0], 'r', t, y[split_point:split_point +time_step * num_days], 'g')
a, b = 4100, 4200
t = np.linspace(a, b, b-a, endpoint=False)
y_pred_plot = y_pred
plt.plot(t, pred_prices[a- split_point:b - split_point, 0], 'r', t, y[a:b]* y_attrs[1]**0.5 + y_attrs[0], 'g')

plt.show()

pred_decisions = output_df['Decisions'].to_numpy()

#%%
'''