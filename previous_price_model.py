#%%
# Import modules
import numpy as np
import pandas as pd
from BatteryClass import Battery
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import matplotlib.pyplot as plt
from run_battey import run_battery, total_revenue, charge_battery

#%%

def load_data(nrows, data_memory, split_point, create_prices = True, load_static = True, create_memory = True, save_memory = False):
    # max rows = 52608
    args = []
    market_df = pd.read_excel('Market Data.xlsx', nrows= nrows)
    args.append(market_df)
    if create_prices:
        prices = market_df[['Market 1 Price [£/MWh]','Market 2 Price [£/MWh]']].to_numpy()
        best_price = select_best_price(nrows, prices)
        args += [prices, best_price]
    if load_static:
        static_df = pd.read_csv('Data/static_optimiser_data.csv', nrows = nrows)
        args.append(static_df)
        if create_memory:
            data_args = create_memory_data(static_df['Decisions'].to_numpy(), prices, nrows, data_memory, normalise)
            X, y = data_args[0:2]
            x_train, x_test, y_train, y_test, t_train, t_test  = test_train_split(X, y, split_point, nrows)
            if save_memory:
                pd.DataFrame(np.concatenate((X[:, :, 0], X[:, :, 1]), axis = 1)).to_csv('memory_data.csv')
            args += [x_train, x_test, y_train, y_test, t_train, t_test]
    return args

def select_best_price(nrows, prices):
    best_price = np.zeros(nrows)
    mean_cost = np.mean(prices)
    for k in range(nrows):
        if mean_cost - prices[k, :].min() > prices[k, :].max() - mean_cost:
            best_price[k] = prices[k, :].min()
        else:
            best_price[k] = prices[k, :].max()
    return best_price
    
def create_memory_data(static_decisions, prices, nrows, data_memory, normalise):
    args = []
    X = np.zeros((nrows, data_memory, 2), dtype=np.float64)
    prices = np.concatenate((np.zeros((data_memory -1, 2), dtype= np.float64), prices))
    for i in range(nrows):
        for j in range(data_memory):
            X[i, :, :] = prices[i:i  + data_memory,:]
    args += [X, static_decisions]
    return args

def test_train_split(X, y, split_point, nrows):
    x_train = X[:split_point]
    x_test = X[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]
    t_train = np.linspace(0, split_point, split_point, endpoint=False)
    t_test = np.linspace(split_point, nrows, X.size - split_point, endpoint=False)
    return x_train, x_test, y_train, y_test, t_train, t_test

# Normalise train data
def normalise(x, y):
    layer = layers.Normalization()
    layer.adapt(x)
    x = layer(x)
    y_attrs = [np.mean(y), np.var(y)]
    y = (y - np.mean(y))/np.var(y)**0.5
    return x, y, y_attrs

def create_model(layers, input_shape):
    # Create and compile model
    model = keras.Sequential(layers)
    model.build(input_shape=input_shape)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model
    
#%%
# Load nrows of data
nrows = 52608
data_memory = 48
time_step = 48
test_size = 0.25
samples = 2
split_point = int(nrows*(1- test_size))
split_point = 40000
args = load_data(nrows, data_memory, split_point)
market_df, prices, best_price, static_df, x_train, x_test, y_train, y_test, t_train, t_test = args

#%%
# Runs battery over static decisions
cell, rev, a = run_battery(nrows, static_df['Decisions']* 2, best_price)
cell.PrintAttributes()
print(rev)
#%%

#%%
# Create rolled y data
y_train_roll = np.roll(prices[:split_point, 1], time_step)
y_test_roll = np.roll(prices[split_point:, 1], time_step)
# Normalise data
x_train_norm, y_train_norm, y_train_attrs = normalise(x_train, y_train_roll)
x_test_norm, y_test_norm, y_test_attrs = normalise(x_test, y_test_roll)
#input_shape=(data_memory, 2)
#%%
# Create model
#, dropout = 0.1, recurrent_dropout = 0.1
# 128, dropout = 0.1, recurrent_dropout = 0.1, return_sequences = True), layers.GRU(128, dropout = 0.1, recurrent_dropout = 0.1)
input_shape = (x_train_norm.shape[0], data_memory, samples)
model_layer = [layers.GRU(64),
               layers.Dense(1)]

model = create_model(model_layer, input_shape)
#%%
# Fit model
history = model.fit(x_train_norm, y_train_norm, validation_split = 0.1, epochs=3  , batch_size=10)


#%%
# Predict the future price 
y_pred = np.roll(np.concatenate((model.predict(x_train_norm[:, :, :])* y_train_attrs[1]**0.5 + y_train_attrs[0], model.predict(x_test_norm[:, :, :])*y_test_attrs[1]**0.5 + y_test_attrs[0])), -time_step)

print(np.mean(y_pred[:48]))
print(np.mean(prices[40000:40048, 1]))
np.mean(y_pred[:48, 0] - prices[40000:40048, 1])
#%%
# Plot the future data
plot_len = 50
plot_start = 40050
y_pred_plot = y_pred[plot_start - time_step: plot_start + plot_len - time_step][:, 0]
y_plot = np.concatenate((y_train_roll, y_test_roll))[plot_start: plot_start + plot_len]
t = np.linspace(0, y_pred_plot.size, y_pred_plot.size, endpoint=False)
plt.plot(t, y_pred_plot, 'r', t, y_plot, 'g')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
axes = plt.gca()
axes.yaxis.label.set_size(17)
axes.xaxis.label.set_size(17)
plt.xlabel('Time step [half hour]')
plt.ylabel('Price [£]')
plt.legend(['Predicted price', 'Combined market price'])
plt.show()
#%%
# Save predictions
pred_df = pd.DataFrame(y_pred)
pred_df.columns = ['Market Price']
pred_df.to_csv('pred_price.csv')
#%%
print('Real diff \n', np.diff(y_plot), '\n \n'
      'Pred diff \n', np.diff(y_pred_plot), '\n \n' 
      'Diff diff \n', np.diff(y_plot) - np.diff(y_pred_plot))
#%%
# Make predictions
y_pred = np.concatenate((model.predict(x_train)* y_train_attrs[1]**0.5 + y_train_attrs[0], model.predict(x_test)*y_test_attrs[1]**0.5 + y_test_attrs[0]))

# Charge battery 
# Charge on training data
charging_rates = y_pred[:40000]/0.475
cell, rev = run_battery(charging_rates.size, charging_rates, best_price)
cell.PrintAttributes()
print(rev)

# Charge on test data
charging_rates = y_pred[40000:]/0.475
cell, rev = run_battery(charging_rates.size, charging_rates, best_price)
cell.PrintAttributes()
print(rev)


# %%
