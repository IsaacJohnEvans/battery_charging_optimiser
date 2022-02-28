#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from BatteryClass import Battery
from run_battey import run_battery, total_revenue, charge_battery
#from previous_price_model import run_battery, total_revenue

# Optimiser
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


# Plot for demo
def plot_day(M, decisions, capacity, balance):
    fig, axs = plt.subplots(4, 1, constrained_layout=True)

    axs[0].plot(M)
    axs[0].set_title('Market Price')

    axs[1].plot(decisions, marker="o")
    axs[1].set_title('Charge/Discharge rate (per 1/2 hour)')

    axs[2].plot(capacity, marker="o")
    axs[2].set_title('Battery Capacity (MWH)')

    axs[3].plot(balance, marker="o")
    axs[3].set_title('Balance (£)')

    plt.savefig("global_opti_static.png")
    plt.show()
    return


#%%
# Data
data = pd.read_excel('Market Data.xlsx', sheet_name=0)
df = pd.read_excel('Market Data.xlsx', sheet_name=0)
data_np = np.array(data)

max_charge_rate = 1
max_discharge_rate = 1
max_storage_vol = 4
charge_efficiency = 0.95
discharge_efficiency = 0.95
lifetime = 5000
storage_col_deg = 0.00001 * max_storage_vol
yearly_cost = 5000

M1_price = data_np[:, 1]
M2_price = data_np[:, 2]

#%%
size = 48
num_days = int(len(M2_price) / size)
num_days = 262
daily_cycles = 1500 / num_days
day_end_balance = np.zeros(num_days)
output_data = np.zeros((4 ,48*num_days))

# Calculating the combined market price
nrows = 52608
prices = df[['Market 1 Price [£/MWh]','Market 2 Price [£/MWh]']].to_numpy()
best_price = np.zeros(nrows)
mean_cost = np.mean(prices)
# mean_cost - prices[k, :].min() > prices[k, :].max() - mean_cost
for k in range(nrows):
    if mean_cost - prices[k, :].min() > prices[k, :].max() - mean_cost:
        best_price[k] = prices[k, :].min()
    else:
        best_price[k] = prices[k, :].max()

#%%
# Read predicted prices
pred_df = pd.read_csv('pred_price.csv')
pred_price = pred_df.to_numpy()

# Run day-wise optimisation over the 3 year period
#num_days = 3


#%%
num_days = 262
for day in range(num_days):
    #M = M2_price[day*size:(day+1)*size]
    #M = best_price[day * size:(day + 1) * size]
    M = pred_price[day * size:(day + 1) * size, 1]
    decisions, capacity, balance = optimize_charging(M,size)
    day_end_balance[day] = balance[-1]
    if day % 10 == 0:
       print("block "+str(day))
    output_data[:, 48*(day): 48*(day + 1)] = np.concatenate((np.array([M]), np.array([decisions]), np.array([capacity]), np.array([balance])), axis = 0)

#%%
# run predicted prices
output_df = pd.DataFrame(np.transpose(output_data))
output_df.columns = ['Market Price', 'Decisions', 'Battery Capacity', 'Balance']
output_df.to_csv('static_optimiser_pred_data.csv')

print(max(day_end_balance))
final_balance = sum(day_end_balance)
final_balance *= 3650/262
print(final_balance)
#%%

charging_rates = output_data[1, :]
run_prices = prices[40000 :40000 +size* num_days, 0]
cell, rev = run_battery(size*num_days, charging_rates, run_prices)

cell.PrintAttributes()
print(rev)
t = np.linspace(0,size * num_days, size * num_days , endpoint=False)
plt.plot(t, output_data[0, :size * num_days], 'r', t, run_prices, 'g')
plt.show()
'''
size = 48
M = M2_price[10010:10010 + size]
M = best_price[10010:10010 + size]
decisions, capacity, balance = optimize_charging(M, size)
plot_day(M, decisions, capacity, balance)
def cycle_cons(x):
  return sum(np.absolute(x))
print(cycle_cons(decisions)/8)'''

#value for mixed market (no  degredation): £178701.42663259295
#%%
static_df = pd.read_csv('static_optimiser_data.csv')


# %%
static_decisions = static_df['Decisions'].to_numpy()[40000:52576]
pred_decisions = output_df['Decisions'].to_numpy()

(static_decisions.round(1) * pred_decisions.round(1) >= 0).sum()

# %%
err = 0.01
con_mat = np.zeros((3, 3))
for k in range(len(pred_decisions)):
    if static_decisions[k] > err:
        i = 0
    elif abs(static_decisions[k]) < err:
        i = 1
    else:
        i =2
    if pred_decisions[k] > err:
        j = 0
    elif abs(pred_decisions[k]) < err:
        j = 1
    else:
        j = 2
    con_mat[i , j] += 1
print(con_mat)
            
#pd.crosstab(static_df['Decisions'], output_df['Decisions'])