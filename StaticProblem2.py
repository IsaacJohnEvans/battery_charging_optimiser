import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

# Define constants
max_charge_rate = 1
max_discharge_rate = 1
max_storage_vol = 4
charge_efficiency = 0.95
discharge_efficiency = 0.95
cycle_limit = 5000
storage_deg = 0.00001
yearly_cost = 5000

# Data
data = pd.read_excel('Market Data.xlsx', sheet_name=0)
data_np = np.array(data)
M1_price = data_np[:, 1]
M2_price = data_np[:, 2]
# Create min and max market price sets for optimal market selection
M_min = np.minimum(M1_price, M2_price)
M_max = np.maximum(M1_price, M2_price)

print(len(M2_price))

# Optimiser
def optimize_charging(M_min, M_max, storage_vol, size=48, cycle_lim=1.368):
    x = np.zeros_like(M_min, dtype='float')

    def func(dec):
        return sum((dec - abs(dec)) * M_max / 2 + (dec + abs(dec)) * M_min / 2)

    bnds = [[-max_charge_rate * discharge_efficiency, max_charge_rate * charge_efficiency] for _ in x]

    # define battery capacity constraint
    linear_constraint = LinearConstraint(np.tri(size, size, 0), 0 * np.ones_like(x), storage_vol * np.ones_like(x))

    # define cycle limit constraint
    def cycle_cons(x):
        return sum(np.absolute(x))

    non_linear_constraint = NonlinearConstraint(cycle_cons, 0, daily_cycles * 8)

    # run optimisation
    res = minimize(func, x0=x, bounds=bnds, constraints=[linear_constraint, non_linear_constraint])

    capacity = np.cumsum(res.x)
    day_end_balance = -1 * res.fun
    return res.x, capacity, day_end_balance


# Plot for demo
def plot_day(M, decisions, capacity):
    balance = np.cumsum(np.where(decisions > 0, -1 * decisions * M_min[:size], -1 * decisions * M_max[:size]))
    buy_dec = np.where(decisions > 0, decisions, np.nan)
    sell_dec = np.where(decisions < 0, decisions, np.nan)
    M1_dec = np.where(M1_price[:size] < M2_price[:size], buy_dec, sell_dec)
    M2_dec = np.where(M1_price[:size] > M2_price[:size], buy_dec, sell_dec)

    M1_col = 'g'
    M2_col = 'Maroon'

    fig, axs = plt.subplots(4, 1, constrained_layout=True)

    axs[0].plot(M[0], M1_col)
    axs[0].plot(M[-1], M2_col)
    #axs[0].set_title('Market Price')
    axs[0].legend(["Market 1 Price","Market 2 Price"])
    axs[0].set_xlabel("Time step [half hour]")
    axs[0].set_xlim([0, 48])
    axs[0].set_ylabel('Market Price [£]')

    markerline, stemlines, baseline = axs[1].stem(M1_dec*2, linefmt=M1_col, markerfmt='ko', basefmt='Grey', use_line_collection=True)
    plt.setp(stemlines, 'linewidth', 3)
    markerline, stemlines, baseline = axs[1].stem(M2_dec*2, linefmt=M2_col, markerfmt='ko', basefmt='Grey', use_line_collection=True)
    plt.setp(stemlines, 'linewidth', 3)
    #axs[1].set_title('Charge/Discharge rate (per 1/2 hour)')
    axs[1].set_xlabel("Time step [half hour]")
    axs[1].set_xlim([0,48])
    axs[1].set_ylabel('Charge rate [MW]')
    axs[1].set_yticks([-2, 0, 2])

    axs[2].plot(capacity, 'k', marker="o")
    #axs[2].set_title('Battery Capacity (MWH)')
    axs[2].set_xlabel("Time step [half hour]")
    axs[2].set_xlim([0,48])
    axs[2].set_ylabel('Capacity [MWH]')

    axs[3].plot(balance, 'k', marker="o")
    #axs[3].set_title('Balance (£)')
    axs[3].set_xlabel("Time step [half hour]")
    axs[3].set_xlim([0, 48])
    axs[3].set_ylabel('Balance [£]')

    fig.align_ylabels(axs[:])
    plt.savefig("global_opti_static.png")
    plt.show()
    return


# Run day-wise optimisation over the 3 year period
size = 48  # 1 day = 48 half hours
num_days = int(len(M2_price) / size)
daily_cycles = (cycle_limit*3) / (num_days*10)  # cycles per day required to expire at 10 year lifetime
day_end_balances = np.zeros(num_days)
storage_vol = max_storage_vol

for day in range(num_days):
    decisions, capacity, day_end_balance = optimize_charging(M_min[day * size:(day + 1) * size],
                                                     M_max[day * size:(day + 1) * size], storage_vol, size=size, cycle_lim=daily_cycles)
    day_end_balances[day] = day_end_balance
    print("day " + str(day))
    # Update storage_vol
    storage_vol -= storage_deg*daily_cycles

three_year_balance = sum(day_end_balances)
final_balance = three_year_balance - yearly_cost * 3  # subtract yearly maintenance cost
print("Balance at end of 3 year period = £" + str(final_balance))

pred_10_year_balance = (three_year_balance * 10/3) - yearly_cost*10
print("Projected balance at end of 10 year period = £" + str(pred_10_year_balance))


# Demo of optimization over day 0 with plot
decisions, capacity, day_end_balance = optimize_charging(M_min[:size],M_max[:size], max_storage_vol, size)
plot_day([M1_price[:size], M2_price[:size]], decisions, capacity)
plt.show()