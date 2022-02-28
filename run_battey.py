from BatteryClass import Battery
import numpy as np

def charge_battery(cell, charging_rate, market_price, charge_time = 0.5):
    if abs(charging_rate) > cell.MaxChargingRate:
        charging_rate = charging_rate /abs(charging_rate) * cell.MaxChargingRate
    if cell.Charge + cell.ChargingEfficiency * charging_rate * charge_time < 0:
        charging_rate = - cell.Charge /(cell.ChargingEfficiency * charge_time)*0.99
    elif cell.Charge + cell.ChargingEfficiency * charging_rate * charge_time > cell.MaxStorage * (1 - cell.MaxStorageLoss * abs(cell.ChargingEfficiency * charging_rate * charge_time * 0.5)):
        charging_rate = (cell.MaxStorage * (1 - cell.MaxStorageLoss * abs(cell.ChargingEfficiency * charging_rate * charge_time * 0.5)) - cell.Charge)/(cell.ChargingEfficiency * charge_time)
    cell.Charging(charging_rate, market_price) 
    return cell.Charge, charging_rate, cell.Bank

def run_battery(nrows, static_decisions, best_price):
    cell = Battery()
    charging_rate = np.zeros(nrows)
    battery_capacity = np.zeros(nrows)
    bank_data = np.zeros(nrows)
    # Run decisions over data
    for k in range(nrows):
        battery_capacity[k], charging_rate[k], bank_data[k] = charge_battery(cell, static_decisions[k], best_price[k])
    rev = total_revenue(cell)
    return cell, rev, charging_rate

def total_revenue(cell):
    if cell.Cycles / cell.MaxCycles > cell.UpTime / cell.Lifetime:
        revenue = cell.Bank/cell.Cycles * cell.MaxCycles
    else:
        revenue = cell.Bank/cell.UpTime * cell.Lifetime
    return revenue