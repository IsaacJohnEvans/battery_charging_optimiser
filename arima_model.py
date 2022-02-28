#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.tree import export_graphviz 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_model(known_end, predict_end, best_price, order, seasonal_order):
    model=SARIMAX(best_price[:known_end], order = order)
    result=model.fit()
    y_pred = result.predict(known_end, predict_end, dynamic = True)
    loss = np.sum(np.abs((y_pred - best_price[known_end: predict_end + 1])))/(predict_end - known_end)
    t = np.linspace(known_end, predict_end, predict_end - known_end + 1)
    fig = plt.plot(t, y_pred, 'r',  t, best_price[known_end:predict_end + 1])
    print('loss', loss)
    fig = 0
    return loss, fig

#%%
known_end = 400
predict_end = 410
order = (48, 2, 24)
#seasonal_order = (1,0,0,48)
best_price = pd.read_excel('Market Data.xlsx')['Market 1 Price [£/MWh]'].to_numpy()

# finding out if it is stationary data 
#print(adfuller(best_price[known_end:predict_end]- np.roll(best_price[known_end:predict_end], 5)))
#fig1=plot_acf(best_price[known_end:predict_end]- np.roll(best_price[known_end:predict_end], 5))
#fig2=plot_pacf(best_price[known_end:predict_end]- np.roll(best_price[known_end:predict_end], 5))

#%%
search_range_p = [44, 52, 4]
search_range_d = [1, 2]
search_range_q = [2, 4, 1]
losses = np.zeros((search_range_p[1]-search_range_p[0], search_range_d[1]-search_range_d[0], search_range_q[1]-search_range_q[0]))
for i in tqdm(range(search_range_p[1]-search_range_p[0])):
    for j in range(search_range_d[1]-search_range_d[0]):
        for k in range(search_range_q[1]-search_range_q[0]):
            order = (i, j, k)
            loss, fig = sarima_model(known_end, predict_end, best_price, order, seasonal_order = None)
            losses[i, j, k] = loss
            print('p', i, 'd', j, 'q', k)
print(losses)
print(np.min(losses))
print(np.argmin(losses))
plt.show()