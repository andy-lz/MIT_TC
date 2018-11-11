
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from TradeAnalysis import TradeAnalysis


# In[9]:

DF = pd.read_csv(os.path.join(os.path.expanduser('~'), 'Downloads', 'btceUSD.csv'), names=['Datetime', 'Price', 'Volume'])
DF['Datetime'] = pd.to_datetime(DF['Datetime'], unit='s')
DF.set_index('Datetime', drop=True, inplace=True)
DF = DF.loc['2015' : ]
del DF['Volume']


# In[10]:

def backtest_autoregressive_strat(look_back, percentile):
    '''
    This strategy predicts the next price by regressing price on the prior "look_back" number of prices.
    Define "edge" as the magnitude of the difference between the current price and the predicted price.
    Calculate "offset" as the "percentile" quantile of all edges in the training data.
    If the current "edge" is larger than "offset", trade.
    Hold the current position until a trade signal in the oppositite direction is received.
    '''
    
    # Storing the prices from previous days
    df = DF.copy()
    df.rename(columns={'Price': 0}, inplace=True)
    for i in range(1, look_back+1):
        df[i] = df[0].shift(i)
    df.dropna(inplace=True)

    daily_pnl_list = []
    # Iterating through each month in the data
    for year in sorted(df.index.year.drop_duplicates().tolist()):
        for month in sorted(df.index.month.drop_duplicates().tolist()):
            start_date = pd.to_datetime('{}-{}-{}'.format(year, month, 1), format='%Y-%m-%d')
            end_date = start_date + relativedelta(months=1)
            start_date, end_date = str(start_date).split(" ")[0], str(end_date).split(" ")[0]
            df1 = df[ : start_date] # the training data
            if not df1.empty:
                df2 = df[start_date: end_date] # the testing data
                # fit my autoregressive model
                ols = LinearRegression().fit(df1[range(1, look_back+1)], df1[0]) 
                df1['Prediction'] = ols.predict(df1[range(1, look_back+1)])
                df1['Edge'] = (df1['Prediction'] - df1[1]).abs()
                # calculate the offset using the training data
                offset = df1['Edge'].quantile(percentile) 
                if not df2.empty:
                    print year, month
                    # predict using the regression model trained on the training data
                    df2['Prediction'] = ols.predict(df2[range(1, look_back+1)])
                    df2['Edge'] = (df2['Prediction'] - df2[1]).abs()
                    df2['Excess Edge'] = df2['Edge'] - offset
                    df2['Direction'] = np.where(df2['Prediction'] - df2[1] >= 0, 1, -1)
                    df2['Position'] = np.nan
                    df2['Position'] = np.where(df2['Excess Edge'] > 0, df2['Direction'], df2['Position'])
                    # If we don't get another signal (Excess Edge > 0), our position doesn't change
                    df2['Position'].fillna(method='ffill', inplace=True)
                    # We start out with a position of zero
                    df2['Position'].fillna(0, inplace=True)
                    df2['Quantity'] = (df2['Position'] - df2['Position'].shift(1)).abs()
                    df2['P&L'] = (df2[0] - df2[1]) * df2['Position']
                    # Aggregate P&L for each day
                    daily_pnl = df2.groupby(df2.index.date)['P&L'].sum().sort_index()
                    # Print some stats for each month
                    print 'Sharpe', daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                    print 'Quantity', df2['Quantity'].sum()
                    print 'R2 train, test', r2_score(df1[0], df1['Prediction']), ',', r2_score(df2[0], df2['Prediction'])
                    # storing the daily P&L timeseries for each month
                    daily_pnl_list.append(daily_pnl)

    print ""
    print ""
    # Aggregate the daily data into a single timeseries
    daily_pnl = pd.concat(daily_pnl_list).fillna(0)
    # Printing the aggregated performance statistics of the strategy
    TradeAnalysis(daily_pnl, None).print_summary_list('P&L', None)
    daily_pnl.cumsum().plot()
    plt.show()


# In[11]:

backtest_autoregressive_strat(look_back=5, percentile=0.98)


# In[ ]:



