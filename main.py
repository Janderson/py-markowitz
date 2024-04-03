# -*- coding: utf-8 -*-
"""
@author: Janderson FFerreira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion, show
import click 

from pandas_datareader import data as pdr
import yfinance as yfin




def run_markowitz(stocks, start_data='2019-01-01'):
    yfin.pdr_override()
    num_stocks = len(stocks) 
    #download daily price data for each of the stocks in the portfolio
    data = pdr.get_data_yahoo(stocks, start=start_data, end='2024-04-01')
    closes = data['Adj Close'].reset_index()[stocks]
    #convert daily stock prices into daily returns
    returns = closes.pct_change()
    
    #calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    
    #set number of runs of random portfolio weights
    num_portfolios = 25000
    
    #set up array to hold results
    #We have increased the size of the array to hold the weight values for each stock
    results = np.zeros((4+num_stocks-1,num_portfolios))
    
    for i in range(num_portfolios):
        #select random weights for portfolio holdings
        weights = np.array(np.random.random(len(stocks)))
        #rebalance weights to sum to 1
        weights /= np.sum(weights)
    
        #calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights))) * np.sqrt(252)
    
        #store results in results array
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2,i] = results[0,i] / results[1,i]
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j+3,i] = weights[j]
    
    columns = ['ret','stdev','sharpe']
    columns.extend(stocks)
    results_frame = pd.DataFrame(results.T, columns=columns)
    
    #locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]

    #locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    print(f"\n\n==================\nmaximo sharpe do portifólio: {max_sharpe_port['sharpe']:.3}"
          f"\n====================\npesos: \n{max_sharpe_port})")
    print(f"\n\n==================\nvol. mínima do portifólio: {(min_vol_port['stdev']*100):.3} %"
          f"\n====================\npesos: \n{min_vol_port})")

    #create scatter plot coloured by Sharpe Ratio
    plt.scatter(results_frame.stdev, results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    #plt.colorbar()
    #plot red star to highlight position of portfolio with highest Sharpe Ratio
    plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=1000)

    #plot green star to highlight position of minimum variance portfolio
    plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=1000)
    plt.savefig("plot.png")
    print("salvo plot.png")


@click.group()
def cli():
    pass


@cli.command("run")
@click.argument("stocks", default='["BOVA11.SA", "DIVO11.SA"]')
def cmd_run_markowitz(stocks):
    import json
    stocks = json.loads(stocks)
    run_markowitz(stocks)

if __name__== "__main__":
    cli()
