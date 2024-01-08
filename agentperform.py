import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
import pandas as pd
import logging

def agent_stock_performance(stock_price_ts: np.ndarray, trade_ts: np.ndarray,
                            stock_name: str, agent_name:str, 
                            display_graph: bool = False, 
                            save_graphic: bool = False,
                            path_file = None):
    """
    Analyzes the trading performance of an agent on a stock dataset.

    Parameters:
    - stock_price_ts (np.ndarray): 1-D array with stock's price at each timestep.
    - trade_ts (np.ndarray): 1-D array with agent's action at each time step. Actions defined as [-1, 0, 1] for [Sell, Hold, Buy] respectively.
    - stock_name (str): Name of the stock, used for labeling plots.
    - agent_name (str): Name of the agent, used for labeling plots.
    - display_graph (bool, optional): Whether to display the generated plot. Defaults to False.
    - save_graphic (bool, optional): Whether to save the plot as an image. Defaults to False.
    - path_file (str, optional): Path to save the generated plot if save_graphic is True. Defaults to None.

    Returns:
    - dict: Dictionary containing agent's trade performance metrics:
        - 'stock': Name of the stock.
        - 'agent_name': Name of the agent.
        - 'n_trades': Number of trades.
        - 'n_wins': Number of winning trades.
        - 'n_losses': Number of losing trades.
        - 'win_percentage': Trade win percentage.
        - 'cumulative_return': Cumulative return.
        - 'sortino': Sortino ratio.
        - 'max_drawdown': Maximum drawdown percentage.
        - 'sharpe': Sharpe ratio.
        - 'trade_dur_avg': Average duration of trades.
        - 'trade_dur_min': Minimum duration of trades.
        - 'trade_dur_max': Maximum duration of trades.
        - 'buy_hold': Buy and Hold return.
    """
    
        
    # Finding index and stock price of Buy Action
    buy_price_idx = np.where(trade_ts == 'B')[0]
    logging.info(buy_price_idx)
    buy_price_idx = buy_price_idx.astype(int)
    buy_price = stock_price_ts[buy_price_idx]
    

    # Finding index and stock price of Sell Action
    sell_price_idx = np.where(trade_ts == 'S')[0]
    sell_price_idx = sell_price_idx.astype(int)
    sell_price = stock_price_ts[sell_price_idx]
    logging.info(sell_price_idx)
    # Error Checking from enviroment
    assert len(buy_price_idx) == len(sell_price_idx),\
          "Trade action input didn't produce equal buy and sell actions"
    

    for i in range(len(buy_price_idx)):
        assert buy_price_idx[i] < sell_price_idx[i],\
            f"Assertion failed at index {i}: {buy_price_idx[i]} is not smaller than {sell_price_idx[i]}"
    

    if len(buy_price_idx) != 0:

        # Calculating Win, Loss, Total Trades
        trade_wins = np.sum(stock_price_ts[buy_price_idx] < stock_price_ts[sell_price_idx])
        trade_loss = np.sum(stock_price_ts[buy_price_idx] > stock_price_ts[sell_price_idx])
        trade_total = int((len(buy_price_idx) + len(sell_price_idx))/2) #Function assumes trade_ts has proper buy-sell patterns 
        win_precentage = trade_wins/trade_total*100

        # Calculating Average Length of Trade
        trade_lengths = np.array(sell_price_idx - buy_price_idx)
        trade_length_min = np.min(trade_lengths)
        trade_length_avg = np.mean(trade_lengths)      
        trade_length_max = np.max(trade_lengths)
        
        # Caculating Financial Performance

        returns_list = []

        for buy_index, sell_index in zip(buy_price_idx,sell_price_idx):
            trade_return = (stock_price_ts[sell_index] - stock_price_ts[buy_index])\
                /stock_price_ts[buy_index]

            
            returns_list.append(trade_return)
        


        returns = pd.Series(returns_list)
        sharpe_ratio = qs.stats.sharpe(returns)
        mdd = qs.stats.max_drawdown(returns)*100
        cumulative_return = returns.add(1).prod()
        sortino_ratio = qs.stats.sortino(returns)

        bad_values = [np.inf]

        if sharpe_ratio in bad_values:
            sharpe_ratio = 0
        
        if sortino_ratio in bad_values:
            sortino_ratio = 0

        if np.isnan(sharpe_ratio):
            sharpe_ratio = 0

        if np.isnan(sortino_ratio):
            sortino_ratio = 0

    else: 
        trade_wins = 0
        trade_loss = 0
        trade_total = 0 
        win_precentage = 0
        trade_lengths = 0
        trade_length_min = 0
        trade_length_avg = 0   
        trade_length_max = 0
        mdd = 0
        sharpe_ratio = 0
        cumulative_return = 0
        sortino_ratio = 0 

    # Buy and Hold
    bh_return = stock_price_ts[-1] / stock_price_ts[0]
 
    results = {'stock': stock_name,
            'agent_name': agent_name,
            "n_trades": trade_total, 
            "n_wins": trade_wins, 
            "n_losses": trade_loss, 
            "win_percentage":win_precentage, 
            "cumulative_return":cumulative_return,
            "sortino": sortino_ratio,
            'max_drawdown': mdd,
            'sharpe': sharpe_ratio,
            'trade_dur_avg': trade_length_avg,
            'trade_dur_min': trade_length_min,
            'trade_dur_max': trade_length_max,
            'buy_hold': bh_return}


    if display_graph is False and save_graphic is False:    
        return results
    
    # Ploting Stock Price and locations of Buy and Sell Actions
    fig, ax = plt.subplots()
    ax.plot(stock_price_ts, color='grey')
    ax.scatter(buy_price_idx,buy_price,color='g',marker="^")
    ax.scatter(sell_price_idx,sell_price,color='r',marker="v")


    hl_start = np.where(trade_ts == 'B')[0]
    hl_end = np.where(trade_ts == 'S')[0]

    for start, end in zip(hl_start, hl_end):
        color = 'grey' if stock_price_ts[start] == stock_price_ts[end] \
            else ('g' if stock_price_ts[start] < stock_price_ts[end] else 'r')
        plt.axvspan(start, end, color=color, alpha=0.15)

    plt.title(f'{agent_name}: {stock_name} Trade Performance')
    plt.ylabel(f'{stock_name} Price')
    plt.xlabel('Time Step$_{Test\ Range}$ ')
    
    # Positioning of Performance Number Text
    # Manual identification of stocks that require test located else where
    
    top_middle_align = ['COKE','BRK']
    
    
    if stock_name in top_middle_align:

    
        plotbox_x = np.median(range(len(trade_ts)))
        plotbox_y = ((np.max(stock_price_ts) - np.min(stock_price_ts))*(2/3)) \
            + np.min(stock_price_ts) 
    
    else:
        plotbox_x = np.min(range(len(trade_ts)))
        plotbox_y = np.min(stock_price_ts)
        
    texbox_content = (f"Trades\n"
        f'(# : W : L : W%):\n'
        f'{trade_total} : {trade_wins} : {trade_loss} : {win_precentage:.1f}\n '
        f'\nTrade Duration\n'
        f'(min : avg : max):\n'
        f'{trade_length_min:.2f} : {trade_length_avg:.2f} : {trade_length_max:.2f}\n'           
        
        f'\nFinancials\n'
        f'(CR : SP : SOR : MDD%):\n'
        f'{cumulative_return:.2f} : {sharpe_ratio:.2f} : {sortino_ratio:.2f} : {mdd:.1f}\n'
        
        f'\nB&H: {bh_return:.2f}'                
    )
    ax.text(plotbox_x,
            plotbox_y,
            texbox_content, 
            bbox=dict(facecolor='yellow', alpha=0.35),
            ha='left',
            va='bottom',
            fontsize=8)      
    if save_graphic:
        assert path_file is not None, "No path/filename provided"

        fig.savefig(path_file)

        if not display_graph:
            plt.close()
            return results
    
    plt.show()
    plt.close()
       
    return results

