import matplotlib.pyplot as plt
import numpy as np

def agent_stock_performance(stock_price_ts: np.ndarray, trade_ts: np.ndarray,stock_name: str, agent_name:str):
    # ---------------------------------------------------------------------------------------------
    # Converts NASDAQ stock csv files from https://www.nasdaq.com/market-activity/quotes/historical
    # to pd.dataframe[date, open, high, low, close, volume] 
    # with dtypes[Datetime,np.float32, np.float32, np.float32, np.float32, np.float32, np.int)
    # in ascentding order
    #----------------------------------------------------------------------------------------------
    ## Parameters:
    #-----------------------------------------------------------------------------------------------
    #   stock_price_ts: (np.narray) 
    #       1-D array with stock's price at each timestep 
    #   trade_ts: (np.narray)
    #       1-D array with agent's action at each time step. Action defined as [-1,0,1] as
    #       [Sell, Hold, Buy] respectively.  
    #   stock_name: (str)
    #       Name of stock, for labeling of plot
    #   agent_name: (str)
    #       Name of agent, for labeling of plot
    #----------------------------------------------------------------------------------------------
    # Returns:
    #----------------------------------------------------------------------------------------------
    #   dict: dict.keys=["n_trades", "n_wins", "n_losses", "win_percentage", "ror"]
    #       dictionary with agent's trade performance:
    #           - number of trades
    #           - number of winning trades
    #           - number losing trades
    #           - trade win percentage
    #           - rate of return (product of all conducted trade returns)
    # ----------------------------------------------------------------------------------------------  

        
    # Finding index and stock price of Buy Action
    buy_price_idx = np.where(trade_ts == 'B')[0]
    buy_price_idx = buy_price_idx.astype(int)
    buy_price = stock_price_ts[buy_price_idx]

    # Finding index and stock price of Sell Action
    sell_price_idx = np.where(trade_ts == 'S')[0]
    sell_price_idx = sell_price_idx.astype(int)
    sell_price = stock_price_ts[sell_price_idx]

    # Ploting Stock Price and locations of Buy and Sell Actions
    fig, ax = plt.subplots()
    ax.plot(stock_price_ts, color='grey')
    ax.scatter(buy_price_idx,buy_price,color='g',marker="^")
    ax.scatter(sell_price_idx,sell_price,color='r',marker="v")

    # Calculating Win, Loss, Total Trades
    trade_wins = np.sum(stock_price_ts[buy_price_idx] < stock_price_ts[sell_price_idx])
    trade_loss = np.sum(stock_price_ts[buy_price_idx] > stock_price_ts[sell_price_idx])
    trade_total = int((len(buy_price_idx) + len(sell_price_idx))/2) #Function assumes trade_ts has proper buy-sell patterns 
    trade_return = np.prod(stock_price_ts[sell_price_idx] / stock_price_ts[buy_price_idx])
    win_precentage = trade_wins/trade_total*100
    
    ror_per_tradeframe = []
    assert len(buy_price_idx) == len(sell_price_idx), "Arrays should have the same length"
    buy_sell_len = len(buy_price_idx)
    for idx in range(buy_sell_len):
        sell_idx = sell_price_idx[idx]
        buy_idx = buy_price_idx[idx]
        assert buy_idx < sell_idx, "Buy time index should occur before sell time index"
        if idx > 0:
            last_buy_idx = buy_price_idx[idx - 1]
            last_sell_idx = sell_price_idx[idx - 1]
            assert sell_idx > last_buy_idx, "Sell time index should occur before last buy time index"
            assert buy_idx > last_sell_idx, "Buy time index should occur after last buy time index"        
        ror = stock_price_ts[sell_idx] / stock_price_ts[buy_idx]
        ror_per_tradeframe.append(ror)

    trade_return = np.prod(ror_per_tradeframe)
    
    hl_start = np.where(trade_ts == 'B')[0]
    hl_end = np.where(trade_ts == 'S')[0]

    for start, end in zip(hl_start, hl_end):
        color = 'grey' if stock_price_ts[start] == stock_price_ts[end] else ('g' if stock_price_ts[start] < stock_price_ts[end] else 'r')
        plt.axvspan(start, end, color=color, alpha=0.15)

    plt.title(f'{agent_name} Agent: {stock_name} Trade Performance')
    plt.ylabel(f'{stock_name} Price')
    plt.xlabel('Time Step')
    
    plotbox_x = 10
    plotbox_y = np.median(stock_price_ts) - (min(stock_price_ts)/4.75) 
        
    texbox_content = (f"Trades:{trade_total:>5}\n"
        f"Wins:{trade_wins:>8}\n"
        f"Loss: {trade_loss:>8}\n"
        f"Win %: {win_precentage:.2f}\n"
        f"ROR: {trade_return:.3f}"
    )
    ax.text(plotbox_x,
            plotbox_y,
            texbox_content, 
            bbox=dict(facecolor='yellow', alpha=0.5),
            horizontalalignment='left',
            verticalalignment='bottom',
            fontsize=8)      
    plt.show()
    
    results = {"n_trades": trade_total, 
               "n_wins": trade_wins, 
               "n_losses": trade_loss, 
               "win_percentage":win_precentage, 
               "ror":trade_return}

    return results