import gym
from gym import spaces
import numpy as np
import pandas as pd

class DiscretizedOHLCVEnv(gym.Env):
       
    def __init__(self, ohlcv_data,bins_per_feature:list,initial_cash=1000):
        self.ohlcv_raw_data = ohlcv_data
        self.initial_cash = initial_cash
        self.bins_per_feature = bins_per_feature
        self.action_space = spaces.Discrete(3, start=-1) # -1: Sell, 0: Hold, 1: Buy
        self.observation_space = spaces.MultiDiscrete([bins_per_feature] * len(ohlcv_data[0])) # Shape = (Open, High, Low, Close, Volume)
        self.ohlcv_binned_data = self.discretize_ohlcv(self.ohlcv_raw_data,self.bins_per_feature)
        self.max_idx = ohlcv_data.shape[0] -1 
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash_in_hand = self.initial_cash
        self.stock_holding = 0
        self.step_info = []  # Initialize an empty list to store step information
        self.stock_price = self.ohlcv_raw_data[self.current_step][3] #Stock price set to closing price
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        self.available_actions = (0,1)
    
    def step(self, action):
        assert action in self.available_actions, f'Action {action} not in {self.available_actions}'

        prev_valuation = self.total_portfolio_value
        step_data = {
            'Step': self.current_step,
            'Portfolio Value': self.total_portfolio_value,
            'Cash': self.cash_in_hand,
            'Stock Value': self.stock_price * self.stock_holding, 
            'Stock Holdings': self.stock_holding,
            'Stock Price': self.stock_price,
            'Input': self.ohlcv_binned_data[self.current_step],
            "Available Actions": self.available_actions,
            "Action": action
        }

       
        if action == -1: # Sell
            self._sell()

        elif action == 0: # Hold
            pass
   
        elif action == 1: # Buy
            self._buy()



        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        reward = self.total_portfolio_value - prev_valuation

        done = self.current_step >= (len(self.ohlcv_raw_data)- 1)
        
        
        self.step_info.append(step_data)
        #print(step_data)

        if not done:
            self.current_step += 1
            self.stock_price = self.ohlcv_raw_data[self.current_step][3] ## Assuming Closing price for stock price, 2nd place implemented...need to simplify
        
        if self.current_step == (self.max_idx - 1):
            if self.stock_holding > 0:
                self.available_actions = (-1,)
            else:
                self.available_actions = (0,)

        

        next_observation = self.get_observation()

        return next_observation, reward, done
    
    def _buy(self):
        self.num_stocks_buy = np.floor(self.cash_in_hand/self.stock_price) # Buy Maximum allowed (Current Method)
        self.cash_in_hand -= self.num_stocks_buy * self.stock_price
        self.stock_holding = self.num_stocks_buy
        self.num_stocks_buy = 0
        self.available_actions = (-1,0)
        return
    
    def _sell(self):
        self.num_stocks_sell = self.stock_holding # Sell all stocks (Current Mehtod)
        self.cash_in_hand += self.num_stocks_sell * self.stock_price  # No commission fee can be added later
        self.stock_holding -= self.num_stocks_sell
        self.num_stocks_sell = 0
        self.available_actions = (0,1)
        return
    

    def get_observation(self):
        return(tuple(self.ohlcv_binned_data[self.current_step]))

    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored step information

    def discretize_ohlcv(self, data, bins_for_feature):
        discretized_data = []
        for column,num_bins in zip(data.T, bins_for_feature):  # Transpose to iterate through columns
            min_val = np.min(column)
            max_val = np.max(column)
            bin_width = (max_val - min_val) / num_bins
            bins = [min_val + i * bin_width for i in range(num_bins)]
            digitized = np.digitize(column, bins)
            discretized_data.append(digitized-1)
        return np.array(discretized_data).T.tolist()  # Transpose back to original shape