import gym
from gym import spaces
import numpy as np
import pandas as pd
import warnings 


class ContinuousOHLCVEnv(gym.Env):
       
    def __init__(self, ohlcv_data: np.array, commission_rate: float = 0.00, initial_cash: float = 1000.0) -> None:
        self.ohlcv_raw_data = ohlcv_data
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        
        # Define Action Space
        self.actions = ('S','H','B') # Sell, Hold, Buy
        self.action_space = spaces.Discrete(3)
        self.action_to_idx_dic = {'S':0, 'H':1, "B": 2}
        self.idx_to_action_dic = {0: 'S', 1: 'H', 2: 'B'}

         # Define the observation space for OHLCV data
        num_features = ohlcv_data.shape[1]  # Assuming OHLCV columns are Open, High, Low, Close, Volume
        feature_min = np.array([0.0] * num_features)  # Assuming minimum value for each feature is 0.0
        feature_max = np.array([np.inf] * num_features)  # Assuming unbounded maximum values for OHLC and Volume
        self.observation_space = spaces.Box(low=feature_min, high=feature_max, dtype=np.float32)

        self.max_idx = ohlcv_data.shape[0] -1 
        self.reset()

    def reset(self):
        self.current_step = 0
        self.last_reward = 0
        self.total_reward =  0
        self.cash_in_hand = self.initial_cash
        self.last_commission_cost = 0
        self.total_commission_cost = 0
        self.stock_holding = 0
        self.step_info = []  # Initialize an empty list to store step information
        self.stock_price = self.ohlcv_raw_data[self.current_step][3] #Stock price set to closing price
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        self.available_actions = ('H','B')
    
    
    def step(self, action):
        assert action in self.available_actions, f'Action {action} not in {self.available_actions}'

        prev_valuation = self.total_portfolio_value
        step_data = {
            'Step': self.current_step,
            'Portfolio Value': np.round(self.total_portfolio_value, 2),
            'Cash': np.round(self.cash_in_hand),
            'Stock Value': np.round(self.stock_price * self.stock_holding,2), 
            'Stock Holdings': self.stock_holding,
            'Stock Price': np.round(self.stock_price,2),
            'Last Reward': np.round(self.last_reward,2),
            'Total Reward': np.round(self.total_reward,2),
            "Last Commission Cost": np.round(self.last_commission_cost,2),
            'Total Commision Cost': np.round(self.total_commission_cost,2),
            'Input': self.ohlcv_raw_data[self.current_step],
            "Available Actions": self.available_actions,
            "Action": action
        }

       
        if action == 'S': # Sell
            self._sell()

        elif action == 'H': # Hold
            pass
   
        elif action == "B": # Buy
            self._buy()



        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        reward = self.total_portfolio_value - prev_valuation
        self.last_reward = reward
        self.total_reward += reward

        done = self.current_step >= (len(self.ohlcv_raw_data)- 1)
        
        
        self.step_info.append(step_data)
        #print(step_data)

        if not done:
            self.current_step += 1
            self.stock_price = self.ohlcv_raw_data[self.current_step][3] ## Assuming Closing price for stock price, 2nd place implemented...need to simplify
        
        if self.current_step == (self.max_idx - 1):
            if self.stock_holding > 0:
                self.available_actions = ('S',)
            else:
                self.available_actions = ('H',)     

        next_observation = self.get_observation()

        return next_observation, reward, done
    
    def _buy(self):
        self.num_stocks_buy = np.floor(self.cash_in_hand/self.stock_price) # Buy Maximum allowed (Current Method)
        self.last_commission_cost = self.num_stocks_buy * self.stock_price * self.commission_rate
        self.total_commission_cost += self.last_commission_cost
        self.cash_in_hand -= self.num_stocks_buy * self.stock_price - self.last_commission_cost
        self.stock_holding = self.num_stocks_buy
        self.num_stocks_buy = 0
        self.available_actions = ('S','H')
        return
    
    def _sell(self):
        self.num_stocks_sell = self.stock_holding # Sell all stocks (Current Mehtod)
        self.last_commission_cost = self.num_stocks_sell * self.stock_price * self.commission_rate
        self.total_commission_cost += self.last_commission_cost
        self.cash_in_hand += self.num_stocks_sell * self.stock_price - self.last_commission_cost
        self.stock_holding -= self.num_stocks_sell
        self.num_stocks_sell = 0
        self.available_actions = ('H','B')
        return  

    def get_observation(self):
        return(tuple(self.ohlcv_raw_data[self.current_step]))

    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored step information



class DiscretizedOHLCVEnv(gym.Env):
       
    def __init__(self, ohlcv_data:pd.DataFrame,bins_per_feature:list,initial_cash=1000000):
        self.ohlcv_raw_data = ohlcv_data
        self.initial_cash = initial_cash
        self.bins_per_feature = bins_per_feature
        self.actions = ('S','H','B') # Sell, Hold, Buy
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([bins_per_feature] * len(ohlcv_data[0])) # Shape = (Open, High, Low, Close, Volume)
        self.ohlcv_binned_data = self.discretize_ohlcv(self.ohlcv_raw_data,self.bins_per_feature)
        self.max_idx = ohlcv_data.shape[0] -1
        self.finish_idx = self.max_idx
        self.start_idx = 0
        self.reset()

    def reset(self):
        self.current_step = self.start_idx
        self.cash_in_hand = self.initial_cash
        self.stock_holding = int(0)
        self.step_info = []  # Initialize an empty list to store step information
        self.stock_price = self.ohlcv_raw_data[self.current_step][3] #Stock price set to closing price
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        if self.cash_in_hand > self.stock_price:
            self.available_actions = ('H','B')
        else:
            self.available_actions = ('H','B')
        self.step_info = []
    
    def step(self, action):
        assert action in self.available_actions, f'Action {action} not in {self.available_actions}'
        if action == 'B' and self.cash_in_hand < self.stock_price:
            action = 'H'
        
        prev_valuation = self.total_portfolio_value
        step_data = {
            'Step': self.current_step - self.start_idx + 1,
            'idx': self.current_step,
            'Portfolio Value': self.total_portfolio_value,
            'Cash': self.cash_in_hand,
            'Stock Value': self.stock_price * self.stock_holding, 
            'Stock Holdings': self.stock_holding,
            'Stock Price': self.stock_price,
            'Input': self.ohlcv_binned_data[self.current_step],
            "Available Actions": self.available_actions,
            "Action": action
        }

       
        if action == 'S': # Sell
            self._sell()

        elif action == 'H': # Hold
            pass
   
        elif action == "B": # Buy
            self._buy()



        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        reward = self.total_portfolio_value - prev_valuation

        done = self.current_step >= (self.finish_idx)
        
        
        self.step_info.append(step_data)
        #print(step_data)

        if not done:
            self.current_step += 1
            self.stock_price = self.ohlcv_raw_data[self.current_step][3] ## Assuming Closing price for stock price, 2nd place implemented...need to simplify
        
        if self.current_step == (self.finish_idx):

            if int(self.stock_holding) > 0:
                self.available_actions = ('S',)
            else:
                self.available_actions = ('H',)
        
        """if self.current_step == (self.finish_idx) and int(self.stock_holding) > 0:
            warnings.warn("Automatic sell action at last timestep, agent did not perform sell action in max_idx - 1")
            self._sell()"""

        

        next_observation = self.get_observation()

        return next_observation, reward, done
    
    def _buy(self):
        self.num_stocks_buy = int(np.floor(self.cash_in_hand/self.stock_price)) # Buy Maximum allowed (Current Method)
        self.cash_in_hand -= self.num_stocks_buy * self.stock_price
        self.stock_holding = self.num_stocks_buy
        self.num_stocks_buy = 0
        self.available_actions = ('S','H')
        return
    
    def _sell(self):
        self.num_stocks_sell = self.stock_holding # Sell all stocks (Current Mehtod)
        self.cash_in_hand += self.num_stocks_sell * self.stock_price  # No commission fee can be added later
        self.stock_holding -= self.num_stocks_sell
        self.num_stocks_sell = 0
        self.available_actions = ('H','B')
        return
    
    def update_idx(self,start_idx:int,final_idx:int):
        assert start_idx < final_idx, f'invalid: start_idx: {start_idx} < final_idx {final_idx}'
        self.start_idx = start_idx
        self.finish_idx = final_idx
        self.current_step = self.start_idx

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