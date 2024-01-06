import gym
from gym import spaces
import numpy as np
import pandas as pd

class ContinuousOHLCVEnv(gym.Env):
    def __init__(self, ohlcv_data, initial_cash=1000):
        self.ohlcv_raw_data = ohlcv_data
        self.initial_cash = initial_cash
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,))
        self.max_idx = ohlcv_data.shape[0] - 1
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash_in_hand = self.initial_cash
        self.stock_holding = 0
        self.step_info = []  # Initialize an empty list to store step information
        self.stock_price = self.ohlcv_raw_data[self.current_step][3]  # Assuming closing price for stock price
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        self.available_actions = (0, 1)  # Can hold or buy initially
        return self.get_observation()
    
    def step(self, action):
        assert self.action_space.contains(action), f'Action {action} not in action space'

        prev_valuation = self.total_portfolio_value

        if action == 1:  # Buy
            self._buy()
        elif action == 2:  # Sell
            self._sell()

        # Update portfolio value
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
        reward = self.total_portfolio_value - prev_valuation
        done = self.current_step >= (len(self.ohlcv_raw_data) - 1)

        step_data = {
            'Step': self.current_step,
            'Portfolio Value': np.round(self.total_portfolio_value, 2),
            'Cash': np.round(self.cash_in_hand, 2),
            'Stock Value': np.round(self.stock_price * self.stock_holding, 2), 
            'Stock Holdings': np.round(self.stock_holding, 0),
            'Stock Price': np.round(self.stock_price, 2),
            'Available Actions': self.available_actions,
            'Action': action
        }
        self.step_info.append(step_data)

        if not done:
            self.current_step += 1
            self.stock_price = self.ohlcv_raw_data[self.current_step][3]

        # Update available actions
        if self.stock_holding > 0:
            self.available_actions = (0, 2)  # Can hold or sell
        else:
            self.available_actions = (0, 1)  # Can hold or buy

        next_observation = self.get_observation()
        info = {'available_actions': self.available_actions}
        return next_observation, reward, done, info
    
    def _buy(self):
        self.num_stocks_buy = np.floor(self.cash_in_hand/self.stock_price) # Buy 
        #Maximum allowed (Current Method)
        self.cash_in_hand -= self.num_stocks_buy * self.stock_price
        self.stock_holding = self.num_stocks_buy
        self.num_stocks_buy = 0
        self.available_actions = (-1,0)
        return
    
    def _sell(self):
        self.num_stocks_sell = self.stock_holding # Sell all stocks (Current Mehtod)
        self.cash_in_hand += self.num_stocks_sell * self.stock_price  # No commission
        #fee can be added later
        self.stock_holding -= self.num_stocks_sell
        self.num_stocks_sell = 0
        self.available_actions = (0,1)
        return
 
    def get_observation(self):
        return(tuple(self.ohlcv_raw_data[self.current_step]))

    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored 
    #step information

####testing##################
    
ohlcv_data = np.random.rand(100, 5)  # 100 timesteps, 5 features (OHLCV)

# Initialize your environment
env = ContinuousOHLCVEnv(ohlcv_data)

# Number of episodes for testing
num_episodes = 10

for episode in range(num_episodes):
    observation = env.reset()
    done = False
    while not done:
        # Random action
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # You can print observations here to see how the environment changes at each step
        print(f"Episode: {episode + 1}, Step: {env.current_step}, Action: {action}, Reward: {reward}, Done: {done}")

        if done:
            break

# After testing, you can analyze the step data
step_data_df = env.get_step_data()
print(step_data_df.head())  # Display the first few rows of the step data

import matplotlib.pyplot as plt
# Generating some mock data for demonstration
# This would be replaced by your actual step data
np.random.seed(0)
step_data = {
    "Step": np.arange(100),
    "Portfolio Value": np.random.uniform(800, 1200, 100),
    "Cash": np.random.uniform(100, 500, 100),
    "Stock Value": np.random.uniform(300, 700, 100),
    "Stock Holdings": np.random.randint(0, 10, 100),
    "Stock Price": np.random.uniform(10, 50, 100),
}
step_data_df = pd.DataFrame(step_data)

# Plotting
plt.figure(figsize=(12, 6))

# Portfolio Value
plt.subplot(2, 2, 1)
plt.plot(step_data_df["Step"], step_data_df["Portfolio Value"], label="Portfolio Value")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Portfolio Value Over Steps")
plt.legend()

# Cash and Stock Value
plt.subplot(2, 2, 2)
plt.plot(step_data_df["Step"], step_data_df["Cash"], label="Cash")
plt.plot(step_data_df["Step"], step_data_df["Stock Value"], label="Stock Value")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Cash and Stock Value Over Steps")
plt.legend()

# Stock Holdings
plt.subplot(2, 2, 3)
plt.bar(step_data_df["Step"], step_data_df["Stock Holdings"])
plt.xlabel("Step")
plt.ylabel("Number of Stocks")
plt.title("Stock Holdings Over Steps")

# Stock Price
plt.subplot(2, 2, 4)
plt.plot(step_data_df["Step"], step_data_df["Stock Price"])
plt.xlabel("Step")
plt.ylabel("Price")
plt.title("Stock Price Over Steps")

plt.tight_layout()
plt.show()

################################################################################

class DiscretizedOHLCVEnv(gym.Env):
       
    def __init__(self, ohlcv_data:pd.DataFrame,bins_per_feature:list,initial_cash=1000):
        self.ohlcv_raw_data = ohlcv_data
        self.initial_cash = initial_cash
        self.bins_per_feature = bins_per_feature
        self.actions = ('S','H','B') # Sell, Hold, Buy
        self.action_space = spaces.Discrete(3)
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
        self.available_actions = ('H','B')
    
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

       
        if action == 'S': # Sell
            self._sell()

        elif action == 'H': # Hold
            pass
   
        elif action == "B": # Buy
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
                self.available_actions = ('S',)
            else:
                self.available_actions = ('H',)

        

        next_observation = self.get_observation()

        return next_observation, reward, done
    
    def _buy(self):
        self.num_stocks_buy = np.floor(self.cash_in_hand/self.stock_price) # Buy Maximum allowed (Current Method)
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