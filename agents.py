import numpy as np
from collections import deque, defaultdict
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import pandas as pd
from stockenv import DiscretizedOHLCVEnv
import torch
import torch.nn as nn



class DQNAgent02:
    def __init__(self, input_shape, n_actions):
        super(DQNAgent02, self).init()
    
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size,512),
            nn.ReLU(),
            nn.Linear(512,n_actions)
        )

        def _get_conv_out(self, shape):
            o = self.conv(toruch.zeros(1, *shape))
            return int(np.prod(o.size))
        
        def forwar(self, x):
            conv_out = self.conv(x).view(x.size())
        

class DQNAgent01:
    def __init__(self, state_size, action_size,available_actions):
        self.state_size = state_size
        self.action_size = action_size
        self.available_actions = available_actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 0.5  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action_type = 'Exploration'
            return random.choice(self.available_actions), action_type
        action_type = 'Expliotation'
        act_values = self.model.predict(state)
        possible_act_value_idx = [act - 1 for act in self.available_actions]
        max_value = np.max(act_values[possible_act_value_idx])
        max_indices = [i for i, q_value in enumerate(act_values) if abs(q_value - max_value) < 1e-8]
        valid_max_indices = list(set(possible_act_value_idx).intersection(set(max_indices)))
        print(act_values,valid_max_indices)
        return np.argmax(act_values[0]), action_type

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Discrete_QtabAgent:
    def __init__(self,ohlcv_data:pd.DataFrame,
                 bins_per_feature:list,
                 training_idxs:list,
                 num_training_episodes:int,
                 testing_idxs:list,
                 epsilon: float = 0.1,
                 alpha: float = 0.1,
                 gamma: float = 0.9) -> None:
        
        
        # Agent Parameters
        self.epsilon = epsilon      ## Eplsion-greed parameter for exploration vs. exploitation
        
        self.alpha = alpha        ## learning rate
        self.gamma = gamma         ## Discount factor for future rewards

        # Training Parameters
        self.training_range = training_idxs
        self.num_training_episodes = num_training_episodes
        
        # Testing Parameters
        self.test_range = testing_idxs

        
        self.state = None
        self.values = defaultdict(float)

        #Initialize Enviroment. 
        self.env = DiscretizedOHLCVEnv(ohlcv_data[["open","high","low",'close',"volume"]].to_numpy(), 
                                         bins_per_feature)

    def sample_env(self,initial_epsilon, final_epsilon, training_episode):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self._epsilon_decay(training_episode,
                                                  self.num_training_episodes,
                                                  initial_epsilon,
                                                  final_epsilon)                                                  : # Explore
            action = random.choice(self.env.available_actions)
        else:
            _ , action = self.best_value_and_action(self.state)

        old_state = self.state
        new_state, reward, is_done = self.env.step(action)
        
        if is_done:
            self.env.reset()
            self.state = None
        else:
            self.state = new_state
        return old_state, action, reward, new_state
    
    def best_value_and_action(self,state):
        best_value, best_action = None, None
        for action in self.env.available_actions:
            action_value = self.values[(state,action)]
            if best_action is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value,best_action
    
    def value_update(self, state, action, reward, next_state):
        best_value, _ =  self.best_value_and_action(next_state)
        new_value = reward + self.gamma * best_value
        old_value = self.values[(state,action)]
        self.values[(state,action)] = old_value * (1-self.alpha) + new_value * self.alpha
    
    
    def train(self, start_idx, end_idx,initial_epsilon: float = None, 
              final_epsilon: float = None):
        
        if initial_epsilon is None:
            initial_epsilon = self.epsilon
        if final_epsilon is None:
            final_epsilon = self.epsilon

        training_episode = 1
                
        for _ in range(self.num_training_episodes):

            self._play_episode(start_idx,end_idx,initial_epsilon,final_epsilon,training_episode)
            training_episode += 1


    def test(self, start_idx, end_idx, test_epsilon: float = None):
        if test_epsilon is None:
            initial_epsilon = self.epsilon
            final_epsilon = self.epsilon
        else:
            initial_epsilon = test_epsilon
            final_epsilon = test_epsilon
            
        self._play_episode(start_idx,end_idx,initial_epsilon,final_epsilon,1)
  

    def _play_episode(self,start_idx,final_idx, initial_epsilon, final_epsilon, training_episode):
        total_reward = 0.0

        self.env.reset()
        self.env.current_step = start_idx
        self.env.max_idx = final_idx
        state = self.env.get_observation()

               
        is_done = False
        while not is_done:
            state, action, reward, new_state = self.sample_env(initial_epsilon, 
                                                               final_epsilon, 
                                                               training_episode)
            _, action = self.best_value_and_action(state)
            self.value_update(state, action, reward, new_state)
            total_reward += reward
            if self.env.current_step == self.env.max_idx:
                is_done = True
            state = new_state
        return total_reward 




    def _epsilon_decay(self,current_epoch, total_epochs, initial_epsilon, final_epsilon):
        decay_rate = np.log(final_epsilon / initial_epsilon) / total_epochs
        epsilon = initial_epsilon * np.exp(-decay_rate * current_epoch)
        return epsilon
    
class Discrete_Buy_Hold_Agent:
        
        def __init__(self,ohlcv_data:pd.DataFrame,
                     bins_per_feature:list = None,
                     testing_idxs:list = None) -> None:
        
            # Agent Paramters 
            self.state = None
            
            # Testing Parameters
            self.test_range = testing_idxs

            #Initialize Enviroment. 
            self.env = DiscretizedOHLCVEnv(ohlcv_data[["open","high","low",'close',"volume"]].to_numpy(), 
                                         bins_per_feature)
            
        
        def sample_env(self,start_idx,final_idx):
            # Choose action using epsilon-greedy policy
            
            if self.env.current_step == start_idx:
                action = 'B'
            elif self.env.current_step != final_idx-1:
                action = 'H'
            else:
                action = 'S'

            old_state = self.state
            new_state, reward, is_done = self.env.step(action)
            
            if is_done:
                self.env.reset()
                self.state = None
            else:
                self.state = new_state
            return old_state, action, reward, new_state
        
        def test(self, start_idx, end_idx):
            self.env.current_step = start_idx  
            self._play_episode(start_idx,end_idx)
  

        def _play_episode(self,start_idx,final_idx):
            total_reward = 0.0

            self.env.reset()
            self.env.current_step = start_idx
            self.env.max_idx = final_idx
            state = self.env.get_observation()

                
            is_done = False
            while not is_done:
                _, _, reward, _ = self.sample_env(start_idx,final_idx)
                total_reward += reward
                if self.env.current_step == self.env.max_idx:
                    is_done = True

            return total_reward 

class Discrete_Random_Agent:
        
        def __init__(self,ohlcv_data:pd.DataFrame,
                     bins_per_feature:list = None,
                     testing_idxs:list = None) -> None:
        
            # Agent Paramters 
            self.state = None
            
            # Testing Parameters
            self.test_range = testing_idxs

            #Initialize Enviroment. 
            self.env = DiscretizedOHLCVEnv(ohlcv_data[["open","high","low",'close',"volume"]].to_numpy(), 
                                         bins_per_feature)
            
        
        def sample_env(self,start_idx,final_idx):
            # Choose action using epsilon-greedy policy

            action = random.choice(self.env.available_actions)

            old_state = self.state
            new_state, reward, is_done = self.env.step(action)
            
            if is_done:
                self.env.reset()
                self.state = None
            else:
                self.state = new_state
            return old_state, action, reward, new_state
        
        def test(self, start_idx, end_idx):
            self.env.current_step = start_idx  
            self._play_episode(start_idx,end_idx)
  

        def _play_episode(self,start_idx,final_idx):
            total_reward = 0.0

            self.env.reset()
            self.env.current_step = start_idx
            self.env.max_idx = final_idx
            state = self.env.get_observation()

                
            is_done = False
            while not is_done:
                _, _, reward, _ = self.sample_env(start_idx,final_idx)
                total_reward += reward
                if self.env.current_step == self.env.max_idx:
                    is_done = True

            return total_reward 


