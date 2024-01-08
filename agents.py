import numpy as np
from collections import deque, defaultdict, namedtuple
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.optimizers import Adam
import random
import pandas as pd
from stockenv import DiscretizedOHLCVEnv, ContinuousOHLCVEnv
#import torch
#import torch.nn as nn




"""class DQNAgent02:
    def __init__(self, 
                 ohlcv_data: pd.DataFrame, 
                 training_idxs:list,
                 num_training_episodes:int,
                 testing_idxs:list,
                 n_actions: int = 3,
                 buffer_size: int = 1000,
                 batch_size: int = 10,
                 epsilon: float = 0.1,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 device: str = 'cpu') -> None:

        # Agent Parameters
        self.epsilon = epsilon    ## Eplsion-greed parameter for exploration vs. exploitation
        self.alpha = alpha        ## learning rate
        self.gamma = gamma        ## Discount factor for future rewards
        self.state = None

        # Training Parameters
        self.training_range = training_idxs
        self.num_training_episodes = num_training_episodes

        # Testing Parameters
        self.test_range = testing_idxs

        # Buffer 
        class ExperienceBuffer:
            def __init__(self,capacity: int) -> None:
                self.buffer = deque(maxlen=capacity)
            
            def __len__(self):
                return len(self.buffer)
            
            def append(self,experience):
                self.buffer.append(experience)
            
            def sample(self, batch_size):
                indices = random.choice(len(self.buffer), batch_size,
                                            replace=False)
                states, actions, rewards, dones, next_states = \
                    zip(*[self.buffer[idx] for idx in indices])
                return np.array(states), np.array(actions), \
                        np.array(rewards, dtype=np.float32), \
                        np.array(dones, dtype=np.uint8), \
                        np.array(next_states)
        
        Experience = namedtuple('Experience', field_names=['state','action',
                                                           'reward','done',
                                                           'new_state'])

        self.exp_buffer = ExperienceBuffer(buffer_size)
        self.batch_size = batch_size #added to keep line 138 going                           
        # Machine Parameters:
        self.device = device    ## Device which will compute tensors 
        
        # Agent NN
        self.input_shape = ohlcv_data.shape[1]

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_shape, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(self.input_shape)

        self.fc = nn.Sequential(
                nn.Linear(conv_out_size,512),
                nn.ReLU(),
                nn.Linear(512,n_actions)
            )

        def _get_conv_out(self, shape):
            o = self.conv(torch.zeros(1, *shape))
            return int(np.prod(o.size))
            
        def forward(self, x):
            conv_out = self.conv(x).view(x.size())
        
        # Initialize Enviroment. 
        self.env = ContinuousOHLCVEnv(
            ohlcv_data[["open","high","low",'close',"volume"]].to_numpy()) 

    

        @torch.no_grad()
        def _play_step(self,initial_epsilon, 
                       final_epsilon, 
                       training_episode, 
                       device):
            # Choose action using epsilon-greedy policy
            if np.random.rand() < self._epsilon_decay(training_episode,
                                                    self.num_training_episodes,
                                                    initial_epsilon,
                                                    final_epsilon)                                                  : # Explore
                action = random.choice(self.env.available_actions) ## letter action
            else:
                state_a = np.array([self.state], copy=False)
                state_v = torch.tensor(state_a).to(device)
                q_vals_v = net(state_v) ## to figure out net
                _, act_v = torch.max(q_vals_v, dim=1) ## need to check this
                action = self.env.idx_to_action_dic[int(act_v.item())] #need to check if assigned properly


            old_state = self.state
            new_state, reward, is_done = self.env.step(action) ## not clear that letter action is correct for replay

            exp = Experience(self.state, action, reward, is_done, new_state)

            self.exp_buffer.append(exp)
            self.state = new_state
            
            if is_done:
                self.env.reset()
                self.state = None
            else:
                self.state = new_state
            return old_state, action, reward, new_state
        
        def calc_loss(batch, net, tgt_net, device, states, actions, 
                      rewards, dones, next_states = self.batch_size): #added self.batch_size to keep going
            states_v = torch.tensor(np.array(states, copy=False)).to(device)
            next_states_v = torch.tensor(np.array(next_states,
                                         copy=False)).to(device)
            actions_v = torch.tensor(actions).to(device)
            rewards_v = torch.tensor(rewards).to(device)
            done_mask = torch.BoolTensor(dones).to(device)

            state_action_values = net(states_v).gather(\
                1, actions_v.unsqueeze(-1)).squeez(-1) ##see page 151
            
            next_state_values = tgt_net(next_states_v).max(1)[0]

            next_state_values[done_mask] = 0.0 ## Needed for convergence 

            next_state_values = next_state_values.detach() ## prevents gradients flowing into NN

            expected_state_action_values = next_state_values * self.gamma + \
                rewards_v

            return nn.MSELoss()(state_action_values,expected_state_action_values)

        def _epsilon_decay(self,current_epoch, total_epochs, initial_epsilon, final_epsilon):
            decay_rate = np.log(final_epsilon / initial_epsilon) / total_epochs
            epsilon = initial_epsilon * np.exp(-decay_rate * current_epoch)
            return epsilon"""

       
  

class Discrete_QtabAgent:
    def __init__(self,ohlcv_data:pd.DataFrame,
                 bins_per_feature:list,
                 bin_padding:float,
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
                                         bins_per_feature,bin_padding)

    def sample_env(self,initial_epsilon, final_epsilon, training_episode):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < self._linear_decay(initial_epsilon,
                                                  final_epsilon,
                                                  training_episode,
                                                  self.num_training_episodes)                                                  : # Explore
            action = random.choice(self.env.available_actions)
        else:
            _ , action = self.best_value_and_action(self.state)

        old_state = self.state
        new_state, reward, is_done = self.env.step(action)
        
        return old_state, action, reward, new_state, is_done
    
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
    
    
    def train(self, start_idx:int, 
              end_idx:int,
              initial_epsilon: float = None, 
              final_epsilon: float = None):
        
        if initial_epsilon is None:
            init_epsilon = self.epsilon
        else:
            init_epsilon = initial_epsilon
        
        if final_epsilon is None:
            end_epsilon = self.epsilon        
        else:
            end_epsilon = final_epsilon

        self.env.update_idx(start_idx,end_idx-1)
        
                
        for training_episode in range(1, self.num_training_episodes+1):
            
            self._play_episode(init_epsilon, end_epsilon,training_episode)
            
           

    def test(self, start_idx, end_idx, test_epsilon: float = None):
        if test_epsilon is None:
            initial_epsilon = self.epsilon
            final_epsilon = self.epsilon
        else:
            initial_epsilon = test_epsilon
            final_epsilon = test_epsilon

        self.env.update_idx(start_idx,end_idx-1)
        
        self._play_episode(initial_epsilon,final_epsilon,1)
  

    def _play_episode(self,initial_epsilon, final_epsilon,training_episode):
        total_reward = 0.0

        self.env.reset()
        
        state = self.env.get_observation()  
        is_done = False
        current_run = 0
        while not is_done:
            
            state, action, reward, new_state, end = self.sample_env(initial_epsilon, 
                                                               final_epsilon, 
                                                               training_episode)
            _, action = self.best_value_and_action(state)
            self.value_update(state, action, reward, new_state)
            total_reward += reward
            is_done = end
            current_run += 1

            state = new_state
            
        return total_reward 


    def reset(self):
        
        self.state = None
        self.values = defaultdict(float)

    def _linear_decay(self, initial_epsilon, final_epsilon, current_epoch, total_epochs):
        if initial_epsilon == final_epsilon:
            return initial_epsilon
        else:
            rate_of_change = (final_epsilon - initial_epsilon) / (total_epochs-1)
            current_epsilon = np.round((initial_epsilon - rate_of_change) + (rate_of_change * current_epoch),3)
            
            if current_epsilon > initial_epsilon or current_epsilon < final_epsilon:
                raise ValueError(f'Epsilon value ({current_epsilon}) out of valid range ({initial_epsilon}:{final_epsilon})')
        
            return current_epsilon

    def _epsilon_decay(self, initial_epsilon, final_epsilon, current_epoch, total_epochs):
        if initial_epsilon == final_epsilon:
            return initial_epsilon
        else:
            print(f'init:{initial_epsilon}, final:{final_epsilon}, c:{current_epoch}, f:{total_epochs}' )
            decay_rate = np.log((initial_epsilon / final_epsilon) * (current_epoch / total_epochs))
            epsilon = np.exp(-decay_rate)

            print(epsilon)
            if epsilon > initial_epsilon or epsilon < final_epsilon:
                raise ValueError("Epsilon value out of valid range")
            
            return epsilon
    
class Discrete_Buy_Hold_Agent:
        
        def __init__(self,ohlcv_data:pd.DataFrame,
                     bins_per_feature:list,
                     bin_padding:float,
                     training_idxs:list = None,
                     testing_idxs:list = None) -> None:
        
            # Agent Paramters 
            self.state = None

            #Training Parameters 
            self.training_range = training_idxs   ## Not used but needed for loop in test_bench

            # Testing Parameters
            self.test_range = testing_idxs

            #Initialize Enviroment. 
            self.env = DiscretizedOHLCVEnv(ohlcv_data[["open","high","low",'close',"volume"]].to_numpy(), 
                                         bins_per_feature,bin_padding)
            
        
        def sample_env(self,start_idx, final_idx):
            # Choose action using epsilon-greedy policy
            
            if self.env.current_step == start_idx:
                action = 'B'
            elif self.env.current_step == final_idx-1:
                action = 'S'
            else:
                action = 'H'

            old_state = self.state
            new_state, reward, is_done = self.env.step(action)
            
            return old_state, action, reward, new_state, is_done
        
        def test(self, start_idx, end_idx):
            
            self._play_episode(start_idx,end_idx)
  

        def _play_episode(self,start_idx,final_idx):
            total_reward = 0.0

            self.env.reset()
            self.env.update_idx(start_idx,final_idx-1)
            state = self.env.get_observation()  
                            
            is_done = False
            while not is_done:
                _, _, reward, _, end = self.sample_env(start_idx,final_idx)
                total_reward += reward
                is_done = end
        
            return total_reward 
        
        def reset(self):
            self.state = None

class Discrete_Random_Agent:
        
        def __init__(self,ohlcv_data:pd.DataFrame,
                     bins_per_feature:list,
                     bin_padding:float,
                     training_idxs:list = None,
                     testing_idxs:list = None) -> None:
        
            # Agent Paramters 
            self.state = None
            
            # Training Parameters
            self.training_range = training_idxs   ## Not used but needed for loop in test_bench

            # Testing Parameters
            self.test_range = testing_idxs

            #Initialize Enviroment. 
            self.env = DiscretizedOHLCVEnv(ohlcv_data[["open","high","low",'close',"volume"]].to_numpy(), 
                                         bins_per_feature,bin_padding)
            
        
        def sample_env(self):
            # Choose action using epsilon-greedy policy

            action = random.choice(self.env.available_actions)

            old_state = self.state
            new_state, reward, is_done = self.env.step(action)
            
            return old_state, action, reward, new_state, is_done
        
        def test(self, start_idx, end_idx):
            
            self._play_episode(start_idx,end_idx)
  

        def _play_episode(self,start_idx,final_idx):
            total_reward = 0.0

            self.env.reset()
            self.env.update_idx(start_idx,final_idx-1)
            state = self.env.get_observation()  
                            
            is_done = False
            while not is_done:
                _, _, reward, _, end = self.sample_env()
                total_reward += reward
                is_done = end
        
            return total_reward 
        
        def reset(self):
            self.state = None



