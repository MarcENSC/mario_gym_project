import random
import torch
import numpy as np

class DQNAgent:
    def __init__(self, config, model):
        # Set all parameters
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95

        ## Train every N steps
        self.train_freq = config['train_freq'] if 'train_freq' in config.keys() else 1
        self.train_warmup = config['train_warmup'] if 'train_warmup' in config.keys() else 1

        ## Replay Buffer
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 64  # Fix batch size to 64
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, device)

        ## Epsilon-greedy strategy
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

        ## DQN and target DQN
        self.model = model
        self.target_model = deepcopy(self.model).to(device)

        ## Loss / learning rate / optimizer
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)

        ## Number of gradient steps to perform on each batch sampled from the replay buffer
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1

        ## Parameter to update the target DQN with a moving average (Polyak average)
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

    def gradient_step(self):
        if len(self.memory) >= self.batch_size:  # Ensure there are enough samples in the buffer
            self.optimizer.zero_grad()
            S, A, R, next_S, D = self.memory.sample(self.batch_size)  # Get a fixed batch size of 64

            with torch.no_grad():
                Q_next_S_max = self.target_model(next_S).max(1)[0].detach()

            td_objective = R + self.gamma * Q_next_S_max * (1 - D)
            Q_to_update = self.model(S).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(Q_to_update, td_objective.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            print(f"Loss: {loss.item()}")  # Debug statement

class DQNAgent:
    def __init__(self, config, model):
        # Set all parameters
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95

        ## Train every N steps
        self.train_freq = config['train_freq'] if 'train_freq' in config.keys() else 1
        self.train_warmup = config['train_warmup'] if 'train_warmup' in config.keys() else 1

        ## Replay Buffer
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 64  # Fix batch size to 64
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, device)

        ## Epsilon-greedy strategy
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

        ## DQN and target DQN
        self.model = model
        self.target_model = deepcopy(self.model).to(device)

        ## Loss / learning rate / optimizer
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)

        ## Number of gradient steps to perform on each batch sampled from the replay buffer
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1

        ## Parameter to update the target DQN with a moving average (Polyak average)
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

    def gradient_step(self):
        if len(self.memory) >= self.batch_size:  # Ensure there are enough samples in the buffer
            self.optimizer.zero_grad()
            S, A, R, next_S, D = self.memory.sample(self.batch_size)  # Get a fixed batch size of 64

            with torch.no_grad():
                Q_next_S_max = self.target_model(next_S).max(1)[0].detach()

            td_objective = R + self.gamma * Q_next_S_max * (1 - D)
            Q_to_update = self.model(S).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(Q_to_update, td_objective.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            print(f"Loss: {loss.item()}")  # Debug statement

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity  # capacity of the buffer
        self.data = []
        self.index = 0  # index of the next cell to be filled
        self.device = device

    def append(self, s, a, r, s_, d):
        # Preprocess states before appending to the buffer
        s = self._preprocess(s)
        s_ = self._preprocess(s_)
        
        # Add new experience to the buffer
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        # Randomly sample a batch of experiences (batch size fixed at 64)
        batch = random.sample(self.data, batch_size)
    
        # Separate the batch into individual components (states, actions, rewards, next states, dones)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        # Preprocess the states and next states before converting to tensors
        states = np.array([self._preprocess(state) for state in states])
        next_states = np.array([self._preprocess(next_state) for next_state in next_states])
    
        # Convert each component to a tensor and move to the appropriate device
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device).long()  # Ensure actions are long type for indexing
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)
    
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.data)
    
    def _preprocess(self, state):
        # Ensure state has 3 channels (RGB), if it's grayscale, replicate the channels
        state = state.astype(np.float32) / 255.0  # Convert to float32 and normalize
        return state

