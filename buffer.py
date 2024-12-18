import random
import torch
import numpy as np

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
        # Randomly sample a batch of experiences
        batch = random.sample(self.data, batch_size)

        # Separate the batch into individual components (states, actions, rewards, next states, dones)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Preprocess the states and next states before converting to tensors
        states = [self._preprocess(state) for state in states]
        next_states = [self._preprocess(next_state) for next_state in next_states]

        # Convert each component to a tensor and move to the appropriate device
        states = torch.Tensor(np.array(states)).to(self.device)
        actions = torch.Tensor(np.array(actions)).to(self.device).long()  # Ensure actions are long type for indexing
        rewards = torch.Tensor(np.array(rewards)).to(self.device)
        next_states = torch.Tensor(np.array(next_states)).to(self.device)
        dones = torch.Tensor(np.array(dones)).to(self.device)

        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.data)
    
    def _preprocess(self, state):
        # Ensure state has 3 channels (RGB), if it's grayscale, replicate the channels


        # Normalize pixel values to [0, 1]
        state = state.astype(np.float32) / 255.0  # Convert to float32 and normalize

        # Convert from (height, width, channels) to (channels, height, width)
         # Swap axes to get channels first

        return state

