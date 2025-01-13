import random
import torch
import numpy as np
import os
from typing import Tuple, List, Any
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class ReplayBuffer:
    def __init__(self, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.data = []
        self.index = 0
        print(f"Initializing ReplayBuffer with capacity {capacity} on device: {device}")

    def append(self, state, action, reward: float, next_state, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state observation (can be numpy array or torch tensor)
            action: Action taken
            reward: Reward received
            next_state: Next state observation (can be numpy array or torch tensor)
            done: Whether episode ended
        """
        try:
            # Convert reward and done to proper types
            reward = float(reward)
            done = bool(done)
            
            transition = (state, action, reward, next_state, done)
            
            if len(self.data) < self.capacity:
                self.data.append(transition)
            else:
                self.data[self.index] = transition
                
            self.index = (self.index + 1) % self.capacity
            
        except Exception as e:
            print(f"Error adding transition to buffer: {str(e)}")
            raise

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.
        """
        if len(self.data) < batch_size:
            raise ValueError(f"Buffer contains {len(self.data)} transitions, "
                           f"cannot sample batch of {batch_size}")
        
        try:
            batch = random.sample(self.data, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors if not already
            if not isinstance(states[0], torch.Tensor):
                states = torch.tensor(np.array(states), dtype=torch.float32)
            else:
                states = torch.stack(states)

            if not isinstance(next_states[0], torch.Tensor):
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            else:
                next_states = torch.stack(next_states)

            # Convert other elements to tensors
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Move everything to the correct device
            states = states.to(self.device)
            next_states = next_states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            
            return states, actions, rewards, next_states, dones
            
        except Exception as e:
            print(f"Error sampling from buffer: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.data)