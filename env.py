import gym
import gym.spaces
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import matplotlib.pyplot as plt
import cv2
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from gym.wrappers import GrayScaleObservation
from nes_py import NESEnv
import sys
import os
import torch

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the MarioNetwork_Dqn class or function
from network.dqn import MarioNetwork_Dqn
from network.dqn import greedy_action
from network.training import DQNAgent

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        # Get the shape after resetting the environment
        obs = env.reset()  # Get the first observation and info
        (oldh, oldw, oldc) = obs.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        return frame

# Initialize environment
env = gym.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=True)
env = ResizeEnv(env, size=84)

def display_frame(state):
    plt.figure(figsize=(8, 8))
    plt.imshow(state[:, :, 0], cmap='gray')
    plt.show()

state = env.reset()  # Reset the environment again to get the grayscale observation
state, reward, done, info = env.step(0)  # Pass an integer action
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
n_action = env.action_space.n
nb_neurons = 24

DQN = MarioNetwork_Dqn()

# DQN config
config = {'nb_actions': env.action_space.n,
          'train_warmup': 10000,
          'train_freq': 10,
          'gradient_steps': 1,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 100000,
          'epsilon_min': 0.1,
          'epsilon_max': 1.,
          'epsilon_decay_period': 1000000,
          'epsilon_delay_decay': 10000,
          'batch_size': 64}

# Train agent
agent = DQNAgent(config, DQN)
scores = agent.train(env, 10000)
plt.plot(scores)
plt.show()
