

import gym
import gym.spaces
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import matplotlib.pyplot as plt
import cv2
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from gym.wrappers import GrayScaleObservation
from nes_py import NESEnv
import sys
import os
import torch
from env_wrapper import SkipFrame

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the MarioNetwork_Dqn class or function
from network.dqn import MarioNetwork_Dqn

from network.training import DQNAgent


# Initialize environment

env = gym.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=True)


env = DummyVecEnv([lambda: env])

env = VecFrameStack(env, 4,channels_order='first')




state = env.reset()  # Reset the environment again to get the grayscale observation
state, reward, done, info = env.step([0])  # Pass an integer action
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
n_action = env.action_space.n


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
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 10000,
          'batch_size': 64}

# Train agent
agent = DQNAgent(config, DQN)
scores = agent.train(env, 40000)
plt.plot(scores)
plt.show()
