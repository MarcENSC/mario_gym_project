import gym 
import gym_super_mario_bros
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, RecordVideo
from gym.wrappers import FrameStack
from models.dqn_network.network.dqn import DQN 
from models.dqn_network.agent.mario_agent import DQNAgent
import torch.nn as nn
import torch 
import matplotlib.pyplot as plt

# INITIALISATION DE L'ENVIRONNEMENT 

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")


env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Apply Wrappers to the environment
# 1. Resize dimension of dimension to 84x84;(240, 256, 3) -> (84, 84, 3)
env = ResizeObservation(env, shape=84)
# 2. Convert RGB image to Grayscale, dimension before : (84, 84, 3) -> (84, 84, 1)
env = GrayScaleObservation(env, keep_dim=True)
# 3. Skip 4 frames to speed up the training process :(84, 84, 1) -> (4, 84, 84, 1)
env = FrameStack(env, num_stack=4)
env = RecordVideo(env, video_folder="video/",episode_trigger=lambda x: x % 500 == 0 or x==2000 -1 or x==0, name_prefix="rl-video")


env.reset()
env.seed(0)
env.render()
state_dim = env.observation_space.shape
next_state, reward, done,info = env.step(action=0)
print(f"{state_dim},\n{next_state.shape},\n {reward},\n {done},\n {info}")


# Declare network


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0]
n_action = env.action_space.n
nb_neurons=24

DQN = DQN()

# DQN config
config = {'nb_actions': env.action_space.n,
          'train_warmup': 10000,
          'train_freq': 10,
          'gradient_steps': 1,
          'learning_rate': 0.00025,
          'gamma':0.99,
          'buffer_size': 100000,
          'epsilon_min': 0.1,
          'epsilon_max': 1,
          'epsilon_decay_period': 1000000,
          'epsilon_delay_decay': 10000,
          'update_target_tau': 0.001,
          'batch_size': 64}

# Train agent
agent = DQNAgent(config, DQN)
scores = agent.train(env, 2000)
plt.plot(scores)
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.savefig("Scores_Training/score.png")
plt.show()

env.close() 