import gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from env import SkipFrame
from env import ResizeEnv
from env import GrayScaleObservation
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.mario_test import MarioTester
if __name__ == "__main__":
    


    # Path to the saved model
    model_path = 'mario_dqn_model.pth'

    # Create the tester
    tester = MarioTester(model_path, env)

    # Test the agent
    episode_returns = tester.test(max_episodes=10)

    # Print the average return over the test episodes
    print(f"Average Return: {np.mean(episode_returns)}")