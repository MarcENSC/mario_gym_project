import gym 
import gym_super_mario_bros
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from env_wrappers.skip_frame_wrapper import SkipFrame



env = gym.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
