import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import GrayScaleObservation wrapper
from gym.wrappers import GrayScaleObservation

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
        obs, _ = env.reset()  # Get the first observation and info
        (oldh, oldw, oldc) = obs.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame    


# Initialize environment
env = gym.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Get initial state
state, _ = env.reset()  # Extract the observation from the reset output
print("RGB scale : ", state.shape)

# Convert to grayscale
env = GrayScaleObservation(env, keep_dim=True)
state, _ = env.reset()  # Reset the environment again to get the grayscale observation
print("Gray scale:", state.shape)
