# Fonctions qui permettent de pre process les outputs de l'environnement 
# Creation de différent wrappers autour de l'ouput de l'environnement 
# Convertir l'image en nuances de gris 
# Redefinission de la taille  de l'image 
# Frame Stacking 

import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


