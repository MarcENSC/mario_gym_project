import torch
import numpy as np
from network.dqn import greedy_action
from network.dqn import MarioNetwork_Dqn
import gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class MarioTester:
    def __init__(self, model_path, env):
        self.model = MarioNetwork_Dqn()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval
        self.env = env

    def test(self, max_episodes=10):
        episode_returns = []
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_cum_reward = 0
            done = False
            while not done:
                print(f"STATE SHAPE IN MARIO TESTER : {state.shape}")
                action = greedy_action(self.model, state)
                next_state, reward, done, _ = self.env.step(action)
                episode_cum_reward += reward
                state = next_state
                self.env.render(mode="human")  # Render the environment to visualize the agent's actions

            episode_returns.append(episode_cum_reward)
            print(f"Episode {episode + 1}, Return: {episode_cum_reward}")

        return episode_returns

    def greedy_action(network, state):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            # Faire une copie du tableau NumPy avant de le convertir en tenseur PyTorch
            state_copy = state.copy()

            # Vérifier la forme de l'état
            print(f"State shape: {state_copy.shape}")

            # Prétraiter l'état si nécessaire
            if len(state_copy.shape) == 3:
                state_copy = np.expand_dims(state_copy, axis=0)  # Ajouter une dimension pour le batch
            if state_copy.shape[1] != 1:
                state_copy = np.mean(state_copy, axis=1, keepdims=True)  # Convertir en niveaux de gris

            # Assurez-vous que l'état a la forme attendue par le modèle
            if state_copy.shape[1] != 84 or state_copy.shape[2] != 84:
                state_copy = np.resize(state_copy, (1, 84, 84, 1))  # Redimensionner l'état

            Q = network(torch.Tensor(state_copy).to(device))
            return torch.argmax(Q).item()
