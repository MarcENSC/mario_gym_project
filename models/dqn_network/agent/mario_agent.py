from collections import deque
from models.dqn_network.network.dqn import DQN
import numpy as np
import random as rd
import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.dqn_network.Buffer.Replay_buffer import ReplayBuffer
from models.dqn_network.agent.greedy_action import greedy_action
from copy import deepcopy

import json

class DQNAgent:
    def __init__(self, config, model):
        # Paramètres initiaux
        self.device = "cuda"
        print(f"Device: {self.device}")
        self.nb_actions = config.get('nb_actions')
        self.gamma = config.get('gamma')

        # Fréquence d'entraînement
        self.train_freq = config.get('train_freq')
        self.train_warmup = config.get('train_warmup')

        # Mémoire tampon (Replay Buffer)
        self.batch_size = config.get('batch_size')
        buffer_size = config.get('buffer_size', int(1e5))
        self.memory = ReplayBuffer(buffer_size, self.device)

        # Stratégie epsilon-greedy
        self.epsilon_max = config.get('epsilon_max')
        self.epsilon_min = config.get('epsilon_min')
        self.epsilon_stop = config.get('epsilon_decay_period')
        self.epsilon_delay = config.get('epsilon_delay_decay')
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

        # DQN et modèle cible
        self.model = model.to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

        # Perte et optimisation
        self.criterion =torch.nn.MSELoss()
        lr = config.get('learning_rate')
        self.optimizer =torch.optim.Adam(self.model.parameters(), lr=lr)

        self.nb_gradient_steps = config.get('gradient_steps')
        self.update_target_tau = config.get('update_target_tau')

        # Logs
        self.log_file = config.get('log_file', 'training_logs/training_logs.json')
        self.model_save_path = config.get('model_save_path', 'trained_model/trained_model.pth')
        self.logs = []

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Modèle sauvegardé dans {self.model_save_path}")

    def save_logs(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=4)
        print(f"Logs sauvegardés dans {self.log_file}")

    def gradient_step(self):
        if len(self.memory) < self.batch_size:
            return  # Pas assez d'expériences dans le buffer pour une étape de gradient

        
        self.optimizer.zero_grad()
        S, A, R, next_S, D = self.memory.sample(self.batch_size)
        S = S.to(self.device, dtype=torch.float32).squeeze().permute(0, 1, 2, 3)
        next_S = next_S.to(self.device, dtype=torch.float32).squeeze().permute(0, 1, 2, 3)
        A = A.to(self.device, dtype=torch.long)
        R = R.to(self.device, dtype=torch.float32)
        D = D.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            Q_next_S = self.target_model(next_S)
            Q_next_S_max = Q_next_S.max(1)[0].detach()
        
        td_objective = R + self.gamma * Q_next_S_max * (1 - D)
        Q_to_update = self.model(S).gather(1, A.to(torch.long).unsqueeze(1))
        loss = self.criterion(Q_to_update, td_objective.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state = env.reset()
        epsilon = self.epsilon_max
        step = 0
        last_mario_position = 0
        mario_bloque_compteur = 0
        max_step_mario_bloque = 25
        

        while episode < max_episode:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            if np.random.rand() < epsilon:
                
                action = env.action_space.sample()
            else:
                
                action = greedy_action(self.model, state)

            next_state, reward, done, info = env.step(action)
            current_pos = info['x_pos']

            # Si mario est bloqué, on arrête l'épisode et on lui inflige un malus de -50
            if abs(current_pos - last_mario_position) < 1:
                mario_bloque_compteur += 1
            else:
                mario_bloque_compteur = 0
                
            
            last_mario_position = current_pos

            if mario_bloque_compteur > max_step_mario_bloque:
                done = True
                reward += -10


           


            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # Entraînement

            if step > self.train_warmup and step % self.train_freq == 0:
                for _ in range(self.nb_gradient_steps):
                    self.gradient_step()

                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.copy_(self.update_target_tau * param.data + (1 - self.update_target_tau) * target_param.data)

            step += 1
            if done:
                state=env.reset()
                episode += 1
                log_entry = {
                    "episode": episode,
                    "steps": step,
                    "epsilon": epsilon,
                    "batch_size": len(self.memory),
                    "episode_return": episode_cum_reward
                }
                self.logs.append(log_entry)
                print(log_entry)

                # Sauvegarde périodique
                self.save_model()
                self.save_logs()

                state = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return
