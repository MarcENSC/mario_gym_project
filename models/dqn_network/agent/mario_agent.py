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

class DQNAgent:
    def __init__(self, config, model):
        # Set all parameters
        self.device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config.get('nb_actions', 4)
        self.gamma = config.get('gamma', 0.95)

        ## Train every N steps
        self.train_freq = config.get('train_freq', 1)
        self.train_warmup = config.get('train_warmup', 1)

        ## Replay Buffer
        self.batch_size = config.get('batch_size', 100)
        buffer_size = config.get('buffer_size', int(1e5))
        self.memory = ReplayBuffer(buffer_size, self.device)

        ## Epsilon-greedy strategy
        self.epsilon_max = config.get('epsilon_max', 1.)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_stop = config.get('epsilon_decay_period', 1000)
        self.epsilon_delay = config.get('epsilon_delay_decay', 20)
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

        ## DQN and target DQN
        self.model = model
        self.target_model = deepcopy(self.model).to(self.device)

        ## Loss / learning rate / optimizer
        self.criterion = config.get('criterion', torch.nn.MSELoss())
        lr = config.get('learning_rate', 0.001)
        self.optimizer = config.get('optimizer', torch.optim.Adam(self.model.parameters(), lr=lr))

        ## Number of gradient steps to perform on each batch sampled from the replay buffer
        self.nb_gradient_steps = config.get('gradient_steps', 1)

        ## Parameter to update the target DQN with a moving average (Polyak average)
        self.update_target_tau = config.get('update_target_tau', 0.005)

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            self.optimizer.zero_grad()
            S, A, R, next_S, D = self.memory.sample(self.batch_size)

            # Ensure the states are in the correct format
            S = S.float().to(self.device).squeeze().permute(0, 1, 2, 3)
            next_S = next_S.float().to(self.device).squeeze().permute(0, 1, 2, 3)
            A = A.to(self.device)
            R = R.to(self.device)
            D = D.to(self.device)

            with torch.no_grad():
                Q_next_S = self.target_model(next_S)
                Q_next_S_max = Q_next_S.max(1)[0].detach()

            td_objective = R + self.gamma * Q_next_S_max * (1 - D)
            Q_to_update = self.model(S).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(Q_to_update, td_objective.unsqueeze(1))
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # Update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            
            # Select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # Step the environment
            next_state, reward, done, info = env.step(action)
            
            # Record transition in replay buffer
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Train the model
            if step > self.train_warmup and step % self.train_freq == 0:
                for _ in range(self.nb_gradient_steps):
                    self.gradient_step()

                # Update target network with Polyak average
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.copy_(self.update_target_tau * param.data + (1 - self.update_target_tau) * target_param.data)

            # Next transition
            step += 1
            if done:
                episode += 1
                print(f"Episode {episode:3d}, steps {step:3d}, epsilon {epsilon:6.2f}, "
                      f"batch size {len(self.memory):5d}, episode return {episode_cum_reward:4.1f}")
                state = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return
