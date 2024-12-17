import sys
import os
from copy import deepcopy
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network.dqn import greedy_action
from buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, config, model):
        # Set all parameters
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95

        ## Train every N steps
        self.train_freq = config['train_freq'] if 'train_freq' in config.keys() else 1
        self.train_warmup = config['train_warmup'] if 'train_warmup' in config.keys() else 1

        ## Replay Buffer
        ### Number of transitions to sample when training
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, device)

        ## Epsilon-greedy strategy
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

        ## DQN and target DQN
        self.model = model
        self.target_model = deepcopy(self.model).to(device)

        ## Loss / learning rate / optimizer
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)

        ## Number of gradient steps to perform on each batch sampled from the replay buffer
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1

        ## Parameter to update the target DQN with a moving average (Polyak average)
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            self.optimizer.zero_grad()
            S, A, R, next_S, D = self.memory.sample(self.batch_size)
            # loss = <your code>
            with torch.no_grad():
                Q_next_S_max = self.target_model(next_S).max(1)[0].detach()

            td_objective = R + self.gamma * Q_next_S_max * (1 - D)
            Q_to_update = self.model(S).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(Q_to_update, td_objective.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            print(f"Loss: {loss.item()}")  # Debug statement

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            print(f"STEP ACTION: {env.step(action)}")
            next_state, reward, done, trunc = env.step(action)
            # record transition in replay buffer
            self.memory.append(state, action, reward, next_state, done)
            print(f"REWARD: {reward}")
            episode_cum_reward += reward

            # train
            if step > self.train_warmup and step % self.train_freq == 0:
                for _ in range(self.nb_gradient_steps):
                    self.gradient_step()

                # update target network with Polyak average
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)

            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode),
                      ", steps ", '{:3d}'.format(step),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", batch size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        # Save the model after training
        torch.save(self.model.state_dict(), 'mario_dqn_model.pth')
        print("Model saved as mario_dqn_model.pth")
    
        return episode_return
