import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from network.ddqn import DDQN
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict
import numpy as np 

class Mario:
    def __init__(self,state_dim,action_dim,save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.gamma = 0.9

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        self.network = DDQN(self.state_dim, self.action_dim).float().to(self.device)  
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000,device=torch.device("cpu")))
        self.batch_size = 32


        self.optimizer=torch.optim.Adam(self.network.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.exploration_rate =1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.current_step = 0
        self.save_every = 5e5
        self.warming_up = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync




    def act(self,state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = np.squeeze(state, axis=-1)
      
            state = torch.tensor(state, device=self.device).float().unsqueeze(0)
            
            action_values = self.network(state, model="updated")
            action = torch.argmax(action_values, axis=1).item()

        self.current_step += 1
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min,self.exploration_rate)
        return action
    


    def learn(self):
        if self.current_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.current_step % self.save_every == 0:
            self.save()

        if self.current_step < self.warming_up:
            return None, None

        if self.current_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def td_estimate(self,state,action):
        current_Q = self.network(state, model="updated")[np.arange(0, self.batch_size), action]
        return current_Q
    


    def cache(self,state,next_state,action,reward,done):
        def transform(x):
            return x[0].__array__() if isinstance(x, tuple) else x
        state = transform(state).__array__()
        next_state = transform(next_state).__array__()
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward],dtype=torch.float32)
        done = torch.tensor([done],dtype=torch.float32)


        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.current_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.current_step}")


    @torch.no_grad()
    def td_target(self,reward,next_state,done):
        next_state_Q = self.network(next_state, model="updated")
        action = torch.argmax(next_state_Q, axis=1)
        next_Q= self.network(next_state, model="target")[np.arange(0, self.batch_size), action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_updated(self,td_estimate,td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.network.target.load_state_dict(self.network.updated.state_dict())