import torch.nn as nn
import torch
import torch.nn.functional as F

class MarioNetwork_Dqn(nn.Module):
    def __init__(self, in_channels=1, n_actions=6):
        super(MarioNetwork_Dqn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Ensure correct format
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) 
        
        return x


def greedy_action(network, state):
    device = "cuda"
    network.to(device)  # Move the network to the GPU
    with torch.no_grad():
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
 # Ensure shape is correct

        Q = network(state_tensor)
        return torch.argmax(Q).item()
