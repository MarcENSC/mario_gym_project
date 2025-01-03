import torch
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch.nn as nn





class DQN(nn.Module):
    def __init__(self, input_channels=4, output_dim=7):
        super(DQN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"input_channels: {input_channels}")
        print(f"output_dim: {output_dim}")
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Adjust based on the final Conv output
        self.fc2 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
