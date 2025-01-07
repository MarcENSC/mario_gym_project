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
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4,padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Adjust based on the final Conv output
        self.fc2 = nn.Linear(512, output_dim)
        self.fc3 = nn.Linear(64 * 7 * 7, 512)
        self.fc4 = nn.Linear(512, 1)
    

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten

        advantage = torch.relu(self.fc1(x))
        advantage = self.fc2(advantage)
        value = torch.relu(self.fc3(x))
        value = self.fc4(value)
        
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"Model loaded from {path}")