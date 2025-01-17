import torch.nn as nn
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class DDQN(nn.Module):
    """mini CNN structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        self.updated_cnn = self.__build_cnn(c, output_dim)
        self.target_cnn = self.__build_cnn(c, output_dim)    
        self.target_cnn.load_state_dict(self.updated_cnn.state_dict())

        for p in self.target_cnn.parameters():
            p.requires_grad = False
    
    def forward(self, x, model):
        if model == "updated":
            return self.updated_cnn(x)
        elif model == "target":
            return self.target_cnn(x)
        else:
            raise ValueError(f"Unknown model: {model}")

    def __build_cnn(self, input_channels, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )