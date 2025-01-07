import torch
import numpy as np

def greedy_action(network, state):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        # Convert state to tensor and move to the appropriate device
        
        state = np.squeeze(state, axis=-1)  # Remove the extra dimension if present
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        state = torch.tensor(state, dtype=torch.float32).permute(0, 1, 3, 2).to(device)  # Change to [batch_size, channels, height, width]
        
        # Get Q-values from the network
        
        Q = network(state)
        
        # Return the action with the highest Q-value
        print(Q)
        print(torch.argmax(Q))
        print(torch.argmax(Q).item())
        return torch.argmax(Q).item()