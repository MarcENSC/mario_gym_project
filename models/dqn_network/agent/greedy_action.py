import torch
import numpy as np

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        # Convert state to tensor and move to the appropriate device
        
        state = np.squeeze(state, axis=-1)  # Remove the extra dimension if present
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        state = torch.tensor(state, dtype=torch.float32).permute(0, 1, 3, 2).to(device)  # Change to [batch_size, channels, height, width]
        
        # Get Q-values from the network
        print(f"state shape greedy : {state.shape}")
        Q = network(state)
        
        # Return the action with the highest Q-value
        return torch.argmax(Q).item()