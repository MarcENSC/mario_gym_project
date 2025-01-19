import torch
from agent.mario import Mario
from env_wrappers.env_wrapper import env  # Import the environment
import numpy as np

# Load the checkpoint
checkpoint_path = "checkpoints/2025-01-18T12-23-36/mario_net_14.chkpt"
checkpoint = torch.load(checkpoint_path)

# Define the required arguments
state_dim = (4, 84, 84)
action_dim = env.action_space.n  # Get the action dimension from the environment
save_dir = "checkpoints"  # Replace with the appropriate save directory

# Recreate the model
model = Mario(state_dim, action_dim, save_dir)
model.network.load_state_dict(checkpoint['model'])  # Load the model's state_dict into the network
model.network.eval()  # Set the network to evaluation mode

# Access the exploration rate if needed
exploration_rate = checkpoint['exploration_rate']
print(f"Exploration rate at save time: {exploration_rate}")

# Run the model in the environment
state = env.reset()
env.render(mode="human")
done = False
total_reward = 0

while not done:
    state = torch.tensor(state.__array__(), device=model.device).float().unsqueeze(0)
    state = np.squeeze(state, axis=-1)
    

    action_values = model.network(state, model="updated")
    action = torch.argmax(action_values, axis=1).item()
    
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    state = next_state

print(f"Total reward: {total_reward}")