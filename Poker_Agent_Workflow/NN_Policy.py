import torch
import torch.nn as nn
import torch.optim as optim

# Define Neural Network Policy + Value Function
class PokerNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(PokerNN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, action_size)  # Policy output
        self.value_head = nn.Linear(64, 1)  # Value output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value