import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque

# Define Poker Environment
class PokerEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Resets the environment and returns the initial state."""
        self.player_chips = [100, 100]  # Two players, 100 chips each
        self.current_bet = 0
        self.pot = 0
        self.history = []  # Track actions
        self.current_player = 0
        return self.get_state()
    
    def get_state(self):
        return (tuple(self.player_chips), self.current_bet, self.pot, self.current_player)
    
    def step(self, action):
        """Executes an action (0: fold, 1: call, 2: raise)."""
        reward = 0
        if action == 0:  # Fold
            winner = 1 - self.current_player
            self.player_chips[winner] += self.pot
            done = True
        elif action == 1:  # Call
            self.pot += self.current_bet
            self.player_chips[self.current_player] -= self.current_bet
            done = False
        elif action == 2:  # Raise
            raise_amount = min(10, self.player_chips[self.current_player])
            self.current_bet += raise_amount
            self.pot += raise_amount
            self.player_chips[self.current_player] -= raise_amount
            done = False
        else:
            raise ValueError("Invalid action")
        
        self.history.append((self.current_player, action))
        self.current_player = 1 - self.current_player  # Switch turns
        return self.get_state(), reward, done, {}

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

# Monte Carlo Tree Search (MCTS)
class MCTS:
    def __init__(self, env, model, simulations=100):
        self.env = env
        self.model = model
        self.simulations = simulations
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
    
    def uct_select(self, state):
        """Select action based on UCT formula."""
        best_action, best_uct = None, -float('inf')
        for action in range(3):  # Actions: Fold, Call, Raise
            Nsa = self.N[(state, action)]
            Qsa = self.Q[(state, action)]
            if Nsa == 0:
                return action
            uct = Qsa / (Nsa + 1e-6) + np.sqrt(np.log(sum(self.N[(state, a)] for a in range(3))) / (Nsa + 1e-6))
            if uct > best_uct:
                best_uct, best_action = uct, action
        return best_action
    
    def simulate(self, state):
        """Run a simulation from state to update Q values."""
        path, rewards = [], []
        done = False
        while not done:
            action = self.uct_select(state)
            path.append((state, action))
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
        
        for (s, a), r in zip(path, rewards):
            self.N[(s, a)] += 1
            self.Q[(s, a)] += r
    
    def get_action(self, state):
        """Perform MCTS simulations and return the best action."""
        for _ in range(self.simulations):
            self.simulate(state)
        return self.uct_select(state)

# Train Poker Bot via Self-Play
def train_poker_bot(episodes=500):
    env = PokerEnv()
    model = PokerNN(state_size=4, action_size=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    mcts = MCTS(env, model)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = mcts.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Convert to tensors for training
            state_tensor = torch.FloatTensor(state)
            policy, value = model(state_tensor)
            value_target = torch.tensor(reward).float()
            loss = nn.MSELoss()(value.squeeze(), value_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        if episode % 50 == 0:
            print(f"Episode {episode}: Training in progress...")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    trained_model = train_poker_bot(episodes=500)
