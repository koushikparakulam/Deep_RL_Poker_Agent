import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque

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
