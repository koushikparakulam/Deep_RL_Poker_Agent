# Deep Reinforcement Learning Poker Agent

## Overview
This repository implements a **Deep Reinforcement Learning (DRL) Poker Agent** that plays poker using a combination of:
- **Neural Networks (NN) for decision-making**
- **Monte Carlo Tree Search (MCTS) for simulations**
- **Self-Play Training to improve strategies**
- **Integration with an SQL database** for storing and managing game data

## Features
- **Poker Environment Simulation**: The `PokerEnv` class models a simple poker game with basic betting mechanics.
- **Deep Learning with PyTorch**: The `PokerNN` class is a neural network that learns poker strategies.
- **MCTS for Decision-Making**: The `MCTS` class enhances decision-making using Monte Carlo simulations.
- **Self-Play Training**: Bots play against each other to improve over time.
- **Database Integration**: Game results, bets, and player actions are stored in an SQL database.

## Files & Structure
### Core AI Modules
- **`Environment.py`**: Defines the `PokerEnv` class, which simulates the poker game logic.
- **`MCTS.py`**: Implements Monte Carlo Tree Search for decision-making.
- **`NN_Policy.py`**: Defines a PyTorch neural network that predicts poker actions and values.
- **`Train_main.py`**: Runs self-play training to optimize the bot's strategy.

### Database & Integration
- **`Database_Integrate.py`**: Handles connection to the MySQL database and executes SQL statements.
- **`Self_Play_Integrate.py`**: Implements AI-driven bots that play in real-time and store game data in the database.

## How It Works
1. **Setup & Training**: The AI is trained using self-play, with actions optimized via reinforcement learning.
2. **Bot Interaction with Database**:
   - Bots register into the database and buy into poker games.
   - Each bot decides its moves using the trained neural network and MCTS.
   - Bets and game results are logged into the database.
3. **Continuous Learning**:
   - The AI model is updated based on game results.
   - Winning strategies emerge over multiple self-play training iterations.

## Getting Started
### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- MySQL Connector for Python

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Deep_RL_Poker_Agent.git
   cd Deep_RL_Poker_Agent
   ```
2. **Install Dependencies**:
   ```bash
   pip install torch numpy mysql-connector-python
   ```
3. **Setup Database**:
   - Run the SQL scripts in `Database_Integrate.py` to create the necessary tables.
   - Ensure you configure your database connection details in `Database_Integrate.py`.

### Training the Model
To train the poker agent using self-play, run:
```bash
python Train_main.py
```

### Running the Poker Bots
To let AI bots play in the database-connected poker game, execute:
```bash
python Self_Play_Integrate.py
```

## Future Enhancements
- Implement **multi-player Texas Hold'em** with advanced rules.
- Improve AI strategy using **Deep Q-Networks (DQN) or Proximal Policy Optimization (PPO)**.
- Visualize game data using **real-time dashboards**.

