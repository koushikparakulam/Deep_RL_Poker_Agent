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