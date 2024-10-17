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