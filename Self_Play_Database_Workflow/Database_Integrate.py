import mysql.connector
import torch
import numpy as np
from datetime import datetime
import random
import time


# Connect to MySQL Database
def connect_db():
    return mysql.connector.connect(
        host="your_mysql_host",
        user="your_mysql_user",
        password="your_mysql_password",
        database="poker_room"
    )


# Poker AI Bot (Simplified to work with DB)
class PokerBot:
    def __init__(self, player_id, model):
        self.player_id = player_id
        self.model = model
        self.chip_stack = 100  # Starting stack
        self.game_account_id = None  # Will be assigned after buy-in

    def buy_in(self, cursor):
        """Attempt to buy into an existing game or create a new one."""
        buy_in_amount = random.choice([10, 20, 50])
        cursor.callproc("SafeBuyIn", (self.player_id, buy_in_amount))
        self.chip_stack = buy_in_amount
        cursor.execute("SELECT LAST_INSERT_ID()")
        self.game_account_id = cursor.fetchone()[0]
        print(f"Bot {self.player_id} bought in with {buy_in_amount} chips (Game Account ID: {self.game_account_id})")

    def decide_action(self, game_state):
        """Use AI model to decide on a poker action (fold, call, raise)."""
        state_tensor = torch.FloatTensor(game_state)
        policy, _ = self.model(state_tensor)
        action = np.argmax(policy.detach().numpy())  # Select best action
        return action

    def make_bet(self, cursor, subround_id):
        """Bot makes a bet based on AI decision."""
        action = self.decide_action([self.chip_stack, random.randint(0, 100), random.randint(0, 500), self.player_id])
        bet_amount = random.choice([5, 10, 20]) if action == 2 else 0  # Bet amount for raises
        bet_type_id = action + 1  # Mapping: Fold (1), Call (2), Raise (3)

        cursor.execute(
            "INSERT INTO Make_Bet (Bet_Time, Subround_ID, Game_Account_ID, Bet_Type_ID, Bet_Amount) "
            "VALUES (%s, %s, %s, %s, %s)",
            (datetime.now(), subround_id, self.game_account_id, bet_type_id, bet_amount)
        )
        print(f"Bot {self.player_id} made a bet: Type {bet_type_id} | Amount {bet_amount}")

    def cash_out(self, cursor):
        """Bot decides to cash out."""
        cash_out_amount = self.chip_stack
        cursor.execute(
            "INSERT INTO Buy_Out (Buy_Out_Time, Game_Account_ID, Player_ID, Buy_Out_Amount) "
            "VALUES (%s, %s, %s, %s)",
            (datetime.now(), self.game_account_id, self.player_id, cash_out_amount)
        )
        print(f"Bot {self.player_id} cashed out with {cash_out_amount}")