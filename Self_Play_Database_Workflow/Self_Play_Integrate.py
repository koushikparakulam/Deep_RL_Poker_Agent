import mysql.connector
import torch
import numpy as np
from datetime import datetime
import random
import time

# Main Poker Game Engine
def play_poker_game(bot_count=3):
    conn = connect_db()
    cursor = conn.cursor()

    # Load AI Model
    model = torch.load("trained_poker_model.pth")  # Load trained AI model

    # Create Poker Bots
    bots = [DatabaseInterface(i + 1, model) for i in range(bot_count)]

    # Bots buy into the game
    for bot in bots:
        bot.buy_in(cursor)
    conn.commit()

    # Simulate rounds of betting
    for _ in range(5):  # Play 5 rounds
        cursor.execute("SELECT MAX(Subround_ID) FROM Subround")
        subround_id = cursor.fetchone()[0]

        for bot in bots:
            bot.make_bet(cursor, subround_id)
        conn.commit()
        time.sleep(2)

    # Bots cash out at the end
    for bot in bots:
        bot.cash_out(cursor)
    conn.commit()

    cursor.close()
    conn.close()
    
play_poker_game(bot_count=3)