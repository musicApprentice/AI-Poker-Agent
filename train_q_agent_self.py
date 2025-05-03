from qlearning_player import QLearningPlayer
from pypokerengine.api.game import setup_config, start_poker
import pickle
import csv

def train_q_agent_self_play():
    num_game = 1000
    max_round = 100
    initial_stack = 10000
    small_blind_amount = 20

    agent = QLearningPlayer(trainable=True)
    opponent = QLearningPlayer(trainable=False, q_table_path="q_table_frozen.pkl")


    config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
    config.register_player(name="f1", algorithm=agent)
    config.register_player(name="f2", algorithm=opponent)

    agent_wins = 0
    opponent_wins = 0

    for game in range(num_game):
        print(f"Self-play game {game+1}/{num_game}")
        result = start_poker(config, verbose=0)

        winner = max(result["players"], key=lambda p: p["stack"])
        if winner["name"] == "f1":
            agent_wins += 1
        else:
            opponent_wins += 1

        if (game + 1) % 10 == 0:
            win_rate = 100.0 * agent_wins / (agent_wins + opponent_wins)
            print(f"After {game + 1} games: Win rate = {win_rate:.2f}%")

    with open("q_table_selfplay.pkl", "wb") as f:
        pickle.dump(agent.agent.q_table, f)

    print("\n Self-play training complete.")
    print(f"Final win rate: {100.0 * agent_wins / num_game:.2f}%")

if __name__ == "__main__":
    train_q_agent_self_play()
