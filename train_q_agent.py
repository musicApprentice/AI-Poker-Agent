# train_q_agent.py

import os
import pickle
import csv
from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from qlearning_player import QLearningPlayer
from allin_player import AllInPlayer
from raise_player import RaisedPlayer

def train_q_agent():
    num_games = 25
    max_round = 100
    initial_stack = 10000
    small_blind_amount = 20
    switch_interval = 10  # switch opponent every 10 games

    agent = QLearningPlayer()

    # overall counters
    agent_wins = 0
    opponent_wins = 0

    # per-opponent counters
    random_games = random_wins = 0
    allin_games = allin_wins = 0

    # prepare CSV
    with open("training_progress.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Game", "Opponent",
            "Agent Wins vs This Opponent", "Games vs This Opponent", "Win Rate vs This Opponent (%)",
            "Total Agent Wins", "Total Games", "Overall Win Rate (%)"
        ])

        for game in range(1, num_games + 1):
            # pick opponent
            block = (game - 1) // switch_interval
            if block % 2 == 0:
                opponent = AllInPlayer()
                opp_name = "AllIn"
                allin_games += 1
            else:
                opponent = AllInPlayer()
                opp_name = "AllIn"
                allin_games += 1

            # build fresh config
            config = setup_config(
                max_round=max_round,
                initial_stack=initial_stack,
                small_blind_amount=small_blind_amount
            )
            config.register_player(name="f1", algorithm=agent)
            config.register_player(name="f2", algorithm=opponent)

            # play
            print(f"Game {game}/{num_games} vs {opp_name}")
            result = start_poker(config, verbose=0)
            winner = max(result["players"], key=lambda p: p["stack"])

            # tally
            if winner["name"] == "f1":
                agent_wins += 1
                if opp_name == "Random":
                    random_wins += 1
                else:
                    allin_wins += 1
            else:
                opponent_wins += 1

            # compute rates
            total_games = agent_wins + opponent_wins
            overall_win_rate = 100 * agent_wins / total_games if total_games else 0.0
            random_rate = 100 * random_wins / random_games if random_games else 0.0
            allin_rate  = 100 * allin_wins  / allin_games  if allin_games  else 0.0

            # log
            writer.writerow([
                game, opp_name,
                random_wins if opp_name=="Random" else allin_wins,
                random_games if opp_name=="Random" else allin_games,
                f"{(random_rate if opp_name=='Random' else allin_rate):.2f}",
                agent_wins, total_games, f"{overall_win_rate:.2f}"
            ])

            # print summary every switch_interval
            if game % switch_interval == 0:
                print(f" After {game} games:")
                print(f"   vs Random:   {random_wins}/{random_games}  ({random_rate:.2f}%)")
                print(f"   vs AllIn:    {allin_wins}/{allin_games}   ({allin_rate:.2f}%)")
                print(f"   Overall:     {agent_wins}/{total_games}  ({overall_win_rate:.2f}%)\n")

    # final summary
    print("=== Final Training Results ===")
    print(f" vs Random:   {random_wins}/{random_games}  ({random_rate:.2f}%)")
    print(f" vs AllIn:    {allin_wins}/{allin_games}   ({allin_rate:.2f}%)")
    print(f" Overall:     {agent_wins}/{num_games}  ({100 * agent_wins / num_games:.2f}%)")

    # save Q-table
    with open("q_table01.pkl", "wb") as f:
        pickle.dump(agent.agent.q_table, f)
    print("Saved final Q-table to q_table01.pkl")

if __name__ == "__main__":
    train_q_agent()
