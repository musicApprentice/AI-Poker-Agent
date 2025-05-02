from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from qlearning_player import QLearningPlayer
import pickle
import csv

# Modified from 
def train_q_agent():
    num_game = 25
    max_round = 100
    initial_stack = 10000
    small_blind_amount = 20

    agent = QLearningPlayer()
    opponent = RandomPlayer()

    config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
    config.register_player(name="f1", algorithm=agent)
    config.register_player(name="f2", algorithm=opponent)

    agent_wins = 0
    opponent_wins = 0

    # Create CSV file for logging win rate
    with open("training_progress.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Game", "Agent Wins", "Opponent Wins", "Win Rate (%)"])

        for game in range(num_game):
            print(f"Training game {game+1}/{num_game}")
            result = start_poker(config, verbose=0)

            # Find out who won this game
            winner = max(result["players"], key=lambda p: p["stack"])
            if winner["name"] == "f1":
                agent_wins += 1
            else:
                opponent_wins += 1

            # Log and print every 100 games
            if (game + 1) % 25 == 0:
                total = agent_wins + opponent_wins
                win_rate = 100.0 * agent_wins / total if total > 0 else 0.0
                print(f"After {game + 1} games: Agent wins = {agent_wins}, Opponent wins = {opponent_wins}, Win rate = {win_rate:.2f}%")
                writer.writerow([game + 1, agent_wins, opponent_wins, f"{win_rate:.2f}"])

    print("\n=== Training Results ===")
    print(f"Agent wins: {agent_wins}")
    print(f"Opponent wins: {opponent_wins}")
    print(f"Win rate: {100.0 * agent_wins / num_game:.2f}%")

    # Save Q-table
    with open("q_table01.pkl", "wb") as f:
        pickle.dump(agent.agent.q_table, f)

    print("\nTraining completed. Q-table saved as q_table.pkl!")

    # Print state summaries
    print("\n=== State Summary ===")
    print("Hand Strength Counts:")
    for k, v in agent.agent.hand_strength_counts.items():
        print(f"{k}: {v}")

    print("\nOpponent Behavior Counts:")
    for k, v in agent.agent.behavior_counts.items():
        print(f"{k}: {v}")

    print("\nPot Bucket Counts:")
    for k, v in agent.agent.pot_bucket_counts.items():
        print(f"{k}: {v}")
    avg_raise = sum(agent.agent.raise_ratios) / len(agent.agent.raise_ratios)
    avg_call = sum(agent.agent.call_ratios) / len(agent.agent.call_ratios)

    print(f"\nAverage Raise Ratio: {avg_raise:.2f}")
    print(f"Average Call Ratio: {avg_call:.2f}")


if __name__ == "__main__":
    train_q_agent()
