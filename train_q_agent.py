from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from allin_player import AllInPlayer
from qlearning_player import QLearningPlayer
import pickle
import csv
import os
import random
import json

QTABLE_PATH = "q_table_agent.pkl"
CSV_PATH = "training_progress.csv"
JSONL_LOG_PATH = "game_log.jsonl"
SAVE_INTERVAL = 10

def read_last_game_index():
    if not os.path.exists(JSONL_LOG_PATH):
        return 0
    with open(JSONL_LOG_PATH, "r") as f:
        lines = f.readlines()
        if not lines:
            return 0
        try:
            last_entry = json.loads(lines[-1])
            return last_entry["game_id"]
        except (json.JSONDecodeError, KeyError):
            return 0

def train_q_agent():
    num_new_games = 10
    max_round = 100
    initial_stack = 10000
    small_blind_amount = 20

    agent = QLearningPlayer(trainable=True, fold_weak_prob=0, fold_medium_prob=0)

    if os.path.exists(QTABLE_PATH):
        with open(QTABLE_PATH, "rb") as f:
            agent.agent.q_table = pickle.load(f)
        print("[Resume] Loaded Q-table from previous run.")

    last_game_id = read_last_game_index()

    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Game ID", "Agent Wins", "Opponent Wins", "Win Rate (%)", "Reward", "Opponent", "Result"])

        agent_wins = opponent_wins = 0
        random_wins = allin_wins = random_losses = allin_losses = 0

        for i in range(num_new_games):
            game_id = last_game_id + i + 1
            print(f"\n=== Game {game_id} ===")
            if (game_id // 10) % 2 == 0:
                opponent = AllInPlayer()
                opponent_type = "AllInPlayer"
            else:
                opponent = AllInPlayer()
                opponent_type = "AllInPlayer"

            config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
            config.register_player(name="f1", algorithm=agent)
            config.register_player(name="f2", algorithm=opponent)

            result = start_poker(config, verbose=0)
            winner = max(result["players"], key=lambda p: p["stack"])

            reward = 0
            for p in result["players"]:
                if p["name"] == "f1":
                    reward = 0.01 * (p["stack"] - initial_stack)
                    break

            if winner["name"] == "f1":
                agent_wins += 1
                if opponent_type == "Random":
                    random_wins += 1
                else:
                    allin_wins += 1
                outcome = f"Agent won against {opponent_type}"
            else:
                opponent_wins += 1
                if opponent_type == "Random":
                    random_losses += 1
                else:
                    allin_losses += 1
                outcome = f"Agent lost to {opponent_type}"

            last_state_key = agent.transition_history[-1][0] if agent.transition_history else None
            hand_strength = last_state_key.split("_")[0] if last_state_key else "unknown"
            action_taken = agent.transition_history[-1][1] if agent.transition_history else None

            profile = agent.opponent_actions if agent.opponent_actions["total"] >= 5 else agent.opponent_stats_total
            if profile["total"] == 0:
                profile = {"raise": 0, "call": 0, "fold": 0, "total": 1}  # minimal placeholder

            was_override = (
                (hand_strength == "weak" and action_taken == "fold" and random.random() < agent.fold_weak_prob) or
                (hand_strength == "medium" and action_taken == "fold" and random.random() < agent.fold_medium_prob)
            )
            for i in range(len(agent.state_action_history)):
                state_key, action = agent.state_action_history[i]
                next_state_key = agent.state_action_history[i + 1][0] if i + 1 < len(agent.state_action_history) else None
                is_terminal = (i == len(agent.state_action_history) - 1)
                prev_q = agent.agent.q_table.get(state_key, {}).get(action, 0.0)
                agent.agent.update(
                    state_key,
                    action,
                    reward if is_terminal else 0,
                    next_state_key,
                    valid_actions=[{"action": "fold"}, {"action": "call"}, {"action": "raise"}]
                )
                new_q = agent.agent.q_table[state_key][action]
                print(f"[Q-UPDATE] {state_key} -> {action}: {prev_q:.4f} -> {new_q:.4f} (reward={reward if is_terminal else 0})")
            # Print all Q-values for this state
                all_qs = agent.agent.q_table.get(state_key, {})
                print(f"[Q-VALUES] State: {state_key}")
                for a in ["fold", "call", "raise"]:
                    print(f"  {a:>5}: {all_qs.get(a, 0.0):.4f}")
            agent.state_action_history.clear()

            random_total = random_wins + random_losses
            allin_total = allin_wins + allin_losses
            random_rate = 100.0 * random_wins / random_total if random_total else 0.0
            allin_rate = 100.0 * allin_wins / allin_total if allin_total else 0.0

            print(f"Opponent Type     : {opponent_type}")
            print(f"Hand Strength     : {hand_strength}")
            print(f"Chosen Action     : {action_taken}" + (" (override)" if was_override else ""))
            print(f"Winner            : {winner['name']}")
            print(f"Reward this game  : {round(reward, 2)}")
            print(f"Total Agent Wins  : {agent_wins}")
            print(f"Total Opponent Wins: {opponent_wins}")
            print(f"RandomPlayer W/L  : {random_wins}W / {random_losses}L ({random_rate:.2f}%)")
            print(f"AllInPlayer W/L   : {allin_wins}W / {allin_losses}L ({allin_rate:.2f}%)")
            print("=" * 30)
            print("Action counts", agent.action_counts.copy())
            print(f"[Summary] G{game_id} | {opponent_type} | {hand_strength} | {action_taken}" + 
                  (" (override)" if was_override else "") + 
                  f" | reward={reward:.2f} | agent={agent_wins} | opp={opponent_wins} | WR(AllIn)={allin_rate:.2f}%")

            log_entry = {
                "game_id": game_id,
                "opponent": opponent_type,
                "hand_strength": hand_strength,
                "action": action_taken,
                "was_override": was_override,
                "winner": winner["name"],
                "reward": reward,
                "agent_wins": agent_wins,
                "opponent_wins": opponent_wins,
                "random_wins": random_wins,
                "random_losses": random_losses,
                "allin_wins": allin_wins,
                "allin_losses": allin_losses,
                "random_winrate": round(random_rate, 2),
                "allin_winrate": round(allin_rate, 2),
                "action_counts": agent.action_counts.copy(),
                "exploited_maniac": agent.exploit_maniac
            }
            agent.exploit_maniac = False

            with open(JSONL_LOG_PATH, "a") as logf:
                logf.write(json.dumps(log_entry) + "\n")

            with open(QTABLE_PATH, "wb") as f:
                pickle.dump(agent.agent.q_table, f)
            print(f"[Checkpoint] Saved Q-table at game {game_id}")

            total = agent_wins + opponent_wins
            win_rate = 100.0 * agent_wins / total if total > 0 else 0.0
            writer.writerow([
                game_id,
                agent_wins,
                opponent_wins,
                f"{win_rate:.2f}",
                reward,
                opponent_type,
                outcome
            ])

    print("\n=== Training Ended ===")
    print(f"Q-table size: {len(agent.agent.q_table)} states")
    total_pairs = sum(len(v) for v in agent.agent.q_table.values())
    print(f"Total state-action pairs: {total_pairs}")

if __name__ == "__main__":
    train_q_agent()