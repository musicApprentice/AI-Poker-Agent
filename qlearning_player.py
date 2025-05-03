import os
import pickle
import random
from pypokerengine.players import BasePokerPlayer
from qlearning_core import QLearningPokerAgent

class QLearningPlayer(BasePokerPlayer):
    def __init__(self, q_table_path="q_table01.pkl", trainable=True,
                 fold_weak_prob=0.5, fold_medium_prob=0.1, lambda_trace=0.8):
        super().__init__()
        self.q_table_path = q_table_path
        self.agent = QLearningPokerAgent(actions=["fold", "call", "raise"])

        # Load Q-table if it exists, otherwise start fresh
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, "rb") as f:
                self.agent.q_table = pickle.load(f)
            print(f"Loaded existing Q-table from {self.q_table_path} ({len(self.agent.q_table)} states).")
        else:
            print(f"No Q-table found at {self.q_table_path}; starting fresh.")

        # Ensure each state has Q-values initialized for all actions
        for state, q_vals in self.agent.q_table.items():
            for a in self.agent.actions:
                q_vals.setdefault(a, 0.0)

        # Set learning parameters and tracking variables
        self.trainable = trainable
        self.fold_weak_prob = fold_weak_prob
        self.fold_medium_prob = fold_medium_prob
        self.lambda_trace = lambda_trace
        self.alpha = self.agent.alpha
        self.gamma = self.agent.gamma

        # Per-hand trackers
        self.initial_stack = None
        self.state_action_history = []
        self.updated_pairs = set()
        self.E = {}  # Eligibility traces

        # Opponent behavior tracking (short- and long-term)
        self.opponent_actions = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.opponent_stats_total = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.profile_decay = 0.9
        self.action_counts = {"fold": 0, "call": 0, "raise": 0}

    # λ-return Q-learning update with eligibility traces
    
    # Perform Q(λ) update using eligibility traces:
    # - Tracks all (state, action) pairs visited this hand.
    # - Assigns each pair an eligibility value that decays over time (γ * λ).
    # - On each update, applies the TD error (δ) scaled by eligibility to all pairs.
    # This allows delayed rewards to influence earlier decisions, improving credit assignment.

    def _q_lambda_update(self, state, action, reward, next_state, valid_actions):
        self.agent.q_table.setdefault(state, {a: 0.0 for a in self.agent.actions})
        self.agent.q_table.setdefault(next_state, {a: 0.0 for a in self.agent.actions})

        Q_sa = self.agent.q_table[state][action]
        max_next_q = max(self.agent.q_table[next_state].values())
        delta = reward + self.gamma * max_next_q - Q_sa

        # Increment eligibility for current (state, action)
        self.E[(state, action)] = self.E.get((state, action), 0.0) + 1.0

        # Apply eligibility trace updates to all visited state-action pairs
        for (s, a), e_val in self.E.items():
            self.agent.q_table[s][a] += self.alpha * delta * e_val
            self.E[(s, a)] = self.gamma * self.lambda_trace * e_val
            self.updated_pairs.add((s, a))

    # Called each time our agent must act
    def declare_action(self, valid_actions, hole_card, round_state):
        # Choose short- or long-term opponent profile
        profile = self.opponent_actions if self.opponent_actions["total"] >= 5 else self.opponent_stats_total
        if profile["total"] == 0:
            profile = {"raise": 0, "call": 0, "fold": 0, "total": 1}

        state_key = self.agent.get_state_key(hole_card, round_state, profile)
        available = [a["action"] for a in valid_actions]

        # TD(λ) update for previous step
        if self.trainable and self.state_action_history:
            last_s, last_a = self.state_action_history[-1]
            self._q_lambda_update(last_s, last_a, 0, state_key, valid_actions)

        # Override policy when facing a "maniac" opponent
        hand_strength = self.agent.evaluate_hand_strength(hole_card)
        behavior = self.agent.abstract_history(round_state, profile)
        action = None
        if behavior == "maniac":
            if hand_strength == "strong":
                action = "raise" if "raise" in available else "call" if "call" in available else None
            elif hand_strength == "medium":
                if "raise" in available:
                    action = "raise"
                elif "call" in available:
                    action = "call"
            elif hand_strength == "weak" and "check" not in available:
                action = "fold"

        # Otherwise use ε-greedy Q-learning policy
        if action is None:
            action = self.agent.choose_action(state_key, valid_actions)

        # Log action for later update
        self.state_action_history.append((state_key, action))
        self.action_counts[action] += 1
        return action

    def receive_game_start_message(self, game_info): pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        # Reset internal trackers at the start of each hand
        for s in seats:
            if s["uuid"] == self.uuid:
                self.initial_stack = s["stack"]
                break
        self.opponent_actions = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.state_action_history.clear()
        self.updated_pairs.clear()
        self.E.clear()
        self.action_counts = {"fold": 0, "call": 0, "raise": 0}

    def receive_street_start_message(self, street, round_state): pass

    def receive_game_update_message(self, action, round_state):
        # Update opponent stats (for profiling)
        if action["player_uuid"] != self.uuid:
            act = action["action"]
            if act in self.opponent_actions:
                self.opponent_actions[act] += 1
                self.opponent_actions["total"] += 1
                for move in ["raise", "call", "fold"]:
                    self.opponent_stats_total[move] = (
                        self.profile_decay * self.opponent_stats_total[move] +
                        (1 if act == move else 0)
                    )
                self.opponent_stats_total["total"] = self.profile_decay * self.opponent_stats_total["total"] + 1

            # Reward shaping if opponent folds after our raise
            if act == "fold" and self.state_action_history:
                last_s, last_a = self.state_action_history[-1]
                if last_a == "raise":
                    profile = self.opponent_actions if self.opponent_actions["total"] >= 5 else self.opponent_stats_total
                    next_s = self.agent.get_state_key([], round_state, profile)
                    self._q_lambda_update(last_s, last_a, 2, next_s, valid_actions=[
                        {"action": "fold"}, {"action": "call"}, {"action": "raise"}
                    ])
                    # print shaping reward

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Final reward update at end of hand
        reward = self._calculate_reward(round_state["seats"])
        profile = self.opponent_actions if self.opponent_actions["total"] >= 5 else self.opponent_stats_total
        next_state = self.agent.get_state_key([], round_state, profile)

        # Final TD(λ) update
        if self.trainable and self.state_action_history:
            last_s, last_a = self.state_action_history[-1]
            self._q_lambda_update(
                last_s, last_a, reward, next_state,
                valid_actions=[{"action": "fold"}, {"action": "call"}, {"action": "raise"}]
            )
            with open(self.q_table_path, "wb") as f:
                pickle.dump(self.agent.q_table, f)

        # Sanity check to ensure every visited (state, action) was updated
        visited = set(self.state_action_history)
        missing = visited - self.updated_pairs
        if missing and self.trainable:
            raise RuntimeError(f"Never updated these state->actions: {missing}")


    # Compute net profit/loss from hand
    def _calculate_reward(self, final_seats):
        for s in final_seats:
            if s["uuid"] == self.uuid:
                return s["stack"] - self.initial_stack
        return 0
