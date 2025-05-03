# qlearning_player.py

import os
import pickle
import random
from pypokerengine.players import BasePokerPlayer
from qlearning_core import QLearningPokerAgent
from raise_player import RaisedPlayer

class QLearningPlayer(BasePokerPlayer):
    def __init__(self,
                 q_table_path="q_table_agent.pkl",
                 trainable=True,
                 fold_weak_prob=0.5,
                 fold_medium_prob=0.1,
                 lambda_trace=0.8):
        super().__init__()
        self.agent = QLearningPokerAgent(actions=["fold", "call", "raise"])

        # Load or initialize Q-table
        if os.path.exists("q_table01.pkl"):
            with open("q_table01.pkl", "rb") as f:
                self.agent.q_table = pickle.load(f)
            print(f"Loaded existing Q-table ({len(self.agent.q_table)} states).")
        else:
            print("No pretrained Q-table found; starting fresh.")

        # Ensure every loaded state has entries for all actions
        for state, action_dict in list(self.agent.q_table.items()):
            for a in self.agent.actions:
                action_dict.setdefault(a, 0.0)

        # Training hyperparameters
        self.trainable = trainable
        self.fold_weak_prob = fold_weak_prob
        self.fold_medium_prob = fold_medium_prob
        self.lambda_trace = lambda_trace    # λ parameter
        self.recent_opponent_actions = []

        # Pull α, γ from your QLearningPokerAgent
        self.alpha = self.agent.alpha
        self.gamma = self.agent.gamma

        # Per-hand trackers
        self.initial_stack = None
        self.state_action_history = []      # list of (state_key, action)
        self.updated_pairs = set()          # for assertion
        self.E = {}                         # eligibility trace dict

        # Opponent profiling stats
        self.opponent_actions = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.opponent_stats_total = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.profile_decay = 0.9 
        self.action_counts = {"fold": 0, "call": 0, "raise": 0}

    def _q_lambda_update(self, state, action, reward, next_state, valid_actions):
        # Initialize Q entries if missing
        self.agent.q_table.setdefault(state, {a:0.0 for a in self.agent.actions})
        self.agent.q_table.setdefault(next_state, {a:0.0 for a in self.agent.actions})

        # Compute TD error delta
        Q_sa = self.agent.q_table[state].setdefault(action, 0.0)
        max_next = max(self.agent.q_table[next_state].values())
        delta = reward + self.gamma * max_next - Q_sa

        # Increment eligibility for this pair
        self.E[(state,action)] = self.E.get((state,action), 0.0) + 1.0

        # Update all traces
        for (s,a), trace_val in list(self.E.items()):
            self.agent.q_table.setdefault(s, {ac:0.0 for ac in self.agent.actions})
            self.agent.q_table[s].setdefault(a, 0.0)
            self.agent.q_table[s][a] += self.alpha * delta * trace_val
            self.E[(s,a)] = trace_val * self.gamma * self.lambda_trace
            self.updated_pairs.add((s,a))

    def declare_action(self, valid_actions, hole_card, round_state):
        # Build opponent profile
        profile = (self.opponent_actions
                   if self.opponent_actions["total"] >= 5
                   else self.opponent_stats_total)
        if profile["total"] == 0:
            profile = {"raise": 0, "call": 0, "fold": 0, "total": 1}

        state_key = self.agent.get_state_key(hole_card, round_state, profile)
        available = [a["action"] for a in valid_actions]

        # λ-trace update on prior step with reward=0
        if self.trainable and self.state_action_history:
            prev_s, prev_a = self.state_action_history[-1]
            self._q_lambda_update(prev_s, prev_a, 0, state_key, valid_actions)
            # print(f"[lambda-step] Updated eligibility traces for {(prev_s,prev_a)}")

        # Maniac override (unchanged)
        hand_strength = self.agent.evaluate_hand_strength(hole_card)
        behavior = self.agent.abstract_history(round_state, profile)
        action = None
        if behavior == "maniac":                
                if hand_strength == "strong" and "raise" in available:
                    action = "raise"
                elif hand_strength == "strong" and "call" in available:
                    action = "call"
                if hand_strength == "medium" and "raise" in available:
                    action = "raise"
                elif hand_strength == "medium" and "call" in available:
                    action = "call"
                elif hand_strength == "weak" and "check" not in available:
                    action = "fold"

        # Fallback to ε-greedy Q-policy
        if action is None:
            action = self.agent.choose_action(state_key, valid_actions)

        # Record the decision
        self.state_action_history.append((state_key, action))
        self.action_counts[action] += 1
        return action

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        # Reset trackers for new hand
        for s in seats:
            if s["uuid"] == self.uuid:
                self.initial_stack = s["stack"]
                break

        self.opponent_actions = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.action_counts = {"fold": 0, "call": 0, "raise": 0}
        self.state_action_history.clear()
        self.updated_pairs.clear()
        self.E.clear()

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        # Track opponent moves
        if action["player_uuid"] != self.uuid:
           act = action["action"]
           if act in self.opponent_actions:
                self.opponent_actions[act] += 1
                self.opponent_actions["total"] += 1

                # EMA for long-term opponent stats
                for move in ["raise", "call", "fold"]:
                    self.opponent_stats_total[move] = (
                        self.profile_decay * self.opponent_stats_total[move]
                        + (1 if act == move else 0)
                    )
                self.opponent_stats_total["total"] = (
                    self.profile_decay * self.opponent_stats_total["total"] + 1
                )

            # Reward shaping (unchanged)
           if act == "fold" and self.state_action_history:
                last_state, last_action = self.state_action_history[-1]
                if last_action == "raise":
                    profile = (
                        self.opponent_actions
                        if self.opponent_actions["total"] >= 5
                        else self.opponent_stats_total
                    )
                    next_state = self.agent.get_state_key([], round_state, profile)
                    shaping_reward = 2
                    self._q_lambda_update(
                        last_state,
                        last_action,
                        shaping_reward,
                        next_state,
                        [{"action": "fold"}, {"action": "call"}, {"action": "raise"}],
                    )
                    print(f"[Shaping] +{shaping_reward} for raise->fold: "
                        f"Q[{last_state!r}]['raise'] = {self.agent.q_table[last_state]['raise']:.3f}")

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Compute terminal reward
        reward = self._calculate_reward(round_state["seats"])

        # Build final next_state
        profile = (self.opponent_actions
                   if self.opponent_actions["total"] >= 5
                   else self.opponent_stats_total)
        next_state = self.agent.get_state_key([], round_state, profile)

        # Final λ-update on last step
        if self.trainable and self.state_action_history:
            last_s, last_a = self.state_action_history[-1]
            self._q_lambda_update(
                last_s,
                last_a,
                reward,
                next_state,
                valid_actions=[{"action": "fold"},
                               {"action": "call"},
                               {"action": "raise"}]
            )
            # print(f"[Lambda-final] Propagated TD error with reward={reward}")

            # Save updated Q-table
            with open("q_table01.pkl", "wb") as f:
                pickle.dump(self.agent.q_table, f)
            # print(f"Saved Q-table ({len(self.agent.q_table)} states).")

        # Assertion: every visited (s,a) got updated
        visited = set(self.state_action_history)
        missing = visited - self.updated_pairs
        if missing:
            raise RuntimeError(f"Never updated these state–actions: {missing}")

        # Decay fold-medium probability
        self.fold_medium_prob = max(self.fold_medium_prob * 0.95, 0.05)

    def _calculate_reward(self, final_seats):
        for s in final_seats:
            if s["uuid"] == self.uuid:
                return s["stack"] - self.initial_stack
        return 0
