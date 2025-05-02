from pypokerengine.players import BasePokerPlayer
from qlearning_core import QLearningPokerAgent
import pickle
import random

class QLearningPlayer(BasePokerPlayer):
    def __init__(self, trainable=True, fold_weak_prob=.5, fold_medium_prob=0.1):
        super().__init__()
        self.agent = QLearningPokerAgent(actions=["fold", "call", "raise"])
        try:
            with open("q_table_agent.pkl", "rb") as f:
                self.agent.q_table = pickle.load(f)
                print("Loaded existing Q-table.")
        except:
            print("No pretrained Q-table found, starting fresh.")

        self.trainable = trainable
        self.fold_weak_prob = fold_weak_prob
        self.fold_medium_prob = fold_medium_prob

        self.exploit_maniac = False
        self.initial_stack = None

        self.state_action_history = []  # For storing full episode history

        self.opponent_actions = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.opponent_stats_total = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.action_counts = {"fold": 0, "call": 0, "raise": 0}

    def declare_action(self, valid_actions, hole_card, round_state):
        state_key = self.agent.get_state_key(hole_card, round_state)
        available_actions = [a["action"] for a in valid_actions]
        can_fold = "fold" in available_actions
        can_check = "check" in available_actions
        can_call = "call" in available_actions
        can_raise = "raise" in available_actions

        self.exploit_maniac = False
        hand_strength = self.agent.evaluate_hand_strength(hole_card)

        # Get opponent behavior
        opponent_profile = (
            self.opponent_actions if self.opponent_actions["total"] >= 5
            else self.opponent_stats_total
        )
        opponent_behavior = self.agent.abstract_history(round_state, opponent_profile)
        current_street = round_state.get("street", "").lower()

        action = None
        was_override = False

        # === EXPLOIT MANIAC (PRE-FLOP FOLDING) ===
        if current_street == "preflop" and opponent_behavior == "maniac" and hand_strength in ["weak", "medium"] and can_fold and not can_check:
            prob = self.fold_medium_prob if hand_strength == "medium" else self.fold_weak_prob
            if random.random() < prob:
                self.exploit_maniac = True
                action = "fold"
                was_override = True

        # === MIMIC MANIAC (AGGRESSIVE REACTION) ===
        if action is None and opponent_behavior == "maniac":
        #     if can_raise:
        #         action = "raise"
        #     elif can_call:
        #         action = "call"
        #     else:
        #         action = "fold"
        #         was_override = True

            # === STRONG HAND RULE ===
            if action is None and hand_strength == "strong":
                if can_raise:
                    action = "raise"
                elif can_call:
                    action = "call"
                was_override = False

            # === MEDIUM HAND STABS ===
            if action is None and hand_strength == "medium":
                if random.random() < 0.4 and can_raise:
                    action = "raise"
                elif random.random() < 0.2 and can_call:
                    action = "call"
                was_override = False

        # === DEFAULT POLICY ===
        if action is None:
            action = self.agent.choose_action(state_key, valid_actions)
            was_override = False

        # === Record and Count ===
        self._record_step(state_key, action)
        self.action_counts[action] += 1

        return action


    def _record_step(self, state_key, action):
        self.state_action_history.append((state_key, action))

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        for s in seats:
            if s['uuid'] == self.uuid:
                self.initial_stack = s['stack']
                break
        self.opponent_actions = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        self.action_counts = {"fold": 0, "call": 0, "raise": 0}
        self.state_action_history = []

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        if action['player_uuid'] != self.uuid:
            act = action['action']
            if act in self.opponent_actions:
                self.opponent_actions[act] += 1
                self.opponent_actions["total"] += 1
                self.opponent_stats_total[act] += 1
                self.opponent_stats_total["total"] += 1

    def receive_round_result_message(self, winners, hand_info, round_state):
        reward = self._calculate_reward(round_state["seats"])
        profile = (
            self.opponent_actions if self.opponent_actions["total"] >= 5
            else self.opponent_stats_total
        )
        next_state_key = self.agent.get_state_key([], round_state, profile)
        self.fold_medium_prob = max(self.fold_medium_prob * 0.95, 0.05)

        if self.trainable:
            for state_key, action in self.state_action_history:
                self.agent.update(
                    state_key,
                    action,
                    reward,
                    next_state_key,
                    valid_actions=[{"action": "fold"}, {"action": "call"}, {"action": "raise"}]
                )

    def _calculate_reward(self, final_seats):
        for s in final_seats:
            if s['uuid'] == self.uuid:
                reward = s['stack'] - self.initial_stack
                return reward*4 if reward >= 0 else reward * 8
        return 0
