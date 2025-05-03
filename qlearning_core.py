import random
import pickle
from collections import defaultdict

class QLearningPokerAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Q-table: maps state -> action -> value
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor (future reward importance)
        self.epsilon = epsilon  # exploration probability
        self.actions = actions  # e.g., ['fold', 'call', 'raise']

        # Tracking stats for analysis/debugging
        self.hand_strength_counts = defaultdict(int)
        self.behavior_counts = defaultdict(int)
        self.pot_bucket_counts = defaultdict(int)

        self.raise_ratios = []
        self.call_ratios = []

    def evaluate_hand_strength(self, hole_card):
        # Estimates hand strength as 'strong', 'medium', or 'weak'
        if len(hole_card) < 2:
            self.hand_strength_counts["weak"] += 1
            return "unknown"

        rank_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                      '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
                      '4': 4, '3': 3, '2': 2}

        ranks = [card[1:] for card in hole_card]
        suits = [card[0] for card in hole_card]
        is_suited = suits[0] == suits[1]
        is_pair = ranks[0] == ranks[1]

        sorted_ranks = sorted(ranks, key=lambda r: rank_order[r], reverse=True)
        r1, r2 = sorted_ranks
        val1, val2 = rank_order[r1], rank_order[r2]
        diff = abs(val1 - val2)

        if is_pair:
            self.hand_strength_counts["strong"] += 1
            return "strong"
        if (r1 == 'A' and val2 >= 9) or (is_suited and r1 == 'A'):
            self.hand_strength_counts["strong"] += 1
            return "strong"
        if (r1, r2) in [('A', 'K'), ('A', 'Q'), ('K', 'Q'), ('K', 'J'), ('Q', 'J')] or (is_suited and val1 >= 11 and val2 >= 10):
            self.hand_strength_counts["strong"] += 1
            return "strong"

        if is_suited and diff == 1 and val1 >= 6:
            self.hand_strength_counts["medium"] += 1
            return "medium"
        if diff == 1 and val1 >= 10:
            self.hand_strength_counts["medium"] += 1
            return "medium"
        if r1 == 'A' or r2 == 'A' or r1 == 'K' or r2 == 'K':
            self.hand_strength_counts["medium"] += 1
            return "medium"

        self.hand_strength_counts["weak"] += 1
        return "weak"

    def abstract_state(self, hole_card, round_state):
        # Creates a simplified state representation string
        strength = self.evaluate_hand_strength(hole_card)
        pot_size = round_state["pot"]["main"]["amount"]
        relative_pot = pot_size / 20  # normalize by big blind

        if relative_pot < 2:
            pot_bucket = "very_small"
        elif relative_pot < 4:
            pot_bucket = "small"
        elif relative_pot < 8:
            pot_bucket = "medium"
        elif relative_pot < 16:
            pot_bucket = "large"
        else:
            pot_bucket = "very_large"

        self.pot_bucket_counts[pot_bucket] += 1
        street = round_state["street"]

        return f"{strength}_{street}_{pot_bucket}"

    def abstract_history(self, round_state, opponent_profile=None):
        # Returns opponent behavior style: 'maniac', 'tight', etc.
        if opponent_profile is None or opponent_profile.get("total", 0) == 0:
            self.behavior_counts["unknown"] += 1
            return "unknown"

        total = opponent_profile["total"]
        raise_ratio = opponent_profile["raise"] / total
        call_ratio = opponent_profile["call"] / total

        self.raise_ratios.append(raise_ratio)
        self.call_ratios.append(call_ratio)

        if raise_ratio > 0.6:
            self.behavior_counts["maniac"] += 1
            return "maniac"
        elif call_ratio > 0.55 and raise_ratio < 0.3:
            self.behavior_counts["calling_station"] += 1
            return "calling_station"
        elif abs(raise_ratio - call_ratio) < 0.2:
            self.behavior_counts["balanced"] += 1
            return "balanced"
        elif raise_ratio > call_ratio:
            self.behavior_counts["aggressive"] += 1
            return "aggressive"
        else:
            self.behavior_counts["tight"] += 1
            return "tight"

    def get_state_key(self, hole_card, round_state, opponent_profile=None):
        # Combines abstracted state and opponent behavior into one key
        state = self.abstract_state(hole_card, round_state)
        behavior = self.abstract_history(round_state, opponent_profile)
        return f"{state}_{behavior}"

    def choose_action(self, state_key, valid_actions):
        # ε-greedy policy
        if state_key not in self.q_table:
            self.q_table[state_key] = {a["action"]: 0.0 for a in valid_actions}

        if random.random() < self.epsilon:
            return random.choice(valid_actions)["action"]
        return max(valid_actions, key=lambda a: self.q_table[state_key].get(a["action"], 0))["action"]

    def update(self, prev_state_key, action, reward, next_state_key, valid_actions):
        # Standard Q-learning update rule
        self.q_table.setdefault(prev_state_key, {})
        self.q_table.setdefault(next_state_key, {})

        for a in valid_actions:
            act = a["action"]
            self.q_table[prev_state_key].setdefault(act, 0.0)
            self.q_table[next_state_key].setdefault(act, 0.0)

        current_q = self.q_table[prev_state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_table[prev_state_key][action] = new_q

    def save(self, filename):
        # Save Q-table to file
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        # Load Q-table from file
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
