import random
import pickle
from collections import defaultdict

class QLearningPokerAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # state -> action -> value
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.actions = actions  # ['fold', 'call', 'raise']

        # Counters for analysis
        self.hand_strength_counts = defaultdict(int)
        self.behavior_counts = defaultdict(int)
        self.pot_bucket_counts = defaultdict(int)

    def evaluate_hand_strength(self, hole_card):
        print("Hole card", hole_card)

        if len(hole_card) < 2:
            self.hand_strength_counts["weak"] += 1
            return "unknown"

        # Poker rank order from high to low for sorting
        rank_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
                      '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
                      '4': 4, '3': 3, '2': 2}

        # Extract rank and suit
        ranks = [card[1:] for card in hole_card]  # e.g., 'HK' -> 'K'
        suits = [card[0] for card in hole_card]   # e.g., 'HK' -> 'H'
        is_suited = suits[0] == suits[1]
        is_pair = ranks[0] == ranks[1]

        # Sort ranks using poker order (not alphabetically)
        sorted_ranks = sorted(ranks, key=lambda r: rank_order[r], reverse=True)
        r1, r2 = sorted_ranks
        val1, val2 = rank_order[r1], rank_order[r2]
        diff = abs(val1 - val2)

        # STRONG: any pair, A-x suited, AK/AQ, Broadway suited
        if is_pair:
            print("Hand: Strong (Pair)")
            self.hand_strength_counts["strong"] += 1
            return "strong"
        if (r1 == 'A' and val2 >= 9) or (is_suited and r1 == 'A'):
            print("Hand: Strong (A-x or suited A)")
            self.hand_strength_counts["strong"] += 1
            return "strong"
        if (r1, r2) in [('A', 'K'), ('A', 'Q'), ('K', 'Q'), ('K', 'J'), ('Q', 'J')] or (is_suited and val1 >= 11 and val2 >= 10):
            print("Hand: Strong (Broadway combo)")
            self.hand_strength_counts["strong"] += 1
            return "strong"

        # MEDIUM: suited connectors like J-T, 9-8; high kicker A-x or K-x; offsuited broadways
        if is_suited and diff == 1 and val1 >= 6:
            print("Hand: Medium (Suited connector)")
            self.hand_strength_counts["medium"] += 1
            return "medium"
        if diff == 1 and val1 >= 10:
            print("Hand: Medium (Offsuit connector)")
            self.hand_strength_counts["medium"] += 1
            return "medium"
        if r1 == 'A' or r2 == 'A' or r1 == 'K' or r2 == 'K':
            print("Hand: Medium (High kicker)")
            self.hand_strength_counts["medium"] += 1
            return "medium"

        print("Hand: Weak")
        self.hand_strength_counts["weak"] += 1
        return "weak"

    def abstract_state(self, hole_card, round_state):
        # Get the strength from our evaluate_hand_strength function
        strength = self.evaluate_hand_strength(hole_card)

        # Get the information from the round_state
        pot_size = round_state["pot"]["main"]["amount"]
        total_stack = sum(seat["stack"] + seat.get("bet", 0) for seat in round_state["seats"])
        relative_pot = pot_size / total_stack if total_stack > 0 else 0

        # Dynamically bucket based on relative pot size
        if relative_pot < 0.1:
            pot_bucket = "very_small"
            print("very small bucket")
        elif relative_pot < 0.25:
            pot_bucket = "small"
            print("small bucket")
        elif relative_pot < 0.5:
            pot_bucket = "medium"
            print("small medium")
        elif relative_pot < 0.75:
            pot_bucket = "large"
            print("large bucket")
        else:
            pot_bucket = "very_large"
            print("very large bucket")

        self.pot_bucket_counts[pot_bucket] += 1

        street = round_state["street"]

        # Return the hand_strength, the street, and the amount in the pot
        return f"{strength}_{street}_{pot_bucket}"

    def abstract_history(self, round_state, opponent_profile=None):
        if opponent_profile is None or opponent_profile.get("total", 0) == 0:
            self.behavior_counts["unknown"] += 1
            return "unknown"

        raise_ratio = opponent_profile["raise"] / opponent_profile["total"]
        call_ratio = opponent_profile["call"] / opponent_profile["total"]

        if raise_ratio > 0.6:
            print("Behavior: Maniac")
            self.behavior_counts["maniac"] += 1
            return "maniac"
        elif raise_ratio > 0.3:
            print("Behavior: Aggressive")
            self.behavior_counts["aggressive"] += 1
            return "aggressive"
        elif call_ratio > 0.5:
            print("Behavior: Calling station")
            self.behavior_counts["calling_station"] += 1
            return "calling_station"
        else:
            print("Behavior: Tight")
            self.behavior_counts["tight"] += 1
            return "tight"

    def get_state_key(self, hole_card, round_state, opponent_profile=None):
        state = self.abstract_state(hole_card, round_state)
        behavior = self.abstract_history(round_state, opponent_profile)
        return f"{state}_{behavior}"

    def choose_action(self, state_key, valid_actions):
        if state_key not in self.q_table:
            self.q_table[state_key] = {a["action"]: 0.0 for a in valid_actions}

        if random.random() < self.epsilon:
            return random.choice(valid_actions)["action"]
        else:
            return max(valid_actions, key=lambda a: self.q_table[state_key].get(a["action"], 0))["action"]

    def update(self, prev_state_key, action, reward, next_state_key, valid_actions):
        if prev_state_key not in self.q_table:
            self.q_table[prev_state_key] = {}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}

        for a in valid_actions:
            act = a["action"]
            self.q_table[prev_state_key].setdefault(act, 0.0)
            self.q_table[next_state_key].setdefault(act, 0.0)

        current_q = self.q_table[prev_state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())

        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_table[prev_state_key][action] = new_q

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
