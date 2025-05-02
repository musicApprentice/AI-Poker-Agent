from pypokerengine.players import BasePokerPlayer
from qlearning_core import QLearningPokerAgent
import pickle

class QLearningPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.agent = QLearningPokerAgent(actions=["fold", "call", "raise"])
        try:
            with open("q_table01.pkl", "rb") as f:
                self.agent.q_table = pickle.load(f)
                print("Loaded existing Q-table.")
        except:
            print("No pretrained Q-table found, starting fresh.")

        self.last_state = None
        self.last_action = None
        self.initial_stack = None

        # Short-term stats (per round)
        self.opponent_actions = {"raise": 0, "call": 0, "fold": 0, "total": 0}
        # Long-term stats (cumulative)
        self.opponent_stats_total = {"raise": 0, "call": 0, "fold": 0, "total": 0}

    def declare_action(self, valid_actions, hole_card, round_state):
        # Use per-hand stats if available; otherwise fall back to long-term stats
        profile = (
            self.opponent_actions if self.opponent_actions["total"] >= 5
            else self.opponent_stats_total
        )
        state_key = self.agent.get_state_key(hole_card, round_state, profile)
        action = self.agent.choose_action(state_key, valid_actions)
        self.last_state = state_key
        self.last_action = action
        return action

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        for s in seats:
            if s['uuid'] == self.uuid:
                self.initial_stack = s['stack']
                break
        self.opponent_actions = {"raise": 0, "call": 0, "fold": 0, "total": 0}

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
        self.agent.update(
            self.last_state,
            self.last_action,
            reward,
            next_state_key,
            valid_actions=[{"action": "fold"}, {"action": "call"}, {"action": "raise"}]
        )

    def _calculate_reward(self, final_seats):
        for s in final_seats:
            if s['uuid'] == self.uuid:
                return s['stack'] - self.initial_stack
        return 0
