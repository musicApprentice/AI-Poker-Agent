# allin_player.py
from pypokerengine.players import BasePokerPlayer
import pprint

class AllInPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # Check if 'raise' is a valid action
        can_raise = False
        for action in valid_actions:
            if action["action"] == "raise":
                can_raise = True
                break

        if can_raise:
            # Found raise is possible, return the action string
            return "raise"
        
        # If 'raise' is not available, check if 'call' is a valid action
        can_call = False
        for action in valid_actions:
            if action["action"] == "call":
                can_call = True
                break

        if can_call:
            # Found call is possible, return the action string
            return "call"

        # If neither 'raise' nor 'call' is available, 'fold' must be the only option
        return "fold"

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
