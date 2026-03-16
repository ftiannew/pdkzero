from __future__ import annotations

from pdkzero.game.engine import InfoSet
from pdkzero.game.move import Move, MoveType


class HeuristicAgent:
    def act(self, infoset: InfoSet) -> Move:
        if not infoset.legal_actions:
            raise ValueError("infoset has no legal actions")

        playable = [action for action in infoset.legal_actions if action.move_type is not MoveType.PASS]
        if not playable:
            return infoset.legal_actions[0]

        for opponent_offset in (1, 2, 3):
            opponent = (infoset.current_player + opponent_offset) % 4
            if infoset.cards_left[opponent] == 1:
                single_actions = [action for action in playable if action.move_type is MoveType.SINGLE]
                if single_actions:
                    return max(single_actions, key=lambda action: action.primary_value)

        def action_key(action: Move) -> tuple[int, int, int]:
            bomb_bias = 1 if action.move_type is MoveType.BOMB else 0
            return (bomb_bias, action.length, action.primary_value)

        return min(playable, key=action_key)
