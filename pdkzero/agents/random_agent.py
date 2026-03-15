from __future__ import annotations

from random import Random

from pdkzero.game.engine import InfoSet
from pdkzero.game.move import Move


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = Random(seed)

    def act(self, infoset: InfoSet) -> Move:
        if not infoset.legal_actions:
            raise ValueError("infoset has no legal actions")
        return self._rng.choice(list(infoset.legal_actions))
