from __future__ import annotations

from random import Random

from pdkzero.env.observation import build_observation
from pdkzero.game.engine import GameEngine


class PaodekuaiEnv:
    def __init__(self, engine: GameEngine | None = None, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = Random(seed) if seed is not None else None
        self.engine = engine or self._deal()

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._seed = seed
            self._rng = Random(seed)
        self.engine = self._deal()
        return self.observe()

    def _deal(self) -> GameEngine:
        if self._rng is None:
            return GameEngine.deal()
        return GameEngine.deal(self._rng)

    def observe(self) -> dict:
        return build_observation(self.engine.infoset())

    def step(self, action) -> tuple[dict | None, int, bool, dict]:
        acting_player = self.engine.current_player
        self.engine.play(action)
        if self.engine.is_game_over:
            assert self.engine.scores is not None
            return None, self.engine.scores[acting_player], True, {"scores": self.engine.scores}
        return self.observe(), 0, False, {}
