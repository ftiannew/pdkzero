from __future__ import annotations

from random import Random

from pdkzero.agents.heuristic_agent import HeuristicAgent
from pdkzero.game.engine import GameEngine


def play_game(agents: tuple, seed: int | None = None, verbose: bool = False) -> dict[int, int]:
    engine = GameEngine.deal() if seed is None else GameEngine.deal(Random(seed))
    while not engine.is_game_over:
        infoset = engine.infoset()
        action = agents[engine.current_player].act(infoset)
        if verbose:
            hand_str = " ".join(c.label for c in infoset.hand_cards)
            move_str = " ".join(c.label for c in action.cards) if action.cards else "PASS"
            print(f"[P{engine.current_player}] {move_str:20} | Hand: {hand_str}")
        engine.play(action)
    assert engine.scores is not None
    if verbose:
        print(f"[Result] Scores: {engine.scores} | Winner: {engine.winner}")
    return engine.scores


def play_many_games(agents: tuple, games: int, seed: int | None = None, verbose: bool = False) -> list[dict[int, int]]:
    if seed is None:
        return [play_game(agents=agents, seed=None, verbose=verbose) for _ in range(games)]
    return [play_game(agents=agents, seed=seed + game, verbose=verbose) for game in range(games)]


def average_scores(score_rows: list[dict[int, int]]) -> dict[int, float]:
    totals = {seat: 0.0 for seat in range(4)}
    for row in score_rows:
        for seat, value in row.items():
            totals[seat] += float(value)
    count = max(len(score_rows), 1)
    return {seat: value / count for seat, value in totals.items()}


def default_heuristic_table() -> tuple[HeuristicAgent, HeuristicAgent, HeuristicAgent, HeuristicAgent]:
    return (HeuristicAgent(), HeuristicAgent(), HeuristicAgent(), HeuristicAgent())
