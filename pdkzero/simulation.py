from __future__ import annotations

import sys
from random import Random
from typing import TYPE_CHECKING, Any

from pdkzero.agents.heuristic_agent import HeuristicAgent
from pdkzero.game.engine import GameEngine

if TYPE_CHECKING:
    from pdkzero.agents.deep_agent import DeepAgent
    from pdkzero.agents.random_agent import RandomAgent

Agent = Any

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

SUIT_COLORS = {
    "H": RED,
    "D": RED,
    "S": BLUE,
    "C": BLUE,
}
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
RANK_SYMBOLS = {3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 15: "2"}


def get_agent_name(agent: Agent) -> str:
    name = type(agent).__name__
    if name == "DeepAgent":
        return "DeepAgent"
    if name == "HeuristicAgent":
        return "Heuristic "
    if name == "RandomAgent":
        return "Random    "
    return name


def format_card(card) -> str:
    color = SUIT_COLORS.get(card.suit, "")
    suit = SUIT_SYMBOLS.get(card.suit, card.suit)
    rank = RANK_SYMBOLS.get(card.rank, str(card.rank))
    return f"{color}{suit}{rank}{RESET}"


def format_hand(cards) -> str:
    return " ".join(format_card(c) for c in cards)


def play_game(agents: tuple[Agent, ...], seed: int | None = None, verbose: bool = False, show_colors: bool = True) -> dict[int, int]:
    engine = GameEngine.deal() if seed is None else GameEngine.deal(Random(seed))
    step = 0
    while not engine.is_game_over:
        infoset = engine.infoset()
        action = agents[engine.current_player].act(infoset)
        if verbose:
            step += 1
            agent_name = get_agent_name(agents[engine.current_player])
            move_str = format_hand(action.cards) if action.cards else "PASS"
            hand_str = format_hand(infoset.hand_cards)
            print(f"[{step:02d}] [P{engine.current_player}]{agent_name} {move_str:20} | Hand: {hand_str}")
        engine.play(action)
    assert engine.scores is not None
    if verbose:
        print(f"[Result] Scores: {engine.scores} | Winner: {engine.winner}")
    return engine.scores


def play_many_games(agents: tuple[Agent, ...], games: int, seed: int | None = None, verbose: bool = False) -> list[dict[int, int]]:
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
