from __future__ import annotations

from dataclasses import dataclass, field
from random import Random, SystemRandom
from typing import Iterable
from collections import Counter

from pdkzero.game.cards import Card, contains_heart_three, make_standard_deck, parse_cards, sort_cards
from pdkzero.game.detector import detect_move
from pdkzero.game.generator import generate_legal_actions
from pdkzero.game.move import Move, MoveType


@dataclass(frozen=True)
class TurnRecord:
    player: int
    move: Move


@dataclass(frozen=True)
class InfoSet:
    player_index: int
    current_player: int
    hand_cards: tuple[Card, ...]
    legal_actions: tuple[Move, ...]
    lead_move: Move | None
    lead_player: int | None
    cards_left: tuple[int, ...]
    played_cards: tuple[tuple[Card, ...], ...]
    history: tuple[TurnRecord, ...]
    is_opening_move: bool
    trick_index: int


@dataclass(frozen=True)
class CompensationCandidate:
    responsible_player: int
    protected_player: int
    trick_index: int


class GameEngine:
    def __init__(
        self,
        hands: tuple[tuple[Card, ...], ...],
        current_player: int | None = None,
        lead_move: Move | None = None,
        lead_player: int | None = None,
        is_opening_move: bool = True,
    ) -> None:
        if len(hands) != 4:
            raise ValueError("4-player Paodekuai is required")

        self.hands = [list(sort_cards(hand)) for hand in hands]
        self.played_cards = [[] for _ in range(4)]
        self.history: list[TurnRecord] = []
        self.lead_move = lead_move
        self.lead_player = lead_player
        self.is_opening_move = is_opening_move
        self.passes_since_lead = 0
        self.trick_index = 0
        self.pending_compensation: CompensationCandidate | None = None
        self.compensation_player: int | None = None

        self.is_game_over = False
        self.winner: int | None = None
        self.scores: dict[int, int] | None = None

        self.starting_player = current_player if current_player is not None else self._find_heart_three_holder()
        self.current_player = self.starting_player

    @classmethod
    def deal(cls, rng: Random | None = None) -> "GameEngine":
        deck = list(make_standard_deck())
        random = rng or SystemRandom()
        random.shuffle(deck)
        hands = tuple(sort_cards(tuple(deck[index * 13 : (index + 1) * 13])) for index in range(4))
        return cls(hands=hands)

    def infoset(self, player_index: int | None = None) -> InfoSet:
        index = self.current_player if player_index is None else player_index
        return InfoSet(
            player_index=index,
            current_player=self.current_player,
            hand_cards=tuple(self.hands[index]),
            legal_actions=self.legal_actions() if index == self.current_player and not self.is_game_over else tuple(),
            lead_move=self.lead_move,
            lead_player=self.lead_player,
            cards_left=tuple(len(hand) for hand in self.hands),
            played_cards=tuple(tuple(cards) for cards in self.played_cards),
            history=tuple(self.history),
            is_opening_move=self.is_opening_move,
            trick_index=self.trick_index,
        )

    def legal_actions(self) -> tuple[Move, ...]:
        if self.is_game_over:
            return tuple()
        return generate_legal_actions(
            hand=tuple(self.hands[self.current_player]),
            lead_move=self.lead_move,
            is_opening_move=self.is_opening_move and self.lead_move is None,
        )

    def play(self, action: str | Iterable[Card] | Move) -> Move:
        move = self._coerce_move(action)
        self._validate_action(move)

        active_player = self.current_player
        self._track_compensation_risk(move)
        self.history.append(TurnRecord(player=active_player, move=move))

        if move.move_type is MoveType.PASS:
            self._apply_pass()
            return move

        self._remove_cards(active_player, move.cards)
        self.played_cards[active_player].extend(move.cards)
        self.lead_move = move
        self.lead_player = active_player
        self.passes_since_lead = 0
        self.is_opening_move = False

        if len(self.hands[active_player]) == 0:
            self.is_game_over = True
            self.winner = active_player
            self._finalize_scores()
            return move

        self.current_player = self._next_player(active_player)
        self._clear_stale_compensation(active_player)
        return move

    def play_pass(self) -> Move:
        return self.play(Move.pass_move())

    def _coerce_move(self, action: str | Iterable[Card] | Move) -> Move:
        if isinstance(action, Move):
            return action
        if isinstance(action, str):
            return detect_move(parse_cards(action))
        return detect_move(tuple(action))

    def _validate_action(self, move: Move) -> None:
        if self.is_opening_move and self.lead_move is None and not move.contains_heart_three:
            raise ValueError("opening move must include H3")
        legal_actions = self.legal_actions()
        if not any(self._same_action_shape(move, legal) for legal in legal_actions):
            raise ValueError(f"illegal action for player {self.current_player}: {move}")

    def _apply_pass(self) -> None:
        if self.lead_move is None:
            raise ValueError("cannot pass without a lead move")
        self.passes_since_lead += 1
        if self.passes_since_lead == 3:
            self.current_player = self.lead_player if self.lead_player is not None else self.current_player
            self.lead_move = None
            self.lead_player = None
            self.passes_since_lead = 0
            self.pending_compensation = None
            self.trick_index += 1
            return

        active_player = self.current_player
        self.current_player = self._next_player(active_player)
        self._clear_stale_compensation(active_player)

    def _remove_cards(self, player: int, cards: tuple[Card, ...]) -> None:
        hand = self.hands[player]
        for card in cards:
            hand.remove(card)

    def _find_heart_three_holder(self) -> int:
        for player, hand in enumerate(self.hands):
            if contains_heart_three(hand):
                return player
        raise ValueError("one player must hold H3")

    def _next_player(self, player: int) -> int:
        return (player + 1) % 4

    def _track_compensation_risk(self, chosen_move: Move) -> None:
        if self.pending_compensation is not None:
            if self.current_player == self.pending_compensation.protected_player:
                return
            self.pending_compensation = None

        if self.lead_move is None or self.lead_move.move_type is not MoveType.SINGLE:
            return

        protected = self._next_player(self.current_player)
        if len(self.hands[protected]) != 1:
            return

        beating_singles = [
            action
            for action in self.legal_actions()
            if action.move_type is MoveType.SINGLE and action.primary_value > self.lead_move.primary_value
        ]
        if not beating_singles:
            return

        highest_single = max(action.primary_value for action in beating_singles)
        if chosen_move.move_type is MoveType.SINGLE and chosen_move.primary_value == highest_single:
            return

        self.pending_compensation = CompensationCandidate(
            responsible_player=self.current_player,
            protected_player=protected,
            trick_index=self.trick_index,
        )

    def _clear_stale_compensation(self, acting_player: int) -> None:
        if self.pending_compensation is None:
            return
        if acting_player == self.pending_compensation.protected_player:
            self.pending_compensation = None

    def _finalize_scores(self) -> None:
        assert self.winner is not None
        if (
            self.pending_compensation is not None
            and self.pending_compensation.protected_player == self.winner
            and self.pending_compensation.trick_index == self.trick_index
        ):
            self.compensation_player = self.pending_compensation.responsible_player

        scores: dict[int, int] = {}
        total_loss = 0
        for player, hand in enumerate(self.hands):
            if player == self.winner:
                continue
            loss = 0 if len(hand) == 1 else len(hand)
            scores[player] = -loss
            total_loss += loss

        if self.compensation_player is not None:
            for player in list(scores):
                scores[player] = 0
            scores[self.compensation_player] = -total_loss

        scores[self.winner] = total_loss
        self.scores = scores

    @staticmethod
    def _same_action_shape(left: Move, right: Move) -> bool:
        return (
            left.move_type is right.move_type
            and Counter(card.rank for card in left.cards) == Counter(card.rank for card in right.cards)
        )
