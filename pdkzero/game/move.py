from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from pdkzero.game.cards import Card, contains_heart_three, sort_cards


class MoveType(Enum):
    INVALID = auto()
    PASS = auto()
    SINGLE = auto()
    PAIR = auto()
    STRAIGHT = auto()
    SERIAL_PAIR = auto()
    TRIPLE = auto()
    TRIPLE_WITH_SINGLE = auto()
    TRIPLE_WITH_TWO_SINGLES = auto()
    TRIPLE_WITH_PAIR = auto()
    FOUR_WITH_KICKERS = auto()
    AIRPLANE = auto()
    AIRPLANE_WITH_SINGLES = auto()
    AIRPLANE_WITH_PAIRS = auto()
    BOMB = auto()


@dataclass(frozen=True)
class Move:
    cards: tuple[Card, ...] = field(default_factory=tuple)
    move_type: MoveType = MoveType.INVALID
    primary_value: int = 0
    length: int = 0
    chain_length: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cards", sort_cards(self.cards))
        if self.length == 0:
            object.__setattr__(self, "length", len(self.cards))

    @property
    def is_pass(self) -> bool:
        return self.move_type is MoveType.PASS

    @property
    def is_bomb(self) -> bool:
        return self.move_type is MoveType.BOMB

    @property
    def contains_heart_three(self) -> bool:
        return contains_heart_three(self.cards)

    @classmethod
    def invalid(cls, cards: tuple[Card, ...]) -> "Move":
        return cls(cards=cards, move_type=MoveType.INVALID, length=len(cards))

    @classmethod
    def pass_move(cls) -> "Move":
        return cls(cards=tuple(), move_type=MoveType.PASS, length=0)
