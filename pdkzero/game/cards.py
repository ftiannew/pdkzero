from __future__ import annotations

from dataclasses import dataclass

SUITS = ("C", "D", "H", "S")
SUIT_ORDER = {suit: index for index, suit in enumerate(SUITS)}
RANK_LABELS = {
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
    15: "2",
}
LABEL_RANKS = {label: rank for rank, label in RANK_LABELS.items()}


@dataclass(frozen=True)
class Card:
    rank: int
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in RANK_LABELS:
            raise ValueError(f"unsupported rank: {self.rank}")
        if self.suit not in SUIT_ORDER:
            raise ValueError(f"unsupported suit: {self.suit}")

    @property
    def label(self) -> str:
        return f"{self.suit}{RANK_LABELS[self.rank]}"

    @property
    def is_heart_three(self) -> bool:
        return self.rank == 3 and self.suit == "H"


def sort_cards(cards: tuple[Card, ...] | list[Card]) -> tuple[Card, ...]:
    return tuple(sorted(cards, key=lambda card: (card.rank, SUIT_ORDER[card.suit])))


def parse_card(token: str) -> Card:
    token = token.strip().upper()
    if len(token) < 2:
        raise ValueError(f"invalid card token: {token!r}")
    suit = token[0]
    rank_token = token[1:]
    if rank_token not in LABEL_RANKS:
        raise ValueError(f"invalid rank token: {rank_token!r}")
    return Card(rank=LABEL_RANKS[rank_token], suit=suit)


def parse_cards(cards: str | list[str] | tuple[str, ...]) -> tuple[Card, ...]:
    if isinstance(cards, str):
        tokens = [token for token in cards.split() if token]
    else:
        tokens = list(cards)
    return sort_cards([parse_card(token) for token in tokens])


def make_standard_deck() -> tuple[Card, ...]:
    return tuple(Card(rank=rank, suit=suit) for rank in range(3, 16) for suit in SUITS)


def contains_heart_three(cards: tuple[Card, ...] | list[Card]) -> bool:
    return any(card.is_heart_three for card in cards)
