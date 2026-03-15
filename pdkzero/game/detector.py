from __future__ import annotations

from collections import Counter
from itertools import groupby

from pdkzero.game.cards import Card, sort_cards
from pdkzero.game.move import Move, MoveType


SPECIAL_SERIAL_PAIR_AA22_VALUE = 100


def detect_move(cards: tuple[Card, ...] | list[Card]) -> Move:
    ordered_cards = sort_cards(cards)
    if not ordered_cards:
        return Move.pass_move()

    ranks = [card.rank for card in ordered_cards]
    counts = Counter(ranks)
    count_values = sorted(counts.values(), reverse=True)
    length = len(ordered_cards)

    if length == 1:
        return Move(ordered_cards, MoveType.SINGLE, primary_value=ranks[0], chain_length=1)

    if length == 2 and len(counts) == 1:
        rank = next(iter(counts))
        return Move(ordered_cards, MoveType.PAIR, primary_value=rank, chain_length=1)

    straight = _detect_straight(ordered_cards, counts)
    if straight is not None:
        return straight

    serial_pair = _detect_serial_pair(ordered_cards, counts)
    if serial_pair is not None:
        return serial_pair

    if length == 3 and len(counts) == 1:
        rank = next(iter(counts))
        return Move(ordered_cards, MoveType.TRIPLE, primary_value=rank, chain_length=1)

    if length == 4 and len(counts) == 1:
        rank = next(iter(counts))
        return Move(ordered_cards, MoveType.BOMB, primary_value=rank, chain_length=1)

    if length == 4 and count_values == [3, 1]:
        triple_rank = _rank_with_count(counts, 3)
        return Move(
            ordered_cards,
            MoveType.TRIPLE_WITH_SINGLE,
            primary_value=triple_rank,
            chain_length=1,
        )

    if length == 5 and count_values == [3, 2]:
        triple_rank = _rank_with_count(counts, 3)
        return Move(
            ordered_cards,
            MoveType.TRIPLE_WITH_PAIR,
            primary_value=triple_rank,
            chain_length=1,
        )

    if length == 5 and count_values == [3, 1, 1]:
        triple_rank = _rank_with_count(counts, 3)
        return Move(
            ordered_cards,
            MoveType.TRIPLE_WITH_TWO_SINGLES,
            primary_value=triple_rank,
            chain_length=1,
        )

    if length in {5, 6, 7} and 4 in count_values:
        quad_rank = _rank_with_count(counts, 4)
        if quad_rank == 3 and length != 5:
            return Move.invalid(ordered_cards)
        return Move(
            ordered_cards,
            MoveType.FOUR_WITH_KICKERS,
            primary_value=quad_rank,
            chain_length=1,
        )

    airplane = _detect_airplane(ordered_cards, counts)
    if airplane is not None:
        return airplane

    return Move.invalid(ordered_cards)


def can_beat(candidate: Move, lead: Move) -> bool:
    if candidate.move_type is MoveType.INVALID or candidate.move_type is MoveType.PASS:
        return False
    if lead.move_type in {MoveType.INVALID, MoveType.PASS}:
        return candidate.move_type is not MoveType.INVALID

    if candidate.move_type is MoveType.BOMB and lead.move_type is not MoveType.BOMB:
        return True
    if lead.move_type is MoveType.BOMB:
        return candidate.move_type is MoveType.BOMB and candidate.primary_value > lead.primary_value

    if (
        lead.move_type is MoveType.TRIPLE_WITH_TWO_SINGLES
        and candidate.move_type is MoveType.TRIPLE_WITH_PAIR
        and candidate.length == lead.length
    ):
        return candidate.primary_value > lead.primary_value

    if candidate.move_type is not lead.move_type:
        return False

    if candidate.length != lead.length:
        return False

    return candidate.primary_value > lead.primary_value


def _detect_straight(cards: tuple[Card, ...], counts: Counter[int]) -> Move | None:
    if len(cards) < 5 or any(count != 1 for count in counts.values()):
        return None

    rank_set = set(counts)
    if 14 in rank_set and 15 in rank_set:
        if len(cards) == 5 and rank_set == {14, 15, 3, 4, 5}:
            return Move(cards, MoveType.STRAIGHT, primary_value=5, chain_length=5)
        return None

    if _is_consecutive(sorted(rank_set)):
        return Move(cards, MoveType.STRAIGHT, primary_value=max(rank_set), chain_length=len(cards))

    low_ranks = sorted(1 if rank == 14 else 2 if rank == 15 else rank for rank in rank_set)
    if 14 not in rank_set and _is_consecutive(low_ranks):
        return Move(cards, MoveType.STRAIGHT, primary_value=max(low_ranks), chain_length=len(cards))

    return None


def _detect_serial_pair(cards: tuple[Card, ...], counts: Counter[int]) -> Move | None:
    if len(cards) < 4 or len(cards) % 2 != 0 or any(count != 2 for count in counts.values()):
        return None

    rank_set = set(counts)
    if rank_set == {14, 15} and len(cards) == 4:
        return Move(
            cards,
            MoveType.SERIAL_PAIR,
            primary_value=SPECIAL_SERIAL_PAIR_AA22_VALUE,
            chain_length=2,
        )

    if 15 in rank_set:
        return None

    ordered_ranks = sorted(rank_set)
    if _is_consecutive(ordered_ranks):
        return Move(
            cards,
            MoveType.SERIAL_PAIR,
            primary_value=max(ordered_ranks),
            chain_length=len(cards) // 2,
        )
    return None


def _detect_airplane(cards: tuple[Card, ...], counts: Counter[int]) -> Move | None:
    triple_ranks = sorted(rank for rank, count in counts.items() if count == 3)
    if len(triple_ranks) < 2 or 15 in triple_ranks or not _is_consecutive(triple_ranks):
        return None

    chain_length = len(triple_ranks)
    remaining = len(cards) - chain_length * 3
    side_counts = [count for rank, count in counts.items() if rank not in triple_ranks]

    if remaining == 0:
        return Move(
            cards,
            MoveType.AIRPLANE,
            primary_value=max(triple_ranks),
            chain_length=chain_length,
        )

    if remaining == chain_length and all(count == 1 for count in side_counts):
        return Move(
            cards,
            MoveType.AIRPLANE_WITH_SINGLES,
            primary_value=max(triple_ranks),
            chain_length=chain_length,
        )

    if remaining == chain_length * 2 and all(count == 2 for count in side_counts):
        return Move(
            cards,
            MoveType.AIRPLANE_WITH_PAIRS,
            primary_value=max(triple_ranks),
            chain_length=chain_length,
        )

    return None


def _rank_with_count(counts: Counter[int], expected_count: int) -> int:
    for rank, count in counts.items():
        if count == expected_count:
            return rank
    raise ValueError(f"no rank with count {expected_count}")


def _is_consecutive(values: list[int]) -> bool:
    if not values:
        return False
    return all(current + 1 == nxt for current, nxt in zip(values, values[1:]))
