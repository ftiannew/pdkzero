from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable

from pdkzero.game.cards import Card, SUIT_ORDER, parse_cards, sort_cards
from pdkzero.game.detector import can_beat, detect_move
from pdkzero.game.move import Move, MoveType


def generate_legal_actions(
    hand: str | Iterable[Card],
    lead_move: Move | None,
    is_opening_move: bool,
) -> tuple[Move, ...]:
    hand_cards = _coerce_cards(hand)
    candidate_moves = _enumerate_candidate_moves(hand_cards)

    if lead_move is None or lead_move.move_type is MoveType.PASS:
        legal_actions = candidate_moves
    else:
        legal_actions = [move for move in candidate_moves if can_beat(move, lead_move)]
        legal_actions.append(Move.pass_move())

    if is_opening_move:
        legal_actions = [move for move in legal_actions if move.contains_heart_three]

    return tuple(sorted(legal_actions, key=_action_sort_key))


def _enumerate_candidate_moves(hand_cards: tuple[Card, ...]) -> tuple[Move, ...]:
    grouped = _group_cards_by_rank(hand_cards)
    moves: list[Move] = []

    moves.extend(_generate_singles(grouped))
    moves.extend(_generate_pairs(grouped))
    moves.extend(_generate_straights(grouped))
    moves.extend(_generate_serial_pairs(grouped))
    moves.extend(_generate_triples(grouped))
    moves.extend(_generate_bombs(grouped))
    moves.extend(_generate_triple_with_single(grouped))
    moves.extend(_generate_triple_with_two_singles(grouped))
    moves.extend(_generate_triple_with_pair(grouped))
    moves.extend(_generate_four_with_kickers(grouped))
    moves.extend(_generate_airplanes(grouped))

    unique = {}
    for move in moves:
        if move.move_type is MoveType.INVALID:
            continue
        key = (move.move_type, _rank_count_signature(move))
        unique.setdefault(key, move)
    return tuple(unique.values())


def _generate_singles(grouped: dict[int, list[Card]]) -> list[Move]:
    return [detect_move((cards[0],)) for cards in grouped.values()]


def _generate_pairs(grouped: dict[int, list[Card]]) -> list[Move]:
    return [detect_move(tuple(cards[:2])) for cards in grouped.values() if len(cards) >= 2]


def _generate_triples(grouped: dict[int, list[Card]]) -> list[Move]:
    return [detect_move(tuple(cards[:3])) for cards in grouped.values() if len(cards) >= 3]


def _generate_bombs(grouped: dict[int, list[Card]]) -> list[Move]:
    return [detect_move(tuple(cards[:4])) for cards in grouped.values() if len(cards) == 4]


def _generate_straights(grouped: dict[int, list[Card]]) -> list[Move]:
    canonical = {rank: cards[0] for rank, cards in grouped.items()}
    available = sorted(rank for rank in grouped if rank != 15)
    moves: list[Move] = []

    for sequence in _consecutive_sequences(available, minimum_length=5):
        for start in range(len(sequence)):
            for end in range(start + 5, len(sequence) + 1):
                ranks = sequence[start:end]
                moves.append(detect_move(tuple(canonical[rank] for rank in ranks)))

    if all(rank in canonical for rank in (14, 15, 3, 4, 5)):
        moves.append(detect_move((canonical[14], canonical[15], canonical[3], canonical[4], canonical[5])))

    if 15 in canonical:
        low_two_sequence = [rank for rank in range(3, 15) if rank in canonical]
        prefix_length = 0
        for expected_rank, actual_rank in enumerate(low_two_sequence, start=3):
            if actual_rank != expected_rank:
                break
            prefix_length += 1
        for length in range(4, prefix_length + 1):
            ranks = list(range(3, 3 + length)) + [15]
            moves.append(detect_move(tuple(canonical[rank] for rank in ranks)))

    return moves


def _generate_serial_pairs(grouped: dict[int, list[Card]]) -> list[Move]:
    pair_ranks = sorted(rank for rank, cards in grouped.items() if len(cards) >= 2 and rank != 15)
    moves: list[Move] = []

    for sequence in _consecutive_sequences(pair_ranks, minimum_length=2):
        for start in range(len(sequence)):
            for end in range(start + 2, len(sequence) + 1):
                ranks = sequence[start:end]
                cards = [card for rank in ranks for card in grouped[rank][:2]]
                moves.append(detect_move(tuple(cards)))

    if 14 in grouped and 15 in grouped and len(grouped[14]) >= 2 and len(grouped[15]) >= 2:
        moves.append(detect_move(tuple(grouped[14][:2] + grouped[15][:2])))

    return moves


def _generate_triple_with_single(grouped: dict[int, list[Card]]) -> list[Move]:
    moves: list[Move] = []
    for triple_rank, cards in grouped.items():
        if len(cards) < 3:
            continue
        triple_cards = cards[:3]
        for kicker_rank, kicker_cards in grouped.items():
            if kicker_rank == triple_rank:
                continue
            moves.append(detect_move(tuple(triple_cards + kicker_cards[:1])))
    return moves


def _generate_triple_with_two_singles(grouped: dict[int, list[Card]]) -> list[Move]:
    moves: list[Move] = []
    ranks = sorted(grouped)
    for triple_rank, cards in grouped.items():
        if len(cards) < 3:
            continue
        triple_cards = cards[:3]
        kicker_ranks = [rank for rank in ranks if rank != triple_rank]
        for left_rank, right_rank in combinations(kicker_ranks, 2):
            kickers = grouped[left_rank][:1] + grouped[right_rank][:1]
            moves.append(detect_move(tuple(triple_cards + kickers)))
    return moves


def _generate_triple_with_pair(grouped: dict[int, list[Card]]) -> list[Move]:
    moves: list[Move] = []
    for triple_rank, cards in grouped.items():
        if len(cards) < 3:
            continue
        triple_cards = cards[:3]
        for pair_rank, pair_cards in grouped.items():
            if pair_rank == triple_rank or len(pair_cards) < 2:
                continue
            moves.append(detect_move(tuple(triple_cards + pair_cards[:2])))
    return moves


def _generate_four_with_kickers(grouped: dict[int, list[Card]]) -> list[Move]:
    moves: list[Move] = []
    for quad_rank, quad_cards in grouped.items():
        if len(quad_cards) != 4:
            continue
        limit = 1 if quad_rank == 3 else 3
        kicker_options = _enumerate_kicker_combinations(grouped, exclude_ranks={quad_rank}, max_cards=limit)
        for kicker_cards in kicker_options:
            if not kicker_cards:
                continue
            moves.append(detect_move(tuple(quad_cards + list(kicker_cards))))
    return moves


def _generate_airplanes(grouped: dict[int, list[Card]]) -> list[Move]:
    triple_ranks = sorted(rank for rank, cards in grouped.items() if len(cards) >= 3 and rank != 15)
    moves: list[Move] = []

    for sequence in _consecutive_sequences(triple_ranks, minimum_length=2):
        for start in range(len(sequence)):
            for end in range(start + 2, len(sequence) + 1):
                chain = sequence[start:end]
                triple_cards = [card for rank in chain for card in grouped[rank][:3]]
                moves.append(detect_move(tuple(triple_cards)))

                single_kickers = _enumerate_distinct_rank_kickers(grouped, exclude_ranks=set(chain), width=len(chain), cards_per_rank=1)
                for kickers in single_kickers:
                    moves.append(detect_move(tuple(triple_cards + list(kickers))))

                pair_kickers = _enumerate_distinct_rank_kickers(grouped, exclude_ranks=set(chain), width=len(chain), cards_per_rank=2)
                for kickers in pair_kickers:
                    moves.append(detect_move(tuple(triple_cards + list(kickers))))
    return moves


def _enumerate_kicker_combinations(
    grouped: dict[int, list[Card]],
    *,
    exclude_ranks: set[int],
    max_cards: int,
) -> list[tuple[Card, ...]]:
    ranks = [rank for rank in sorted(grouped) if rank not in exclude_ranks]
    results: list[tuple[Card, ...]] = []

    def backtrack(index: int, chosen: list[Card]) -> None:
        if 1 <= len(chosen) <= max_cards:
            results.append(tuple(chosen))
        if len(chosen) == max_cards or index >= len(ranks):
            return

        rank = ranks[index]
        available = grouped[rank]
        max_take = min(len(available), max_cards - len(chosen))
        for take in range(0, max_take + 1):
            if take:
                chosen.extend(available[:take])
            backtrack(index + 1, chosen)
            if take:
                del chosen[-take:]

    backtrack(0, [])
    return results


def _enumerate_distinct_rank_kickers(
    grouped: dict[int, list[Card]],
    *,
    exclude_ranks: set[int],
    width: int,
    cards_per_rank: int,
) -> list[tuple[Card, ...]]:
    candidate_ranks = [
        rank for rank, cards in grouped.items() if rank not in exclude_ranks and len(cards) >= cards_per_rank
    ]
    results: list[tuple[Card, ...]] = []
    for ranks in combinations(sorted(candidate_ranks), width):
        cards = [card for rank in ranks for card in grouped[rank][:cards_per_rank]]
        results.append(tuple(cards))
    return results


def _consecutive_sequences(ranks: list[int], minimum_length: int) -> list[list[int]]:
    if not ranks:
        return []
    sequences: list[list[int]] = []
    current = [ranks[0]]
    for rank in ranks[1:]:
        if rank == current[-1] + 1:
            current.append(rank)
        else:
            if len(current) >= minimum_length:
                sequences.append(current)
            current = [rank]
    if len(current) >= minimum_length:
        sequences.append(current)
    return sequences


def _group_cards_by_rank(hand_cards: tuple[Card, ...]) -> dict[int, list[Card]]:
    grouped: dict[int, list[Card]] = defaultdict(list)
    for card in hand_cards:
        grouped[card.rank].append(card)

    for rank in grouped:
        grouped[rank].sort(
            key=lambda card: (
                0 if card.is_heart_three else 1,
                SUIT_ORDER[card.suit],
            )
        )
    return dict(grouped)


def _coerce_cards(hand: str | Iterable[Card]) -> tuple[Card, ...]:
    if isinstance(hand, str):
        return parse_cards(hand)
    return sort_cards(tuple(hand))


def _rank_count_signature(move: Move) -> tuple[tuple[int, int], ...]:
    by_rank: dict[int, int] = defaultdict(int)
    for card in move.cards:
        by_rank[card.rank] += 1
    return tuple(sorted(by_rank.items()))


def _action_sort_key(move: Move) -> tuple[int, int, int, tuple[str, ...]]:
    labels = tuple(card.label for card in move.cards)
    pass_bias = 1 if move.move_type is MoveType.PASS else 0
    return (pass_bias, move.move_type.value, move.length, move.primary_value, labels)
