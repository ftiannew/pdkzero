from __future__ import annotations

import numpy as np

from pdkzero.game.cards import Card
from pdkzero.game.engine import InfoSet
from pdkzero.game.move import Move, MoveType

RANKS = tuple(range(3, 16))
RANK_INDEX = {rank: index for index, rank in enumerate(RANKS)}
ENCODED_MOVE_TYPES = (
    MoveType.PASS,
    MoveType.SINGLE,
    MoveType.PAIR,
    MoveType.STRAIGHT,
    MoveType.SERIAL_PAIR,
    MoveType.TRIPLE,
    MoveType.TRIPLE_WITH_SINGLE,
    MoveType.TRIPLE_WITH_TWO_SINGLES,
    MoveType.TRIPLE_WITH_PAIR,
    MoveType.FOUR_WITH_KICKERS,
    MoveType.AIRPLANE,
    MoveType.AIRPLANE_WITH_SINGLES,
    MoveType.AIRPLANE_WITH_PAIRS,
    MoveType.BOMB,
)
MOVE_TYPE_INDEX = {move_type: index for index, move_type in enumerate(ENCODED_MOVE_TYPES)}
ACTION_VECTOR_DIM = len(RANKS) + len(ENCODED_MOVE_TYPES) + 5
STATE_VECTOR_DIM = len(RANKS) + len(RANKS) * 3 + 3 + ACTION_VECTOR_DIM + 4
HISTORY_LENGTH = 8


def build_observation(infoset: InfoSet) -> dict[str, np.ndarray | tuple[Move, ...] | int]:
    x_no_action = encode_state(infoset)
    z = encode_history(infoset)
    num_actions = len(infoset.legal_actions)
    action_vectors = np.empty((num_actions, ACTION_VECTOR_DIM), dtype=np.float32)
    for index, action in enumerate(infoset.legal_actions):
        action_vectors[index] = encode_move(action)
    x_batch = np.empty((num_actions, STATE_VECTOR_DIM + ACTION_VECTOR_DIM), dtype=np.float32)
    x_batch[:, :STATE_VECTOR_DIM] = x_no_action
    x_batch[:, STATE_VECTOR_DIM:] = action_vectors
    z_batch = np.broadcast_to(z, (num_actions, HISTORY_LENGTH, ACTION_VECTOR_DIM)).copy()
    return {
        "position": infoset.current_player,
        "legal_actions": infoset.legal_actions,
        "action_batch": action_vectors,
        "x_batch": x_batch,
        "z_batch": z_batch,
        "x_no_action": x_no_action,
        "z": z,
    }


def encode_state(infoset: InfoSet) -> np.ndarray:
    player = infoset.player_index
    encoded = np.zeros(STATE_VECTOR_DIM, dtype=np.float32)
    offset = 0

    _encode_cards_into(encoded[offset : offset + len(RANKS)], infoset.hand_cards)
    offset += len(RANKS)

    for relative_player in (1, 2, 3):
        _encode_cards_into(
            encoded[offset : offset + len(RANKS)],
            infoset.played_cards[(player + relative_player) % 4],
        )
        offset += len(RANKS)

    for relative_player in (1, 2, 3):
        encoded[offset] = infoset.cards_left[(player + relative_player) % 4] / 13.0
        offset += 1

    encoded[offset : offset + ACTION_VECTOR_DIM] = encode_move(infoset.lead_move)
    offset += ACTION_VECTOR_DIM
    encoded[offset] = 1.0 if infoset.is_opening_move else 0.0
    encoded[offset + 1] = 1.0 if infoset.lead_move is not None else 0.0
    encoded[offset + 2] = 1.0 if infoset.cards_left[(player + 1) % 4] == 1 else 0.0
    encoded[offset + 3] = 1.0 if infoset.cards_left[(player - 1) % 4] == 1 else 0.0
    return encoded


def encode_history(infoset: InfoSet) -> np.ndarray:
    history = np.zeros((HISTORY_LENGTH, ACTION_VECTOR_DIM), dtype=np.float32)
    recent_records = infoset.history[-HISTORY_LENGTH:]
    start = HISTORY_LENGTH - len(recent_records)
    for index, record in enumerate(recent_records, start=start):
        history[index] = encode_move(record.move)
    return history


def encode_move(move: Move | None) -> np.ndarray:
    encoded = np.zeros(ACTION_VECTOR_DIM, dtype=np.float32)
    if move is None:
        return encoded

    _encode_cards_into(encoded[: len(RANKS)], move.cards)
    type_offset = len(RANKS)
    if move.move_type in MOVE_TYPE_INDEX:
        encoded[type_offset + MOVE_TYPE_INDEX[move.move_type]] = 1.0
    scalar_offset = type_offset + len(ENCODED_MOVE_TYPES)
    encoded[scalar_offset] = move.primary_value / 15.0
    encoded[scalar_offset + 1] = move.length / 13.0
    encoded[scalar_offset + 2] = move.chain_length / 13.0
    encoded[scalar_offset + 3] = 1.0 if move.contains_heart_three else 0.0
    encoded[scalar_offset + 4] = 1.0 if move.move_type is MoveType.BOMB else 0.0
    return encoded


def encode_cards(cards: tuple[Card, ...] | list[Card]) -> np.ndarray:
    encoded = np.zeros(len(RANKS), dtype=np.float32)
    _encode_cards_into(encoded, cards)
    return encoded


def _encode_cards_into(target: np.ndarray, cards: tuple[Card, ...] | list[Card]) -> None:
    for card in cards:
        target[RANK_INDEX[card.rank]] += 1.0
