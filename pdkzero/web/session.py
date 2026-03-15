from __future__ import annotations

from pathlib import Path
from random import Random
from threading import Lock

import torch

from pdkzero.agents.deep_agent import DeepAgent
from pdkzero.dmc.models import CandidateScoringModel
from pdkzero.env.observation import ACTION_VECTOR_DIM, STATE_VECTOR_DIM
from pdkzero.game.cards import parse_cards
from pdkzero.game.engine import GameEngine, InfoSet
from pdkzero.game.move import Move, MoveType

MOVE_TYPE_LABELS = {
    MoveType.INVALID: "非法",
    MoveType.PASS: "过牌",
    MoveType.SINGLE: "单牌",
    MoveType.PAIR: "对子",
    MoveType.STRAIGHT: "顺子",
    MoveType.SERIAL_PAIR: "连对",
    MoveType.TRIPLE: "三不带",
    MoveType.TRIPLE_WITH_SINGLE: "三带一",
    MoveType.TRIPLE_WITH_TWO_SINGLES: "三带二",
    MoveType.TRIPLE_WITH_PAIR: "三带对子",
    MoveType.FOUR_WITH_KICKERS: "四带",
    MoveType.AIRPLANE: "裸飞机",
    MoveType.AIRPLANE_WITH_SINGLES: "飞机带单",
    MoveType.AIRPLANE_WITH_PAIRS: "飞机带对",
    MoveType.BOMB: "炸弹",
}


class WebGameSession:
    def __init__(self, checkpoint_path: Path, human_seat: int = 0, seed: int | None = None) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self.human_seat = human_seat
        self._rng = Random(seed) if seed is not None else None
        self._lock = Lock()
        self.turn_id = 0
        self.model = _load_model(self._checkpoint_path)
        self.agents = tuple(
            DeepAgent(self.model, device="cpu", seed=None if seed is None else seed + seat)
            for seat in range(4)
        )
        self.engine: GameEngine | None = None
        self.new_game()

    def new_game(self) -> dict:
        with self._lock:
            self.engine = GameEngine.deal() if self._rng is None else GameEngine.deal(self._rng)
            self._advance_ai_turns()
            self.turn_id += 1
            return self._serialize_state()

    def get_state(self) -> dict:
        with self._lock:
            return self._serialize_state()

    def play(self, action_index: int, turn_id: int | None = None, action_id: str | None = None) -> dict:
        with self._lock:
            if self.engine is None:
                raise RuntimeError("game session is not initialized")
            if self.engine.is_game_over:
                raise ValueError("game is already over")
            if self.engine.current_player != self.human_seat:
                raise PermissionError("it is not the human player's turn")
            if turn_id != self.turn_id:
                raise PermissionError("stale action payload")

            infoset = self.engine.infoset()
            if action_index < 0 or action_index >= len(infoset.legal_actions):
                raise IndexError("invalid action index")
            expected_action_id = _serialize_move(infoset.legal_actions[action_index])["action_id"]
            if action_id != expected_action_id:
                raise PermissionError("stale action payload")

            self.engine.play(infoset.legal_actions[action_index])
            self._advance_ai_turns()
            self.turn_id += 1
            return self._serialize_state()

    def play_cards(self, cards: list[str] | tuple[str, ...], turn_id: int | None = None) -> dict:
        with self._lock:
            if self.engine is None:
                raise RuntimeError("game session is not initialized")
            if self.engine.is_game_over:
                raise ValueError("game is already over")
            if self.engine.current_player != self.human_seat:
                raise PermissionError("it is not the human player's turn")
            if turn_id != self.turn_id:
                raise PermissionError("stale action payload")
            if not cards:
                raise ValueError("no cards selected")

            self.engine.play(parse_cards(cards))
            self._advance_ai_turns()
            self.turn_id += 1
            return self._serialize_state()

    def _advance_ai_turns(self) -> None:
        assert self.engine is not None
        while not self.engine.is_game_over and self.engine.current_player != self.human_seat:
            infoset = self.engine.infoset()
            action = self.agents[self.engine.current_player].act(infoset)
            self.engine.play(action)

    def _serialize_state(self) -> dict:
        assert self.engine is not None
        infoset = self.engine.infoset(self.human_seat)
        human_turn = not self.engine.is_game_over and self.engine.current_player == self.human_seat
        legal_actions = self.engine.infoset().legal_actions if human_turn else tuple()
        return {
            "human_seat": self.human_seat,
            "current_player": self.engine.current_player,
            "turn_id": self.turn_id,
            "hand_cards": [card.label for card in infoset.hand_cards],
            "all_hands": [
                [card.label for card in self.engine.infoset(seat).hand_cards]
                for seat in range(4)
            ],
            "cards_left": list(self.engine.infoset().cards_left),
            "lead_move": _serialize_move(self.engine.lead_move),
            "lead_player": self.engine.lead_player,
            "legal_actions": [_serialize_move(action) for action in legal_actions],
            "history": [
                {"player": record.player, "move": _serialize_move(record.move)}
                for record in self.engine.history[-20:]
            ],
            "is_game_over": self.engine.is_game_over,
            "winner": self.engine.winner,
            "scores": self.engine.scores,
            "status_text": _status_text(self.engine, human_turn),
        }


def _load_model(checkpoint_path: Path) -> CandidateScoringModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dim = checkpoint.get("state_dim")
    action_dim = checkpoint.get("action_dim")
    if state_dim is None or action_dim is None or "state_dict" not in checkpoint:
        raise ValueError(f"invalid checkpoint format: {checkpoint_path}")
    if state_dim != STATE_VECTOR_DIM or action_dim != ACTION_VECTOR_DIM:
        raise ValueError(
            f"incompatible checkpoint dimensions: got state_dim={state_dim}, action_dim={action_dim}, "
            f"expected state_dim={STATE_VECTOR_DIM}, action_dim={ACTION_VECTOR_DIM}"
        )
    state_dict = checkpoint["state_dict"]
    history_hidden_dim = checkpoint.get("history_hidden_dim")
    hidden_dim = checkpoint.get("hidden_dim")
    if history_hidden_dim is None:
        history_hidden_dim = int(state_dict["history_encoder.bias_ih_l0"].shape[0] // 4)
    if hidden_dim is None:
        hidden_dim = int(state_dict["value_head.0.weight"].shape[0])
    model = CandidateScoringModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        history_hidden_dim=history_hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _serialize_move(move: Move | None) -> dict | None:
    if move is None:
        return None
    cards = [card.label for card in move.cards]
    if move.move_type is MoveType.PASS:
        text = "PASS"
    else:
        text = " ".join(cards)
    return {
        "text": text,
        "cards": cards,
        "move_type_name": MOVE_TYPE_LABELS[move.move_type],
        "move_type": move.move_type.name,
        "action_id": f"{move.move_type.name}:{text}",
    }


def _status_text(engine: GameEngine, human_turn: bool) -> str:
    if engine.is_game_over:
        if engine.winner == 0:
            return "本局结束，你赢了"
        return f"本局结束，{engine.winner} 号座位获胜"
    if human_turn:
        return "轮到你出牌"
    return f"AI[{engine.current_player}] 行动中"
