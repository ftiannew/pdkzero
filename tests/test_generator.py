from pdkzero.game.generator import generate_legal_actions
from pdkzero.game.move import MoveType
from pdkzero.game.detector import detect_move
from pdkzero.game.cards import parse_cards


def test_generate_legal_actions_opening():
    hand = "H3 D4 C5 H6 D7"
    legal = generate_legal_actions(hand, lead_move=None, is_opening_move=True)
    assert all(move.contains_heart_three for move in legal)
    assert len(legal) > 0


def test_generate_legal_actions_single_lead():
    hand = "H3 D4 C5 H6 D7"
    lead = detect_move(parse_cards("H5"))
    legal = generate_legal_actions(hand, lead_move=lead, is_opening_move=False)
    assert any(move.move_type is MoveType.PASS for move in legal)
    playable = [m for m in legal if m.move_type is not MoveType.PASS]
    assert all(m.primary_value > 5 for m in playable)


def test_generate_legal_actions_bomb_can_beat_non_bomb():
    hand = "H3 D3 C3 S3 D5"
    lead = detect_move(parse_cards("H8"))
    legal = generate_legal_actions(hand, lead_move=lead, is_opening_move=False)
    bomb_actions = [m for m in legal if m.move_type is MoveType.BOMB]
    assert len(bomb_actions) > 0


def test_generate_legal_actions_no_duplicates():
    hand = "H3 D3 C3 H4 D4 C4 H5 D5 C5"
    legal = generate_legal_actions(hand, lead_move=None, is_opening_move=False)
    moves_by_key = {}
    for m in legal:
        key = (m.move_type, m.primary_value, m.length)
        moves_by_key[key] = m
    assert len(legal) == len(moves_by_key)


def test_generate_legal_actions_empty_hand():
    legal = generate_legal_actions((), lead_move=None, is_opening_move=False)
    assert len(legal) == 0


def test_generate_legal_actions_triple_with_pair():
    hand = "H3 D3 C3 H5 D5"
    lead = detect_move(parse_cards("H3 D3 C3 H7 D8"))
    legal = generate_legal_actions(hand, lead_move=lead, is_opening_move=False)
    triple_with_pair = [m for m in legal if m.move_type is MoveType.TRIPLE_WITH_PAIR]
    assert len(triple_with_pair) > 0


def test_generate_legal_actions_triple_with_two_beats_triple_with_pair():
    hand = "H3 D3 C3 H5 D5 H7"
    lead = detect_move(parse_cards("H4 D4 C4 H6 D6"))
    legal = generate_legal_actions(hand, lead_move=lead, is_opening_move=False)
    triple_with_two = [m for m in legal if m.move_type is MoveType.TRIPLE_WITH_TWO_SINGLES]
    assert len(triple_with_two) > 0


def test_generate_legal_actions_pass_when_cannot_beat():
    hand = "H3 D4 C5 H6"
    lead = detect_move(parse_cards("HJ"))
    legal = generate_legal_actions(hand, lead_move=lead, is_opening_move=False)
    assert any(move.move_type is MoveType.PASS for move in legal)


def test_generate_legal_actions_3333_four_with_one():
    hand = "H3 D3 C3 S3 D5"
    legal = generate_legal_actions(hand, lead_move=None, is_opening_move=False)
    four_with_one = [m for m in legal if m.move_type is MoveType.FOUR_WITH_KICKERS]
    assert len(four_with_one) > 0
    four_with_three = [m for m in legal if m.move_type is MoveType.FOUR_WITH_KICKERS and m.length == 7]
    assert len(four_with_three) == 0
