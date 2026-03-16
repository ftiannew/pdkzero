from collections import Counter
from pdkzero.game.cards import Card, parse_cards, make_standard_deck, contains_heart_three
from pdkzero.game.detector import detect_move, can_beat, SPECIAL_SERIAL_PAIR_AA22_VALUE
from pdkzero.game.move import MoveType


def test_detect_single():
    cards = parse_cards("H3")
    move = detect_move(cards)
    assert move.move_type == MoveType.SINGLE
    assert move.primary_value == 3


def test_detect_pair():
    cards = parse_cards("H3 D3")
    move = detect_move(cards)
    assert move.move_type == MoveType.PAIR
    assert move.primary_value == 3


def test_detect_triple():
    cards = parse_cards("H3 D3 C3")
    move = detect_move(cards)
    assert move.move_type == MoveType.TRIPLE
    assert move.primary_value == 3


def test_detect_bomb():
    cards = parse_cards("H3 D3 C3 S3")
    move = detect_move(cards)
    assert move.move_type == MoveType.BOMB
    assert move.primary_value == 3


def test_detect_straight():
    cards = parse_cards("H3 D4 C5 H6 D7")
    move = detect_move(cards)
    assert move.move_type == MoveType.STRAIGHT
    assert move.primary_value == 7


def test_detect_straight_with_2():
    cards = parse_cards("H3 D4 C5 H14 D15")
    move = detect_move(cards)
    assert move.move_type == MoveType.STRAIGHT
    assert move.primary_value == 5


def test_detect_serial_pair():
    cards = parse_cards("H3 D3 H4 D4")
    move = detect_move(cards)
    assert move.move_type == MoveType.SERIAL_PAIR
    assert move.primary_value == 4


def test_detect_serial_pair_aa22():
    cards = parse_cards("H14 D14 H15 D15")
    move = detect_move(cards)
    assert move.move_type == MoveType.SERIAL_PAIR
    assert move.primary_value == SPECIAL_SERIAL_PAIR_AA22_VALUE


def test_detect_triple_with_single():
    cards = parse_cards("H3 D3 C3 D5")
    move = detect_move(cards)
    assert move.move_type == MoveType.TRIPLE_WITH_SINGLE
    assert move.primary_value == 3


def test_detect_triple_with_pair():
    cards = parse_cards("H3 D3 C3 H5 D5")
    move = detect_move(cards)
    assert move.move_type == MoveType.TRIPLE_WITH_PAIR
    assert move.primary_value == 3


def test_detect_triple_with_two_singles():
    cards = parse_cards("H3 D3 C3 D5 H7")
    move = detect_move(cards)
    assert move.move_type == MoveType.TRIPLE_WITH_TWO_SINGLES
    assert move.primary_value == 3


def test_detect_airplane():
    cards = parse_cards("H3 D3 C3 H4 D4 C4")
    move = detect_move(cards)
    assert move.move_type == MoveType.AIRPLANE
    assert move.primary_value == 4
    assert move.chain_length == 2


def test_detect_airplane_with_singles():
    cards = parse_cards("H3 D3 C3 H4 D4 C4 D6 H7")
    move = detect_move(cards)
    assert move.move_type == MoveType.AIRPLANE_WITH_SINGLES
    assert move.primary_value == 4
    assert move.chain_length == 2


def test_detect_airplane_with_pairs():
    cards = parse_cards("H3 D3 C3 H4 D4 C4 D6 D6 H7 H7")
    move = detect_move(cards)
    assert move.move_type == MoveType.AIRPLANE_WITH_PAIRS
    assert move.primary_value == 4
    assert move.chain_length == 2


def test_detect_four_with_kickers():
    cards = parse_cards("H3 D3 C3 S3 D5 H7")
    move = detect_move(cards)
    assert move.move_type == MoveType.FOUR_WITH_KICKERS
    assert move.primary_value == 3


def test_3333_must_be_four_with_one():
    cards = parse_cards("H3 D3 C3 S3 H5")
    move = detect_move(cards)
    assert move.move_type == MoveType.FOUR_WITH_KICKERS


def test_3333_invalid_four_with_three():
    cards = parse_cards("H3 D3 C3 S3 H5 D6 H7")
    move = detect_move(cards)
    assert move.move_type == MoveType.INVALID


def test_can_beat_single():
    lead = detect_move(parse_cards("H3"))
    follow = detect_move(parse_cards("H5"))
    assert can_beat(follow, lead) is True


def test_can_beat_single_fail():
    lead = detect_move(parse_cards("H5"))
    follow = detect_move(parse_cards("H3"))
    assert can_beat(follow, lead) is False


def test_bomb_beats_non_bomb():
    bomb = detect_move(parse_cards("H3 D3 C3 S3"))
    single = detect_move(parse_cards("H5"))
    assert can_beat(bomb, single) is True
    assert can_beat(single, bomb) is False


def test_bomb_vs_bomb():
    small_bomb = detect_move(parse_cards("H3 D3 C3 S3"))
    large_bomb = detect_move(parse_cards("H5 D5 C5 S5"))
    assert can_beat(large_bomb, small_bomb) is True
    assert can_beat(small_bomb, large_bomb) is False


def test_triple_with_pair_beats_triple_with_two():
    lead = detect_move(parse_cards("H3 D3 C3 D5 H7"))
    follow = detect_move(parse_cards("H4 D4 C4 H5 D5"))
    assert can_beat(follow, lead) is True


def test_contains_heart_three():
    cards = parse_cards("H3 D5 C7")
    assert contains_heart_three(cards) is True


def test_no_heart_three():
    cards = parse_cards("D5 C7 H9")
    assert contains_heart_three(cards) is False


def test_standard_deck():
    deck = make_standard_deck()
    assert len(deck) == 52
    ranks = [card.rank for card in deck]
    assert Counter(ranks)[3] == 4
    assert Counter(ranks)[15] == 4
