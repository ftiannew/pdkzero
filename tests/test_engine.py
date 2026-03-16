from random import Random
from pdkzero.game.engine import GameEngine
from pdkzero.game.cards import parse_cards, Card
from pdkzero.game.move import MoveType


def test_deal_creates_4_players():
    engine = GameEngine.deal()
    assert len(engine.hands) == 4
    for hand in engine.hands:
        assert len(hand) == 13


def test_heart_three_holder_starts():
    hands = (
        parse_cards("H3 D4 C5 H6 D7"),
        parse_cards("H5 D6 C7 H8 D9"),
        parse_cards("HJ DQ CK HA D2"),
        parse_cards("H2 D8 C9 HK D2"),
    )
    engine = GameEngine(hands=hands)
    assert engine.current_player == 0
    assert engine.starting_player == 0


def test_legal_actions_opening():
    hands = (
        parse_cards("H3 D4 C5 H6 D7"),
        parse_cards("H5 D6 C7 H8 D9"),
        parse_cards("HJ DQ CK HA D2"),
        parse_cards("H2 D8 C9 HK D2"),
    )
    engine = GameEngine(hands=hands)
    legal = engine.legal_actions()
    assert all(move.contains_heart_three for move in legal)
    assert len(legal) > 0


def test_play_single():
    hands = (
        parse_cards("H3 D4 C5 H6 D7"),
        parse_cards("H5 D6 C7 H8 D9"),
        parse_cards("HJ DQ CK HA D2"),
        parse_cards("H2 D8 C9 HK D2"),
    )
    engine = GameEngine(hands=hands)
    engine.play("H3")
    assert len(engine.hands[0]) == 12
    assert engine.current_player == 1


def test_play_pass():
    hands = (
        parse_cards("H3 D4 C5 H6 D7"),
        parse_cards("H5 D6 C7 H8 D9"),
        parse_cards("HJ DQ CK HA D2"),
        parse_cards("H2 D8 C9 HK D2"),
    )
    engine = GameEngine(hands=hands)
    engine.play("H3")
    engine.play_pass()
    assert engine.current_player == 2
    engine.play_pass()
    assert engine.current_player == 3
    engine.play_pass()
    assert engine.lead_move is None


def test_game_over_when_last_card_played():
    hand0 = [Card(3, "H"), Card(4, "D"), Card(5, "C"), Card(6, "H")]
    hand1 = parse_cards("H5 D6 C7 H8 D9")
    hand2 = parse_cards("HJ DQ CK HA D2")
    hand3 = parse_cards("H2 D8 C9 HK D2")
    engine = GameEngine(hands=(tuple(hand0), hand1, hand2, hand3), current_player=0)
    engine.play("H3")
    engine.play_pass()
    engine.play_pass()
    engine.play_pass()
    engine.play("H4")
    assert engine.is_game_over is True
    assert engine.winner == 0


def test_lead_move_changes():
    hands = (
        parse_cards("H3 D4 C5 H6 D7"),
        parse_cards("H5 D6 C7 H8 D9"),
        parse_cards("HJ DQ CK HA D2"),
        parse_cards("H2 D8 C9 HK D2"),
    )
    engine = GameEngine(hands=hands)
    engine.play("H3")
    assert engine.lead_move is not None
    engine.play_pass()
    engine.play_pass()
    engine.play_pass()
    assert engine.lead_move is None


def test_infoset_legal_actions():
    hands = (
        parse_cards("H3 D4 C5 H6 D7"),
        parse_cards("H5 D6 C7 H8 D9"),
        parse_cards("HJ DQ CK HA D2"),
        parse_cards("H2 D8 C9 HK D2"),
    )
    engine = GameEngine(hands=hands)
    infoset = engine.infoset()
    assert len(infoset.legal_actions) > 0


def test_random_deal_reproducible():
    rng = Random(42)
    engine1 = GameEngine.deal(rng)
    rng2 = Random(42)
    engine2 = GameEngine.deal(rng2)
    assert engine1.hands == engine2.hands
