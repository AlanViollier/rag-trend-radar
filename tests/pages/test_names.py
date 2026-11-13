from rag_trend_radar.pages.radar.names import WIRE_NAME, rough

GAS = ["gas", "pipeline", "storage"]
POWER = ["power", "california", "grid"]


def test_a_name_uses_the_best_words_that_say_something() -> None:
    assert rough([0], [["enron", "gas", "thanks", "pipeline", "storage", "said"]]) == [
        (0, "gas pipeline storage")
    ]


def test_word_order_alone_doesnt_change_a_name() -> None:
    words = [GAS] * 12 + [list(reversed(GAS))] * 12
    assert len(rough(list(range(24)), words)) == 1


def test_a_passing_subject_doesnt_rename_a_blob() -> None:
    words = [GAS] * 12 + [POWER] * 2 + [GAS] * 12
    assert [name for _, name in rough(list(range(26)), words)] == ["gas pipeline storage"]


def test_a_subject_that_lasts_does() -> None:
    words = [GAS] * 12 + [POWER] * 30
    runs = rough(list(range(42)), words)
    assert [name for _, name in runs] == ["gas pipeline storage", "power california grid"]


def test_press_clips_get_one_plain_name() -> None:
    assert rough([0], [["mentions", "dow", "jones", "company"]]) == [(0, WIRE_NAME)]


def test_a_week_with_nothing_borrows_the_nearest_keywords() -> None:
    assert rough([0, 20], [GAS, GAS])[0] == (0, "gas pipeline storage")
