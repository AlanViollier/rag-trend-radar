from datetime import UTC, datetime, timedelta

from rag_trend_radar.ingest.dedup import content_hash, resolve_sent_at

SENT = datetime(2001, 5, 4, 15, 51, tzinfo=UTC)


def test_copies_match_across_whitespace_and_subject_case() -> None:
    assert content_hash("RE: Plan", "see you\n at 3 ") == content_hash("re: plan ", "see you at 3")


def test_a_body_opening_on_a_blank_line_is_the_same_email() -> None:
    assert content_hash("plan", "\n\nat 3") == content_hash("plan", "at 3")


def test_different_bodies_differ() -> None:
    assert content_hash("plan", "at 3") != content_hash("plan", "at 4")


def test_one_date_is_certain() -> None:
    assert resolve_sent_at([SENT, SENT]) == (SENT, False)


def test_a_date_stamped_to_the_second_wins() -> None:
    server = SENT.replace(second=47) - timedelta(hours=5)
    assert resolve_sent_at([SENT, SENT - timedelta(hours=12), server]) == (server, False)


def test_disagreeing_minute_dates_keep_the_earliest_flagged() -> None:
    early = SENT - timedelta(hours=10)
    assert resolve_sent_at([SENT, early]) == (early, True)
