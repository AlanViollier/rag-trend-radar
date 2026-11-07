from datetime import date, timedelta

from rag_trend_radar.trends import alerts, quiet

START = date(2001, 1, 7)
TOPIC = quiet.Trait(chunks=200, top_sender=0.1, footers=0.0, empty_subjects=0, chatty_words=0)
LETTER = quiet.Trait(chunks=200, top_sender=0.9, footers=0.0, empty_subjects=0, chatty_words=0)
EVERYTHING = 5000
## 400 chunks against a normal of 50: eight times its usual, 8% of all mail.
FAST, SLOW = 350.0, 100.0


def _weeks(surges: list[float], **changes: object) -> list[alerts.Week]:
    base: dict[str, object] = {"recent": 400, "other_seed": True, "trait": TOPIC}
    return [
        alerts.Week(i, START + timedelta(weeks=i), surge=s, **(base | changes))  # type: ignore[arg-type]
        for i, s in enumerate(surges)
    ]


def _days(weeks: list[alerts.Week], everything: int = EVERYTHING) -> list[date]:
    every = {START + timedelta(weeks=i): everything for i in range(40)}
    return [a.day for a in alerts.surges({0: weeks}, every)]


def test_a_blob_four_times_its_normal_fires_that_week() -> None:
    assert _days(_weeks([SLOW, FAST])) == [START + timedelta(weeks=1)]


def test_a_blob_under_four_times_its_normal_stays_quiet() -> None:
    assert _days(_weeks([SLOW, 299.0])) == []


def test_a_fast_blob_under_one_percent_of_all_mail_stays_quiet() -> None:
    assert _days(_weeks([FAST]), everything=50_000) == []


def test_a_blob_the_other_seed_missed_never_fires() -> None:
    assert _days(_weeks([FAST], other_seed=False)) == []


def test_a_blob_fires_again_only_after_eight_quiet_weeks() -> None:
    assert _days(_weeks([FAST] * 12)) == [START, START + timedelta(weeks=9)]


def test_a_newsletter_so_far_never_fires() -> None:
    assert _days(_weeks([FAST], trait=LETTER)) == []


def test_a_blob_known_as_a_topic_then_fires_whatever_it_becomes_later() -> None:
    early = _weeks([FAST])
    later = early + _weeks([SLOW] * 8, trait=LETTER)[1:]
    assert _days(early) == _days(later) == [START]


def test_the_alert_records_the_blobs_mail_against_its_normal() -> None:
    every = {START: EVERYTHING}
    assert alerts.surges({7: _weeks([FAST])}, every) == [alerts.Alert(START, 7, 400, 50.0)]
