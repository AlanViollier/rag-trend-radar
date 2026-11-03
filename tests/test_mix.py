import pytest

from rag_trend_radar.ingest.kinds import Source, source
from rag_trend_radar.schema import SearchHit
from rag_trend_radar.search.mix import MAX_PRESS, mix

PRESS = "Subject: Enron Mentions -- 01/25/02\nFrom: ann.schmidt@enron.com\n\nThe stock fell again."
MAIL = "Subject: Re: stock\nFrom: john.arnold@enron.com\n\nI love it, I've died and gone to heaven."
BULK = "Subject: Real Deals from Travelocity.com\n\nFly to Denver for $99."


def _hit(chunk_id: int, text: str, score: float) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, document_id=chunk_id, text=text, score=score)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (PRESS, Source.PRESS),
        (MAIL, Source.MAIL),
        (BULK, Source.BULK),
        ("Subject: FW: news\n\nHOUSTON -(Dow Jones)- Enron said on Monday.", Source.PRESS),
        ("Subject: Hi\nFrom: noreply@ccomad3.uu.commissioner.com\n\nWeek 8 scores.", Source.BULK),
        ("Subject: prices\n\nCheck Bloomberg for the close.", Source.MAIL),
        ("Mary: can someone else handle this?", Source.MAIL),
        (
            "Subject: Demand Ken Lay Donate Proceeds from Enron Stock Sales\n\nDear Mr. Lay,",
            Source.BULK,
        ),
    ],
    ids=[
        "digest",
        "person",
        "deals",
        "forwarded-wire",
        "sender",
        "terminal",
        "no-lead",
        "form-letter",
    ],
)
def test_a_chunks_source_comes_from_its_own_text(text: str, kind: Source) -> None:
    assert source(text) is kind


def test_press_is_capped_and_mail_moves_up() -> None:
    press = [_hit(n, PRESS, 0.99) for n in range(8)]
    mail = [_hit(n, MAIL, 0.5) for n in range(8, 16)]
    chosen = mix([*press, *mail], 10)
    assert sum(source(hit.text) is Source.PRESS for hit in chosen) == MAX_PRESS
    assert len(chosen) == 10


def test_press_comes_back_when_little_else_is_left() -> None:
    press = [_hit(n, PRESS, 0.99) for n in range(8)]
    assert len(mix([*press, _hit(9, MAIL, 0.5)], 6)) == 6


def test_noise_never_takes_the_place_of_a_press_article() -> None:
    press = [_hit(n, PRESS, 0.9) for n in range(5)]
    chosen = mix([*press, _hit(9, MAIL, 0.01)], 5)
    assert [hit.chunk_id for hit in chosen] == [0, 1, 2, 3, 4]


def test_noise_fills_the_list_when_nothing_else_is_left() -> None:
    chosen = mix([_hit(1, PRESS, 0.9), _hit(2, MAIL, 0.01)], 5)
    assert [hit.chunk_id for hit in chosen] == [1, 2]


def test_the_list_stays_best_first() -> None:
    chosen = mix([_hit(n, PRESS, 0.9 - n / 100) for n in range(5)] + [_hit(9, MAIL, 0.5)], 5)
    assert [hit.score for hit in chosen] == sorted((hit.score for hit in chosen), reverse=True)
