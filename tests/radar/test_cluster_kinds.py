import numpy as np

from rag_trend_radar.radar import cluster_kinds


def _text(subject: str, sender: str, body: str = "Gas prices rose.") -> str:
    return f"Subject: {subject}\nFrom: {sender}\nDate: 2001-01-02\n\n{body}"


def _trait(**changes: float) -> cluster_kinds.Trait:
    base = {
        "chunks": 100,
        "top_sender": 0.1,
        "footers": 0.0,
        "empty_subjects": 0,
        "chatty_words": 0,
    }
    return cluster_kinds.Trait(**(base | changes))  # type: ignore[arg-type]


def test_a_header_gives_the_sender_and_the_subject_without_re() -> None:
    mail = cluster_kinds.read([_text("Re: Gas", "A@b.c")])
    assert (mail.senders[0], mail.subjects[0]) == ("a@b.c", "gas")


def test_a_week_is_read_from_the_mail_the_cluster_held() -> None:
    mail = cluster_kinds.read(
        [
            _text("Gas", "a@b.c"),
            _text("Gas", "a@b.c", "This message is intended only for you."),
            _text("Hi", "c@d.e"),
            "From: c@d.e\n\nNo subject here.",
        ]
    )
    found = cluster_kinds.trait(mail, np.array([0, 1, 3], dtype=np.int64), ["thanks", "gas", "fyi"])
    assert found.chunks == 3
    assert found.top_sender == 2 / 3
    assert found.footers == 1 / 3
    assert found.empty_subjects == 1
    assert found.chatty_words == 2


def test_a_subject_people_write_about_is_a_topic() -> None:
    assert cluster_kinds.kind([_trait()] * 5) is cluster_kinds.Kind.TOPIC


def test_one_sender_makes_a_newsletter() -> None:
    assert cluster_kinds.kind([_trait(top_sender=0.9)] * 5) is cluster_kinds.Kind.NEWSLETTER


def test_mostly_footers_make_a_footer() -> None:
    assert (
        cluster_kinds.kind([_trait(footers=0.8)] * 3 + [_trait()] * 2) is cluster_kinds.Kind.FOOTER
    )


def test_many_empty_subjects_make_chatter_with_no_word_list() -> None:
    assert cluster_kinds.kind([_trait(empty_subjects=20)] * 5) is cluster_kinds.Kind.CHATTER


def test_chatty_keywords_in_most_weeks_make_chatter_and_in_few_dont() -> None:
    assert (
        cluster_kinds.kind([_trait(chatty_words=2)] * 3 + [_trait()] * 2)
        is cluster_kinds.Kind.CHATTER
    )
    assert cluster_kinds.kind([_trait(chatty_words=2)] + [_trait()] * 4) is cluster_kinds.Kind.TOPIC
