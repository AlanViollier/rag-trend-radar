from datetime import date

from rag_trend_radar.evals.judge import POOL, pool, quoted
from rag_trend_radar.evals.retrieval import relevant, score
from rag_trend_radar.evals.review import review
from rag_trend_radar.schema import EvalQuestion, Judgment, RecallRun, RunManifest, Verdict


def _q(qid: str, label: str) -> EvalQuestion:
    return EvalQuestion(
        id=qid, question="?", as_of=date(2001, 9, 30), person="p", relevant=(label,)
    )


def _j(qid: str, doc: str, verdict: Verdict, quoted: bool = True) -> Judgment:
    return Judgment(question_id=qid, external_id=doc, verdict=verdict, quote="x", quoted=quoted)


def test_without_judgments_both_label_sets_are_the_originals() -> None:
    strict, judged = relevant([_q("a", "1"), _q("b", "2")])
    assert strict == judged == {"a": {"1"}, "b": {"2"}}


def test_judgments_widen_the_labels_and_drop_questions_their_own_document_fails() -> None:
    judgments = [
        _j("a", "1", Verdict.ANSWERS),
        _j("a", "9", Verdict.ANSWERS),
        _j("a", "8", Verdict.PARTLY),
        _j("a", "7", Verdict.ANSWERS, quoted=False),
        _j("b", "2", Verdict.NO),
    ]
    strict, judged = relevant([_q("a", "1"), _q("b", "2")], judgments)
    assert strict == {"a": {"1"}}
    assert judged == {"a": {"1", "9"}}


def test_any_answering_document_first_is_a_hit_where_recall_would_count_half() -> None:
    ranked = {"a": ["9", "1"]}
    assert score({"a": {"1"}}, ranked, ["hit_rate@1"]) == {"hit_rate@1": 0.0}
    assert score({"a": {"1", "9"}}, ranked, ["hit_rate@1"]) == {"hit_rate@1": 1.0}
    assert score({"a": {"1", "9"}}, ranked, ["recall@1"]) == {"recall@1": 0.5}


def test_the_pool_takes_what_a_judged_method_ranked_above_the_label() -> None:
    run = RecallRun(
        manifest=RunManifest(
            questions="q.jsonl",
            count=1,
            model="m",
            dims=8,
            depth=50,
            ef_search=40,
            reranker="",
            retype="",
            commit="c",
            chunks=1,
            started="2026-10-30T00:00:00Z",
        ),  # fmt: skip
        rankings={
            "hybrid + rerank": {"a": ["5", "1", "6"]},
            "vector + rerank": {"a": [str(n) for n in range(10, 10 + POOL + 5)]},
            "keyword": {"a": ["7", "1"]},
        },
        latency_ms={},
    )
    picked = pool([_q("a", "1")], run)["a"]
    assert picked == {"1", "5"} | {str(n) for n in range(10, 10 + POOL)}


def test_a_quote_counts_only_when_the_document_holds_it() -> None:
    document = "Subject: Hotels\nFrom: anita.martin@enron.com\n\nThe  recruiting\nteam pays."
    assert quoted("the recruiting team pays.", document)
    assert quoted("From: anita.martin@enron.com", document)
    assert not quoted("Sales pays.", document)
    assert not quoted("", document)


def test_the_review_lists_answers_found_elsewhere_and_dropped_questions() -> None:
    run = RecallRun(
        manifest=RunManifest(
            questions="q.jsonl",
            count=2,
            model="m",
            dims=8,
            depth=50,
            ef_search=40,
            reranker="r",
            retype="",
            commit="c",
            chunks=1,
            started="2026-10-30T00:00:00Z",
        ),  # fmt: skip
        rankings={"hybrid + rerank": {"a": ["9", "1"], "b": ["2"]}},
        latency_ms={},
    )
    judgments = [
        _j("a", "1", Verdict.ANSWERS),
        _j("a", "9", Verdict.ANSWERS),
        _j("b", "2", Verdict.NO),
    ]
    texts = {"1": "Label text", "9": "Another answer", "2": "Off topic"}
    page = review([_q("a", "1"), _q("b", "2")], run, judgments, texts)
    found, dropped, missed = page.split("\n## ")[1:4]
    assert "**Judged** `9`" in found
    assert "**Judged** `1`" not in found
    assert "**Judged** `2`" in dropped
    assert "Ranked first** `9`" in missed
