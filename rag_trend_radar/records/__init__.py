"""Every record that crosses a module boundary, one file per area; import them from here."""

from rag_trend_radar.records.agent import AgentQuestion, AgentRun, AgentScore, AgentStep
from rag_trend_radar.records.base import Record
from rag_trend_radar.records.evals import EvalQuestion, FactQuestion, FactResult, RadarEvent
from rag_trend_radar.records.mail import (
    Copy,
    DayFile,
    Document,
    EmailHeaders,
    ParsedEmail,
    QuotedBlock,
    RawFile,
    Rejected,
    Scan,
    SplitBody,
)
from rag_trend_radar.records.runs import RadarRun, RecallRun, RunManifest, RunStage, StageSummary
from rag_trend_radar.records.search import Answer, Citation, Ranked, SearchHit, SearchStage

__all__ = [
    "AgentQuestion",
    "AgentRun",
    "AgentScore",
    "AgentStep",
    "Answer",
    "Citation",
    "Copy",
    "DayFile",
    "Document",
    "EmailHeaders",
    "EvalQuestion",
    "FactQuestion",
    "FactResult",
    "ParsedEmail",
    "QuotedBlock",
    "RadarEvent",
    "RadarRun",
    "Ranked",
    "RawFile",
    "RecallRun",
    "Record",
    "Rejected",
    "RunManifest",
    "RunStage",
    "Scan",
    "SearchHit",
    "SearchStage",
    "SplitBody",
    "StageSummary",
]
