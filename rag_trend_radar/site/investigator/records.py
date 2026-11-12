"""What page 2 holds: the chosen runs, each step as the page replays it, the emails it names."""

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from rag_trend_radar.schema import Citation, Record, SearchStage


class ChosenRun(Record):
    """A run picked for the page, a line of `evals/investigator_runs.jsonl`, in page order.

    `id` is the question's id in the agent's sets; `asker` names an employee asking about their own
    mailbox, empty for the investigator."""

    id: str
    label: str
    group: str
    asker: str = ""


class Baseline(Record):
    """The same question asked to the same model without the mail or tools: "ask without RAG"."""

    text: str
    found: Literal["yes", "no", "partly"] | None
    seconds: float
    cost: float
    system: str


class Thought(Record):
    """Claude's summarised thinking before the turn that follows it."""

    kind: Literal["think"] = "think"
    text: str


class Note(Record):
    """What Claude wrote between tool calls."""

    kind: Literal["note"] = "note"
    text: str


class Turn(Record):
    """One call to Claude, as long as it took."""

    kind: Literal["turn"] = "turn"
    seconds: float


class Hit(Record):
    """A search result: its email, and whether it was people's own mail, press, bulk or planted."""

    document_id: int
    source: str


class ToolCall(Record):
    """A tool call as recorded, and what it showed when run again on the store.

    A search has `stages` and `hits`; read_email the email it `read`; other tools their `text`."""

    kind: Literal["tool"] = "tool"
    tool: str
    arguments: dict[str, JsonValue]
    summary: str
    seconds: float
    stages: tuple[SearchStage, ...] = ()
    hits: tuple[Hit, ...] = ()
    read: int | None = None
    text: str = ""


Step = Annotated[Thought | Note | Turn | ToolCall, Field(discriminator="kind")]


class PageEmail(Record):
    """An email the page names, as its asker can read it. `quoted`: known only from replies."""

    kind: Literal["email", "quoted", "planted"]
    date: date | None
    subject: str
    sender: str
    to: tuple[str, ...] = ()
    cc: int = 0


class PageRun(Record):
    """One recorded run, whole, ready to replay."""

    id: str
    label: str
    question: str
    person: str
    asker: str
    staged: bool
    steps: tuple[Step, ...]
    answer: str
    found: Literal["yes", "no", "partly"] | None
    citations: tuple[Citation, ...]
    cost_usd: float
    seconds: float
    model: str
    baseline: Baseline


class PageGroup(Record):
    """A heading of the question list and its runs."""

    title: str
    runs: tuple[PageRun, ...]


class InvestigatorPage(Record):
    """Everything inside page 2: the runs by group, and every email they name by id."""

    groups: tuple[PageGroup, ...]
    emails: dict[int, PageEmail]
