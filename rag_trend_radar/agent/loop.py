"""The agent: Claude picks a tool, reads what comes back, and goes on until it answers.

A plain loop on the Anthropic SDK (STATE, 2026-11-10): one agent and seven tools don't need a
graph framework. Every step lands in the run's trace as it happens, for the terminal and page 2.
"""

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaToolChoiceAutoParam,
    BetaToolChoiceNoneParam,
    BetaToolParam,
    BetaToolResultBlockParam,
)

from rag_trend_radar import claude
from rag_trend_radar.agent import tools
from rag_trend_radar.agent.desk import Desk
from rag_trend_radar.agent.prompt import MAX_TOOL_CALLS, PROMPT_VERSION, system, verdict
from rag_trend_radar.answer import SOURCE_PREFIX
from rag_trend_radar.exceptions import AnswerRefused, ToolRefused
from rag_trend_radar.run_log import commit
from rag_trend_radar.schema import AgentRun, AgentStep, Citation

# A question that runs past this stops calling tools and answers with what it has. A normal
# question is expected at 5 to 8 cents (2026-11-10 estimate, before the smoke run).
MAX_COST_USD = 0.40
OUT_OF_CALLS = "No more tool calls: answer now from what you already have."
NO_TOOLS: BetaToolChoiceNoneParam = {"type": "none"}
ANY_TOOL: BetaToolChoiceAutoParam = {"type": "auto"}

type Watcher = Callable[[AgentStep], None]


def run(
    desk: Desk,
    question: str,
    model: str,
    client: anthropic.Anthropic | None = None,
    watch: Watcher | None = None,
) -> AgentRun:
    """Work `question` as `desk.person` on `desk.as_of` and return the whole run.

    A run already made with the same question, person, day, model, prompt and tools is read back
    from disk at no cost (rule 6). `watch` sees every step as it happens; `client` is the
    Anthropic client (default: the one on the key in .env).
    """
    offered = tools.definitions(desk)
    prompt = system(desk)
    cached = _cache_path(desk, question, model, prompt, offered)
    if cached.exists():
        saved = AgentRun.model_validate_json(cached.read_text())
        for step in saved.steps:
            (watch or _ignore)(step)
        return saved
    client = client or claude.client()
    started, clock = datetime.now(UTC), time.perf_counter()
    steps: list[AgentStep] = []

    def record(step: AgentStep) -> None:
        steps.append(step)
        (watch or _ignore)(step)

    messages: list[BetaMessageParam] = [{"role": "user", "content": question}]
    calls, cost = 0, 0.0
    while True:
        out_of_calls = calls >= MAX_TOOL_CALLS or cost >= MAX_COST_USD
        begun = time.perf_counter()
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=prompt,
            tools=offered,
            tool_choice=NO_TOOLS if out_of_calls else ANY_TOOL,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            # Multi-step tool use: medium keeps the thinking short between calls (claude-api
            # guidance for Sonnet 5.5); the summaries are what page 2 shows.
            output_config={"effort": "medium"},
            # The history only grows, so each call reads the last one's prefix from the cache.
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        turn = _turn(model, response, time.perf_counter() - begun)
        cost += turn.cost_usd
        calling = response.stop_reason == "tool_use"
        for block in response.content:
            if block.type == "thinking" and block.thinking:
                record(AgentStep(kind="thinking", text=block.thinking))
            elif block.type == "text" and calling and block.text.strip():
                record(AgentStep(kind="note", text=block.text.strip()))
        record(turn)
        if response.stop_reason == "refusal":
            raise AnswerRefused(f"{model} declined: {response.stop_details}")
        # Appended unchanged: thinking blocks are only valid in the conversation that made them.
        messages.append({"role": "assistant", "content": response.content})
        uses = [block for block in response.content if block.type == "tool_use"]
        if response.stop_reason != "tool_use" or not uses:
            break
        results: list[BetaToolResultBlockParam] = []
        for use in uses:
            calls += 1
            results.append(_use(desk, use.id, use.name, use.input, out_of_calls, record))
        messages.append({"role": "user", "content": results})
    answer, found = verdict("".join(b.text for b in response.content if b.type == "text"))
    result = AgentRun(
        question=question,
        person=desk.person,
        as_of=desk.as_of,
        model=model,
        prompt_version=PROMPT_VERSION,
        steps=tuple(steps),
        answer=answer,
        found=found,
        citations=_citations(response),
        cost_usd=cost,
        seconds=time.perf_counter() - clock,
        started=started,
        commit=commit(),
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(result.model_dump_json())
    return result


def _use(
    desk: Desk, use_id: str, name: str, arguments: object, refused: bool, record: Watcher
) -> BetaToolResultBlockParam:
    # One tool call: run it and record it. A refusal goes back to Claude as an error result, so
    # it can correct its arguments or move on.
    begun = time.perf_counter()
    sent = json.dumps(arguments, ensure_ascii=False)
    try:
        if refused:
            raise ToolRefused(OUT_OF_CALLS)
        done = tools.call(desk, name, arguments)
    except ToolRefused as error:
        seconds = time.perf_counter() - begun
        record(
            AgentStep(
                kind="tool",
                tool=name,
                arguments=sent,
                summary=str(error),
                error=True,
                seconds=seconds,
            )
        )
        return {
            "type": "tool_result",
            "tool_use_id": use_id,
            "content": str(error),
            "is_error": True,
        }
    record(
        AgentStep(
            kind="tool",
            tool=name,
            arguments=sent,
            summary=done.summary,
            documents=done.documents,
            seconds=time.perf_counter() - begun,
        )
    )
    return {"type": "tool_result", "tool_use_id": use_id, "content": list(done.blocks)}


def _turn(model: str, response: BetaMessage, seconds: float) -> AgentStep:
    usage = response.usage
    written, read = usage.cache_creation_input_tokens or 0, usage.cache_read_input_tokens or 0
    return AgentStep(
        kind="turn",
        seconds=seconds,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read=read,
        cache_write=written,
        cost_usd=claude.cost_usd(model, usage.input_tokens, usage.output_tokens, written, read),
    )


def _citations(response: BetaMessage) -> tuple[Citation, ...]:
    # Citations name the search result's `source` ("document 123"): that is how one finds its
    # document, whichever tool call returned it.
    return tuple(
        Citation(document_id=int(c.source.removeprefix(SOURCE_PREFIX)), cited_text=c.cited_text)
        for block in response.content
        if block.type == "text"
        for c in block.citations or []
        if c.type == "search_result_location" and c.source.startswith(SOURCE_PREFIX)
    )


def _cache_path(
    desk: Desk, question: str, model: str, prompt: str, offered: list[BetaToolParam]
) -> Path:
    # A changed tool description or schema is a different agent: its runs aren't reused.
    tools_hash = hashlib.blake2b(
        json.dumps(offered, sort_keys=True).encode(), digest_size=8
    ).hexdigest()
    key = json.dumps(
        [PROMPT_VERSION, model, desk.person, desk.as_of.isoformat(), question, prompt, tools_hash]
    )
    return claude.cache_path("agent", key)


def _ignore(_: AgentStep) -> None:
    pass
