"""The agent's tools: which ones a person gets, as Claude sees them, and one way to run any."""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from anthropic.types.beta import BetaToolParam
from pydantic import ValidationError

from rag_trend_radar.agent.desk import Desk, ToolInput, ToolResult
from rag_trend_radar.agent.tools import (
    alerts,
    find_person,
    mail_counts,
    read_email,
    search,
    top_topics,
    topic_history,
)
from rag_trend_radar.exceptions import ToolRefused

# Bump when any tool's behaviour changes without its description changing: saved runs are keyed
# by it (a fixed mail_counts was read back as the broken run's answer, 2026-11-10).
TOOLS_VERSION = 2


@dataclass(frozen=True)
class Tool:
    """One tool: its name and description for Claude, the arguments it takes, what runs it."""

    name: str
    description: str
    arguments: type[ToolInput]
    run: Callable[[Desk, Any], ToolResult]
    investigator_only: bool


def _tool(module: ModuleType, investigator_only: bool = False) -> Tool:
    name = module.__name__.rsplit(".", 1)[-1]
    return Tool(name, module.DESCRIPTION, module.Input, module.run, investigator_only)


# The radar's tables mix everyone's mail: row-level security already gives an employee no rows,
# and an employee isn't offered the tools either.
TOOLS = {
    tool.name: tool
    for tool in (
        _tool(search),
        _tool(read_email),
        _tool(find_person),
        _tool(mail_counts),
        _tool(top_topics, investigator_only=True),
        _tool(topic_history, investigator_only=True),
        _tool(alerts, investigator_only=True),
    )
}


def offered(desk: Desk) -> list[Tool]:
    """The tools this person may call, in a fixed order (a changed order breaks the prompt cache)."""
    return [tool for tool in TOOLS.values() if desk.investigator or not tool.investigator_only]


def definitions(desk: Desk) -> list[BetaToolParam]:
    """The offered tools as the API takes them, each schema from its pydantic arguments."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.arguments.model_json_schema(),
        }
        for t in offered(desk)
    ]


def call(desk: Desk, name: str, arguments: object) -> ToolResult:
    """Run tool `name` on Claude's arguments, checked against its schema first.

    Raises `ToolRefused` for a tool this person isn't offered or arguments that don't fit; the
    loop hands the message back to Claude as an error result so it can try again.
    """
    tool = next((t for t in offered(desk) if t.name == name), None)
    if tool is None:
        raise ToolRefused(f"no tool {name!r} for {desk.person}")
    try:
        given = tool.arguments.model_validate(arguments)
    except ValidationError as error:
        raise ToolRefused(f"{name}: {error}") from error
    return tool.run(desk, given)
