"""Shared state types for the multi-agent research workflow."""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


Route = Literal["llm", "web", "rag"]


class Source(TypedDict):
    title: str
    url: str


class WebResult(TypedDict):
    query: str
    title: str
    url: str
    content: str
    score: float | None


def resettable_add(existing: list, update: list) -> list:
    """Append parallel results, or clear them at the start of a new user turn."""
    if update and isinstance(update[0], dict) and update[0].get("__reset__"):
        return []
    return existing + update


class AgentState(TypedDict, total=False):
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    route: Route
    route_reason: str
    route_history: list[Route]
    reflection_count: int
    information_sufficient: bool
    research_queries: list[str]
    web_results: Annotated[list[WebResult], resettable_add]
    gathered_context: str
    sources: list[Source]
    final_answer: str


class WebSearchState(TypedDict):
    search_query: str
