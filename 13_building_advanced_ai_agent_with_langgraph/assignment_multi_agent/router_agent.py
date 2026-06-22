"""Router agent and general-knowledge LLM node."""

from __future__ import annotations

import argparse
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import SETTINGS, get_llm
from state import AgentState, Route


class ReflectionDecision(BaseModel):
    action: Literal["llm", "web", "rag", "summarize"] = Field(
        description="The next specialist to call, or summarize when evidence is sufficient"
    )
    information_sufficient: bool = Field(
        description="Whether the collected evidence can support a complete answer"
    )
    reason: str = Field(description="A brief evidence-gap or sufficiency explanation")


ROUTER_PROMPT = """You are a reflective router for a research assistant.

First inspect the question, evidence gathered so far, sources, and routes already
attempted. Decide whether the evidence is sufficient for a complete, accurate answer.

- If it is sufficient, choose `summarize` and set information_sufficient=true.
- If it is insufficient, set information_sufficient=false and choose the best next
  specialist. Identify the concrete information gap in your reason.

Specialists:
- rag: questions answerable from the paper 'Attention Is All You Need', including
  Transformers, self-attention, multi-head attention, positional encoding, or the
  architecture/experiments described in that paper.
- web: questions requiring current, latest, live, recently changed, or externally
  verifiable information.
- llm: stable general knowledge, explanation, reasoning, or conversation that does
  not require the paper or current web information.

Do not repeat a route unless its evidence clearly failed to address the question.
Never choose summarize before any evidence has been gathered. Use conversation history
only to resolve references in the newest query."""


def reflect_on_information(
    query: str,
    messages: list | None = None,
    gathered_context: str = "",
    sources: list | None = None,
    route_history: list | None = None,
) -> ReflectionDecision:
    history = messages or []
    prompt_messages = [SystemMessage(content=ROUTER_PROMPT)]
    prompt_messages.extend(history[-6:])
    prompt_messages.append(
        HumanMessage(
            content=f"""Current question:
{query}

Routes already attempted: {route_history or []}

Evidence gathered so far:
{gathered_context or 'None yet'}

Sources gathered so far:
{sources or []}"""
        )
    )
    router = get_llm().with_structured_output(ReflectionDecision)
    return router.invoke(prompt_messages)


def classify_route(query: str, messages: list | None = None) -> ReflectionDecision:
    """Backward-compatible helper for a first-pass routing decision."""
    return reflect_on_information(query, messages)


def router_node(
    state: AgentState,
) -> Command[Literal["llm_agent", "web_planner", "rag_agent", "summarize"]]:
    reflection_count = state.get("reflection_count", 0)
    max_reflections = SETTINGS.reflection.max_steps
    has_evidence = bool(state.get("gathered_context", "").strip())

    if has_evidence and reflection_count >= max_reflections:
        return Command(
            update={
                "information_sufficient": True,
                "route_reason": (
                    "Reflection limit reached; synthesizing the best available evidence."
                ),
            },
            goto="summarize",
        )

    decision = reflect_on_information(
        query=state["query"],
        messages=state.get("messages", []),
        gathered_context=state.get("gathered_context", ""),
        sources=state.get("sources", []),
        route_history=state.get("route_history", []),
    )
    if (
        decision.action == "summarize"
        and decision.information_sufficient
        and has_evidence
    ):
        return Command(
            update={
                "information_sufficient": True,
                "route_reason": decision.reason,
                "reflection_count": reflection_count + 1,
            },
            goto="summarize",
        )

    destinations = {
        "llm": "llm_agent",
        "web": "web_planner",
        "rag": "rag_agent",
    }
    # A premature summarize decision is redirected to the general LLM route.
    action: Route = decision.action if decision.action != "summarize" else "llm"
    return Command(
        update={
            "route": action,
            "route_history": [*state.get("route_history", []), action],
            "route_reason": decision.reason,
            "information_sufficient": False,
            "reflection_count": reflection_count + 1,
        },
        goto=destinations[action],
    )


def llm_agent_node(state: AgentState) -> dict:
    system = SystemMessage(
        content=(
            "Answer the user's stable general-knowledge question accurately. "
            "This is an intermediate research note for a summarizer, so be factual, "
            "concise, and do not invent citations."
        )
    )
    response = get_llm().invoke([system, *state.get("messages", [])[-8:]])
    existing = state.get("gathered_context", "")
    note = f"[General LLM analysis]\n{response.content}"
    return {
        "gathered_context": f"{existing}\n\n{note}".strip(),
        "sources": state.get("sources", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the router agent")
    parser.add_argument("query", nargs="+", help="Question to classify")
    args = parser.parse_args()
    query = " ".join(args.query)
    decision = classify_route(query, [HumanMessage(content=query)])
    print(
        f"Next action: {decision.action}\n"
        f"Information sufficient: {decision.information_sufficient}\n"
        f"Reason: {decision.reason}"
    )


if __name__ == "__main__":
    main()
