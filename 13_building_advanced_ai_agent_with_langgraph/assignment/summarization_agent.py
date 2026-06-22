"""Final synthesis agent for all three research routes."""

from __future__ import annotations

import argparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from config import get_llm
from state import AgentState


SUMMARIZER_PROMPT = """You are the final synthesis agent.
Answer the user's question using only the supplied research context and relevant
conversation context. Produce clear Markdown with:
1. a direct answer first;
2. concise supporting details;
3. inline source markers such as [1] when numbered web sources are supplied, or
   paper/page citations when PDF passages are supplied;
4. a short Sources section when sources exist.

Do not fabricate facts or sources. If the context is insufficient, state what is
missing. Keep the response proportional to the question."""


def summarization_node(state: AgentState) -> dict:
    sources = state.get("sources", [])
    source_list = "\n".join(
        f"[{number}] {source['title']} — {source['url']}"
        for number, source in enumerate(sources, start=1)
    )
    user_prompt = f"""Question:
{state['query']}

Selected route: {state.get('route', 'standalone')}

Research context:
{state.get('gathered_context', '') or 'No research context was returned.'}

Available sources:
{source_list or 'None'}"""
    response = get_llm().invoke(
        [
            SystemMessage(content=SUMMARIZER_PROMPT),
            *state.get("messages", [])[-6:-1],
            HumanMessage(content=user_prompt),
        ]
    )
    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def build_summarization_graph():
    builder = StateGraph(AgentState)
    builder.add_node("summarize", summarization_node)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the summarization agent")
    parser.add_argument("query", help="Original question")
    parser.add_argument("context", help="Research context to summarize")
    args = parser.parse_args()
    result = build_summarization_graph().invoke(
        {"query": args.query, "gathered_context": args.context, "sources": []}
    )
    print(result["final_answer"])


if __name__ == "__main__":
    main()

