"""CLI and full LangGraph workflow for multi-agent research and summarization."""

from __future__ import annotations

import argparse
import sqlite3
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config import MEMORY_DB
from rag_agent import rag_agent_node
from router_agent import llm_agent_node, router_node
from state import AgentState
from summarization_agent import summarization_node
from web_research_agent import (
    dispatch_web_searches,
    web_planner_node,
    web_reduce_node,
    web_search_node,
)


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("llm_agent", llm_agent_node)
    builder.add_node("web_planner", web_planner_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("web_reduce", web_reduce_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("summarize", summarization_node)

    builder.add_edge(START, "router")
    # router_node uses Command(goto=...) instead of conditional edges.
    # Every specialist returns evidence to the reflective router. Only the router
    # can approve the final hand-off to the summarization agent.
    builder.add_edge("llm_agent", "router")
    builder.add_conditional_edges(
        "web_planner", dispatch_web_searches, ["web_search"]
    )
    builder.add_edge("web_search", "web_reduce")
    builder.add_edge("web_reduce", "router")
    builder.add_edge("rag_agent", "router")
    builder.add_edge("summarize", END)
    return builder.compile(checkpointer=checkpointer)


class ResearchAssistant:
    """Owns the persistent SQLite connection and compiled graph."""

    def __init__(self, memory_path=MEMORY_DB):
        self.connection = sqlite3.connect(str(memory_path), check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.graph = build_graph(self.checkpointer)

    def ask(self, query: str, thread_id: str) -> AgentState:
        initial_state: AgentState = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "research_queries": [],
            # The reducer recognizes this marker and clears prior-turn web evidence.
            "web_results": [{"__reset__": True}],
            "gathered_context": "",
            "sources": [],
            "final_answer": "",
            "route_history": [],
            "reflection_count": 0,
            "information_sufficient": False,
        }
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(initial_state, config=config)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent research and summarization assistant"
    )
    parser.add_argument("query", nargs="*", help="Question to answer")
    parser.add_argument(
        "--thread-id",
        default=str(uuid.uuid4()),
        help="Reuse this ID to continue a persisted conversation",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Keep asking in the same thread"
    )
    args = parser.parse_args()

    with ResearchAssistant() as assistant:
        if args.query:
            result = assistant.ask(" ".join(args.query), args.thread_id)
            print(result["final_answer"])
            print(f"\nRoute: {result['route']} ({result['route_reason']})")

        if args.interactive or not args.query:
            print(f"Thread: {args.thread_id} (type 'exit' to stop)")
            while True:
                query = input("\nYou: ").strip()
                if query.lower() in {"exit", "quit", "q"}:
                    break
                if not query:
                    continue
                result = assistant.ask(query, args.thread_id)
                print(f"\nAssistant: {result['final_answer']}")
                print(f"\n[route={result['route']}: {result['route_reason']}]")


if __name__ == "__main__":
    main()
