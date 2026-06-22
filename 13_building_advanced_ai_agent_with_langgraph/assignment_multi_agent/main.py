"""CLI and full LangGraph workflow for multi-agent research and summarization."""

from __future__ import annotations

import argparse
import sqlite3
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config import MEMORY_DB, SETTINGS
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

    @staticmethod
    def _trace_update(node: str, update: object) -> None:
        """Print a compact, safe summary of one LangGraph node update."""
        print(f"\n[agent] {node}")
        if not isinstance(update, dict):
            print(f"  update: {type(update).__name__}")
            return

        if node == "router":
            sufficient = update.get("information_sufficient", False)
            print(f"  information sufficient: {sufficient}")
            if update.get("route"):
                print(f"  selected route: {update['route']}")
            if update.get("route_reason"):
                print(f"  reflection: {update['route_reason']}")
            if update.get("reflection_count") is not None:
                print(f"  reflection step: {update['reflection_count']}")
            destination = "summarize" if sufficient else {
                "llm": "llm_agent",
                "rag": "rag_agent",
                "web": "web_planner",
            }.get(update.get("route"), "specialist")
            print(f"  flow: router -> {destination}")
            return

        if node == "web_planner":
            queries = update.get("research_queries", [])
            print(f"  planned searches ({len(queries)}):")
            for query in queries:
                print(f"    - {query}")
            print("  flow: web_planner -> parallel web_search workers")
            return

        if node == "web_search":
            results = update.get("web_results", [])
            search_query = results[0].get("query") if results else "unknown"
            print(f"  query: {search_query}")
            print(f"  results returned: {len(results)}")
            print("  flow: web_search -> web_reduce")
            return

        if node in {"llm_agent", "rag_agent", "web_reduce"}:
            context = update.get("gathered_context", "")
            sources = update.get("sources", [])
            print(f"  accumulated evidence: {len(context):,} characters")
            print(f"  accumulated sources: {len(sources)}")
            print(f"  flow: {node} -> router")
            return

        if node == "summarize":
            answer = update.get("final_answer", "")
            print(f"  final answer: {len(answer):,} characters")
            print("  flow: summarize -> END")

    def ask(
        self, query: str, thread_id: str, verbose: bool | None = None
    ) -> AgentState:
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
        show_trace = SETTINGS.execution.verbose if verbose is None else verbose
        if not show_trace:
            return self.graph.invoke(initial_state, config=config)

        print(f"\n[flow] START -> router (thread={thread_id})")
        for event in self.graph.stream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node, update in event.items():
                self._trace_update(node, update)
        return self.graph.get_state(config).values

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
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show or hide agent-by-agent execution tracing",
    )
    args = parser.parse_args()

    with ResearchAssistant() as assistant:
        if args.query:
            result = assistant.ask(
                " ".join(args.query), args.thread_id, verbose=args.verbose
            )
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
                result = assistant.ask(query, args.thread_id, verbose=args.verbose)
                print(f"\nAssistant: {result['final_answer']}")
                print(f"\n[route={result['route']}: {result['route_reason']}]")


if __name__ == "__main__":
    main()
