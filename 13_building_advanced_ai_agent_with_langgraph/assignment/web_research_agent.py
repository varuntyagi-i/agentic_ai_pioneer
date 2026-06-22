"""Tavily web research agent using LangGraph Send for map-reduce."""

from __future__ import annotations

import argparse

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from tavily import TavilyClient

from config import SETTINGS, get_llm, require_env
from state import AgentState, Source, WebResult, WebSearchState


class SearchPlan(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description="One to three focused and non-overlapping web search queries",
    )


def web_planner_node(state: AgentState) -> dict:
    planner = get_llm().with_structured_output(SearchPlan)
    plan = planner.invoke(
        [
            SystemMessage(
                content=(
                    "Create up to three focused Tavily searches that together answer "
                    "the user's question. Include dates or freshness terms when useful."
                )
            ),
            HumanMessage(content=state["query"]),
        ]
    )
    return {
        "research_queries": plan.queries,
        "web_results": [{"__reset__": True}],
    }


def dispatch_web_searches(state: AgentState) -> list[Send]:
    return [
        Send("web_search", {"search_query": query})
        for query in state["research_queries"]
    ]


def web_search_node(state: WebSearchState) -> dict:
    client = TavilyClient(api_key=require_env("TAVILY_API_KEY"))
    response = client.search(
        query=state["search_query"],
        search_depth=SETTINGS.tavily.search_depth,
        max_results=SETTINGS.tavily.max_results,
        include_answer=False,
        include_raw_content=False,
    )
    results: list[WebResult] = []
    for item in response.get("results", []):
        results.append(
            {
                "query": state["search_query"],
                "title": item.get("title", "Untitled result"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score"),
            }
        )
    return {"web_results": results}


def web_reduce_node(state: AgentState) -> dict:
    unique: dict[str, WebResult] = {}
    for result in state.get("web_results", []):
        key = result["url"] or f'{result["query"]}:{result["title"]}'
        if key not in unique or (result.get("score") or 0) > (
            unique[key].get("score") or 0
        ):
            unique[key] = result

    ranked = sorted(unique.values(), key=lambda item: item.get("score") or 0, reverse=True)
    context_parts: list[str] = []
    sources: list[Source] = list(state.get("sources", []))
    known_urls = {source["url"] for source in sources}
    source_numbers = {
        source["url"]: index for index, source in enumerate(sources, start=1)
    }
    for number, result in enumerate(ranked[:10], start=1):
        citation_number = source_numbers.get(result["url"], len(sources) + 1)
        context_parts.append(
            f'[{citation_number}] {result["title"]}\n'
            f'URL: {result["url"]}\n{result["content"]}'
        )
        if result["url"] not in known_urls:
            sources.append({"title": result["title"], "url": result["url"]})
            known_urls.add(result["url"])
            source_numbers[result["url"]] = citation_number

    existing = state.get("gathered_context", "")
    web_context = "\n\n".join(context_parts)
    return {
        "gathered_context": (
            f"{existing}\n\n[Web research]\n{web_context}".strip()
        ),
        "sources": sources,
    }


def build_web_research_graph():
    builder = StateGraph(AgentState)
    builder.add_node("web_planner", web_planner_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("web_reduce", web_reduce_node)
    builder.add_edge(START, "web_planner")
    builder.add_conditional_edges(
        "web_planner", dispatch_web_searches, ["web_search"]
    )
    builder.add_edge("web_search", "web_reduce")
    builder.add_edge("web_reduce", END)
    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the web research agent")
    parser.add_argument("query", nargs="+", help="Question to research")
    args = parser.parse_args()
    query = " ".join(args.query)
    result = build_web_research_graph().invoke(
        {"query": query, "messages": [HumanMessage(content=query)], "web_results": []}
    )
    print(result.get("gathered_context", "No results found."))


if __name__ == "__main__":
    main()
