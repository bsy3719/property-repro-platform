from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.retriever_table_agent import RetrieverTableAgent
from src.services.vector_db_service import LocalVectorDB


TOPIC_QUERIES = {
    "model": "What model architecture does the paper use for LFL prediction?",
    "feature": "What feature representation/input features are used for LFL prediction?",
    "hyperparameter": "What hyperparameters are reported for the LFL prediction model?",
    "training": "What training setup is used for LFL prediction (split, seed, optimizer, lr, epoch, batch, loss, validation, early stopping)?",
    "metrics": "What evaluation metrics and reported values are provided for LFL prediction?",
}


class RagState(TypedDict, total=False):
    markdown: str
    source_id: str
    top_k: int
    vector_info: dict[str, Any]
    retrieved: list[dict[str, Any]]
    retrieved_by_topic: dict[str, list[dict[str, Any]]]
    summary_markdown: str
    error: str


def _build_vector_db(state: RagState) -> RagState:
    if state.get("error"):
        return state

    vdb = LocalVectorDB(embedding_model="text-embedding-3-large")
    info = vdb.build_from_markdown(
        markdown=state.get("markdown", ""),
        source_id=state.get("source_id"),
    )
    return {"vector_info": info, "source_id": info["source_id"]}


def _retrieve(state: RagState) -> RagState:
    if state.get("error"):
        return state

    vdb = LocalVectorDB(embedding_model="text-embedding-3-large")
    top_k = int(state.get("top_k", 5))

    retrieved_by_topic: dict[str, list[dict[str, Any]]] = {}
    flattened: list[dict[str, Any]] = []

    for topic, query in TOPIC_QUERIES.items():
        rows = vdb.retrieve(source_id=state["source_id"], query=query, top_k=top_k)
        topic_rows: list[dict[str, Any]] = []
        for row in rows:
            enriched = dict(row)
            enriched["topic"] = topic
            enriched["query"] = query
            topic_rows.append(enriched)
            flattened.append(enriched)
        retrieved_by_topic[topic] = topic_rows

    return {"retrieved_by_topic": retrieved_by_topic, "retrieved": flattened}


def _to_summary(state: RagState) -> RagState:
    if state.get("error"):
        return state

    agent = RetrieverTableAgent(model_name="gpt-5-mini")
    result = agent.invoke(retrieved_by_topic=state.get("retrieved_by_topic", {}))
    if result.get("error"):
        return {"error": result["error"]}

    return {"summary_markdown": result.get("summary_markdown", "")}


def build_graph():
    graph = StateGraph(RagState)
    graph.add_node("build_vector_db", _build_vector_db)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("to_summary", _to_summary)

    graph.set_entry_point("build_vector_db")
    graph.add_edge("build_vector_db", "retrieve")
    graph.add_edge("retrieve", "to_summary")
    graph.add_edge("to_summary", END)

    return graph.compile()


RAG_GRAPH = build_graph()


def run_rag_pipeline(markdown: str, source_id: str | None = None, top_k: int = 5) -> RagState:
    return RAG_GRAPH.invoke(
        {
            "markdown": markdown,
            "source_id": source_id,
            "top_k": top_k,
        }
    )
