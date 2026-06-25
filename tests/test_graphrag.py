from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from verilume.core.agentic_planner import BUILD_GRAPH_CONTEXT, AgenticPlanner
from verilume.core.graphrag import GraphRAGRetriever
from verilume.core.knowledge_graph import KnowledgeGraph
from verilume.core.query_interpreter import InterpretedQuery
from verilume.settings import AppSettings


class GraphRAGTests(unittest.TestCase):
    def test_graphrag_context_expands_seed_entities(self) -> None:
        with TemporaryDirectory() as tmp:
            graph = KnowledgeGraph(Path(tmp) / "kg.sqlite")
            graph.index_chunk(
                "Christophe Ley from University of Luxembourg studies Bayesian inference.",
                document="profile.pdf",
                page=1,
                chunk_id="profile-1",
            )
            retriever = GraphRAGRetriever(graph)

            context = retriever.retrieve_graph_context("Who is Christophe Ley?")

            self.assertIn("Christophe Ley", context.seed_entities)
            self.assertTrue(any("University of Luxembourg" in item for item in context.expanded_entities))
            self.assertIn("profile.pdf", context.related_documents)

    def test_graphrag_retrieves_mention_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            graph = KnowledgeGraph(Path(tmp) / "kg.sqlite")
            graph.index_chunk(
                "Hamiltonian Monte Carlo appears in the Bayesian model selection document.",
                document="methods.pdf",
                page=3,
                chunk_id="methods-3",
            )
            retriever = GraphRAGRetriever(graph)

            context = retriever.retrieve_graph_context("Which documents mention Hamiltonian Monte Carlo?")
            sources = retriever.retrieve_graph_chunks("Which documents mention Hamiltonian Monte Carlo?", context)

            self.assertTrue(sources)
            self.assertEqual(sources[0].document, "methods.pdf")
            self.assertEqual(sources[0].metadata["source_type"], "knowledge_graph")

    def test_planner_adds_graph_context_for_entity_questions(self) -> None:
        question = "Who is Christophe Ley?"
        interpretation = InterpretedQuery(
            original_question=question,
            resolved_question=question,
            intent="person",
            search_queries=[question],
        )

        plan = AgenticPlanner().plan(question, interpretation, AppSettings(enable_web_search=False))

        self.assertIn(BUILD_GRAPH_CONTEXT, plan.actions)


if __name__ == "__main__":
    unittest.main()
