from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from verilume.core.knowledge_graph import KnowledgeGraph, candidate_entity_names, extract_entities


class KnowledgeGraphTests(unittest.TestCase):
    def test_extract_entities_finds_people_orgs_and_topics(self) -> None:
        entities = extract_entities(
            "Damian Mingo Ndiwago works with Christophe Ley at University of Luxembourg "
            "on Bayesian inference and Hamiltonian Monte Carlo.",
            document="thesis.pdf",
        )
        names = {name for name, _, _ in entities}

        self.assertIn("Christophe Ley", names)
        self.assertIn("University of Luxembourg", names)
        self.assertIn("Bayesian inference", names)
        self.assertIn("Hamiltonian Monte Carlo", names)

    def test_graph_indexes_chunk_mentions_and_documents(self) -> None:
        with TemporaryDirectory() as tmp:
            graph = KnowledgeGraph(Path(tmp) / "knowledge_graph.sqlite")
            graph.index_chunk(
                "Christophe Ley is affiliated with University of Luxembourg. "
                "The document discusses Bayesian inference.",
                document="thesis.pdf",
                page=12,
                chunk_id="chunk-12",
            )

            results = graph.search_entity("Christophe Ley")
            context = graph.graph_context_for_query("Which documents mention Christophe Ley?")

            self.assertTrue(results)
            self.assertIn("thesis.pdf", context.related_documents)
            self.assertIn("chunk-12", context.related_chunks)

    def test_graph_neighbors_include_affiliation_relation(self) -> None:
        with TemporaryDirectory() as tmp:
            graph = KnowledgeGraph(Path(tmp) / "knowledge_graph.sqlite")
            graph.index_chunk(
                "Christophe Ley from University of Luxembourg studies statistics.",
                document="profile.pdf",
                page=1,
                chunk_id="profile-1",
            )
            person = graph.search_entity("Christophe Ley")[0]
            neighbors = graph.neighbors(person.id)

            self.assertTrue(any(item.relation == "affiliated_with" for item in neighbors))
            self.assertTrue(any("University of Luxembourg" in item.entity.name for item in neighbors))

    def test_candidate_entity_names_uses_question_topics(self) -> None:
        names = candidate_entity_names("Which topics are linked to Bayesian inference?")

        self.assertIn("Bayesian inference", names)

    def test_delete_document_removes_its_evidence_but_keeps_shared_entities(self) -> None:
        with TemporaryDirectory() as tmp:
            graph = KnowledgeGraph(Path(tmp) / "knowledge_graph.sqlite")
            # A shared person appears in both documents; a topic only in the diploma.
            graph.index_chunk(
                "Damian Mingo Ndiwago completed the diploma in statistics with distinction.",
                document="diploma.pdf",
                chunk_id="diploma-1",
            )
            graph.index_chunk(
                "Damian Mingo Ndiwago studies Bayesian inference at University of Luxembourg.",
                document="thesis.pdf",
                chunk_id="thesis-1",
            )
            self.assertIn("diploma.pdf", graph.documents())

            graph.delete_document("diploma.pdf")

            # The diploma is gone from the graph and from query results...
            self.assertNotIn("diploma.pdf", graph.documents())
            context = graph.graph_context_for_query("Damian Mingo Ndiwago")
            self.assertNotIn("diploma.pdf", context.related_documents)
            self.assertNotIn("diploma-1", context.related_chunks)
            # ...but the shared person is preserved via the surviving document.
            self.assertTrue(graph.search_entity("Damian Mingo Ndiwago"))
            self.assertIn("thesis.pdf", graph.documents())

    def test_delete_document_is_a_noop_for_unknown_document(self) -> None:
        with TemporaryDirectory() as tmp:
            graph = KnowledgeGraph(Path(tmp) / "knowledge_graph.sqlite")
            graph.index_chunk("Bayesian inference is discussed.", document="thesis.pdf")
            graph.delete_document("does-not-exist.pdf")
            self.assertIn("thesis.pdf", graph.documents())


if __name__ == "__main__":
    unittest.main()
