import numpy as np

from planning import plan_then_execute, tree_search
from retrieval import (add_vector_memory, chunk_words, cosine_top_k,
                       expected_pass_at_k, pass_at_k, recall_vector_memory,
                       retrieval_prompt)
from solutions import (debiased_pairwise_score, exact_unbiased_value,
                       overlapping_chunks, tiny_rag_prompt)


def test_chunking_keeps_overlap_between_neighbors():
    chunks = overlapping_chunks("one two three four five six seven eight nine ten")
    assert chunks == [
        "one two three four five six",
        "five six seven eight nine ten",
    ]
    assert chunk_words("a b c d", size=3, overlap=1) == ["a b c", "c d"]


def test_hashed_cosine_retrieval_finds_the_tool_document():
    docs = [
        "LayerNorm normalizes each token representation.",
        "A tool schema validates function-call arguments before dispatch.",
        "Perplexity is exponentiated cross entropy.",
    ]
    top = cosine_top_k("validate tool call schema", docs, k=2, dims=128)
    assert top[0][0] == 1
    prompt = tiny_rag_prompt("How does a tool schema validate calls?", docs)
    assert "tool schema" in prompt.lower()
    assert "Question: How does a tool schema validate calls?" in prompt


def test_vector_memory_recalls_related_past_notes():
    memory = []
    add_vector_memory(memory, "The user prefers NumPy examples.", dims=128)
    add_vector_memory(memory, "The build uses Node tests.", dims=128)
    recalled = recall_vector_memory(memory, "Please write a NumPy listing.", dims=128)
    assert recalled[0][0] == "The user prefers NumPy examples."


def test_pass_at_k_matches_combinatorics_and_is_unbiased():
    np.testing.assert_allclose(pass_at_k(10, 3, 1), 0.3)
    np.testing.assert_allclose(pass_at_k(10, 3, 2), 1 - (7 / 10) * (6 / 9))
    assert pass_at_k(10, 8, 3) == 1.0
    for n, k, p in [(5, 2, 0.2), (6, 3, 0.4), (7, 1, 0.7)]:
        expected = exact_unbiased_value(n, k, p)
        np.testing.assert_allclose(expected, 1 - (1 - p) ** k)


def test_tree_search_uses_value_estimates_to_pick_a_plan():
    graph = {0: [1, 2], 1: [3, 4], 2: [5, 6], 6: [7], 3: [], 4: [], 5: [], 7: []}
    values = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 1, 6: 5, 7: 8}
    path = tree_search(0, lambda state: graph[state], values.get, depth=3, beam=2)
    assert path == [0, 2, 6, 7]


def test_plan_then_execute_and_pairwise_judge_swapping():
    observations = plan_then_execute("ship", lambda goal: [f"test {goal}", f"send {goal}"], str.upper)
    assert observations == ["TEST SHIP", "SEND SHIP"]

    def biased_judge(first, second):
        return (len(first) - len(second)) + 1

    assert debiased_pairwise_score(biased_judge, "aaaa", "bb") == 2
    assert "Use only this context" in retrieval_prompt("q", ["a document"], k=1)
