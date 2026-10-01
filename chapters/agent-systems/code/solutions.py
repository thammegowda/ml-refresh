"""Worked solutions to the Chapter 43 exercises."""
from retrieval import chunk_words, expected_pass_at_k, pass_at_k, retrieval_prompt


# tag::chunks[]
def overlapping_chunks(text):
    return chunk_words(text, size=6, overlap=2)
# end::chunks[]


# tag::pass-derive[]
def failed_subset_ratio(n, c, k):
    return 1.0 - pass_at_k(n, c, k)
# end::pass-derive[]


# tag::judge-swap[]
def debiased_pairwise_score(judge, answer_a, answer_b):
    first = judge(answer_a, answer_b)
    second = judge(answer_b, answer_a)
    return (first - second) / 2
# end::judge-swap[]


# tag::rag[]
def tiny_rag_prompt(question, documents):
    return retrieval_prompt(question, documents, k=1)
# end::rag[]


# tag::unbiased[]
def exact_unbiased_value(n, k, p):
    return expected_pass_at_k(n, k, p)
# end::unbiased[]
