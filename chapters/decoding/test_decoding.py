import numpy as np

from scratch.decoding import (acceptance_probability, expected_accepted_prefix,
                              filtered_distribution, mask_logits, softmax,
                              speculative_sample, tiny_beam_search)
from solutions import (digit_only_distribution, grammar_step,
                       speculative_output_probability)


def rng(seed=39):
    return np.random.default_rng(seed)


def test_temperature_and_filters_transform_one_distribution():
    logits = np.array([4.0, 2.0, 1.0, -1.0])
    cold = filtered_distribution(logits, temperature=0.5)
    hot = filtered_distribution(logits, temperature=2.0)
    assert cold[0] > softmax(logits)[0] > hot[0]
    np.testing.assert_allclose(filtered_distribution(logits, top_k=2)[2:], [0, 0])
    top_p = filtered_distribution(logits, top_p=0.9)
    assert np.count_nonzero(top_p) == 2
    min_p = filtered_distribution(logits, min_p=0.2)
    assert min_p[-1] == 0 and np.isclose(min_p.sum(), 1)


def test_greedy_beam_and_constrained_decoding():
    transitions = {
        (): np.log([0.45, 0.40, 0.15]),
        (0,): np.log([0.10, 0.10, 0.80]),
        (1,): np.log([0.95, 0.03, 0.02]),
        (2,): np.log([0.20, 0.20, 0.60]),
    }
    beams = tiny_beam_search(lambda prefix: transitions[prefix], (), 2, 2)
    assert beams[0][0] == (1, 0)
    assert beams[1][0] == (0, 2)
    masked = mask_logits(np.array([1.0, 2.0, 3.0]), [True, False, True])
    assert np.argmax(masked) == 2 and np.isneginf(masked[1])
    vocab = ["A", "7", "b", "42"]
    digits = digit_only_distribution(np.array([5.0, 1.0, 5.0, 2.0]), vocab)
    np.testing.assert_allclose(digits[[0, 2]], 0)
    step = grammar_step(np.zeros(13), ["[", "3"], list("[],0123456789"))
    assert step[1] > 0 and step[2] > 0 and step[[0, *range(3, 13)]].sum() == 0


def test_speculative_sampling_is_exact_empirically_and_by_identity():
    p = np.array([0.50, 0.20, 0.20, 0.10])
    q = np.array([0.15, 0.55, 0.20, 0.10])
    np.testing.assert_allclose(speculative_output_probability(p, q), p)
    # The identity above is exact; sampling guards against gross errors. 20,000 one-at-a-time
    # draws keep the test fast, and 0.015 is about four standard errors for these frequencies.
    draws, rate = speculative_sample(p, q, 20_000, rng())
    frequencies = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_allclose(frequencies, p, atol=0.015)
    np.testing.assert_allclose(rate, acceptance_probability(p, q), atol=0.015)


def test_expected_accepted_tokens_per_verification_step():
    alphas = np.array([0.8, 0.7, 0.5])
    np.testing.assert_allclose(expected_accepted_prefix(alphas), 0.8 + 0.56 + 0.28)
    trials = 200_000
    uniforms = rng().random((trials, len(alphas)))
    accepted = (uniforms < alphas).cumprod(axis=1).sum(axis=1)
    np.testing.assert_allclose(accepted.mean(), 1.64, atol=0.01)
