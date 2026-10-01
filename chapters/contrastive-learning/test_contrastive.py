import numpy as np

from solutions import one_positive_three_negative_bias, positive_probability
from scratch.contrastive_learning import (clip_loss_and_grad, cosine_similarity,
                                          hard_negative_indices,
                                          info_nce_loss_and_grad,
                                          linear_pair_loss_and_grad,
                                          make_synthetic_pairs,
                                          pairwise_contrastive_loss,
                                          retrieval_recall_at_k,
                                          siglip_loss_and_grad,
                                          train_linear_pair,
                                          triplet_loss_hard_negative)
from scratch.gradcheck import check_gradient


def test_cosine_and_margin_losses_have_expected_values():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    b = np.array([[2.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    np.testing.assert_allclose(cosine_similarity(a, b), [[1.0, 0.0, 0.8]] * 3)
    same = np.array([1.0, 0.0, 0.0])
    loss = pairwise_contrastive_loss(a, b, same, margin=0.5)
    np.testing.assert_allclose(loss, (0.0 + 0.0 + 0.3**2) / 3)


def test_triplet_loss_mines_the_closest_wrong_item():
    anchor = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    positive = np.array([[0.9, 0.1], [0.1, 0.9], [1.0, 0.8]])
    assert hard_negative_indices(anchor, positive).tolist() == [2, 2, 0]
    assert triplet_loss_hard_negative(anchor, positive, margin=0.3) > 0.0


def test_info_nce_gradient_matches_finite_differences():
    similarity = np.array([[1.2, -0.4, 0.3], [0.1, 0.8, -0.2], [-0.7, 0.2, 0.9]])
    loss, grad = info_nce_loss_and_grad(similarity, temperature=0.7)
    assert loss > 0
    check_gradient(lambda s: info_nce_loss_and_grad(s, 0.7)[0], similarity, grad)


def test_clip_and_siglip_gradients_match_finite_differences():
    similarity = np.array([[0.4, -0.2, 0.1], [0.3, 0.7, -0.5], [-0.1, 0.2, 0.5]])
    _, clip_grad = clip_loss_and_grad(similarity, temperature=0.9)
    check_gradient(lambda s: clip_loss_and_grad(s, 0.9)[0], similarity, clip_grad)
    loss, sig_grad, grad_bias = siglip_loss_and_grad(similarity, bias=-0.3)
    assert loss > 0
    check_gradient(lambda s: siglip_loss_and_grad(s, -0.3)[0], similarity, sig_grad)
    check_gradient(lambda b: siglip_loss_and_grad(similarity, float(b[0]))[0],
                   np.array([-0.3]), np.array([grad_bias]))


def test_linear_pair_gradient_and_retrieval_training():
    images, texts = make_synthetic_pairs(n=32, seed=3)
    rng = np.random.default_rng(4)
    wi = 0.1 * rng.standard_normal((images.shape[1], 4))
    wt = 0.1 * rng.standard_normal((texts.shape[1], 4))
    _, grad_i, grad_t = linear_pair_loss_and_grad(images[:6], texts[:6], wi, wt)
    check_gradient(lambda w: linear_pair_loss_and_grad(images[:6], texts[:6], w, wt)[0],
                   wi, grad_i, tolerance=2e-7)
    check_gradient(lambda w: linear_pair_loss_and_grad(images[:6], texts[:6], wi, w)[0],
                   wt, grad_t, tolerance=2e-7)
    before = retrieval_recall_at_k(images[:, :4], texts[:, :4], k=1)
    image_z, text_z = train_linear_pair(images, texts, steps=220, lr=0.7, seed=5)
    after1 = retrieval_recall_at_k(image_z, text_z, k=1)
    after5 = retrieval_recall_at_k(image_z, text_z, k=5)
    assert before <= 0.1
    assert after1 >= 0.7
    assert after5 >= 0.9


def test_exercise_solution_numbers_are_computed():
    assert round(positive_probability(0.4, 0.1), 4) == 0.982
    assert round(positive_probability(0.4, 0.5), 4) == 0.69
    np.testing.assert_allclose(one_positive_three_negative_bias(), 1 / 6)
