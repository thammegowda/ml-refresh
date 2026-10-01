import numpy as np

from distill import (best_of_n_accuracy, distillation_loss_and_grad,
                     forward_kl_teacher_to_student, majority_vote_accuracy,
                     reverse_kl_student_to_teacher, softmax)
from scratch.gradcheck import check_gradient
from solutions import scaled_and_unscaled_gradients, softened_teacher, vote_values


def test_temperature_distillation_gradient_is_checked_and_t2_scaled():
    student = np.array([0.3, -0.4, 1.2], dtype=np.float64)
    teacher = np.array([1.5, 0.2, -0.7], dtype=np.float64)
    temperature = 3.0
    loss, grad = distillation_loss_and_grad(student, teacher, temperature)
    assert loss > 0
    check_gradient(lambda z: distillation_loss_and_grad(z, teacher, temperature)[0],
                   student, grad)
    scaled, unscaled = scaled_and_unscaled_gradients(student, teacher, temperature)
    np.testing.assert_allclose(scaled, temperature ** 2 * unscaled)


def test_soft_targets_become_less_peaked_as_temperature_rises():
    teacher = np.array([4.0, 1.0, -1.0])
    cold = softened_teacher(teacher, 1.0)
    warm = softened_teacher(teacher, 4.0)
    assert cold[0] > warm[0]
    assert warm[-1] > cold[-1]
    np.testing.assert_allclose(cold.sum(), 1.0)
    np.testing.assert_allclose(warm.sum(), 1.0)


def test_forward_and_reverse_kl_are_directional():
    teacher = np.array([0.48, 0.48, 0.04])
    broad_student = np.array([0.34, 0.33, 0.33])
    narrow_student = np.array([0.90, 0.08, 0.02])
    assert forward_kl_teacher_to_student(teacher, broad_student) < 0.27
    assert forward_kl_teacher_to_student(teacher, narrow_student) > 0.5
    assert reverse_kl_student_to_teacher(narrow_student, teacher) < 0.45
    assert reverse_kl_student_to_teacher(broad_student, teacher) > 0.45


def test_majority_vote_and_best_of_n_are_exact_binomial_calculations():
    majority, best = vote_values()
    np.testing.assert_allclose(majority, 0.68256)
    np.testing.assert_allclose(best, 0.98976)
    for n in [1, 3, 5, 7]:
        assert majority_vote_accuracy(0.7, n) >= majority_vote_accuracy(0.7, n - 2) if n > 1 else True
    np.testing.assert_allclose(best_of_n_accuracy(0.6, 1), 0.6)
    np.testing.assert_allclose(majority_vote_accuracy(0.6, 1), 0.6)
