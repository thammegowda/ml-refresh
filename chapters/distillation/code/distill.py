"""Distillation and test-time compute helpers (Chapter 38)."""
import math

import numpy as np


def log_softmax(logits):
    logits = np.asarray(logits)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def softmax(logits):
    return np.exp(log_softmax(logits))


# tag::temperature[]
def distillation_loss_and_grad(student_logits, teacher_logits, temperature=1.0,
                               scale_t2=True):
    """KL teacher_T || student_T, with optional T^2 multiplier."""
    student_logits = np.asarray(student_logits, dtype=np.float64)
    teacher_logits = np.asarray(teacher_logits, dtype=np.float64)
    teacher = softmax(teacher_logits / temperature)
    log_student = log_softmax(student_logits / temperature)
    loss = -np.sum(teacher * log_student)
    grad = (softmax(student_logits / temperature) - teacher) / temperature
    if scale_t2:
        loss *= temperature ** 2
        grad *= temperature ** 2
    return float(loss), grad
# end::temperature[]


# tag::kl-directions[]
def forward_kl_teacher_to_student(teacher_probs, student_probs):
    """KL(teacher || student): mass-covering and teacher-sampled."""
    teacher_probs = np.asarray(teacher_probs, dtype=np.float64)
    student_probs = np.asarray(student_probs, dtype=np.float64)
    return float(np.sum(teacher_probs * (np.log(teacher_probs) -
                                         np.log(student_probs))))


def reverse_kl_student_to_teacher(student_probs, teacher_probs):
    """KL(student || teacher): on-policy and mode-seeking."""
    student_probs = np.asarray(student_probs, dtype=np.float64)
    teacher_probs = np.asarray(teacher_probs, dtype=np.float64)
    return float(np.sum(student_probs * (np.log(student_probs) -
                                         np.log(teacher_probs))))
# end::kl-directions[]


# tag::voting[]
def best_of_n_accuracy(p, n):
    """Probability that at least one of n independent samples is correct."""
    return 1.0 - (1.0 - p) ** n


def majority_vote_accuracy(p, n):
    """Strict-majority accuracy for n independent samples, each correct with prob p."""
    threshold = n // 2 + 1
    total = 0.0
    for k in range(threshold, n + 1):
        total += math.comb(n, k) * p ** k * (1.0 - p) ** (n - k)
    return total
# end::voting[]
