"""Solutions for Chapter 38 exercises."""
import numpy as np

from distill import (best_of_n_accuracy, distillation_loss_and_grad,
                     majority_vote_accuracy, softmax)


# tag::soft-targets[]
def softened_teacher(teacher_logits, temperature):
    return softmax(np.asarray(teacher_logits) / temperature)
# end::soft-targets[]


# tag::t2-gradient[]
def scaled_and_unscaled_gradients(student_logits, teacher_logits, temperature):
    _, scaled = distillation_loss_and_grad(student_logits, teacher_logits,
                                           temperature, scale_t2=True)
    _, unscaled = distillation_loss_and_grad(student_logits, teacher_logits,
                                             temperature, scale_t2=False)
    return scaled, unscaled
# end::t2-gradient[]


# tag::vote-values[]
def vote_values():
    return majority_vote_accuracy(0.6, 5), best_of_n_accuracy(0.6, 5)
# end::vote-values[]
