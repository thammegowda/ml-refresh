import numpy as np


# tag::quadratic[]
def quadratic_value(theta, curvature):
    return 0.5 * float(np.sum(curvature * theta * theta))


def quadratic_gradient(theta, curvature):
    return curvature * theta


def gradient_descent(theta, curvature, lr, steps):
    path = [theta.astype(np.float64).copy()]
    for _ in range(steps):
        theta = theta - lr * quadratic_gradient(theta, curvature)
        path.append(theta.copy())
    return np.array(path)
# end::quadratic[]
