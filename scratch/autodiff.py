"""A tiny reverse-mode automatic differentiation engine (Chapter 10)."""
import numpy as np

from scratch.arrays import unbroadcast

__chapter__ = "autodiff"


def as_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


# tag::tensor-core[]
class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = tuple(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out
# end::tensor-core[]

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-as_tensor(other))

    def __rsub__(self, other):
        return as_tensor(other) + (-self)

    # tag::tensor-ops[]
    def __matmul__(self, other):
        other = as_tensor(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,))

        def _backward():
            self.grad += out.grad * (self.data > 0.0)
        out._backward = _backward
        return out
    # end::tensor-ops[]

    # tag::backward[]
    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        topo, seen = [], set()

        def build(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        for node in topo:
            node.grad = np.zeros_like(node.data)
        self.grad = np.asarray(gradient, dtype=np.float64)
        for node in reversed(topo):
            node._backward()
        return self.grad
    # end::backward[]
