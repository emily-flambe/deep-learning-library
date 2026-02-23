from typing import Tuple

import numpy as np

from .tensor import NDArray, Tensor


class Function:
    def __call__(self, *args):  # Takes inputs
        requires_grad = any(t.requires_grad for t in args)
        inputs_data = [t.data for t in args]  # Gets .data
        output_data = self.forward(*inputs_data)  # Calls forward
        # Wrap in Tensor
        output_tensor = Tensor(output_data, requires_grad=requires_grad)
        if requires_grad:
            output_tensor._op = self  # Save operation
            output_tensor._inputs = args  # Save parents
        return output_tensor

    def forward(self, *args):
        """Computes the forward pass of the operation.
        Args:
            *args: One or more NumPy arrays
        """
        raise NotImplementedError()

    def backward(self, out_grad, node):
        """Calculates backward pass (gradients)
        Args:
            out_grad: upstream gradient flowing from output to input
            node: Value object holding inputs from forward pass
        """
        raise NotImplementedError()


class Add(Function):
    def forward(self, a: NDArray, b: NDArray):
        return a + b

    def backward(self, out_grad, node):
        # derivative of (a + b) wrt a is 1
        # derivative of (a + b) wrt b is 1
        # this is local derivative
        return out_grad, out_grad


def add(a, b):
    return Add()(a, b)  # `__call__`


class Mul(Function):
    def forward(self, a, b):
        return a * b

    def backward(self, out_grad, node):
        a, b = node._inputs
        # derivative of (a*b) wrt a is b
        # derivative of (a*b) wrt b is a
        return out_grad * b, out_grad * a


def mul(a, b):
    return Mul()(a, b)


# --- Arithmetic (two inputs) ---


class Sub(Function):
    def forward(self, a, b):
        return a - b

    def backward(self, out_grad, node):
        pass  # TODO


def sub(a, b):
    return Sub()(a, b)


class Div(Function):
    def forward(self, a, b):
        return a / b

    def backward(self, out_grad, node):
        pass  # TODO


def div(a, b):
    return Div()(a, b)


class Pow(Function):
    def forward(self, a, b):
        return np.power(a, b)

    def backward(self, out_grad, node):
        pass  # TODO


def power(a, b):
    return Pow()(a, b)


# --- Unary math (one input) ---


class Negate(Function):
    def forward(self, a):
        return -1 * a

    def backward(self, out_grad, node):
        pass  # TODO


def negate(a):
    return Negate()(a)


class Log(Function):
    def forward(self, a):
        return np.log(a)

    def backward(self, out_grad, node):
        pass  # TODO


def log(a):
    return Log()(a)


class Exp(Function):
    def forward(self, a):
        return np.exp(a)

    def backward(self, out_grad, node):
        pass  # TODO


def exp(a):
    return Exp()(a)


class Sqrt(Function):
    def forward(self, a):
        return np.sqrt(a)

    def backward(self, out_grad, node):
        pass  # TODO


def sqrt(a):
    return Sqrt()(a)


class Abs(Function):
    def forward(self, a):
        return np.abs(a)

    def backward(self, out_grad, node):
        pass  # TODO


def abs_(a):
    return Abs()(a)


# --- Activation functions (one input) ---


class ReLU(Function):
    def forward(self, a):
        return np.maximum(0, a)

    def backward(self, out_grad, node):
        pass  # TODO


def relu(a):
    return ReLU()(a)


class Sigmoid(Function):
    def forward(self, a):
        return 1 / (1 + np.exp(-a))

    def backward(self, out_grad, node):
        pass  # TODO


def sigmoid(a):
    return Sigmoid()(a)


class Tanh(Function):
    def forward(self, a):
        np.tanh(a)

    def backward(self, out_grad, node):
        pass  # TODO


def tanh(a):
    return Tanh()(a)


# --- Shape operations ---


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        pass  # TODO: np.reshape or a.reshape

    def backward(self, out_grad, node):
        pass  # TODO: reshape out_grad back to original shape


def reshape(a, shape):
    return Reshape(shape)(a)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, a):
        pass  # TODO: np.transpose

    def backward(self, out_grad, node):
        pass  # TODO: transpose out_grad back


def transpose(a, axes=None):
    return Transpose(axes)(a)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        pass  # TODO: np.broadcast_to

    def backward(self, out_grad, node):
        pass  # TODO: sum out_grad back to original shape


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, a):
        pass  # TODO: np.sum

    def backward(self, out_grad, node):
        pass  # TODO: broadcast out_grad back to original shape


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(Function):
    def forward(self, a, b):
        pass  # TODO

    def backward(self, out_grad, node):
        pass  # TODO: out_grad @ B^T, A^T @ out_grad


def matmul(a, b):
    return MatMul()(a, b)


# --- Scalar operations ---


class AddScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a):
        pass  # TODO

    def backward(self, out_grad, node):
        pass  # TODO


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class MulScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a):
        pass  # TODO

    def backward(self, out_grad, node):
        pass  # TODO


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class DivScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a):
        pass  # TODO

    def backward(self, out_grad, node):
        pass  # TODO


def div_scalar(a, scalar):
    return DivScalar(scalar)(a)


class PowerScalar(Function):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a):
        pass  # TODO

    def backward(self, out_grad, node):
        pass  # TODO


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)
