import math

import marimo as mo


class Value:
    """
    This class will enable us to look at forward and backward propagation
    through a neural network using scalars before later applying those
    concepts to Tensors
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self.label = label
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "×")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * self.data

        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1

        for node in reversed(topo):
            node._backward()


def trace(root):
    """Builds a set of all nodes and edges in a graph."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    """Renders a computation graph as a Mermaid diagram."""
    nodes, edges = trace(root)
    lines = ["graph LR"]
    leaf_ids = []
    result_ids = []
    op_ids = []

    for n in nodes:
        uid = f"n{id(n)}"
        data_str = f"data: {n.data:.1f}<br>grad: {n.grad:.1f}"
        label = f"{n.label}<br>{data_str}" if n.label else data_str
        lines.append(f'    {uid}["{label}"]')
        if n._op:
            op_uid = f"{uid}_op"
            lines.append(f'    {op_uid}(("{n._op}"))')
            lines.append(f"    {op_uid} --> {uid}")
            result_ids.append(uid)
            op_ids.append(op_uid)
        else:
            leaf_ids.append(uid)

    for n1, n2 in edges:
        lines.append(f"    n{id(n1)} --> n{id(n2)}_op")

    lines.append("    classDef leaf fill:#dbeafe,stroke:#93c5fd,color:#1e3a5f")
    lines.append("    classDef result fill:#f1f5f9,stroke:#94a3b8,color:#334155")
    lines.append("    classDef op fill:#fef3c7,stroke:#f59e0b,color:#92400e")
    for nid in leaf_ids:
        lines.append(f"    class {nid} leaf")
    for nid in result_ids:
        lines.append(f"    class {nid} result")
    for nid in op_ids:
        lines.append(f"    class {nid} op")

    return mo.mermaid("\n".join(lines))
