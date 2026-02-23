import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.class_definition
class Value:
    """
    This class will enable us to look at forward and backward propagation through a neural network using scalars, before later extending those concepts to Tensor objects
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '×')
        return out


@app.cell
def _():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10, label='c')
    e = a*b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')

    L = d*f
    L.label = 'L'
    return L, d


@app.cell
def _():
    import marimo as mo

    def trace(root):
        """
        Builds a set of all nodes and edges in a graph
        """
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
        nodes, edges = trace(root)
        lines = ["graph LR"]
        leaf_ids = []
        result_ids = []
        op_ids = []

        for n in nodes:
            uid = f"n{id(n)}"
            label = f"{n.label}: {n.data:.4f}" if n.label else f"{n.data:.4f}"
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

    return (draw_dot,)


@app.cell
def _(L, draw_dot):
    draw_dot(L)
    return


@app.cell
def _(d):
    d
    return


if __name__ == "__main__":
    app.run()
