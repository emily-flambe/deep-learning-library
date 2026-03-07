import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    from utils import Value, draw_dot

    return Value, draw_dot, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Automated backpropagation
    """)
    return


@app.cell(hide_code=True)
def _(Value):
    # inputs x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights w1,w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of the neuron
    b = Value(6.7, label='b')

    # x1*w1 + x2+w2 + b
    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'

    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh()
    return (o,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of calling `_backward()` on each node manually, we can automate backpropagation by building a topological ordering of the graph — ensuring every node comes after its children — then walking it in reverse, so each node's gradient is already set before it propagates further back.
    """)
    return


@app.cell
def _(o):
    o.grad = 1

    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(o)

    for node in reversed(topo):
        node._backward()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This iteration through a reversed set of topologically sorted nodes is what is happening inside the `Value` class's `backward()` method. This means you can just call backward on the output node like this:
    """)
    return


@app.cell
def _(draw_dot, o):
    o.backward()

    draw_dot(o)
    return


if __name__ == "__main__":
    app.run()
