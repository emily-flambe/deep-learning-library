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
    # Manual backpropagation

    n = x1\*w1 + x2\*w2 + b
    o = tanh(n)

    n is a neuron representing the combined output of weighted and summed inputs plus a scalar bias factor.

    o represents the final output of the neuron, which applies a tanh **activation function** to squash *n* into a value between -1 and +1. This makes it so that no matter how large or small are the input values to the function, every neuron's output is on the same scale, so that their gradients can all contribute to the network's learning regardless of the scale of the original inputs.
    """)
    return


@app.cell
def _(Value, draw_dot):
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
    draw_dot(o)
    return o, x1w1, x1w1x2w2, x2w2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Manual backpropagation

    Below, we initialize the gradient of the final output o to `1.0`, then call backward() on each node working backwards through the graph, populating gradients one step at a time.
    """)
    return


@app.cell
def _(draw_dot, o):
    o.grad = 1.0
    o._backward()
    draw_dot(o)
    return


app._unparsable_cell(
    r"""
    n._backward()
    draw_dot(o)b
    """,
    name="_"
)


@app.cell
def _(draw_dot, o, x1w1x2w2):
    x1w1x2w2._backward()
    draw_dot(o)
    return


@app.cell
def _(draw_dot, o, x1w1, x2w2):
    x1w1._backward()
    x2w2._backward()
    draw_dot(o)
    return


if __name__ == "__main__":
    app.run()
