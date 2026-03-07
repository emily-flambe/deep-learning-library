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

    # writing out tanh
    e = (2*n).exp()
    o = (e - 1)/(e + 1)

    o.backward()
    draw_dot(o)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using Pytorch
    """)
    return


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _(torch):
    x1 = torch.Tensor([2.0]).double(); x1.requires_grad=True
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
