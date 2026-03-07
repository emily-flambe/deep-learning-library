import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    from utils import Value, draw_dot

    return Value, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural Network
    """)
    return


@app.cell(hide_code=True)
def _(Value, random):
    class Neuron:
        def __init__(self, nin):
            self.value=[Value(random.uniform(-1,1)) for _ in range(nin)]
            self.b=Value(random.uniform(-1,1))

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
