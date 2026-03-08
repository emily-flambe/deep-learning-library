import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    import torch
    from utils import Value, draw_dot

    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backpropogation using Pytorch Tensors

    We get the same values. Wow
    """)
    return


@app.cell
def _(torch):
    x1 = torch.Tensor([2.0]).double();               x1.requires_grad=True
    x2 = torch.Tensor([0.0]).double();               x2.requires_grad=True
    w1 = torch.Tensor([-3.0]).double();              w1.requires_grad=True
    w2 = torch.Tensor([1.0]).double();               w2.requires_grad=True
    b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad=True # bias chosen so that tanh'(n) = 0.5, which is convenient for manual backpropagation (not really relevant here but it's fine)
    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)

    print(o.data.item)
    o.backward()

    # These gradients will match what came out in the backprop we did with our Value class. Wow
    print('----')
    print('x2', x2.grad.item())
    print('w2', w2.grad.item())
    print('x1', x1.grad.item())
    print('w1', w1.grad.item())
    return


if __name__ == "__main__":
    app.run()
