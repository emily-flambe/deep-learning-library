import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
async def _():
    import sys
    import marimo as mo
    import math
    if sys.platform == "emscripten":
        import pyodide.http, pathlib
        resp = await pyodide.http.pyfetch("/utils.py")
        pathlib.Path("/utils.py").write_text(await resp.string())
        sys.path.insert(0, "/")
    from utils import Value, draw_dot
    try:
        import torch
    except ImportError:
        torch = None

    return Value, draw_dot, math, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Automated Backpropagation

    Instead of calling `_backward()` on each node manually, we can automate backpropagation by building a topological ordering of the graph — ensuring every node comes after its children — then walking it in reverse, so each node's gradient is already set before it propagates further back.
    """)
    return


@app.cell
def _(Value):
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.7, label='b')

    x1w1 = x1*w1; x1w1.label = 'x1w1'
    x2w2 = x2*w2; x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    return b, n, o, w1, w2, x1, x1w1, x1w1x2w2, x2, x2w2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The topo sort + reversed walk is what `backward()` does internally. You can just call it on the output node:
    """)
    return


@app.cell
def _(draw_dot, o):
    o.backward()
    draw_dot(o)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Expanding tanh into Primitives

    `tanh` is a compound operation. We can break it into ops the engine already knows (`exp`, `+`, `*`, `/`) and get the same gradients — confirming the chain rule works through arbitrary graphs.
    """)
    return


@app.cell
def _(Value, draw_dot):
    _x1 = Value(2.0, label='x1')
    _x2 = Value(0.0, label='x2')
    _w1 = Value(-3.0, label='w1')
    _w2 = Value(1.0, label='w2')
    _b = Value(6.7, label='b')

    _x1w1 = _x1*_w1; _x1w1.label = 'x1w1'
    _x2w2 = _x2*_w2; _x2w2.label = 'x2w2'
    _n = _x1w1 + _x2w2 + _b; _n.label = 'n'

    # tanh expanded: (e^2x - 1) / (e^2x + 1)
    _e = (2*_n).exp()
    _o = (_e - 1) / (_e + 1); _o.label = 'o'

    _o.backward()
    draw_dot(_o)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verifying with PyTorch

    Same expression in PyTorch. The gradients should match exactly.
    """)
    return


@app.cell
def _(torch):
    pt_x1 = torch.Tensor([2.0]).double();                pt_x1.requires_grad = True
    pt_x2 = torch.Tensor([0.0]).double();                pt_x2.requires_grad = True
    pt_w1 = torch.Tensor([-3.0]).double();               pt_w1.requires_grad = True
    pt_w2 = torch.Tensor([1.0]).double();                pt_w2.requires_grad = True
    pt_b  = torch.Tensor([6.8813735870195432]).double(); pt_b.requires_grad  = True

    pt_n = pt_x1*pt_w1 + pt_x2*pt_w2 + pt_b
    pt_o = torch.tanh(pt_n)
    pt_o.backward()

    print('x1', pt_x1.grad.item())
    print('x2', pt_x2.grad.item())
    print('w1', pt_w1.grad.item())
    print('w2', pt_w2.grad.item())
    return pt_b, pt_n, pt_o, pt_w1, pt_w2, pt_x1, pt_x2


if __name__ == "__main__":
    app.run()
