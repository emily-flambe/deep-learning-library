import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import random
    from utils import Value, Neuron, Layer, MLP, draw_dot

    return MLP, draw_dot, mo, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gradient Descent

    In the previous notebook, we built an MLP, computed a loss, and ran `backward()` to get gradients. Now we'll use those gradients to actually improve the network's predictions.
    """)
    return


@app.cell
def _(MLP):
    n = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]
    return n, xs, ys


@app.cell
def _(n, xs, ys):
    # forward pass
    ypred0 = [n(x) for x in xs]
    loss0 = sum((ygt - yout)**2 for ygt, yout in zip(ys, ypred0))

    # backward
    loss0.backward()

    loss0
    return loss0, ypred0


@app.cell
def _(n):
    # We can inspect the gradient on the first weight of the first neuron:
    n.layers[0].neurons[0].w[0].grad
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The update rule

    Each gradient tells us the direction of *increased* loss. To *decrease* loss, we nudge each parameter in the opposite direction:

    ```
    p.data += -step_size * p.grad
    ```

    The **step size** (learning rate) controls how big each nudge is. Too large and the network overshoots; too small and it converges slowly.
    """)
    return


@app.cell
def _(loss0, n, xs, ys):
    # nudge every parameter by a small step opposite the gradient
    for _p in n.parameters():
        _p.data += -0.01 * _p.grad

    # re-run forward pass — loss should be lower
    _ypred = [n(x) for x in xs]
    loss1 = sum((ygt - yout)**2 for ygt, yout in zip(ys, _ypred))

    print(f"loss before: {loss0.data:.4f}")
    print(f"loss after:  {loss1.data:.4f}")
    return (loss1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zeroing gradients

    Before each call to `backward()`, we must reset all gradients to zero. Otherwise gradients **accumulate** across backward passes, which gives us incorrect values. This is a common bug — PyTorch requires the same thing with `optimizer.zero_grad()`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training loop

    Now we put it all together: forward pass, zero grads, backward, update — repeated many times.
    """)
    return


@app.cell
def _(n, xs, ys):
    for k in range(20):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((ygt - yout)**2 for ygt, yout in zip(ys, ypred))

        # zero gradients, then backward
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # update weights
        for p in n.parameters():
            p.data += -0.01 * p.grad

        print(k, loss.data)
    return loss, ypred


@app.cell
def _(ypred, ys):
    # Final predictions vs targets
    list(zip(ys, [y.data for y in ypred]))
    return


@app.cell
def _(draw_dot, loss):
    draw_dot(loss)
    return


if __name__ == "__main__":
    app.run()
