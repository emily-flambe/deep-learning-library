import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MicroGrad Demo

    A tiny scalar-valued autograd engine training a binary classifier on the moons dataset.
    Replicates [karpathy/micrograd demo.ipynb](https://github.com/karpathy/micrograd/blob/master/demo.ipynb).
    """)
    return


@app.cell
def _():
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    return mo, np, plt, random


@app.cell
def _():
    from micrograd.engine import Value
    from micrograd.nn import Neuron, Layer, MLP
    return Layer, MLP, Neuron, Value


@app.cell
def _(np, random):
    np.random.seed(1337)
    random.seed(1337)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset

    Two interleaved half-circles — a classic non-linearly separable binary classification problem.
    Labels are ±1 (SVM convention).
    """)
    return


@app.cell
def _(np, plt):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1  # make y be -1 or 1

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
    ax.set_title("Training data")
    fig
    return X, ax, fig, make_moons, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model

    A 2-layer MLP with 16 hidden units per layer and a single scalar output.
    The last layer is linear (no activation) — the sign of the output determines the class.
    """)
    return


@app.cell
def _(MLP):
    model = MLP(2, [16, 16, 1])
    print(model)
    print("number of parameters", len(model.parameters()))
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss Function

    SVM-style hinge loss with L2 regularization:

    - **Data loss**: `(1 + -y_i * score_i).relu()` — penalizes predictions with wrong sign or insufficient margin
    - **Reg loss**: `α * Σ(p²)` — keeps weights small
    """)
    return


@app.cell
def _(Value, X, model, np, y):
    def loss(batch_size=None):
        # inline DataLoader
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]
        inputs = [list(map(Value, xrow)) for xrow in Xb]

        # forward pass
        scores = list(map(model, inputs))

        # svm "max-margin" loss
        losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))

        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p*p for p in model.parameters()))
        total_loss = data_loss + reg_loss

        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    total_loss0, acc0 = loss()
    print(f"initial loss: {total_loss0.data:.4f}, accuracy: {acc0*100:.1f}%")
    return acc0, loss, total_loss0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    SGD with a linearly decaying learning rate (1.0 → 0.1 over 100 steps).
    Each step: forward → zero grads → backward → update.
    """)
    return


@app.cell
def _(loss, model):
    for k in range(100):
        total_loss, acc = loss()

        model.zero_grad()
        total_loss.backward()

        learning_rate = 1.0 - 0.9*k/100
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 10 == 0:
            print(f"step {k:3d}  loss {total_loss.data:.4f}  accuracy {acc*100:.1f}%")

    print(f"\nfinal  loss {total_loss.data:.4f}  accuracy {acc*100:.1f}%")
    return acc, k, learning_rate, total_loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision Boundary

    Evaluate the model on a dense grid and shade each region by predicted class.
    """)
    return


@app.cell
def _(Value, X, acc, model, np, plt, total_loss, y):
    # total_loss and acc are used only to ensure this cell runs after training
    _ = total_loss, acc

    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    boundary_fig, boundary_ax = plt.subplots(figsize=(6, 5))
    boundary_ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    boundary_ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    boundary_ax.set_xlim(xx.min(), xx.max())
    boundary_ax.set_ylim(yy.min(), yy.max())
    boundary_ax.set_title("Decision boundary after training")
    boundary_fig
    return Z, Xmesh, boundary_ax, boundary_fig, h, inputs, scores, x_max, x_min, xx, y_max, y_min, yy


if __name__ == "__main__":
    app.run()
