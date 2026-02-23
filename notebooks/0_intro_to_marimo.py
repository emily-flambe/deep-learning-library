import marimo

__generated_with = "0.20.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Building Micrograd from Scratch
    """)
    return


@app.cell
def _():
    y=4
    return (y,)


@app.cell
def _(y):
    def f(x):
        return 3*x**y- 4*x + 5

    return (f,)


@app.cell
def _(f):
    f(3.0)
    return


@app.cell
def _(f, np, plt):
    xs = np.arange(-3,5,0.5)
    ys = f(xs)
    plt.plot(xs,ys)
    return


if __name__ == "__main__":
    app.run()
