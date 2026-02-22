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
    # Hello marimo! 🧠

    This is your first marimo notebook for the micrograd tutorial.
    """)
    return


@app.cell
def _(np, plt):
    # Plot a simple function: f(x) = 3x^2 - 4x + 5
    xs = np.arange(-5, 5, 0.25)
    ys = 3 * xs**2 - 4 * xs + 5
    plt.plot(xs, ys)
    plt.title("f(x) = 3x² - 4x + 5")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
