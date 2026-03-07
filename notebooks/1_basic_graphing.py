import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    from utils import Value, draw_dot

    return Value, draw_dot, math, mo


@app.cell
def _(Value):
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10, label='c')
    e = a*b
    e.label = 'e'
    d = e + c
    d.label = 'd'
    f = Value(-2.0, label='f')

    L = d*f
    L.label = 'L'
    return (L,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## L = (a*b + c) * f
    """)
    return


@app.cell(hide_code=True)
def _(L, draw_dot):
    draw_dot(L)
    return


if __name__ == "__main__":
    app.run()
