import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import math
    import random
    import torch
    from utils import Value, draw_dot

    return Value, draw_dot, mo, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Let's make a Neuron
    """)
    return


@app.cell
def _(Value, random):
    class Neuron:

        def __init__(self, nin):
            self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
            self.b = Value(random.uniform(-1,1))

        def __call__(self, x):
            # w*x + b
            act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
            out = act.tanh()
            return out

        def parameters(self):
            return self.w + [self.b]

    return (Neuron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Let's make a Neural Network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layer: a class combining multiple neurons
    """)
    return


@app.cell
def _(Neuron):
    class Layer:

        def __init__(self, nin, nout):
            self.neurons = [Neuron(nin) for _ in range(nout)]

        def __call__(self, x):
            outs = [n(x) for n in self.neurons]
            return outs[0] if len(outs) == 1 else outs

        def parameters(self):
            return [p for neuron in self.neurons for p in neuron.parameters()]

    return (Layer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-Layer Perceptron (MLP):

    A network consisting of at least three Layers (input, hidden, and output)
    """)
    return


@app.cell
def _(Layer):
    class MLP:

        def __init__(self, nin, nouts):
            """
            nouts = list of output sizes
            """
            sz = [nin] + nouts
            self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def parameters(self):
            return [p for layer in self.layers for p in layer.parameters()]

    return (MLP,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implement a Multilayer Perceptron

    Following the image below, we will be implementing the structure shown on the right:

    - 3 input neurons
    - 2 layers of 4 neurons
    - 1 output layer of 1 neuron
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/MLP.png")
    return


@app.cell
def _(MLP):
    x = [2.0, 3.0, -1.0] # input layer of 3 neurons
    n = MLP(3, [4,4,1]) # 3 = size of input, [4,4,1] = sizes of the remaining hidden layers & output layer
    n(x)
    return n, x


@app.cell
def _(n):
    # This shows all the weights AND biases in the entire neural net
    n.parameters()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The graph is getting gnarly
    """)
    return


@app.cell
def _(draw_dot, n, x):
    draw_dot(n(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural net with multiple input layers

    The following is an example of a very basic binary classifier neural net. We start with a set of 4 inputs (xs) that each have a corresponding target values (ys).
    """)
    return


@app.cell
def _(n):
    # four inputs to the neural net
    xs = [
        [2.0, 3.0, -1.0], #desired target of ys[0]
        [3.0, -1.0, 0.5], #desired target of ys[1] ... etc (these correspond to the ys)
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    # desired targets (xs[n] => ys[n])
    ys = [1.0, -1.0, -1.0, 1.0]

    # n is the neural network that ...
    # ypred is the set of predicted values for the input xs from the starting state of the neural network.
    # These predictions will be shitty, so the act of training will modify the weights of the model and stuff
    ypred = [n(x) for x in xs]
    ypred
    return ypred, ys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loss

    In order to tune the weights to get better predictions, we calculate a single number measuring the total performance of the neural net: the "loss."

    To do this, we will implement mean squared loss. For each predicted value, we can square the amount it differs from its target:

    `[(ygt - yout)**2 for ygt, yout in zip(ys, ypred)]`

    This gives us an array that shows us how far off each prediction was from its target.
    """)
    return


@app.cell(hide_code=True)
def _(ypred, ys):
    [(ygt - yout)**2 for ygt, yout in zip(ys, ypred)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We sum up these squared differences in order to compute the final loss.
    """)
    return


@app.cell
def _(ypred, ys):
    # pair up the target values (ys) with the predicted values (ypred) to coumpute the loss
    loss = sum([(ygt - yout)**2 for ygt, yout in zip(ys, ypred)])
    loss
    return (loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our goal is to **minimize this loss value** by tuning the weights of the neural network.

    Then we run `backward()` on this loss, backpropagating gradients all along the neural network.
    """)
    return


@app.cell
def _(loss):
    loss.backward()
    return


@app.cell
def _(n):
    # We can inspect the value of the first weight in the first neuron of the first layer:
    n.layers[0].neurons[0].w[0].data
    return


@app.cell
def _(n):
    # And now that weight has a gradient because of the backward() pass:
    n.layers[0].neurons[0].w[0].grad
    return


@app.cell
def _(draw_dot, loss):
    draw_dot(loss)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we want handy code to gather up all the parameters of the neural net so we can operate on them simultaneously.

    Each one will be nudged a tiny amount based on the gradients.

    For this, we need to add `parameters` to the Neuron, Layer, and MLP classes.

    Neuron:

    ```
        def parameters(self):
            return self.w + [self.b]
    ```

    Layer:

    ```
        def parameters(self):
            return [p for neuron in self.neurons for p in neuron.parameters()]
    ```

    MLP:

    ```
        def parameters(self):
            return [p for layer in self.layers for p in layer.parameters()]
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the neural network's `parameters` is big beautiful vector containing **all of the weights and biases of the neural net:**
    """)
    return


@app.cell
def _(n):
    n.parameters()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Next up: gradient descent

    We now have a neural network that computes a forward pass, calculates loss, and backpropagates gradients. In the next notebook, we'll use those gradients to actually *train* the network.
    """)
    return


if __name__ == "__main__":
    app.run()
