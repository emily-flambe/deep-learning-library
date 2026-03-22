import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
async def _():
    import sys
    import marimo as mo
    import numpy as np

    # Fetch Shakespeare text
    shakespeare_url = "https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw"
    if sys.platform == "emscripten":
        import pyodide.http
        resp = await pyodide.http.pyfetch(shakespeare_url)
        text = await resp.string()
    else:
        import urllib.request
        text = urllib.request.urlopen(shakespeare_url).read().decode("utf-8")

    # Strip sonnet numbers (e.g. "                    42\n") to reduce noise
    import re
    text = re.sub(r'\n\s+\d+\n', '\n\n', text)

    # Use first 100k chars to keep training fast
    text = text[:100_000]
    print(f"Loaded {len(text)} characters of Shakespeare")
    return mo, np, re, text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Character-Level RNN

    We're going to build a **recurrent neural network** (RNN) from scratch using only numpy — no PyTorch, no frameworks.

    The task: learn to generate Shakespeare-like text, one character at a time.

    ## Why RNNs?

    The neural networks we built previously (MLPs) take a fixed-size input and produce a fixed-size output. But language is *sequential* — the meaning of a word depends on what came before it. RNNs handle this by maintaining a **hidden state** that gets updated at each time step, giving the network a form of memory.

    ## The plan

    1. Build a vocabulary from individual characters (no tokenizer needed!)
    2. Initialize an RNN with random weights → generate text (gibberish)
    3. Train the RNN on Shakespeare
    4. Generate text again → something more readable
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Character Vocabulary

    Instead of using a tokenizer to break text into words or subwords, we treat each unique character as its own token. This is the simplest possible approach — our "vocabulary" is just the set of characters that appear in the text.
    """)
    return


@app.cell
def _(text):
    # Build vocabulary from all unique characters
    chars = sorted(set(text))
    vocab_size = len(chars)

    # Mappings between characters and integers
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Characters: {''.join(chars[:40])}...")
    return char_to_ix, chars, ix_to_char, vocab_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: The RNN

    Here's the core idea. At each time step $t$, the RNN:

    1. Takes the current character $x_t$ (as a one-hot vector)
    2. Combines it with the previous hidden state $h_{t-1}$
    3. Produces a new hidden state $h_t$ and an output $y_t$

    The equations:

    $$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

    $$y_t = W_{hy} \cdot h_t + b_y$$

    The hidden state $h_t$ is the RNN's "memory" — it accumulates information from all previous characters. The output $y_t$ is a score for each character in the vocabulary, which we convert to probabilities with softmax.

    Three weight matrices, two bias vectors — that's the entire model.
    """)
    return


@app.cell
def _(np, vocab_size):
    class CharRNN:
        def __init__(self, vocab_size, hidden_size, seed=42):
            rng = np.random.default_rng(seed)
            scale = 0.01

            self.hidden_size = hidden_size
            self.vocab_size = vocab_size

            # Weight matrices
            self.Wxh = rng.standard_normal((hidden_size, vocab_size)) * scale   # input -> hidden
            self.Whh = rng.standard_normal((hidden_size, hidden_size)) * scale  # hidden -> hidden
            self.Why = rng.standard_normal((vocab_size, hidden_size)) * scale   # hidden -> output
            self.bh = np.zeros((hidden_size, 1))                                # hidden bias
            self.by = np.zeros((vocab_size, 1))                                 # output bias

        def forward_and_loss(self, inputs, targets, h_prev):
            """
            Forward pass through a sequence, computing loss.
            inputs/targets: lists of integer character indices
            h_prev: previous hidden state
            Returns: loss, gradients, last hidden state
            """
            xs, hs, ys, ps = {}, {}, {}, {}
            hs[-1] = h_prev.copy()
            loss = 0.0

            # Forward pass: one character at a time
            for t in range(len(inputs)):
                # One-hot encode input
                xs[t] = np.zeros((self.vocab_size, 1))
                xs[t][inputs[t]] = 1.0

                # Hidden state update: h = tanh(Wxh @ x + Whh @ h_prev + bh)
                hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)

                # Output scores: y = Why @ h + by
                ys[t] = self.Why @ hs[t] + self.by

                # Softmax to get probabilities
                exp_ys = np.exp(ys[t] - np.max(ys[t]))  # subtract max for numerical stability
                ps[t] = exp_ys / exp_ys.sum()

                # Cross-entropy loss: -log(probability of correct next char)
                loss += -np.log(ps[t][targets[t], 0])

            # Backward pass: backpropagation through time (BPTT)
            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dWhy = np.zeros_like(self.Why)
            dbh = np.zeros_like(self.bh)
            dby = np.zeros_like(self.by)
            dh_next = np.zeros_like(hs[0])

            for t in reversed(range(len(inputs))):
                # Gradient of loss w.r.t. output scores
                dy = ps[t].copy()
                dy[targets[t]] -= 1  # softmax + cross-entropy gradient

                # Gradients for Why and by
                dWhy += dy @ hs[t].T
                dby += dy

                # Gradient flowing back into hidden state
                dh = self.Why.T @ dy + dh_next

                # Backprop through tanh: d/dx tanh(x) = 1 - tanh(x)^2
                dh_raw = (1 - hs[t] ** 2) * dh

                # Gradients for Wxh, Whh, bh
                dWxh += dh_raw @ xs[t].T
                dWhh += dh_raw @ hs[t-1].T
                dbh += dh_raw

                # Pass gradient to previous time step
                dh_next = self.Whh.T @ dh_raw

            # Clip gradients to prevent explosion
            grads = [dWxh, dWhh, dWhy, dbh, dby]
            for i in range(len(grads)):
                np.clip(grads[i], -5, 5, out=grads[i])

            return loss, grads, hs[len(inputs) - 1]

        def sample(self, h, seed_ix, n, temperature=1.0, rng=None):
            """
            Generate n characters starting from seed_ix.
            h: initial hidden state
            temperature: higher = more random, lower = more conservative
            """
            if rng is None:
                rng = np.random.default_rng()
            x = np.zeros((self.vocab_size, 1))
            x[seed_ix] = 1.0
            indices = []

            for _ in range(n):
                h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
                y = self.Why @ h + self.by

                # Apply temperature
                y = y / temperature
                exp_y = np.exp(y - np.max(y))
                p = exp_y / exp_y.sum()

                # Sample from the distribution
                ix = rng.choice(range(self.vocab_size), p=p.ravel())
                x = np.zeros((self.vocab_size, 1))
                x[ix] = 1.0
                indices.append(ix)

            return indices

    # Create the model
    hidden_size = 128
    rnn = CharRNN(vocab_size, hidden_size)
    print(f"RNN created: {vocab_size} vocab × {hidden_size} hidden units")
    total_params = (
        hidden_size * vocab_size +    # Wxh
        hidden_size * hidden_size +   # Whh
        vocab_size * hidden_size +    # Why
        hidden_size +                  # bh
        vocab_size                     # by
    )
    print(f"Total parameters: {total_params:,}")
    return CharRNN, hidden_size, rnn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Generate Text Before Training

    With random weights, the RNN has no idea what Shakespeare looks like. Let's see what it produces — this should be pure gibberish.
    """)
    return


@app.cell
def _(char_to_ix, hidden_size, ix_to_char, np, rnn):
    # Generate text with the untrained model
    h_init = np.zeros((hidden_size, 1))
    sample_ixs = rnn.sample(h_init, char_to_ix['\n'], 500, rng=np.random.default_rng(0))
    untrained_text = ''.join(ix_to_char[ix] for ix in sample_ixs)
    print("=== BEFORE TRAINING (random weights) ===\n")
    print(untrained_text)
    return h_init, untrained_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Yep — total gibberish. The model is just sampling characters at near-random from the vocabulary.

    ## Step 4: Training

    We train using **Adagrad** (adaptive gradient descent). For each chunk of 25 characters, we:

    1. Run the forward pass to compute loss
    2. Backpropagate gradients through the sequence
    3. Update weights using the gradients

    The key insight: after each chunk, we carry over the hidden state to the next chunk. This lets the RNN "remember" context across chunks, even though we only backpropagate within each 25-character window. (This is called **truncated backpropagation through time**.)
    """)
    return


@app.cell
def _(char_to_ix, hidden_size, ix_to_char, mo, np, rnn, text):
    # Training hyperparameters
    seq_length = 25    # characters per training chunk
    learning_rate = 1e-1
    num_iterations = 20_000

    # Encode the full text as integers
    data = [char_to_ix[ch] for ch in text]
    data_size = len(data)

    # Adagrad memory (sum of squared gradients)
    params = [rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by]
    memory = [np.zeros_like(p) for p in params]

    # Training loop
    smooth_loss = -np.log(1.0 / rnn.vocab_size) * seq_length  # initial loss estimate
    h_prev = np.zeros((hidden_size, 1))
    pointer = 0
    loss_history = []

    for iteration in range(num_iterations):
        # Reset if we've gone past the data
        if pointer + seq_length + 1 >= data_size:
            pointer = 0
            h_prev = np.zeros((hidden_size, 1))

        # Grab a chunk: inputs and targets (shifted by 1)
        inputs = data[pointer:pointer + seq_length]
        targets = data[pointer + 1:pointer + seq_length + 1]

        # Forward + backward
        loss, grads, h_prev = rnn.forward_and_loss(inputs, targets, h_prev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Adagrad update
        for param, grad, mem in zip(params, grads, memory):
            mem += grad * grad
            param -= learning_rate * grad / (np.sqrt(mem) + 1e-8)

        pointer += seq_length

        if iteration % 2000 == 0:
            loss_history.append((iteration, smooth_loss))

    loss_history.append((num_iterations, smooth_loss))

    mo.md(f"""
    **Training complete!**

    - Iterations: {num_iterations:,}
    - Starting loss: {loss_history[0][1]:.1f}
    - Final loss: {loss_history[-1][1]:.1f}
    """)
    return data, loss_history, seq_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loss Over Training

    The loss should decrease over time as the model learns character patterns.
    """)
    return


@app.cell
def _(loss_history, mo):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    iters, losses = zip(*loss_history)
    ax.plot(iters, losses, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Smooth Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.output.replace(mo.as_html(fig))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: Generate Text After Training

    Now let's see what the RNN produces after training. It should look noticeably more "Shakespeare-ish" — real words appearing, some structure, maybe even fragments of verse. Not coherent, but clearly better than the random gibberish we started with.
    """)
    return


@app.cell
def _(char_to_ix, hidden_size, ix_to_char, np, rnn):
    # Generate text with the TRAINED model
    _h = np.zeros((hidden_size, 1))
    _sample_ixs = rnn.sample(_h, char_to_ix['\n'], 500, temperature=0.8, rng=np.random.default_rng(42))
    trained_text = ''.join(ix_to_char[ix] for ix in _sample_ixs)
    print("=== AFTER TRAINING ===\n")
    print(trained_text)
    return (trained_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the RNN Learned

    Even this simple RNN picks up real patterns:

    - **Character frequencies**: common letters appear more often
    - **Common sequences**: "th", "he", "in", "ou" — it learns English digrams
    - **Word boundaries**: spaces appear in roughly the right places
    - **Line structure**: it learns that newlines happen periodically

    What it *can't* do well: long-range coherence. The hidden state is too small and the vanilla RNN suffers from the **vanishing gradient problem** — gradients shrink exponentially as they backpropagate through time, making it hard to learn long-distance dependencies.

    This is exactly what **LSTMs** and **Transformers** were invented to fix.

    ## The Architecture at a Glance

    ```
    Input (one-hot char) ──→ [Wxh] ──┐
                                      ├──→ tanh ──→ hidden state ──→ [Why] ──→ output scores ──→ softmax ──→ next char
    Previous hidden state ──→ [Whh] ──┘                    │
                                                           └── fed back as input to next step
    ```

    That feedback loop is what makes it "recurrent" — the same weights are applied at every time step, but the hidden state carries information forward through the sequence.
    """)
    return


@app.cell(hide_code=True)
def _(mo, trained_text, untrained_text):
    mo.md(f"""
    ## Side-by-Side Comparison

    **Before training** (random weights):
    ```
    {untrained_text[:300]}
    ```

    **After training** ({20_000} iterations):
    ```
    {trained_text[:300]}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Generation

    Try different temperatures to see how they affect the output:
    - **Low temperature (0.3)**: more conservative, repetitive but "safer" choices
    - **Medium (0.8)**: balanced — readable but varied
    - **High (1.5)**: more creative/chaotic, more errors
    """)
    return


@app.cell
def _(mo):
    temp_slider = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=0.8,
        label="Temperature"
    )
    temp_slider
    return (temp_slider,)


@app.cell
def _(char_to_ix, hidden_size, ix_to_char, np, rnn, temp_slider):
    _h = np.zeros((hidden_size, 1))
    _ixs = rnn.sample(_h, char_to_ix['\n'], 500, temperature=temp_slider.value, rng=np.random.default_rng(7))
    _generated = ''.join(ix_to_char[ix] for ix in _ixs)
    print(f"Temperature = {temp_slider.value:.1f}\n")
    print(_generated)
    return


if __name__ == "__main__":
    app.run()
