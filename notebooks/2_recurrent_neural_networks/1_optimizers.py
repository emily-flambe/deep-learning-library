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

    # Clean up: strip sonnet numbers, lowercase everything
    import re
    text = re.sub(r'\n\s+\d+\n', '\n\n', text)
    text = text.lower()

    # Use first 100k chars to keep training fast
    text = text[:100_000]

    # Build vocabulary from all unique characters
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # Encode full text as integers
    data = [char_to_ix[ch] for ch in text]

    print(f"Loaded {len(text)} characters, vocabulary size: {vocab_size}")
    return char_to_ix, data, ix_to_char, mo, np, vocab_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimizers: Why Adam Beats Adagrad for RNN Training

    In the previous notebook we trained a character-level RNN on Shakespeare using Adam. But *why* Adam? What makes it better than simpler optimizers?

    We're going to answer that question empirically. We'll train the **exact same RNN architecture** three times — identical weights, identical data, identical number of iterations — but with three different optimizers:

    1. **Vanilla SGD** (stochastic gradient descent)
    2. **Adagrad** (adaptive gradient)
    3. **Adam** (adaptive moment estimation) with cosine learning rate decay

    Then we'll compare the loss curves, the generated text quality, and the effective learning rates to understand what's actually happening under the hood.
    """)
    return


@app.cell
def _(np, vocab_size):
    class CharRNN:
        """Character-level RNN, identical to 0_char_rnn.py."""

        def __init__(self, vocab_size, hidden_size, seed=42):
            rng = np.random.default_rng(seed)
            scale = 0.01

            self.hidden_size = hidden_size
            self.vocab_size = vocab_size

            # Weight matrices
            self.Wxh = rng.standard_normal((hidden_size, vocab_size)) * scale
            self.Whh = rng.standard_normal((hidden_size, hidden_size)) * scale
            self.Why = rng.standard_normal((vocab_size, hidden_size)) * scale
            self.bh = np.zeros((hidden_size, 1))
            self.by = np.zeros((vocab_size, 1))

        def params(self):
            return [self.Wxh, self.Whh, self.Why, self.bh, self.by]

        def forward_and_loss(self, inputs, targets, h_prev):
            xs, hs, ys, ps = {}, {}, {}, {}
            hs[-1] = h_prev.copy()
            loss = 0.0

            for t in range(len(inputs)):
                xs[t] = np.zeros((self.vocab_size, 1))
                xs[t][inputs[t]] = 1.0
                hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)
                ys[t] = self.Why @ hs[t] + self.by
                exp_ys = np.exp(ys[t] - np.max(ys[t]))
                ps[t] = exp_ys / exp_ys.sum()
                loss += -np.log(ps[t][targets[t], 0])

            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dWhy = np.zeros_like(self.Why)
            dbh = np.zeros_like(self.bh)
            dby = np.zeros_like(self.by)
            dh_next = np.zeros_like(hs[0])

            for t in reversed(range(len(inputs))):
                dy = ps[t].copy()
                dy[targets[t]] -= 1
                dWhy += dy @ hs[t].T
                dby += dy
                dh = self.Why.T @ dy + dh_next
                dh_raw = (1 - hs[t] ** 2) * dh
                dWxh += dh_raw @ xs[t].T
                dWhh += dh_raw @ hs[t-1].T
                dbh += dh_raw
                dh_next = self.Whh.T @ dh_raw

            grads = [dWxh, dWhh, dWhy, dbh, dby]
            for i in range(len(grads)):
                np.clip(grads[i], -5, 5, out=grads[i])

            return loss, grads, hs[len(inputs) - 1]

        def sample(self, h, seed_ix, n, temperature=1.0, rng=None):
            if rng is None:
                rng = np.random.default_rng()
            x = np.zeros((self.vocab_size, 1))
            x[seed_ix] = 1.0
            indices = []

            for _ in range(n):
                h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
                y = self.Why @ h + self.by
                y = y / temperature
                exp_y = np.exp(y - np.max(y))
                p = exp_y / exp_y.sum()
                ix = rng.choice(range(self.vocab_size), p=p.ravel())
                x = np.zeros((self.vocab_size, 1))
                x[ix] = 1.0
                indices.append(ix)

            return indices

    hidden_size = 128
    seq_length = 50
    num_iterations = 10_000
    log_every = 200

    print(f"RNN config: vocab_size={vocab_size}, hidden_size={hidden_size}")
    print(f"Training: {num_iterations} iterations, seq_length={seq_length}")
    return CharRNN, hidden_size, log_every, num_iterations, seq_length


@app.cell
def _(CharRNN, data, hidden_size, log_every, np, num_iterations, seq_length, vocab_size):
    def train_rnn(optimizer_name, update_fn):
        """
        Train a fresh CharRNN and return (loss_history, rnn, update_rms_history).

        optimizer_name: label for printing
        update_fn: callable(params, grads, iteration) that mutates params in-place
                   and returns the list of parameter deltas applied
        """
        rnn = CharRNN(vocab_size, hidden_size, seed=42)
        params = rnn.params()

        smooth_loss = -np.log(1.0 / vocab_size) * seq_length
        h_prev = np.zeros((hidden_size, 1))
        pointer = 0
        data_size = len(data)

        loss_history = []
        update_rms_history = []

        for iteration in range(num_iterations):
            if pointer + seq_length + 1 >= data_size:
                pointer = 0
                h_prev = np.zeros((hidden_size, 1))

            inputs = data[pointer:pointer + seq_length]
            targets = data[pointer + 1:pointer + seq_length + 1]

            loss, grads, h_prev = rnn.forward_and_loss(inputs, targets, h_prev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            deltas = update_fn(params, grads, iteration)

            if iteration % log_every == 0:
                loss_history.append((iteration, smooth_loss))
                # RMS of actual parameter updates across all weights
                all_deltas = np.concatenate([d.ravel() for d in deltas])
                rms = np.sqrt(np.mean(all_deltas ** 2))
                update_rms_history.append((iteration, rms))

            pointer += seq_length

        # Final data point
        loss_history.append((num_iterations, smooth_loss))

        print(f"{optimizer_name}: final per-char loss = {smooth_loss / seq_length:.3f}")
        return loss_history, rnn, update_rms_history

    return (train_rnn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vanilla SGD

    The simplest optimizer. Every parameter gets the same learning rate, every step:

    $$\theta \leftarrow \theta - \eta \cdot \nabla_\theta L$$

    where $\eta$ is the learning rate and $\nabla_\theta L$ is the gradient.

    This is fine for convex problems but struggles with the loss landscapes of deep networks, where different parameters may need very different step sizes. A learning rate that works for one weight matrix may be too large for another and too small for a third.
    """)
    return


@app.cell
def _(np, train_rnn):
    # --- SGD ---
    _sgd_lr = 0.1

    def _sgd_update(params, grads, iteration):
        deltas = []
        for p, g in zip(params, grads):
            delta = -_sgd_lr * g
            p += delta
            deltas.append(delta)
        return deltas

    sgd_loss, sgd_rnn, sgd_rms = train_rnn("SGD", _sgd_update)
    return sgd_loss, sgd_rms, sgd_rnn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adagrad (Adaptive Gradient)

    Adagrad (Duchi et al., 2011) gives each parameter its own learning rate based on its gradient history. Parameters with large gradients get smaller steps; parameters with small gradients get larger steps.

    $$G_t = G_{t-1} + g_t^2$$
    $$\theta \leftarrow \theta - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t$$

    The key insight: **the accumulator $G_t$ only ever grows**. It's a running sum of squared gradients from the beginning of training. This means the effective learning rate $\frac{\eta}{\sqrt{G_t} + \epsilon}$ monotonically decreases.

    For sparse problems (like NLP with rare words), this is a feature: common features get their learning rate aggressively reduced while rare features keep learning. But for RNN training, it's a bug: after enough iterations, the learning rate approaches zero for *all* parameters and the model stops learning entirely.
    """)
    return


@app.cell
def _(np, train_rnn):
    # --- Adagrad ---
    _adagrad_lr = 0.1

    def _make_adagrad():
        mem = {}

        def update(params, grads, iteration):
            deltas = []
            for i, (p, g) in enumerate(zip(params, grads)):
                if i not in mem:
                    mem[i] = np.zeros_like(p)
                # Accumulate squared gradients forever
                mem[i] += g * g
                delta = -_adagrad_lr * g / (np.sqrt(mem[i]) + 1e-8)
                p += delta
                deltas.append(delta)
            return deltas

        return update

    _adagrad_update = _make_adagrad()
    adagrad_loss, adagrad_rnn, adagrad_rms = train_rnn("Adagrad", _adagrad_update)
    return adagrad_loss, adagrad_rms, adagrad_rnn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adam (Adaptive Moment Estimation)

    Adam (Kingma & Ba, 2014) combines two ideas:

    1. **Momentum** (first moment $m$): an exponential moving average of the gradients, which smooths out noise and accelerates convergence in consistent directions
    2. **RMSprop-style scaling** (second moment $v$): an exponential moving average of squared gradients, which adapts the step size per-parameter

    $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
    $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
    $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
    $$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

    The **bias correction** (the hat terms) is important: since $m$ and $v$ are initialized to zero, they're biased toward zero in early training. The correction compensates for this.

    The critical difference from Adagrad: instead of accumulating squared gradients forever, Adam uses an exponential moving average that "forgets" old gradients. This means the effective learning rate doesn't decay to zero — it adapts to the *recent* gradient landscape.

    We also add a **cosine learning rate schedule** that smoothly anneals the base learning rate from $\eta$ to $0$ over training. This helps prevent late-training instability where the model oscillates around a good solution.

    ### The optimizer family tree

    > SGD $\rightarrow$ Momentum $\rightarrow$ Adagrad $\rightarrow$ RMSprop (Hinton, unpublished lecture notes) $\rightarrow$ **Adam**
    """)
    return


@app.cell
def _(np, train_rnn):
    # --- Adam with cosine LR schedule ---
    _adam_base_lr = 1e-3
    _adam_beta1 = 0.9
    _adam_beta2 = 0.999
    _adam_eps = 1e-8
    _adam_num_iters = 10_000  # must match num_iterations

    def _make_adam():
        m_state = {}
        v_state = {}

        def update(params, grads, iteration):
            t_step = iteration + 1
            # Cosine learning rate decay
            lr = _adam_base_lr * 0.5 * (1 + np.cos(np.pi * iteration / _adam_num_iters))

            deltas = []
            for i, (p, g) in enumerate(zip(params, grads)):
                if i not in m_state:
                    m_state[i] = np.zeros_like(p)
                    v_state[i] = np.zeros_like(p)
                # Exponential moving averages
                m_state[i] = _adam_beta1 * m_state[i] + (1 - _adam_beta1) * g
                v_state[i] = _adam_beta2 * v_state[i] + (1 - _adam_beta2) * g * g
                # Bias correction
                m_hat = m_state[i] / (1 - _adam_beta1 ** t_step)
                v_hat = v_state[i] / (1 - _adam_beta2 ** t_step)
                delta = -lr * m_hat / (np.sqrt(v_hat) + _adam_eps)
                p += delta
                deltas.append(delta)
            return deltas

        return update

    _adam_update = _make_adam()
    adam_loss, adam_rnn, adam_rms = train_rnn("Adam", _adam_update)
    return adam_loss, adam_rms, adam_rnn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss Comparison

    Here's the money shot. All three optimizers started from identical weights — the only difference is how they update those weights.
    """)
    return


@app.cell
def _(adam_loss, adagrad_loss, mo, seq_length, sgd_loss):
    import matplotlib.pyplot as plt

    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))

    for label, history, color, ls in [
        ("SGD (lr=0.1)", sgd_loss, "#2196F3", "-"),
        ("Adagrad (lr=0.1)", adagrad_loss, "#FF9800", "-"),
        ("Adam (lr=1e-3, cosine)", adam_loss, "#4CAF50", "-"),
    ]:
        iters, losses = zip(*history)
        per_char = [l / seq_length for l in losses]
        ax_loss.plot(iters, per_char, color=color, linestyle=ls, linewidth=2, label=label)

    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss per Character")
    ax_loss.set_title("Training Loss: SGD vs Adagrad vs Adam")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=11)

    plt.tight_layout()
    mo.output.replace(mo.as_html(fig_loss))
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the three distinct behaviors:

    - **SGD** (blue): learns slowly and steadily. The fixed learning rate means it can't adapt to the curvature of the loss landscape.
    - **Adagrad** (orange): drops fast early on because of per-parameter learning rates, but then **plateaus**. The squared gradient accumulator grows without bound, choking off learning.
    - **Adam** (green): gets the best of both worlds. Fast early progress *and* continued improvement throughout training. The cosine schedule gently reduces the learning rate to stabilize the endgame.

    ## Generated Text Comparison

    Loss numbers are abstract. Let's make the difference concrete by generating text from each trained model.
    """)
    return


@app.cell
def _(adam_rnn, adagrad_rnn, char_to_ix, hidden_size, ix_to_char, mo, np, sgd_rnn):
    _gen_results = {}
    for name, model in [("SGD", sgd_rnn), ("Adagrad", adagrad_rnn), ("Adam", adam_rnn)]:
        _h = np.zeros((hidden_size, 1))
        _ixs = model.sample(_h, char_to_ix['\n'], 300, temperature=0.8, rng=np.random.default_rng(42))
        _gen_results[name] = ''.join(ix_to_char[ix] for ix in _ixs)

    mo.md(f"""
    **SGD** (10,000 iterations, lr=0.1):
    ```
    {_gen_results["SGD"]}
    ```

    **Adagrad** (10,000 iterations, lr=0.1):
    ```
    {_gen_results["Adagrad"]}
    ```

    **Adam** (10,000 iterations, lr=1e-3, cosine schedule):
    ```
    {_gen_results["Adam"]}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effective Learning Rate Over Training

    To understand *why* these optimizers behave differently, we can look at the RMS (root mean square) of the actual parameter updates at each logging step. This measures how much the model is actually changing its weights — the "effective step size" regardless of the optimizer's internal mechanics.
    """)
    return


@app.cell
def _(adam_rms, adagrad_rms, mo, plt, sgd_rms):
    fig_lr, ax_lr = plt.subplots(figsize=(10, 5))

    for label, history, color in [
        ("SGD", sgd_rms, "#2196F3"),
        ("Adagrad", adagrad_rms, "#FF9800"),
        ("Adam (cosine)", adam_rms, "#4CAF50"),
    ]:
        iters, rms_vals = zip(*history)
        ax_lr.plot(iters, rms_vals, color=color, linewidth=2, label=label)

    ax_lr.set_xlabel("Iteration")
    ax_lr.set_ylabel("RMS of Parameter Updates")
    ax_lr.set_title("Effective Step Size Over Training")
    ax_lr.legend(fontsize=11)
    ax_lr.grid(True, alpha=0.3)
    ax_lr.set_yscale("log")
    plt.tight_layout()
    mo.output.replace(mo.as_html(fig_lr))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This plot tells the whole story:

    - **SGD**: flat line. Same step size throughout. No adaptation whatsoever.
    - **Adagrad**: the effective step size decays toward zero as the squared gradient accumulator grows without bound. By the end of training, the model is barely moving.
    - **Adam (cosine)**: starts with meaningful steps, adapts throughout, and the cosine schedule provides a smooth ramp-down at the end. The model keeps learning for much longer than Adagrad.

    ## Summary

    | Optimizer | Mechanism | Strength | Weakness |
    |-----------|-----------|----------|----------|
    | **SGD** | Fixed learning rate for all params | Simple, well-understood | Can't adapt to loss landscape |
    | **Adagrad** | Per-param LR based on sum of squared gradients | Fast early learning, good for sparse data | LR decays to zero, stops learning |
    | **Adam** | Exponential moving averages of gradient + squared gradient | Adapts AND keeps learning | More hyperparameters (usually the defaults work) |

    **Adam is the default choice for most deep learning.** The standard hyperparameters ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) work well across a wide range of problems. Adding a cosine learning rate schedule on top is a common and effective practice.

    The lineage of ideas: **SGD** $\rightarrow$ **Momentum** (Polyak, 1964) $\rightarrow$ **Adagrad** (Duchi et al., 2011) $\rightarrow$ **RMSprop** (Hinton, unpublished lecture notes, 2012) $\rightarrow$ **Adam** (Kingma & Ba, 2014). Each step fixed a limitation of the previous one.
    """)
    return


if __name__ == "__main__":
    app.run()
