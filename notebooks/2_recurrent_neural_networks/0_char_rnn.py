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

    # Clean up: strip sonnet numbers, lowercase, trim before copyright notice
    import re
    text = re.sub(r'\n\s+\d+\n', '\n\n', text)
    text = text.lower()
    text = text[:98_000]

    # Build vocabulary
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Loaded {len(text)} characters of Shakespeare")
    print(f"Vocabulary: {vocab_size} unique characters — {''.join(chars)}")
    return char_to_ix, chars, ix_to_char, mo, np, re, text, vocab_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Character-Level RNN: From Gibberish to Shakespeare

    We're going to build a **recurrent neural network** (RNN) from scratch using only numpy — no PyTorch, no frameworks — and train it to generate Shakespeare one character at a time.

    ## Why RNNs?

    The neural networks we built previously (MLPs) take a fixed-size input and produce a fixed-size output. But language is *sequential* — the meaning of a word depends on what came before it. RNNs handle this by maintaining a **hidden state** that gets updated at each time step, giving the network a form of memory.

    ## The plan

    1. Build a vocabulary from individual characters (no tokenizer needed!)
    2. Build the RNN and generate text with random weights (gibberish)
    3. Train and watch it learn — snapshots at 1, 5, and 10 epochs
    4. Compare two optimizers: Adagrad vs Adam
    5. Interactive generation with a temperature slider
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Character Vocabulary

    Instead of a tokenizer, we treat each unique character as its own "token." We lowercase everything so the model doesn't waste capacity learning that 'T' and 't' are related.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The RNN

    At each time step $t$, the RNN takes a character $x_t$ and the previous hidden state $h_{t-1}$, and produces a new hidden state and an output:

    $$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

    $$y_t = W_{hy} \cdot h_t + b_y$$

    The hidden state $h_t$ is the RNN's "memory." The output $y_t$ gives scores for each character, which softmax converts to probabilities.

    Three weight matrices, two bias vectors — that's the entire model.

    ```
    Input (one-hot char) ──→ [Wxh] ──┐
                                      ├──→ tanh ──→ hidden state ──→ [Why] ──→ softmax ──→ next char
    Previous hidden state ──→ [Whh] ──┘         │
                                                └── fed back as input to next step
    ```

    That feedback loop is what makes it "recurrent."
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

            self.Wxh = rng.standard_normal((hidden_size, vocab_size)) * scale   # input -> hidden
            self.Whh = rng.standard_normal((hidden_size, hidden_size)) * scale  # hidden -> hidden
            self.Why = rng.standard_normal((vocab_size, hidden_size)) * scale   # hidden -> output
            self.bh = np.zeros((hidden_size, 1))
            self.by = np.zeros((vocab_size, 1))

        def params(self):
            return [self.Wxh, self.Whh, self.Why, self.bh, self.by]

        def forward_and_loss(self, inputs, targets, h_prev):
            """Forward pass + backpropagation through time (BPTT)."""
            xs, hs, ys, ps = {}, {}, {}, {}
            hs[-1] = h_prev.copy()
            loss = 0.0

            # Forward: process each character
            for t in range(len(inputs)):
                xs[t] = np.zeros((self.vocab_size, 1))
                xs[t][inputs[t]] = 1.0
                hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)
                ys[t] = self.Why @ hs[t] + self.by
                exp_ys = np.exp(ys[t] - np.max(ys[t]))
                ps[t] = exp_ys / exp_ys.sum()
                loss += -np.log(ps[t][targets[t], 0])

            # Backward: compute gradients
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
            """Generate n characters."""
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

    print(f"RNN: {vocab_size} vocab × {hidden_size} hidden")
    total_params = hidden_size * vocab_size + hidden_size * hidden_size + vocab_size * hidden_size + hidden_size + vocab_size
    print(f"Total parameters: {total_params:,}")
    return CharRNN, hidden_size, seq_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Before Training: Random Weights

    With random weights, the RNN has no idea what Shakespeare looks like.
    """)
    return


@app.cell
def _(CharRNN, char_to_ix, hidden_size, ix_to_char, np, vocab_size):
    _rnn_untrained = CharRNN(vocab_size, hidden_size, seed=42)
    _h = np.zeros((hidden_size, 1))
    _ixs = _rnn_untrained.sample(_h, char_to_ix['\n'], 300, rng=np.random.default_rng(0))
    untrained_text = ''.join(ix_to_char[ix] for ix in _ixs)
    print("=== RANDOM WEIGHTS ===\n")
    print(untrained_text)
    return (untrained_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pure gibberish. Now let's train it and watch it learn.

    ## Training: Watching the RNN Learn

    **Iterations vs epochs:** an iteration is one weight update (one batch of ~50 characters). An epoch is one full pass through the entire dataset. With our 98k characters and sequence length of 50:

    $$1 \text{ epoch} = \frac{98{,}000}{50} = 1{,}960 \text{ iterations}$$

    We'll train for **10 epochs** (~19,600 iterations) and snapshot the output at each epoch to watch the RNN gradually learn English.
    """)
    return


@app.cell
def _(CharRNN, char_to_ix, hidden_size, ix_to_char, mo, np, seq_length, text, vocab_size):
    # Encode text
    _data = [char_to_ix[ch] for ch in text]
    _data_size = len(_data)
    _iters_per_epoch = _data_size // seq_length
    _num_epochs = 10
    _num_iters = _num_epochs * _iters_per_epoch

    # Create fresh model with Adam optimizer
    rnn = CharRNN(vocab_size, hidden_size, seed=42)
    _params = rnn.params()
    _base_lr = 1e-3
    _beta1, _beta2, _eps = 0.9, 0.999, 1e-8
    _m = [np.zeros_like(p) for p in _params]
    _v = [np.zeros_like(p) for p in _params]

    _smooth_loss = -np.log(1.0 / vocab_size) * seq_length
    _h_prev = np.zeros((hidden_size, 1))
    _ptr = 0
    adam_loss_history = []
    adam_snapshots = {}

    for _i in range(_num_iters):
        if _ptr + seq_length + 1 >= _data_size:
            _ptr = 0
            _h_prev = np.zeros((hidden_size, 1))

        _inputs = _data[_ptr:_ptr + seq_length]
        _targets = _data[_ptr + 1:_ptr + seq_length + 1]
        _loss, _grads, _h_prev = rnn.forward_and_loss(_inputs, _targets, _h_prev)
        _smooth_loss = _smooth_loss * 0.999 + _loss * 0.001

        # Adam with cosine LR decay
        _lr = _base_lr * 0.5 * (1 + np.cos(np.pi * _i / _num_iters))
        _t = _i + 1
        for _p, _g, _mi, _vi in zip(_params, _grads, _m, _v):
            _mi[:] = _beta1 * _mi + (1 - _beta1) * _g
            _vi[:] = _beta2 * _vi + (1 - _beta2) * _g * _g
            _mhat = _mi / (1 - _beta1 ** _t)
            _vhat = _vi / (1 - _beta2 ** _t)
            _p -= _lr * _mhat / (np.sqrt(_vhat) + _eps)

        _ptr += seq_length

        if _i % 100 == 0:
            adam_loss_history.append((_i, _smooth_loss / seq_length))

        # Snapshot at each epoch
        if (_i + 1) % _iters_per_epoch == 0:
            _epoch = (_i + 1) // _iters_per_epoch
            _h0 = np.zeros((hidden_size, 1))
            _ixs = rnn.sample(_h0, char_to_ix['\n'], 300, temperature=0.8, rng=np.random.default_rng(42))
            adam_snapshots[_epoch] = {
                'text': ''.join(ix_to_char[ix] for ix in _ixs),
                'loss': _smooth_loss / seq_length,
            }

    adam_loss_history.append((_num_iters, _smooth_loss / seq_length))

    mo.md(f"**Training complete:** {_num_epochs} epochs ({_num_iters:,} iterations), final per-char loss: {_smooth_loss / seq_length:.3f}")
    return adam_loss_history, adam_snapshots, rnn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loss Curve
    """)
    return


@app.cell
def _(adam_loss_history, mo):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    _iters, _losses = zip(*adam_loss_history)
    ax.plot(_iters, _losses, color='#4CAF50', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss per Character')
    ax.set_title('Training Loss (Adam, 10 epochs)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.output.replace(mo.as_html(fig))
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generated Text at Each Epoch

    Watch the RNN gradually learn English:
    """)
    return


@app.cell
def _(adam_snapshots, mo, untrained_text):
    _sections = [f"""**Epoch 0** (random weights) — loss: 3.81
```
{untrained_text[:250]}
```
"""]
    for _epoch in sorted(adam_snapshots.keys()):
        _s = adam_snapshots[_epoch]
        _sections.append(f"""**Epoch {_epoch}** — loss: {_s['loss']:.3f}
```
{_s['text'][:250]}
```
""")
    mo.md("\n".join(_sections))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's Happening at Each Stage

    - **Epoch 0** (random): Uniform gibberish. Every character is roughly equally likely.
    - **Epoch 1**: Spaces appear. Common letters like 'e', 't', 'h' dominate. Word-like chunks emerge.
    - **Epoch 3**: Real English digrams ("th", "he", "in"). Many recognizable words: "the", "and", "thou".
    - **Epoch 5**: Sentence-like structure. Commas and line breaks in plausible places. Shakespeare vocabulary: "love", "beauty", "sweet".
    - **Epoch 10**: Longer coherent phrases. Still makes up words, but the rhythm and vocabulary are distinctly Shakespearean.

    With 50 epochs (~5 min) the output gets even better — here's what that looks like:

    ```
    that thou prien someting afrowed defander to fool
    with me things happete mear,
    all the time mady's words to forthard shade desire
    may the world i speed,
    thou hath the willuss them say their sweeted and cllame,
    when you more with that her life, my stall that i
    ag false it again,
    and but the...
    ```

    Not Shakespeare — but recognizably *trying* to be. A vanilla RNN's ceiling is limited by the **vanishing gradient problem**, where gradients shrink exponentially as they flow backward through time. **LSTMs** and **Transformers** were invented to fix this.

    ---

    ## Why Adam? Comparing Optimizers

    We used **Adam** as our optimizer above. But what is an optimizer, and why does the choice matter?

    During training, we compute gradients that tell us which direction to nudge each weight. The optimizer decides *how much* to nudge. The simplest approach — vanilla gradient descent — uses the same fixed learning rate for every weight. **Adagrad** and **Adam** both give each weight its own adaptive learning rate. But they do it very differently.

    Let's train the same RNN with Adagrad and compare.
    """)
    return


@app.cell
def _(CharRNN, char_to_ix, hidden_size, ix_to_char, np, seq_length, text, vocab_size):
    # Train a second RNN with Adagrad for comparison
    _data2 = [char_to_ix[ch] for ch in text]
    _data_size2 = len(_data2)
    _iters_per_epoch2 = _data_size2 // seq_length
    _num_iters2 = 10 * _iters_per_epoch2  # same 10 epochs

    _rnn_ag = CharRNN(vocab_size, hidden_size, seed=42)  # identical starting weights
    _params_ag = _rnn_ag.params()
    _lr_ag = 0.1
    _mem_ag = [np.zeros_like(p) for p in _params_ag]

    _sl_ag = -np.log(1.0 / vocab_size) * seq_length
    _hp_ag = np.zeros((hidden_size, 1))
    _ptr_ag = 0
    adagrad_loss_hist = []
    _ag_snaps = {}

    for _i in range(_num_iters2):
        if _ptr_ag + seq_length + 1 >= _data_size2:
            _ptr_ag = 0
            _hp_ag = np.zeros((hidden_size, 1))

        _inp = _data2[_ptr_ag:_ptr_ag + seq_length]
        _tgt = _data2[_ptr_ag + 1:_ptr_ag + seq_length + 1]
        _loss, _grads, _hp_ag = _rnn_ag.forward_and_loss(_inp, _tgt, _hp_ag)
        _sl_ag = _sl_ag * 0.999 + _loss * 0.001

        for _p, _g, _m in zip(_params_ag, _grads, _mem_ag):
            _m += _g * _g  # accumulate forever — this is the problem
            _p -= _lr_ag * _g / (np.sqrt(_m) + 1e-8)

        _ptr_ag += seq_length

        if _i % 100 == 0:
            adagrad_loss_hist.append((_i, _sl_ag / seq_length))

        if (_i + 1) % _iters_per_epoch2 == 0:
            _epoch = (_i + 1) // _iters_per_epoch2
            if _epoch in {1, 5, 10}:
                _h0 = np.zeros((hidden_size, 1))
                _ixs = _rnn_ag.sample(_h0, char_to_ix['\n'], 300, temperature=0.8, rng=np.random.default_rng(42))
                _ag_snaps[_epoch] = ''.join(ix_to_char[ix] for ix in _ixs)

    adagrad_loss_hist.append((_num_iters2, _sl_ag / seq_length))
    adagrad_snaps = _ag_snaps
    print(f"Adagrad done: final per-char loss {_sl_ag / seq_length:.3f}")
    return adagrad_loss_hist, adagrad_snaps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adagrad (Duchi, Hazan & Singer, 2011)

    Adagrad gives each weight its own learning rate, based on a running **sum** of all past squared gradients:

    $$G_t = G_{t-1} + g_t^2$$
    $$\theta \leftarrow \theta - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t$$

    Weights with historically large gradients get smaller steps. Weights barely touched get bigger steps.

    **The fatal flaw:** $G_t$ only ever grows. The effective learning rate monotonically shrinks toward zero. The model eventually stops learning.

    ### Adam (Kingma & Ba, 2014)

    Adam uses **exponential moving averages** instead of sums — old gradients fade away:

    $$m_t = 0.9 \cdot m_{t-1} + 0.1 \cdot g_t \quad \text{(momentum — smooths direction)}$$
    $$v_t = 0.999 \cdot v_{t-1} + 0.001 \cdot g_t^2 \quad \text{(velocity — adapts step size)}$$
    $$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

    Because $v_t$ decays, the learning rate doesn't die. Adam keeps learning where Adagrad gives up.

    **The family tree:** SGD → Momentum (Polyak, 1964) → Adagrad (2011) → RMSprop (Hinton, unpublished lecture, 2012) → Adam (2014). Each step fixed a limitation of the previous one.
    """)
    return


@app.cell
def _(adam_loss_history, adagrad_loss_hist, mo, plt):
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    _ai, _al = zip(*adagrad_loss_hist)
    ax2.plot(_ai, _al, color='#FF9800', linewidth=2, label='Adagrad (lr=0.1)')
    _ji, _jl = zip(*adam_loss_history)
    ax2.plot(_ji, _jl, color='#4CAF50', linewidth=2, label='Adam (lr=1e-3, cosine)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss per Character')
    ax2.set_title('Adagrad vs Adam: Same RNN, Same Data, Same 10 Epochs')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.output.replace(mo.as_html(fig2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Adagrad (orange) drops fast then **flatlines**. Adam (green) keeps improving throughout.

    ### Side-by-Side: Generated Text After 10 Epochs
    """)
    return


@app.cell
def _(adam_snapshots, adagrad_snaps, mo):
    mo.md(f"""
**Adagrad** (10 epochs):
```
{adagrad_snaps.get(10, '(training still running)')}
```

**Adam** (10 epochs):
```
{adam_snapshots[10]['text']}
```

Adam produces noticeably more coherent text — more real words, better structure — because it kept learning for the full 10 epochs while Adagrad effectively stopped after ~3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Generation

    Try different temperatures:
    - **Low (0.3)**: conservative, repetitive, but recognizable words
    - **Medium (0.8)**: balanced
    - **High (1.5)**: creative/chaotic
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
    print(f"Temperature = {temp_slider.value:.1f}\n")
    print(''.join(ix_to_char[ix] for ix in _ixs))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What We Learned

    1. **RNNs learn sequentially** — the hidden state carries information from character to character, allowing the network to learn patterns like common letter pairs, word boundaries, and even punctuation habits.

    2. **Training is iterative** — the model starts with random gibberish and gradually learns. After 1 epoch it knows spaces exist; after 10 it produces Shakespearean fragments.

    3. **The optimizer matters** — Adagrad's accumulator kills learning; Adam's exponential averages keep it alive. Adam is the default choice for modern deep learning.

    4. **Vanilla RNNs hit a ceiling** — the vanishing gradient problem limits how far back the hidden state can remember. This is exactly what **LSTMs** (1997) and **Transformers** (2017) were designed to fix.
    """)
    return


if __name__ == "__main__":
    app.run()
