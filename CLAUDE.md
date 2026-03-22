# Deep Learning Library

Interactive marimo notebooks teaching deep learning from scratch, deployed to grad.emilycogsdill.com via Cloudflare Workers.

## Notebooks

- `notebooks/1_intro_to_gradient_descent/` — MLPs, backprop, gradient descent (Karpathy Zero to Hero lecture 1)
- `notebooks/2_recurrent_neural_networks/` — Character-level RNN, optimizer comparison

## Development

- `make build` — export all notebooks to HTML-WASM in `dist/`
- `make deploy` — build + deploy to Cloudflare Workers
- `.venv/bin/marimo edit <notebook.py>` — edit a notebook locally

## Rules

### Always verify notebooks by running them

After creating or modifying any marimo notebook, you MUST run `marimo export html <notebook.py> -o /dev/null` and verify it completes without errors. Check the output for `MarimoExceptionRaisedError` or any tracebacks. Do NOT deploy a notebook you haven't verified.

Common marimo pitfalls:
- Variables prefixed with `_` are private to their cell and NOT exported to downstream cells
- Cell function parameters must exactly match variable names returned by upstream cells
- `replace_all` edits can double-prefix variables (e.g. `_foo` → `bar_foo` → `bar_bar_foo`)

### WASM compatibility

Notebooks run in the browser via Pyodide. Only use numpy — no torch, no C extensions. Fetch external data with `pyodide.http.pyfetch` when `sys.platform == "emscripten"`.

### Training time budgets

Keep notebook training runs under ~2 minutes on native CPU. WASM is 3-5x slower. If you need more training, show pre-computed results as static text.
