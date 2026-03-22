"""Generate dist/index.html linking to all exported notebooks."""

notebooks = [
    ("0-intro",          "0. Intro to Marimo",          "Introduction to the Marimo notebook environment"),
    ("1-derivatives",    "1. Derivatives",              "Graphing functions and visualizing derivatives"),
    ("2-manual-backprop","2. Manual Backpropagation",   "Manual backprop through a scalar expression"),
    ("3-autograd-engine","3. Autograd Engine",          "Automated backprop, expanding tanh, verifying against PyTorch"),
    ("4-neural-network", "4. Neural Network",           "Building Neuron, Layer, and MLP classes"),
    ("5-gradient-descent","5. Gradient Descent",        "Training loop with gradient descent"),
    ("demo",             "Demo",                        "Binary classifier on the moons dataset"),
    ("6-char-rnn",       "6. Character-Level RNN",      "Train an RNN from scratch to generate Shakespeare"),
]

items = "\n".join(
    f'    <li><a href="/{slug}/">{title}</a> — {desc}</li>'
    for slug, title, desc in notebooks
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>The Spelled-Out Intro to Neural Networks</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 680px; margin: 4rem auto; padding: 0 1.5rem; color: #1a1a1a; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
    p.subtitle {{ color: #555; margin-top: 0; }}
    ul {{ line-height: 2; padding-left: 1.25rem; }}
    a {{ color: #0066cc; }}
    footer {{ margin-top: 3rem; font-size: 0.85rem; color: #888; }}
  </style>
</head>
<body>
  <h1>The Spelled-Out Intro to Neural Networks</h1>
  <p class="subtitle">
    Following Andrej Karpathy's
    <a href="https://karpathy.ai/zero-to-hero.html">Zero to Hero</a> series —
    <a href="https://www.youtube.com/watch?v=VMj-3S1tku0">lecture 1</a>
  </p>
  <ul>
{items}
  </ul>
  <footer>
    Source: <a href="https://github.com/emilycogsdill/deep-learning-library">github.com/emilycogsdill/deep-learning-library</a>
  </footer>
</body>
</html>
"""

with open("dist/index.html", "w") as f:
    f.write(html)

print("Wrote dist/index.html")
