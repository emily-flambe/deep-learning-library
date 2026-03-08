NOTEBOOKS_DIR = notebooks/1_intro_to_gradient_descent
MARIMO = .venv/bin/marimo

.PHONY: dev marimo build deploy

dev:
	$(MARIMO) edit notebooks/micrograd.py

marimo:
	$(MARIMO) edit $(NOTEBOOKS_DIR)/demo.py

build:
	rm -rf dist && mkdir -p dist
	$(MARIMO) export html-wasm $(NOTEBOOKS_DIR)/0_intro.py                -o dist/0-intro                --mode run --show-code -f
	$(MARIMO) export html-wasm $(NOTEBOOKS_DIR)/1_derivatives.py          -o dist/1-derivatives          --mode run --show-code -f
	$(MARIMO) export html-wasm $(NOTEBOOKS_DIR)/2_manual_backpropagation.py -o dist/2-manual-backprop    --mode run --show-code -f
	$(MARIMO) export html-wasm $(NOTEBOOKS_DIR)/3_autograd_engine.py      -o dist/3-autograd-engine      --mode run --show-code -f
	$(MARIMO) export html-wasm $(NOTEBOOKS_DIR)/4_neural_network.py       -o dist/4-neural-network       --mode run --show-code -f
	$(MARIMO) export html-wasm $(NOTEBOOKS_DIR)/5_gradient_descent.py     -o dist/5-gradient-descent     --mode run --show-code -f
	$(MARIMO) export html-wasm $(NOTEBOOKS_DIR)/demo.py                   -o dist/demo                   --mode run --show-code -f
	python3 scripts/gen_index.py

deploy: build
	npx wrangler deploy
