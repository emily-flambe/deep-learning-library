.PHONY: dev deploy marimo

dev:
	.venv/bin/marimo edit notebooks/micrograd.py

marimo:
	.venv/bin/marimo edit notebooks/demo.py

deploy:
	.venv/bin/marimo export html-wasm notebooks/micrograd.py -o dist --mode edit --include-cloudflare
	CLOUDFLARE_ACCOUNT_ID=facf6619808dc039df729531bbb26d1d npx wrangler deploy
