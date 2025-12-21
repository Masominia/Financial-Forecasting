.PHONY: install rust test lint run data all

install:
	pip install -e ".[dev]"

rust:  ## build and install the Rust Monte Carlo engine (optional, NumPy fallback otherwise)
	maturin build --release -m rust/Cargo.toml -o dist && pip install --force-reinstall dist/*.whl

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts
	cd rust && cargo fmt --check && cargo clippy --release --all-targets -- -D warnings

test:
	pytest
	cd rust && cargo test --release

run:
	python -m plancast run

data:
	python scripts/fetch_data.py

all: lint test run
