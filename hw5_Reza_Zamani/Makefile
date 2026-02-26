# Makefile — a collection of shortcut commands for common development tasks.
# Usage: run any target with `make <target>`, e.g. `make test` or `make build`.

# .PHONY tells Make that these targets are not actual files on disk.
# Without this, Make might get confused if a file with the same name exists.
.PHONY: test lint build run

# ─── test ───────────────────────────────────────────────────────────────────
# Run the project's test suite using pytest.
#   tests/   — the directory where all test files live
#   -v       — verbose output: shows each individual test name and its result
test:
	pytest tests/ -v

# ─── lint ────────────────────────────────────────────────────────────────────
# Run all code quality checks in sequence. Any failure will stop the chain.
lint:
	# black: auto-formats Python code to a consistent style.
	# --line-length=100 allows lines up to 100 characters before wrapping.
	black . --line-length=100

	# flake8: checks for style violations and potential errors (PEP 8).
	# --max-line-length=100 keeps flake8 in sync with black's line length.
	flake8 . --max-line-length=100

	# isort: automatically sorts and groups import statements.
	# --profile black ensures isort's style is compatible with black's output.
	isort . --profile black

	# mypy: static type checker — catches type errors without running the code.
	# --ignore-missing-imports suppresses errors for third-party libs without type stubs.
	mypy . --ignore-missing-imports

# ─── build ───────────────────────────────────────────────────────────────────
# Build a Docker image from the Dockerfile in the current directory.
#   -t fastapi-app  — tags (names) the resulting image "fastapi-app"
build:
	docker build -t fastapi-app .

# ─── run ─────────────────────────────────────────────────────────────────────
# Start a container from the built image in the background.
#   -d            — detached mode: runs the container in the background
#   -p 8000:8000  — maps port 8000 on your machine to port 8000 in the container
#                   so you can reach the app at http://localhost:8000
run:
	docker run -d -p 8000:8000 fastapi-app
