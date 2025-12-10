# Essential Commands

## Environment Management

**CRITICAL: Ensure the virtual environment is active for all Python operations**

### Package Management
```bash
uv add package-name              # Add dependency
uv remove package-name           # Remove dependency
uv sync                          # Sync from lock file
```

### Running Code
```bash
uv run python main.py            # Run main application
uv run pytest tests/             # Run tests
uv run ruff check .              # Linting
uv run ruff format .             # Formatting
```

**For other package managers:**
- **pip/venv:** Activate environment first (`source .venv/bin/activate`), then run commands without prefix
- **conda:** Activate environment (`conda activate env-name`), then run commands without prefix
- **poetry:** Use `poetry run` prefix (e.g., `poetry run python script.py`)

See `tech-stack.instructions.md` for your project's specific tool and commands.
pytest tests/                    # Run tests
```

### Using poetry
```bash
# Package management
poetry add package-name          # Add dependency
poetry remove package-name       # Remove dependency
poetry install                   # Install from lock file

# Execution - poetry automatically manages environment
poetry run python script.py      # Run scripts
poetry run pytest tests/         # Run tests
poetry run ruff check .          # Linting
```

## Testing

```bash
uv run pytest tests/                    # All tests
uv run pytest tests/test_module.py      # Specific module
uv run pytest tests/ -v                 # Verbose output
uv run pytest tests/ --cov=src          # With coverage
```

## Project-Specific Commands

```bash
uv run python main.py                   # Launch GUI application
uv run python -m src.monte_carlo        # Run Monte Carlo simulation
uv run python -m src.risk_analysis      # Run risk analysis
```