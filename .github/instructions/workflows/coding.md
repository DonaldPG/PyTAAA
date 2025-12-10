# Coding Standards

## Code Style

- **Python**: PEP8, 4-space indentation
- **Naming**: snake_case (functions), PascalCase (classes), UPPER_SNAKE_CASE (constants)
- **Type Hints**: Optional. Enable in `tech-stack.instructions.md` → "Code Style"
- **Imports**: Follow tool in `tech-stack.instructions.md` (ruff, isort, black)
- **Tone**: Professional, no emojis in code/docstrings

## Docstrings

**Format**: Python docstrings and markdown (required for public functions/classes)

```python
def process_data(file_path, mode="strict"):
    """
    Process data file with validation.

    Parameters
    ----------
    file_path : Path
        Path to input data file
    mode : str, optional
        Processing mode (default: 'strict')

    Returns
    -------
    ProcessingResult
        Validated data

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValidationError
        If data invalid
    """
```

## Comments

- **DO NOT**: Explain WHAT (code should be self-documenting)
- **DO**: Explain WHY (business logic, non-obvious decisions)

```python
# Good: WHY
# Backward compatibility for legacy format
data = process_legacy(file)

# Bad: WHAT (obvious)
# Loop through items
for item in items:
```

## Error Handling

```python
def process_data(file_path):
    """See tech-stack.instructions.md for project-specific exceptions."""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Not found: {file_path}")

        with open(file_path) as f:
            return validate(parse_data(f))

    except ValueError as e:
        logger.error(f"Invalid format: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

**Project exceptions**: Define in `tech-stack.instructions.md` → "Error Handling"
