# Testing Patterns

## Test-Driven Development (TDD)

**TDD Cycle**: Red (failing test) → Green (minimal code) → Refactor (clean up)

**When to write tests**:
- Before: New features (TDD)
- During: Edge cases discovered
- After: Bug fixes (reproduce first)

## Test Data Location

Specify in `tech-stack.instructions.md` under "Testing":
- Small datasets: `tests/test_data/` (in repo)
- Large datasets: `tests/fixtures/` (generated)
- Temporary: Use pytest `tmp_path` fixture

## Test Structure

```python
def test_feature():
    # Arrange
    data = create_test_data()
    # Act
    result = process(data)
    # Assert
    assert result.is_valid()
```

## Testing Strategies

- **File I/O**: Use `tmp_path`, mock for unit tests, real files for integration
- **External Dependencies**: Mock for unit tests, real for integration
- **Data Processing**: Test validation, formats, edge cases

## Mocking

```python
@pytest.fixture
def mock_data():
    return MockClass(field="value")

@patch('module.function')
def test_feature(mock_func):
    mock_func.return_value = expected_value
    # test implementation
```

**Test commands**: See `@.github/instructions/core/commands.md`

- **Exclusions**: Large test files excluded from Git via `.gitignore`