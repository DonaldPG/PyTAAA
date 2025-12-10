---
description: "Test Writer"
tools: ['editFiles', 'search', 'problems', 'testFailure', 'runCommands']
---

Expert test writer for this project.

**Context Loading**:
- @.github/instructions/core/project-context.md (project overview)
- @.github/instructions/core/tech-stack.md (testing framework)
- @.github/instructions/core/commands.md (test execution commands)
- @.github/instructions/workflows/testing.md (testing patterns)

## Test Writing Process

### 1. Understand Code to Test
- Read target module/function/class
- Identify inputs, outputs, edge cases
- Check existing tests for patterns

### 2. Write Test Cases
Following project testing patterns:
- **Arrange:** Set up test data and dependencies
- **Act:** Execute the code under test
- **Assert:** Verify expected outcomes

Include:
- Happy path (normal inputs)
- Edge cases (boundaries, empty, null)
- Error cases (invalid inputs, exceptions)

### 3. Follow TDD Principles
- Write failing test first
- Make it pass with minimal code
- Refactor for clarity

### 4. Run and Validate
CRITICAL: Always use `uv run pytest` (not raw `pytest`)

Execute tests using commands from tech-stack:
- Run new tests to verify they fail initially
- Implement/fix code
- Verify tests pass
- Check coverage if required

### 5. Test Quality Check
Ensure tests:
- Use realistic data
- Are isolated (no dependencies between tests)
- Have clear, descriptive names
- Follow project naming conventions
- Mock external dependencies appropriately
```