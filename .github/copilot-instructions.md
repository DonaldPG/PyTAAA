# GitHub Copilot Custom Instructions for Python Project


## Dependency Management
- This Python project uses **uv** to manage dependencies.
- Use "uv run python" to run python scripts
- Dependencies are recorded in `pyproject.toml`, not `requirements.txt`.
- Always follow this when generating or modifying dependency code.

## Testing
- Use **pytest** as the testing framework.
- Write clear and maintainable test functions.
- Use fixtures for setup and teardown.

## Commit Message Conventions
- Use conventional commit prefixes:
  - `feat:` for features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `style:` for formatting and whitespace only
  - `refactor:` for code restructuring without behavior change
  - `test:` for tests
  - `chore:` for miscellaneous tasks
- Format: `<type>(optional scope): <description>`

## Python Code Style
- Follow PEP 8 guidelines.
- Use 4 spaces for indentation; no tabs.
- Use snake_case for functions and variables.
- Use CamelCase for classes.
- Constants in UPPER_SNAKE_CASE.
- Max line length: 79 characters for code, 72 for comments.
- Two blank lines before top-level functions/classes.
- Use double quotes for strings.
- Add meaningful docstrings following PEP 257.
- Use type annotations.
- Use descriptive names.
- Keep functions focused and simple.

## Logging
- Use the standard `logging` module for all logging purposes.
- Configure loggers with appropriate log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- Use print statements for tracking progress and showing results as computations progress
- Include contextual information in log messages to aid debugging.
- Follow consistent log message formatting.
- Follow consistent print statement formatting.

## Comment Style
- Comments must be complete sentences starting with a capital letter.
- Limit comment lines to 72 characters.
- Use block comments indented to code level, starting with `#############`.
- Separate paragraphs in block comments with a single `#` line.
- Use inline comments for natural sub-tasks, separated by two spaces, starting with `# `.
- Use obvious comments sparingly; explain *why* not *what*.
- Write comments in English.
- Keep comments up to date.
- **Ensure comments are updated promptly to reflect any code changes.**

## Environment Setup
- Always set the `PYTHONPATH` to the root of the project directory for testing and development.
- Example: `export PYTHONPATH=$(pwd)` (for macOS/Linux) or `set PYTHONPATH=%cd%` (for Windows).
- This ensures that all modules in the project are correctly recognized.

## Copilot Session Documentation
- At the end of significant Copilot sessions, create a summary document in `docs/copilot_sessions/`.
- Use the naming format: `YYYY-MM-DD_brief-description.md` (e.g., `2025-10-04_daily-abacus-update-fix.md`).
- Include the following sections in session summaries:
  - **Date and Context**: When the session occurred and what prompted it
  - **Problem Statement**: What issue or task was being addressed
  - **Solution Overview**: High-level description of the solution implemented
  - **Key Changes**: List of files modified and what changed
  - **Technical Details**: Important implementation details for future reference
  - **Testing**: How the solution was verified
  - **Follow-up Items**: Any remaining tasks or considerations
- Keep summaries concise but informative for future reference.
- Session summaries help maintain project knowledge and facilitate onboarding.
