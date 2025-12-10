```instructions
# AI Coding Instructions

**CRITICAL: Follow environment management patterns in `@.github/instructions/core/commands.md`**
(uv/poetry use `[tool] run` prefix, pip/conda require environment activation first)

## AI Response Guidelines

**Token Efficiency**:
- Minimize output tokens - prefer concise responses (1-4 lines when possible)
- No preamble or postamble ("Here is...", "I will...", "The answer is...")
- Direct answers without unnecessary elaboration
- One-word/short answers when appropriate

**Workflow**:
- Use search tools extensively before making changes
- Verify solutions with tests and linting
- Mark tasks complete immediately (don't batch)
- Never commit changes unless explicitly asked, although do suggest them when a new agent (e.g., `clean`) is selected
- Follow existing code conventions precisely
- Never assume library availability - check codebase first
- Mimic existing patterns, naming, and structure

## Context-Aware Loading System

This project uses **micro-instructions** to minimize context bloat. Load only what you need:

### Core Instructions (Always Load)
- **Tech Stack**: `@.github/instructions/tech-stack.instructions.md` - Project dependencies and standards

### Agents (Recommended)
All agents automatically load the right instructions for each task type:
- **research** - Explore solutions and technologies before planning
- **plan-product** - Plan features and roadmap
- **create-spec** - Generate technical specifications
- **execute-tasks** - Implement features with TDD
- **test** - Write comprehensive tests
- **analyse-product** - Analyze and document code
- **critic** - Review for bloat and unnecessary complexity
- **clean** - Refactor code applying Clean Code and SOLID principles
- **reprompt** - Refine prompts and improve instruction files
```