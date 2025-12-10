---
description: Generate an implementation plan for new features or refactoring existing code.
tools: ['changes', 'codebase', 'extensions', 'fetch', 'findTestFiles', 'githubRepo', 'problems', 'search', 'searchResults', 'terminalLastCommand', 'terminalSelection', 'testFailure', 'usages', 'vscodeAPI']
model: Claude Sonnet 4
---
# Planning mode instructions
Plan new features for this project.

**Context Loading**:
- @.github/instructions/core/project-context.md (project overview)
- @.github/instructions/core/tech-stack.md (technical constraints)
- @.github/instructions/templates/planning.md (planning templates)
- @.agentic-docs/product/mission.md (if exists - product vision)
- @.agentic-docs/product/roadmap.md (if exists - current roadmap)

## Product Planning Process

### 1. Gather User Input
Ask for (if not provided):
- Main product idea
- Key features (minimum 3)
- Target users
- Tech stack preferences

Check `.github/instructions/tech-stack.instructions.md` for defaults.

### 2. Create Documentation Structure
Create `.agentic-docs/product/` with files:
- mission.md (product vision)
- tech-stack.md (technical architecture)
- roadmap.md (development phases)
- decisions.md (decision log)

### 3. Create mission.md
Sections: Pitch, Users, Problem, Differentiators, Key Features
- **Pitch:** 1-2 sentence elevator pitch
- **Users:** Customer segments and personas
- **Problem:** 2-4 problems with solutions
- **Differentiators:** 2-3 competitive advantages
- **Key Features:** 8-10 features grouped by category

### 4. Create tech-stack.md
Document: framework, storage, GUI, environment/dependency management, testing, CI/CD, environments, repository

Check user input, then config files for missing items.

### 5. Create roadmap.md
5 phases with 3-7 features each:
- Phase 1: Core MVP
- Phase 2: Differentiators
- Phase 3: Scale & polish
- Phase 4: Advanced features
- Phase 5: Enterprise features

Format: `- [ ] Feature - Description \`effort\``

### 6. Create decisions.md
Empty template for logging future decisions with format: Date, ID, Status, Category, Decision, Context, Consequences

### 7. Verification
Confirm all files created, provide next steps (use `create-spec` agent for first feature)
```