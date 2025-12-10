---
description: Analyse an existing product and generate documentation for it.
tools: ['changes', 'codebase', 'editFiles', 'extensions', 'fetch', 'findTestFiles', 'githubRepo', 'problems', 'runCommands', 'runTasks', 'runTests', 'search', 'searchResults', 'terminalLastCommand', 'terminalSelection', 'testFailure', 'usages', 'vscodeAPI' ]
model: Claude Sonnet 4
---
Analyse the codebase and generate documentation for this project.

**Context Loading**:
- @.github/instructions/core/project-context.md (project overview)
- @.github/instructions/core/tech-stack.md (architecture and standards)
- @.github/instructions/workflows/coding.md (coding patterns)
- @.github/instructions/templates/analysis.md (analysis templates)

## Product Analysis Process

### 1. Analyze Existing Codebase
Examine:
- **Structure:** Directories, modules, file organization
- **Tech Stack:** Frameworks, libraries, dependencies (check package files)
- **Features:** Completed functionality, WIP, integrations
- **Patterns:** Coding style, naming conventions, architecture decisions

### 2. Gather Product Context
Ask user about:
- Product vision and target users
- Features not obvious from code
- Planned roadmap
- Important technical decisions to document
- Team coding standards

### 3. Execute plan-product Workflow
Use gathered information to run `plan-product` agent:
- Main idea from analysis + user input
- Key features (implemented + planned)
- Target users from context
- Tech stack detected from codebase

### 4. Customize Generated Documentation
Update files to reflect reality:
- **roadmap.md:** Add "Phase 0: Already Completed" with implemented features
- **tech-stack.md:** Verify versions match actual dependencies
- **decisions.md:** Add historical architectural decisions

### 5. Verification Summary
Report:
- Tech stack found
- Completed features count
- Code patterns detected
- Files created in `.agentic-docs/product/`
- Next steps for user
```