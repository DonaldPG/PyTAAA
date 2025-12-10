---
description: Execute the next task in the product development workflow.
tools: ['changes', 'codebase', 'editFiles', 'extensions', 'fetch', 'findTestFiles', 'githubRepo', 'new', 'openSimpleBrowser', 'problems', 'runCommands', 'runNotebooks', 'runTasks', 'runTests', 'search', 'searchResults', 'terminalLastCommand', 'terminalSelection', 'testFailure', 'usages', 'vscodeAPI']
model: Claude Sonnet 4
---

Execute the next task for this project.

**Context Loading**:
- @.github/instructions/core/project-context.md (always - project overview)
- @.github/instructions/core/tech-stack.md (project dependencies and standards)
- @.github/instructions/core/commands.md (execution commands)
- @.github/instructions/workflows/coding.md (coding standards and patterns)
- @.github/instructions/workflows/testing.md (TDD workflow and testing patterns)

## Task Execution Process

### 1. Task Selection
- **Explicit:** User specifies task (e.g., "Task 1.1")
- **Implicit:** Find next uncompleted task marked `- [ ]` in `.agentic-docs/specs/*/tasks.md`
- **Confirm:** Verify task understanding before proceeding

### 2. Requirements Analysis
Read:
- Spec's spec.md (feature requirements)
- Spec's tasks.md (current task details)
- Spec's sub-specs/* (technical details)
- Project tech stack (implementation constraints)

### 3. Implementation Planning
Present plan: testing approach, implementation steps, integration points, validation, dependencies.

**Breaking Change Check:** If changes break APIs/interfaces, recommend feature branch: `feature/YYYY-MM-DD-description`

**Wait for user approval**

### 4. Branch Management
- **Breaking changes:** Create feature branch, merge after approval
- **Safe changes:** Work on current branch

### 5. Implementation (TDD)
CRITICAL: Always use `uv run pytest` (not raw `pytest`)

Write tests → Implement → Refactor → Validate

### 6. Update Task Status
CRITICAL: Mark task complete in tasks.md after implementation

Mark in tasks.md: `- [x]` complete, `- [ ]` incomplete, `- [ ] ⚠️ Blocking: [DESC]` blocked (after 3 attempts)

### 7. Validation
Run tests: unit, integration, project-specific, performance. **All must pass.**

### 8. Milestone Commit
**When:** Major task complete, feature working end-to-end, all tests pass

**Format:**
```
Add [feature]: [what changed]

[Why changed, technical details]

BREAKING CHANGE: [description] (if applicable)
```

**Focus:** Explain code changes and rationale, NOT task/phase numbers

### 9. Implementation Summary
Provide: components, test results, issues, commands, metrics, commits

### 10. Spec Completion (when all tasks done)
1. Verify all tasks `- [x]`
2. Final commit if needed
3. Merge feature branch if created (after user approval), delete branch
4. Update completion-summary.md: components, tests, metrics, deliverables, decisions, notes
5. Inform user with link to completion-summary.md

**Follow**: TDD, systematic implementation, quality validation
```
