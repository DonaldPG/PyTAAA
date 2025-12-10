---
description: 'Create technical specifications for features and components.'
tools: ['changes', 'codebase', 'editFiles', 'search', 'usages', 'problems', 'runCommands']
---

Create technical specifications for features in this project.

**Context Loading**:
- @.github/instructions/core/project-context.md (project overview)
- @.github/instructions/core/tech-stack.md (technical constraints)
- @.github/instructions/workflows/coding.md (implementation patterns)
- @.agentic-docs/product/mission.md (if exists - product context)
- @.agentic-docs/product/roadmap.md (if exists - feature priorities)

## Spec Creation Process

### 1. Gather Requirements
- If user asks "what's next?", check @.agentic-docs/product/roadmap.md for next uncompleted item
- Otherwise accept user's spec description in any format
- Clarify scope, technical approach, and requirements if needed

### 2. Create Spec Folder
- **Format:** `.agentic-docs/specs/YYYY-MM-DD-feature-name/`
- **Date:** Use current date in YYYY-MM-DD format
- **Name:** Use kebab-case, max 5 words, descriptive

### 3. Create spec.md
Required sections:
- **Overview:** 1-2 sentence goal
- **User Stories:** 1-3 stories with workflows
- **Spec Scope:** 1-5 features (numbered)
- **Out of Scope:** Excluded functionality
- **Expected Deliverable:** 1-3 testable outcomes
- **Planning Decisions:** Key decisions/rationale (optional)

### 4. Create Sub-Specs (only if complex/staged)
In `sub-specs/` folder:
- **technical-spec.md:** Requirements, approach, dependencies
- **tests.md:** Test coverage, mocking strategy

### 5. Create tasks.md
- **Structure:** 1-5 major tasks, subtasks as 1.1, 1.2
- **Format:** Checklist `- [ ]`
- **Pattern:** First = tests, last = verify tests pass
- **Order:** By dependencies, follow TDD

### 6. Create completion-summary.md
Empty template with sections: Implemented Components, Test Results, Quality Metrics, Deliverables, Implementation Decisions, Notes

### 7. Update Cross-References in spec.md
Link to: tasks.md, completion-summary.md, sub-specs/* (if created)

### 8. Request Review
Present files to user. **Suggest `critic` agent** for bloat review.

### 9. Ready for Execution
After approval, present first task summary and ask if user wants to proceed with implementation.
```