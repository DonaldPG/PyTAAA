---
description: Summarize completed work by creating a detailed progress report markdown file for future reference.
tools: ['editFiles', 'search', 'codebase', 'changes', 'usages']
model: Claude Sonnet 4
---

Summarize major coding work completed for this project.

**Context Loading**:
- @.github/instructions/core/project-context.md (project overview)
- @.github/instructions/core/tech-stack.md (technical context)
- Recent git changes or modified files (to understand what was accomplished)

## Summarize Work Mode Instructions

You are in summarize-work mode. Your primary task is to create a comprehensive markdown summary report after a major piece of coding has been completed. This report serves as documentation for future reference, allowing the work to be understood, repeated, or undone with minimal additional planning.

**Report Location**: Save the report as a new markdown file in `.github/progress-reports/` with a descriptive filename (e.g., `2025-12-10-backtest-refactoring-summary.md`).

**Report Structure**:

### 1. Header Information
- **Author**: [Your name or identifier, e.g., "GitHub Copilot assisted by Donald P."]
- **Date Range**: [Start date - End date of the work, e.g., "December 8-10, 2025"]
- **High-Level Summary**: 1-2 sentences describing the overall accomplishment (e.g., "Refactored the backtesting module to improve performance and maintainability by implementing new data structures and optimizing calculations.")

### 2. Major Steps (Maximum 6)
Present as a numbered list of the key phases or milestones:
1. **Step 1**: Brief description
2. **Step 2**: Brief description
... (up to 6)

Focus on high-level actions, not minute details. Use git commits, file changes, or logical groupings as guides.

### 3. Technical Implementation Details (For Future Reference)
This section is primarily for Copilot/AI assistants to understand the changes in a way that enables easy repetition or reversal. Include:
- **Files Modified**: List with brief change summaries (e.g., "functions/dailyBacktest_pctLong.py: Refactored calculation logic to use vectorized operations")
- **Key Functions/Classes Added/Modified**: Names and purposes
- **Dependencies/Integrations**: New libraries, APIs, or data sources introduced
- **Configuration Changes**: Updates to settings, environment, or build files
- **Testing Approach**: How changes were validated
- **Potential Reversal Steps**: High-level guidance on how to undo changes if needed (e.g., "Revert commits X, Y, Z; remove added functions; restore original configurations")
- **Repeatability Notes**: Prerequisites or context needed to replicate this work

**Guidelines**:
- Be concise but thorough - aim for 300-600 words total
- Use markdown formatting for readability (headers, lists, code blocks if needed)
- Base the summary on actual code changes, git history, and context provided
- If work spans multiple sessions, consolidate into coherent steps
- Ensure the report is self-contained for future developers or AI assistants

After creating the report, confirm its location and briefly note any follow-up recommendations.