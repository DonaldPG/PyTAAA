---
description: "Setup Agentic Docs framework for a new or existing project"
tools: ['editFiles', 'search', 'runCommands', 'codebase']
---

Interactive setup to customize the Agentic Docs framework for your project.

**What this does:**
- Gathers information about your project and technical stack
- Customizes framework instruction files with your specific details
- Updates agents with your project's commands and tools
- Creates a setup spec for tracking and future reference

**You'll be asked about:**
- Project details (name, description, capabilities)
- Language and tooling (version, package manager, testing framework)
- Specific commands (run tests, run code, add dependencies)

**Context Loading**:
- @.github/instructions/core/project-context.md (template to customize)
- @.github/instructions/core/tech-stack.md (template to customize)
- @.github/instructions/core/commands.md (commands to customize)

## Interactive Setup Workflow

### 1. Gather Project Details (ask individually, show for review)
1. Project name?
2. Project description (1-2 sentences)?
3. Key capabilities (3-5 bullet points)?
4. Key libraries/dependencies (3-5)?
5. New project or existing codebase?

### 2. Gather Configuration (ask together)
6. Primary programming language?
7. Language version?
8. Package/build tool?
9. Testing framework?
10. Documentation format?
11. Development environment?
12. Project type?
13. Command to run tests?
14. Command to run code?
15. Command to add dependency?

### 3. Create Setup Spec
Create `.agentic-docs/specs/YYYY-MM-DD-framework-setup/` with:
- spec.md (configuration summary)
- tasks.md (6 customization tasks)
- completion-summary.md (template)

### 4. Execute Customizations
Use answers to customize files. Show proposed changes, ask user to review before applying.

**Task 1 - Update `.github/instructions/core/project-context.md`:**
- Fill project description, capabilities, architecture, data formats
- Uses answers: Q1-5

**Task 2 - Update `.github/instructions/core/tech-stack.md`:**
- Fill language version, package tool, testing framework, key libraries, deployment
- Uses answers: Q4, Q6-12

**Task 3 - Update `.github/instructions/core/commands.md`:**
- Replace command examples with user's actual commands
- Uses answers: Q13-15

**Task 4 - Update `.github/instructions/workflows/coding.md`:**
- Replace docstring format and language-specific patterns
- Uses answer: Q10

**Task 5 - Update `.github/agents/execute-tasks.agent.md` and `test.agent.md`:**
- Replace CRITICAL test command reminder
- Uses answer: Q13

**Task 6 - Create `.vscode/settings.json`:**
- Create VS Code settings file pointing to copilot-instructions.md
- Content:
  ```json
  {
      "github.copilot.chat.instructionFiles": [
          ".github/instructions/copilot-instructions.md"
      ]
  }
  ```

**Task 7 - Create `.gitignore`:**
- Create language-appropriate .gitignore
- Always include: `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`
- Python: Add `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.pytest_cache/`, `*.egg-info/`
- JavaScript/TypeScript: Add `node_modules/`, `dist/`, `build/`, `.npm/`, `.eslintcache`
- Rust: Add `target/`, `Cargo.lock` (for binaries)
- Go: Add `bin/`, `*.exe`, `*.test`
- Java: Add `target/`, `*.class`, `*.jar`
- Always add: `.agentic-docs/` as comment (users can uncomment if they don't want to track)

**Task 8 - Verify all placeholders replaced:**
- Search for remaining `[e.g.,`, `[Your`, `[PROJECT` placeholders
- Ask user how to fill any remaining ones

**Task 9 - Initialize Product Documentation (Recommended):**
Based on project type (Q5):

If new project:
- "Since this is a new project, I recommend creating product documentation (vision, users, roadmap) to guide development."
- Ask: "Would you like to use @plan-product agent now?"
  - Yes: "Great! @plan-product will ask about your product idea, target users, and key features. Ready?"
    - Confirmed: Hand off to @plan-product agent
    - Declined: Skip to completion
  - No: "You can run @plan-product anytime. It creates `.agentic-docs/product/` documentation."

If existing codebase:
- "For existing projects, you can either document product vision or analyze current codebase."
- Present options:
  - @plan-product: "Document product vision, roadmap, and architecture"
  - @analyse-product: "Analyze existing code and create technical documentation"
  - Skip: "Run these agents later as needed"

Mark each task [x] after user confirms changes (or makes choice for Task 9).

### 5. Complete Setup
- Generate completion summary with all configuration
- Show list of customized files
- If Task 9 completed → Show created `.agentic-docs/product/` files
- Inform framework ready

### 6. Final Suggestions
If product docs created:
- "Product documentation complete! Recommended next steps:"
  - "@create-spec - Plan your first feature"
  - "@critic - Review entire framework setup and product docs"
  - "Start implementing - You're ready to begin development"

If product docs skipped:
- "Setup complete! Recommended next steps:"
  - "@plan-product - Create product vision and roadmap"
  - "@create-spec - Plan a specific feature"
  - "@critic - Review framework setup"
```
````
