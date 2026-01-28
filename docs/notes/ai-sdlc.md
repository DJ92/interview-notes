# AI SDLC: Context-as-Code

A framework for steering AI agents (Cursor, Claude, Antigravity) to produce consistent, high-quality code by treating **context as code**.

## Core Concept
Instead of repeating instructions in every chat session, we commit **markdown configuration files** to the repository. These files define the rules, skills, and current state for the AI.

## 1. Rules (`.cursorrules` / `prompts`)
**Purpose**: Define global constraints, style guides, and tech stack preferences.

**Location**: `.cursorrules` (root) or `.github/copilot-instructions.md`.

**Example Content**:
- **Tech Stack**: "Use Java 21, Spring Boot 3.2, and Gradle (Kotlin DSL)."
- **Style**: "Prefer composition over inheritance. Use immutable data structures."
- **Behavior**: "Always cite filenames. Don't remove comments unless asked."

## 2. Skills (`.agent/skills/`)
**Purpose**: Teach the AI how to perform specific, complex tasks using a standard operating procedure (SOP).

**Format**: Markdown files (e.g., `migrate-to-helm.md`).

**Structure**:
```markdown
# Skill: Migrate Service to Helm
1. Create a new chart in `charts/`.
2. Copy values from `k8s/deployment.yaml`.
3. Run `helm lint`.
```

## 3. Notepads (`SCRATCHPAD.md`)
**Purpose**: Maintain state and memory across disjointed chat sessions.

**Usage**:
- The AI should read this file at the start of a task to understand the current context.
- The AI should update this file at the end of a task with progress.

**Structure**:
```markdown
# Current Task: OAUTH Implementation
- [x] Add dependency
- [ ] Implement TokenVerifier <-- CURRENT
- [ ] Write tests
```

## 4. Commands
**Purpose**: Aliases for complex agent workflows.
- `/doc`: "Read the code and generate Javadoc/PyDoc."
- `/test`: "Generate unit tests for the selected code."
- `/refactor`: "Apply 'Clean Code' principles to this function."

---

## Workflow

1. **Start**: Agent reads `.cursorrules` (Style) and `SCRATCHPAD.md` (Context).
2. **Execute**: Agent uses **Skills** to perform tasks.
3. **Commit**: Agent updates `SCRATCHPAD.md` with new state.
