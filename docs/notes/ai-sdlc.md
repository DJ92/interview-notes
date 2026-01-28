# AI SDLC: Context-as-Code

A framework for steering AI agents (Cursor, Claude, Antigravity) to produce consistent, high-quality code by treating **context as code**.

## Core Concept
Instead of repeating instructions in every chat session, we commit **markdown configuration files** to the repository. These files define the rules, skills, and current state for the AI.

## 1. Rules

Rules provide system-level instructions to the Agent. They can be scoped to specific projects or applied globally.

**Types**:
- **Project Rules** (`.cursor/rules/*.md`): Version-controlled, scoped by path patterns (globs). Best for project-specific conventions.
- **User Rules** (`~/.cursor/rules` / Dashboard): Global to your environment. Best for personal preferences.
- **Team Rules** (Dashboard): Enforced for all team members.

**Best Practices**:
- Keep rules focused and under 500 lines.
- Reference files instead of duplicating code.
- Avoid vague instructions; be concrete.

## 2. Skills (Tool Use)

Skills extend the agent's capabilities by providing it with "tools" (functions) it can call to interact with external systems or perform complex logic.

### **Concept**:
- **Tool Definitions**: JSON schemas defining the function signature (name, description, parameters).
- **Logic**: The actual code (e.g., Python script, API call) that executes when the tool is called.

### **Claude/MCP Integration**:
- Define tools using the **Model Context Protocol (MCP)** or Claude's native tool format.
- Agents can "call" these tools to fetch documentation, query databases, or run commands.
- **Example**: A "Migrate to Helm" skill might be a composite of file operations and `helm lint` commands exposed as a comprehensive tool or workflow.

## 3. Context: PRDs & RFCs

Instead of maintaining a scratchpad, provide context through structured documentation.

### **Usage**
- **PRDs (Product Requirement Docs)**: Pass the full PRD to the agent at the start of a feature.
- **RFCs (Request for Comments)**: Use technical design docs to align the agent on architectural decisions.

### **Workflow**
1. **Prepare**: Ensure your PRD or RFC is comprehensive and up-to-date.
2. **Inject**: `@mention` the relevant PRD/RFC file in the chat or include it in the context window.
3. **Instruct**: Ask the agent to "Implement the feature described in @prd.md" or "Review this code against @rfc.md".

## 4. Commands (`/command`)

Custom commands allow you to trigger reusable workflows with a simple `/` prefix in the chat.

**Locations**:
- **Project**: `.cursor/commands` (Version controlled).
- **Global**: `~/.cursor/commands` (Personal).
- **Team**: Managed in Cursor Dashboard (Shared).

**Usage**:
- `/doc`: "Read the code and generate Javadoc/PyDoc."
- `/test`: "Generate unit tests for the selected code."
- **With Parameters**: `/commit "fix login bug"` (Passes context to the command).

---

## 5. MCP Support

**Model Context Protocol (MCP)** provides a standard interface for agents to connect with external data and tools.

| Category | Supported Integrations |
| :--- | :--- |
| **Documentation** | **Google Docs**, **Google Drive** (Read specs, access loose notes). |
| **Code & Deploy** | **GitHub** (PRs, Issues, Actions), **Docker** (Container management). |
| **Project Management** | **JIRA**, **Linear** (Ticket tracking, sprint planning). |
| **Communication** | **Slack** (Team updates, incident alerts). |

### MCP Setup Config
*Add these to your MCP configuration file (e.g., `claude_desktop_config.json`).*

**Google Docs & Drive**:
```json
"google-drive": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-google-drive"]
}
```

**GitHub**:
```json
"github": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_TOKEN>"
  }
}
```

**Docker**:
```json
"docker": {
  "command": "docker",
  "args": ["run", "-i", "--rm", "mcp/docker"]
}
```

**JIRA**:
```json
"jira": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-jira"],
  "env": {
    "JIRA_API_TOKEN": "<YOUR_TOKEN>",
    "JIRA_EMAIL": "<YOUR_EMAIL>",
    "JIRA_INSTANCE_URL": "<YOUR_INSTANCE_URL>"
  }
}
```

**Linear**:
```json
"linear": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-linear"],
  "env": {
    "LINEAR_API_KEY": "<YOUR_KEY>"
  }
}
```

**Slack**:
```json
"slack": {
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-slack"],
  "env": {
    "SLACK_BOT_TOKEN": "<YOUR_TOKEN>",
    "SLACK_SIGNING_SECRET": "<YOUR_SECRET>"
  }
}
```

---

## Workflow

1. **Start**: Agent reads `.cursorrules` (Style) and consumes context from **PRDs/RFCs**.
2. **Execute**: Agent uses **Skills** and **MCP Tools** to perform tasks.
3. **Commit**: Agent validates changes against the PRD/RFC requirements.
