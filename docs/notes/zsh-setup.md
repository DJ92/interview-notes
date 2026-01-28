# MacOS Dev Environment Setup

My local Zsh configuration using **Oh My Zsh**, **Powerlevel10k**, and various version managers.

## 1. Shell & Theme

- **Shell**: Zsh (Default on macOS)
- **Framework**: [Oh My Zsh](https://ohmyz.sh/)
- **Theme**: [Powerlevel10k](https://github.com/romkatv/powerlevel10k)
    - *Configuration*: `.p10k.zsh` (Run `p10k configure` to customize)

## 2. Plugins

Plugins configured in `.zshrc`:

| Plugin | Purpose |
| :--- | :--- |
| **`git`** | Aliases and functions for git (e.g., `gst` for `git status`). |
| **`you-should-use`** | Reminds you of existing aliases for commands you type. |
| **`zsh-autosuggestions`** | Suggests commands as you type based on history. |
| **`zsh-syntax-highlighting`** | Highlights commands in green (valid) or red (invalid). |
| **`zsh-bat`** | Integration with `bat` (a better `cat` with syntax highlighting). |
| **`nvm`** | Zsh plugin for NVM lazy loading. |

## 3. Version Managers

### Node.js (NVM)
- **Tool**: `nvm` (Node Version Manager)
- **Path**: `~/.nvm`
- **Usage**:
    ```bash
    nvm install --lts   # Install latest LTS
    nvm use <version>   # Switch version
    ```

### Java / Kotlin (SDKMAN)
- **Tool**: [SDKMAN!](https://sdkman.io/)
- **Path**: `~/.sdkman`
- **Usage**:
    ```bash
    sdk list java       # List available JDKs
    sdk install java 21.0.2-tem # Install Temurin JDK 21
    ```

### Python (Pyenv)
- **Tool**: `pyenv`
- **Path**: `~/.pyenv`
- **Usage**:
    ```bash
    pyenv install 3.11.0
    pyenv global 3.11.0
    ```

## 4. Custom Paths & Exports

- **Antigravity**: Added to path.
- **Hugging Face CLI**: Added to path.

```bash
# Example alias
alias zshconfig="nano ~/.zshrc"
alias ohmyzsh="nano ~/.oh-my-zsh"
```
