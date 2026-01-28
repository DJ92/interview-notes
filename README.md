# GenAI + ML Systems & Engineering Notebook

This repository contains the knowledge, designs & source code for my personal knowledge base and engineering notebook.

**Live Site:** [https://dj92.github.io/interview-notes](https://dj92.github.io/interview-notes)

## About
This project is built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). It serves as a public repository of my thoughts on:
*   Machine Learning Systems
*   Applied ML
*   Generative AI
*   Engineering Tradeoffs

## Local Development

1.  **Clone the repo**
    ```bash
    git clone https://github.com/DJ92/interview-notes.git
    cd interview-notes
    ```

2.  **Install dependencies**
    ```bash
    pip install mkdocs-material
    ```

3.  **Run locally**
    ```bash
    mkdocs serve
    ```
    Open `http://localhost:8000`

## Deployment

Deployment is automated via GitHub Actions (or manually via `mkdocs gh-deploy`).

```bash
mkdocs gh-deploy
```
