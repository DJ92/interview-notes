# Engineering Notes

> "The best systems are boring."

Welcome to my personal knowledge base. This site documents **production-grade patterns**, **trade-offs**, and **system designs** for building scalable software and AI systems.

The goal is to move beyond surface-level tutorials and codify the "Context-as-Code" required to steer autonomous agents.

## 🧠 Agentic AI & SDLC

Standardizing how we build software with AI agents.

- **[AI SDLC](notes/ai-sdlc.md)**: A framework for **Context-as-Code** (`.cursor/rules`, Skills, PRDs) and **MCP** integration to build consistent, high-quality software with agents.

## 🔬 LLM Foundations & Research

Deep dives into transformer architectures, training, and alignment techniques.

- **[Transformer Architecture](notes/transformer-architecture.md)**: From attention mechanisms to GPT - understanding self-attention, positional encodings, and modern variants (RoPE).
- **[LLM Pre-training](notes/llm-pretraining.md)**: CLM vs MLM objectives, training dynamics, warmup schedules, and scaling laws.
- **[Alignment: RLHF vs DPO](notes/alignment-rlhf-dpo.md)**: How LLMs learn to be helpful - reward modeling, PPO, and direct preference optimization.
- **[Parameter-Efficient Fine-Tuning](notes/peft-lora-quantization.md)**: LoRA, quantization (4-bit/8-bit), and QLoRA for efficient fine-tuning.
- **[LLM Evaluation](notes/llm-evaluation.md)**: Beyond BLEU scores - automated metrics, LLM-as-judge, and production patterns.

## ⚙️ Machine Learning Systems

Productionalizing ML models is harder than training them.

- **[MLflow on Cloud Run](notes/mlflow-cloud-run.md)**: Serverless architecture for ML metadata and model registry.
- **[ONNX Java Serving](notes/onnx-java-serving.md)**: High-performance CPU & GPU inference in Java environments.

## 🛠️ Developer Experience

Tools and patterns for efficiency.

- **[Cloud Cheatsheet](notes/cloud-cheatsheet.md)**: CLI references for AWS/GCP/Azure.
- **[Zsh Setup](notes/zsh-setup.md)**: A blazingly fast terminal environment (Powerlevel10k, SDKMAN).

---

### [About Me](about.md)
Maintained by **Dheeraj Joshi**, a Staff Systems & Machine Learning Engineer focused on large-scale personalization and agentic AI systems.
