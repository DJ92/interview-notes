# Alignment Methods: RLHF vs DPO

> How do we teach LLMs to be helpful, harmless, and honest? Two approaches dominate: RLHF and DPO.

Post-training alignment is what transforms a base LLM into a useful assistant. This post explores the two main techniques and their trade-offs.

## The Alignment Problem

**Base LLM** (after pre-training):
- Completes text based on internet training data
- No notion of "helpful" vs "harmful"
- Might continue toxic prompts, refuse reasonable requests

**Aligned LLM** (after post-training):
- Follows instructions reliably
- Refuses harmful requests
- Provides helpful, safe responses

**How do we get there?** Enter post-training alignment.

## Three-Stage Pipeline

Modern LLMs follow a three-stage process:

```
1. Pre-training (CLM/MLM)
   ↓
2. Supervised Fine-Tuning (SFT)
   ↓
3. Preference Alignment (RLHF or DPO)
```

Let's explore each stage.

## Stage 1: Supervised Fine-Tuning (SFT)

**Goal**: Teach the model to follow instructions.

**Data**: Human-written (prompt, response) pairs:
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris."
}
```

**Training**: Standard next-token prediction, but only on the response:

```python
class SupervisedFineTuner:
    def compute_loss(self, model, input_ids, labels):
        """
        Only compute loss on response tokens (ignore prompt)

        Input:  [PROMPT] What is the capital of France? [RESPONSE] Paris.
        Labels: [-100, -100, -100, -100, -100, -100, -100, Paris, .]
        """
        logits = model(input_ids)

        # Mask prompt tokens with -100 (ignore_index)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        return loss
```

**Why mask the prompt?**
- We already know the prompt (given by user)
- We want to learn **responses**, not repeat prompts

**Limitation**: SFT teaches *what* to say, but not how to rank responses by quality.

## Stage 2: Preference Alignment

After SFT, we have a model that can follow instructions. But how do we make it **better**?

**Key Insight**: It's easier to compare two responses than write a perfect one.

**Human Preference Data**:
```json
{
  "prompt": "Explain photosynthesis",
  "chosen": "Photosynthesis is the process where plants convert light...",
  "rejected": "idk google it"
}
```

Two approaches dominate: **RLHF** and **DPO**.

---

## RLHF: Reinforcement Learning from Human Feedback

**Used by**: GPT-4, Claude, early ChatGPT

RLHF has three steps:

### Step 1: Train a Reward Model

Build a model that predicts human preferences:

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # Frozen SFT model
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids, attention_mask):
        """
        Returns: scalar reward for the entire sequence
        """
        # Get hidden states from base model
        hidden_states = self.base_model(input_ids, attention_mask)

        # Get last token's hidden state
        last_token_idx = attention_mask.sum(dim=1) - 1
        last_hidden = hidden_states[range(len(input_ids)), last_token_idx]

        # Project to scalar reward
        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward
```

**Training Objective** (Bradley-Terry Model):
```python
def compute_reward_loss(self, chosen_ids, rejected_ids):
    """
    Train reward model to prefer chosen over rejected

    Loss = -log σ(r_chosen - r_rejected)
    """
    r_chosen = self.reward_model(chosen_ids)
    r_rejected = self.reward_model(rejected_ids)

    # Reward model should predict: r_chosen > r_rejected
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss
```

**Bradley-Terry Model**: `P(chosen > rejected) = σ(r_chosen - r_rejected)`

### Step 2: Optimize Policy with PPO

Use the reward model to fine-tune the SFT model:

```python
class RLHFTrainer:
    def __init__(self, policy_model, reward_model, ref_model, beta=0.01):
        self.policy = policy_model      # Model being trained
        self.reward_model = reward_model  # Frozen reward predictor
        self.ref_model = ref_model      # Frozen SFT model (reference)
        self.beta = beta                # KL penalty coefficient

    def compute_rl_loss(self, prompts):
        """
        PPO objective with KL penalty

        reward = r(x, y) - β * KL(π_θ || π_ref)
        """
        # Generate responses from current policy
        responses = self.policy.generate(prompts)

        # Get reward from reward model
        rewards = self.reward_model(prompts + responses)

        # Compute KL divergence from reference model
        policy_logprobs = self.policy.log_prob(prompts, responses)
        ref_logprobs = self.ref_model.log_prob(prompts, responses)
        kl_penalty = (policy_logprobs - ref_logprobs).mean()

        # Combined objective
        rl_loss = -(rewards - self.beta * kl_penalty).mean()

        return rl_loss
```

**Why KL penalty?**
- Prevents policy from drifting too far from SFT model
- Without it: model might exploit reward model bugs, generate nonsense
- β controls exploration vs exploitation

### Step 3: Iterate

Collect more preference data from the updated policy, retrain reward model, repeat.

---

### RLHF Challenges

❌ **Complexity**: Three models (policy, reward, reference), RL training is tricky
❌ **Instability**: PPO is sensitive to hyperparameters
❌ **Reward Hacking**: Policy might exploit reward model weaknesses
❌ **Slow**: Requires sampling from policy during training
❌ **Memory**: Need to keep 3 models in memory

---

## DPO: Direct Preference Optimization

**Used by**: LLaMA 2, Mistral 7B, modern open-source models

**Key Insight**: We can skip the reward model and optimize preferences directly!

### The DPO Trick

RLHF objective:
```
max E[r(x,y)] - β * KL(π_θ || π_ref)
```

DPO insight: This has a **closed-form optimal solution**:
```
π*(y|x) ∝ π_ref(y|x) * exp(r(x,y) / β)

Rearranging:
r(x,y) = β * log(π*(y|x) / π_ref(y|x))
```

**Implication**: Reward is implicitly defined by the policy ratio!

### DPO Loss

```python
class DirectPreferenceOptimization:
    def __init__(self, policy_model, ref_model, beta=0.1):
        self.policy = policy_model
        self.ref_model = ref_model  # Frozen SFT model
        self.beta = beta

    def compute_loss(self, prompts, chosen, rejected):
        """
        DPO Loss: Maximize likelihood ratio for chosen > rejected

        loss = -E[log σ(β * (log π_θ(chosen) / π_ref(chosen)
                           - log π_θ(rejected) / π_ref(rejected)))]
        """
        # Compute log probs for policy
        policy_chosen_logps = self.policy.log_prob(prompts, chosen)
        policy_rejected_logps = self.policy.log_prob(prompts, rejected)

        # Compute log probs for reference
        with torch.no_grad():
            ref_chosen_logps = self.ref_model.log_prob(prompts, chosen)
            ref_rejected_logps = self.ref_model.log_prob(prompts, rejected)

        # Compute log ratios (implicit rewards)
        chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

        # DPO loss
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -F.logsigmoid(logits).mean()

        return loss
```

### DPO Intuition

**What's happening?**
1. Increase probability of chosen responses
2. Decrease probability of rejected responses
3. Stay close to reference model (implicit KL penalty via log ratio)

**No reward model needed!** The preference signal is baked directly into the loss.

---

## RLHF vs DPO: Head-to-Head

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Models Needed** | 3 (policy, reward, ref) | 2 (policy, ref) |
| **Training** | RL (PPO) | Supervised (standard backprop) |
| **Stability** | Tricky (RL instability) | Stable (classification-like) |
| **Memory** | High (3 models) | Medium (2 models) |
| **Speed** | Slow (sampling required) | Fast (offline data) |
| **Reward Hacking** | Possible | Less likely |
| **Hyperparameters** | Many (PPO params) | Few (β only) |
| **Implementation** | Complex | Simple |
| **Performance** | Slightly better (when tuned) | Competitive |

---

## Which Method to Use?

### Use RLHF if:
- You have extensive RL expertise
- You can afford the compute (3 models)
- You need iterative improvement (online learning)
- You're OpenAI/Anthropic with massive resources

### Use DPO if:
- You want simplicity and stability
- You have limited compute
- You're training open-source models
- You prefer standard supervised learning

**Modern trend**: DPO is winning for open-source models due to simplicity.

---

## Implementation Tips

### Data Quality Matters

**Good preference data**:
```json
{
  "prompt": "Write a poem about summer",
  "chosen": "Golden rays dance on waves so bright...",
  "rejected": "summer is hot lol"
}
```

**Bad preference data** (too similar):
```json
{
  "prompt": "What is 2+2?",
  "chosen": "2+2 equals 4.",
  "rejected": "2+2 is 4."
}
```

**Rule**: Chosen and rejected should be clearly different in quality.

### Hyperparameter Tuning

**β (KL penalty coefficient)**:
- **Too small** (β=0.001): Model drifts too far, generates nonsense
- **Too large** (β=1.0): Model barely changes, stays close to SFT
- **Sweet spot**: 0.01-0.1 for most tasks

**Learning rate**:
- **RLHF**: 1e-6 to 1e-5 (RL is sensitive)
- **DPO**: 1e-6 to 5e-6 (stable)

### Evaluation

Don't just trust loss! Evaluate with:

**1. Win Rate**:
```python
def compute_win_rate(model_a, model_b, test_prompts):
    """Human eval: How often does model_a beat model_b?"""
    wins = 0
    for prompt in test_prompts:
        response_a = model_a.generate(prompt)
        response_b = model_b.generate(prompt)

        # Ask human: which is better?
        if human_prefers(response_a, response_b):
            wins += 1

    return wins / len(test_prompts)
```

**2. Safety Benchmarks**:
- TruthfulQA (factuality)
- BBQ (bias)
- RealToxicityPrompts (toxicity)

**3. Capability Benchmarks**:
- MMLU (knowledge)
- HumanEval (coding)
- GSM8K (math)

---

## Advanced: Constitutional AI

**Anthropic's approach**: Use AI feedback instead of human feedback.

**Process**:
1. Generate responses
2. AI critiques based on "constitution" (principles)
3. AI revises responses
4. Train reward model on AI preferences

**Advantages**:
- Scalable (no human labeling bottleneck)
- Consistent (AI applies principles uniformly)
- Iterative (easy to update constitution)

**Implementation**:
```python
def constitutional_ai_revision(self, response, principle):
    """
    Critique and revise based on constitutional principle

    Principle: "Responses should be harmless and avoid stereotypes"
    """
    # Generate critique
    critique_prompt = f"""
    Response: {response}
    Principle: {principle}

    Does this response violate the principle? If so, how?
    """
    critique = self.model.generate(critique_prompt)

    # Generate revision
    revision_prompt = f"""
    Original: {response}
    Critique: {critique}
    Principle: {principle}

    Revise the response to align with the principle:
    """
    revision = self.model.generate(revision_prompt)

    return revision
```

See my [Constitutional AI project](https://github.com/DJ92/ai-research-portfolio/tree/main/06-constitutional-ai) for full implementation.

---

## Production Considerations

**Compute Costs**:
- **RLHF**: 2-3× more expensive than SFT (3 models + sampling)
- **DPO**: Similar to SFT (offline training)

**Data Requirements**:
- **Minimum**: 10k preference pairs
- **Good**: 100k preference pairs
- **SOTA**: 1M+ preference pairs (GPT-4)

**Iteration Speed**:
- SFT: 1-2 days
- RLHF: 1-2 weeks (reward model + PPO tuning)
- DPO: 2-3 days

---

## Key Takeaways

1. **Three stages**: Pre-training → SFT → Preference Alignment
2. **SFT**: Teach instruction following with (prompt, response) pairs
3. **RLHF**: Train reward model, optimize with PPO, complex but effective
4. **DPO**: Direct optimization, simpler and more stable
5. **KL penalty**: Prevents drift from SFT model
6. **β parameter**: Controls exploration vs exploitation
7. **Modern trend**: DPO winning for open-source due to simplicity
8. **Constitutional AI**: Scale alignment with AI feedback

---

## Further Reading

- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [GitHub: Post-Training Implementation](https://github.com/DJ92/ai-research-portfolio/tree/main/11-post-training-methods)
- [GitHub: Constitutional AI Implementation](https://github.com/DJ92/ai-research-portfolio/tree/main/06-constitutional-ai)

---

*Part of my [AI Research Portfolio](https://github.com/DJ92/ai-research-portfolio) - implementing alignment techniques from scratch to understand how LLMs become helpful assistants.*
