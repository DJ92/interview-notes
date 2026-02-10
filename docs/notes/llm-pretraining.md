# LLM Pre-training: CLM vs MLM

> Understanding the objectives that teach language models to understand language.

Pre-training is where LLMs learn the statistical structure of language from massive text corpora. This post explores the two dominant approaches and their trade-offs.

## The Pre-training Problem

**Goal**: Learn general language representations from unlabeled text.

**Challenge**: How do you create a learning signal without labels?

**Solution**: Self-supervised learning - create labels from the data itself.

## Causal Language Modeling (CLM)

**Used by**: GPT, LLaMA, Claude, GPT-4

**Objective**: Predict the next token given all previous tokens.

```python
class CausalLanguageModeling:
    def compute_loss(self, model, input_ids, attention_mask):
        """
        CLM: Predict next token for each position

        Input:  "The cat sat on"
        Labels: "cat sat on the" (shifted by 1)
        """
        # Forward pass
        logits = model(input_ids, attention_mask)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # Remove last token
        shift_labels = input_ids[:, 1:].contiguous()   # Remove first token

        # Cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_token_id
        )

        return loss
```

### How CLM Works

Given text: `"The quick brown fox jumps"`

| Position | Input Context | Predict | Label |
|----------|---------------|---------|-------|
| 0 | [BOS] | → | The |
| 1 | [BOS] The | → | quick |
| 2 | [BOS] The quick | → | brown |
| 3 | [BOS] The quick brown | → | fox |
| 4 | [BOS] The quick brown fox | → | jumps |

Each position predicts the **next token** using all **previous tokens**.

### Causal Masking

The key trick: prevent the model from "cheating" by looking ahead:

```python
def create_causal_mask(seq_len):
    """
    Lower triangular matrix:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]

    Position i can only attend to positions <= i
    """
    return torch.tril(torch.ones(seq_len, seq_len))
```

**Why causal?** Because we only have access to past tokens during generation!

### Advantages of CLM

✅ **Natural generation**: Training objective matches inference (autoregressive)
✅ **Simple**: No masking strategy needed
✅ **Scalable**: Works well with billions of parameters
✅ **Long-form generation**: Learns to generate coherent long text

### Disadvantages

❌ **Unidirectional**: Can only look at left context, not right
❌ **Less efficient**: Only one prediction per token
❌ **Task mismatch**: Not ideal for classification (no bidirectional context)

## Masked Language Modeling (MLM)

**Used by**: BERT, RoBERTa, ALBERT

**Objective**: Predict masked tokens using bidirectional context.

```python
class MaskedLanguageModeling:
    def __init__(self, mask_token_id, vocab_size, mask_prob=0.15):
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

    def create_masked_input(self, input_ids):
        """
        MLM Masking Strategy (80/10/10):
        - 80% of time: Replace with [MASK]
        - 10% of time: Replace with random token
        - 10% of time: Keep unchanged

        Why? Prevents model from learning "only predict [MASK] tokens"
        """
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)

        # Select 15% of tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% → [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10% → random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        indices_random &= masked_indices & ~indices_replaced
        random_tokens = torch.randint(len(self.vocab), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_tokens[indices_random]

        # 10% → unchanged (already original)

        # Only compute loss on masked tokens
        labels[~masked_indices] = -100  # ignore_index

        return input_ids, labels
```

### How MLM Works

Original: `"The quick brown fox jumps"`

Masked:  `"The [MASK] brown [MASK] jumps"` (15% masked)

Predict: `quick` and `fox` using bidirectional context

### 80/10/10 Masking Strategy

**Why not always use [MASK]?**
- At fine-tuning/inference, there's no [MASK] token!
- Model would overfit to predicting when it sees [MASK]

**Solution**: Mix it up!
- 80%: Use [MASK] (main training signal)
- 10%: Use random token (robustness to noise)
- 10%: Keep original (distribute across all tokens)

### Example Implementation

```python
def compute_loss(self, model, input_ids, attention_mask):
    """
    MLM: Predict only masked tokens with bidirectional context
    """
    # Create masked input and labels
    masked_input_ids, labels = self.create_masked_input(input_ids)

    # Forward pass with bidirectional attention
    logits = model(masked_input_ids, attention_mask)

    # Loss only on masked positions
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100  # Don't compute loss on non-masked tokens
    )

    return loss
```

### Advantages of MLM

✅ **Bidirectional context**: See both left and right context
✅ **Better representations**: Learns richer embeddings
✅ **Efficient training**: Predict multiple tokens per sample
✅ **Great for classification**: Ideal for tasks needing bidirectional understanding

### Disadvantages

❌ **Train-test mismatch**: [MASK] token doesn't appear during fine-tuning
❌ **Not generative**: Can't naturally do autoregressive generation
❌ **Complex masking**: 80/10/10 strategy adds complexity
❌ **Pretrain-finetune gap**: Two-stage training can be suboptimal

## CLM vs MLM: Head-to-Head

| Aspect | CLM (GPT) | MLM (BERT) |
|--------|-----------|------------|
| **Context** | Unidirectional (left only) | Bidirectional (left + right) |
| **Training** | Next-token prediction | Masked token prediction |
| **Generation** | Natural (autoregressive) | Not designed for it |
| **Classification** | Requires prompting | Direct fine-tuning |
| **Embeddings** | Contextualized (left) | Contextualized (bidirectional) |
| **Use Cases** | Chat, completion, writing | Classification, QA, NER |
| **Modern Usage** | GPT-4, Claude, LLaMA | BERT variants declining |

## Training Dynamics

### Learning Rate Scheduling

Both CLM and MLM use **warmup + cosine decay**:

```python
class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def step(self, current_step):
        if current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale
```

**Why warmup?**
- Large learning rates at initialization → exploding gradients
- Warmup allows model to stabilize before aggressive updates

**Why cosine decay?**
- Smooth annealing to zero
- Better final performance than step decay

### Perplexity: The Key Metric

```python
def compute_perplexity(loss):
    """
    Perplexity = exp(cross-entropy loss)

    Interpretation: "How many choices on average?"
    - PPL of 10: Model is confused between ~10 tokens
    - PPL of 2: Model is highly confident
    """
    return torch.exp(loss)
```

**Good Perplexity Values**:
- Raw text (no training): ~10,000+ (totally random)
- Well-trained LLM: 10-30 (depends on domain)
- Overfitting: <5 (too confident, memorizing)

## Modern Hybrid Approaches

### Prefix LM (T5, UL2)
Combine CLM + MLM benefits:
```
Prefix:   [BOS] The quick brown  (bidirectional)
Targets:  fox jumps over         (causal)
```

### Span Corruption (T5)
Mask spans of tokens instead of individual tokens:
```
Original: The quick brown fox jumps
Masked:   The [X] fox [Y]
Targets:  [X] quick brown [Y] jumps
```

## Implementation Tips

### Memory-Efficient Training

```python
# Gradient accumulation for large batch sizes
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision (FP16)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():  # Compute in FP16
    loss = model(input_ids)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**: 2× faster, 2× less memory

## Scaling Laws

**Chinchilla Scaling Laws** (2022):
- Optimal model size : training tokens ≈ 1:20
- GPT-3 (175B params): Should train on 3.5T tokens
- Most models are undertrained!

**Implications**:
- Smaller models trained longer > larger models trained less
- Data quality > data quantity (garbage in = garbage out)

## Which Objective to Choose?

**Use CLM if**:
- Building a chatbot or writing assistant
- Need text generation capabilities
- Want simple, unified training

**Use MLM if**:
- Building a classifier or encoder
- Need strong embeddings
- Don't need generation

**Modern trend**: CLM dominates (GPT-4, Claude, LLaMA)
- Unified architecture (no separate encoder/decoder)
- Scales better to large models
- Can do both generation AND classification (with prompting)

## Key Takeaways

1. **CLM (GPT-style)**: Predict next token, unidirectional, great for generation
2. **MLM (BERT-style)**: Predict masked tokens, bidirectional, great for understanding
3. **80/10/10 masking**: Prevents overfitting to [MASK] token
4. **Warmup + cosine decay**: Standard LR schedule for stable training
5. **Perplexity**: exp(loss), measures model uncertainty
6. **Modern trend**: CLM winning for general-purpose LLMs
7. **Scale matters**: More data + compute = better models

## Production Considerations

**Training Costs**:
- GPT-3: ~$5M to train (estimated)
- LLaMA 65B: ~$3M on 1.4T tokens
- Use smaller models + longer training (Chinchilla optimal)

**Data Quality**:
- Deduplication: Remove duplicates (improves quality)
- Filtering: Remove toxic/low-quality content
- Mixing: Balance domains (code, web, books)

**Checkpointing**:
- Save checkpoints every N steps
- Resume from failures (cloud preemption)
- Evaluate intermediate checkpoints

## Further Reading

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [GitHub: Pre-training Implementation](https://github.com/DJ92/ai-research-portfolio/tree/main/10-llm-pre-training)

---

*Part of my [AI Research Portfolio](https://github.com/DJ92/ai-research-portfolio) - implementing pre-training from scratch to understand LLM foundations.*
