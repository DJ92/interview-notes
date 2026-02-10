# Transformer Architecture: From Attention to GPT

> "Attention is all you need" - but understanding *why* requires going deeper.

This post explores transformer architecture from first principles, covering the core mechanisms that power modern LLMs like GPT, BERT, and Claude.

## Why Transformers?

Before transformers (2017), sequence models relied on RNNs and LSTMs:
- **Sequential processing**: Can't parallelize, slow to train
- **Limited context**: Gradient vanishing limits long-range dependencies
- **Fixed memory**: Hidden state has bounded capacity

Transformers solved these with **self-attention**: every token directly attends to every other token in O(1) steps (though O(n²) memory).

## Core Mechanism: Scaled Dot-Product Attention

Attention is fundamentally a **soft dictionary lookup**:

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: Queries [batch, seq_len, d_k]
        K: Keys    [batch, seq_len, d_k]
        V: Values  [batch, seq_len, d_v]
    Returns:
        output: [batch, seq_len, d_v]
    """
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

### Why Scaling by √d_k?

The scaling factor `1/√d_k` is critical:
- Dot products grow with dimensionality: `Q·K ∼ O(d_k)`
- Large logits → extreme softmax outputs → vanishing gradients
- Scaling normalizes variance to ~1, keeping gradients healthy

**Without scaling**: 99% attention weight on one token, learning collapses.

## Multi-Head Attention: Multiple Perspectives

Multi-head attention runs attention in parallel with different learned projections:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # Learned projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention on each head
        x, attn = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.W_o(x)
```

**Why multiple heads?**
- Different heads learn different patterns (syntax, semantics, long-range)
- Empirically: 8-16 heads work best for most models
- Each head has smaller dimension (d_k = d_model / num_heads) → same total params

## Positional Encodings: Teaching Position

Self-attention is **permutation invariant** - token order doesn't matter! We need to inject position information.

### Three Approaches

**1. Sinusoidal Positional Encoding (Original Transformer)**
```python
def sinusoidal_positional_encoding(seq_len, d_model):
    """Fixed sinusoidal patterns for each position"""
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```
- **Pros**: No learned params, can extrapolate to longer sequences
- **Cons**: Fixed, not adaptive to data

**2. Learned Positional Embeddings (GPT)**
```python
self.position_embeddings = nn.Embedding(max_seq_len, d_model)
```
- **Pros**: Adapts to data patterns
- **Cons**: Can't extrapolate beyond `max_seq_len`

**3. Rotary Positional Embeddings (RoPE) - Modern LLMs**
Used in LLaMA, GPT-NeoX, PaLM:
```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotation to Q, K based on position"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
- **Pros**: Relative position encoding, great extrapolation, used in SOTA models
- **Cons**: Slightly more complex

## Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization (Pre-LN is more stable)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN: Normalize BEFORE attention
        attn_output = self.attention(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + self.dropout(attn_output)

        # Pre-LN: Normalize BEFORE FFN
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)

        return x
```

**Pre-LN vs Post-LN**:
- **Post-LN** (original): Normalize after residual connection → training instability
- **Pre-LN** (modern): Normalize before sub-layer → much more stable, used in GPT-3+

## GPT Architecture: Decoder-Only

Modern LLMs like GPT use **decoder-only** transformers:

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(input_ids.size(1)))
        x = token_emb + pos_emb

        # Causal mask (prevent attending to future tokens)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=causal_mask)

        # Final layer norm + projection to vocab
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
```

## Autoregressive Generation

```python
def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50):
    """Generate text token by token"""
    for _ in range(max_new_tokens):
        # Get logits for next token
        logits = self.forward(input_ids)
        next_token_logits = logits[:, -1, :] / temperature

        # Top-k sampling
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')

        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
```

## Architecture Variants

| Model | Architecture | Use Case |
|-------|--------------|----------|
| **GPT** | Decoder-only | Text generation, completion |
| **BERT** | Encoder-only | Classification, embeddings |
| **T5** | Encoder-Decoder | Translation, summarization |
| **LLaMA** | Decoder-only + RoPE | Modern LLMs, chat |

## Complexity Analysis

**Time Complexity**: O(n² · d) per layer
- n² from attention matrix (every token attends to every token)
- d from embedding dimension

**Space Complexity**: O(n²) for attention weights

**Why this matters**:
- GPT-3: 2048 context → 4M attention scores per layer
- GPT-4: 32k context → 1B attention scores per layer!

**Solutions**:
1. **Sparse Attention** (Longformer): Only attend to local + global tokens
2. **Linear Attention** (Performer): Approximate attention in O(n)
3. **FlashAttention**: Optimized CUDA kernels, 2-4× faster

## Key Takeaways

1. **Attention as soft lookup**: Q queries K to retrieve V
2. **Scaling matters**: `1/√d_k` prevents gradient vanishing
3. **Multi-head = multiple perspectives**: Learn different patterns
4. **Position encoding is critical**: Self-attention has no notion of order
5. **Pre-LN is more stable**: Modern models normalize before sub-layers
6. **Causal masking enables autoregression**: GPT's secret to generation
7. **O(n²) is the bottleneck**: Long context is expensive

## Production Considerations

**Training**:
- Use mixed precision (FP16) for 2× speedup
- Gradient checkpointing for memory efficiency
- Pre-LN for stable training at scale

**Inference**:
- KV caching: Don't recompute attention for previous tokens
- Batch processing: Amortize fixed costs
- FlashAttention for 2-4× speedup

**Context Length**:
- 2k tokens: ~4MB attention weights
- 32k tokens: ~1GB attention weights
- Choose wisely based on use case!

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [GitHub: Transformer Architecture Implementation](https://github.com/DJ92/ai-research-portfolio/tree/main/09-transformer-architecture)

---

*Part of my [AI Research Portfolio](https://github.com/DJ92/ai-research-portfolio) - building transformers from scratch to understand LLM internals.*
