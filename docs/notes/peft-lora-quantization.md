# Parameter-Efficient Fine-Tuning: LoRA & Quantization

> Fine-tuning 70B models on consumer GPUs? Yes, with PEFT techniques.

Modern LLMs have billions of parameters. Full fine-tuning is expensive. This post explores techniques to fine-tune efficiently with minimal memory and compute.

## The Fine-Tuning Problem

**Full Fine-Tuning** (naive approach):
```python
# Update ALL 7 billion parameters
model = GPT2LMHeadModel.from_pretrained("gpt2")  # 7B params
model.train()
optimizer = Adam(model.parameters(), lr=1e-5)

for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradients for 7B params
    optimizer.step()  # Update 7B params
```

**Memory requirement**: ~28 GB just for model weights (4 bytes × 7B)
- Add gradients: +28 GB
- Add optimizer states (Adam): +56 GB
- **Total**: ~112 GB for 7B model

**Problem**: Most GPUs have 16-40 GB. Even A100 (80GB) struggles with large models.

**Solution**: Parameter-Efficient Fine-Tuning (PEFT)

---

## LoRA: Low-Rank Adaptation

**Used by**: Stable Diffusion, LLaMA adapters, open-source fine-tuning

**Key Insight**: Weight updates during fine-tuning are **low-rank**.

### The Math

Full fine-tuning updates:
```
W_new = W_0 + ΔW
```

Where `ΔW` is a full-rank update matrix.

**LoRA approximation**:
```
W_new = W_0 + ΔW
      ≈ W_0 + BA

where:
- B: (d × r)  "down-projection"
- A: (r × k)  "up-projection"
- r << min(d, k)  (typically r = 4-64)
```

**Parameters reduced**: `d×k → r×(d+k)`

Example for GPT-2:
- Original: 768 × 768 = 589,824 params
- LoRA (r=8): 8 × (768+768) = 12,288 params
- **Reduction**: 98% fewer parameters!

### Implementation

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()

        # Frozen pre-trained weight
        self.weight = nn.Parameter(pretrained_weight, requires_grad=False)
        self.bias = nn.Parameter(pretrained_bias, requires_grad=False) if has_bias else None

        # LoRA matrices (trainable)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Scaling factor
        self.scaling = alpha / rank

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # Start with identity

    def forward(self, x):
        # h = W_0 x + (α/r) BA x
        result = F.linear(x, self.weight, self.bias)  # Frozen weights
        lora_out = self.lora_B(self.lora_A(x))         # LoRA path
        result = result + self.scaling * lora_out      # Combine
        return result
```

**Key details**:
1. **W_0 frozen**: Original weights never updated (saves memory)
2. **B initialized to zero**: Start with identity (no change initially)
3. **Scaling α/r**: Keeps magnitude consistent across ranks

### Which Layers to Apply LoRA?

**Option 1: Query & Value only** (most common)
```python
# Apply LoRA to attention Q and V projections
model.transformer.h[i].attn.q_proj = LoRALinear(q_proj)
model.transformer.h[i].attn.v_proj = LoRALinear(v_proj)
```

**Option 2: All attention** (Q, K, V, O)
```python
# More parameters, better quality
for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
    model.transformer.h[i].attn[proj] = LoRALinear(...)
```

**Option 3: Attention + FFN** (full coverage)
```python
# Maximum quality, most parameters
# Apply to all linear layers in transformer
```

**Trade-off**: More LoRA layers = better quality but more params.

### Hyperparameter: Rank (r)

**Low rank** (r=1-4):
- ✅ Minimal parameters (~0.01% of model)
- ❌ May underfit, limited expressiveness

**Medium rank** (r=8-16):
- ✅ Sweet spot for most tasks
- ✅ ~0.1% of model parameters
- ✅ Good quality/efficiency trade-off

**High rank** (r=32-64):
- ✅ Maximum quality
- ❌ More parameters (~1% of model)
- ❌ Diminishing returns

**Rule of thumb**: Start with r=8, increase if quality lacking.

### Hyperparameter: Alpha (α)

**α controls the magnitude of LoRA updates**:

```python
scaling = alpha / rank
```

**Low alpha** (α = 1-8):
- ✅ Small, conservative updates
- ❌ May underfit

**Medium alpha** (α = 16-32):
- ✅ Standard choice (α = 2×r is common)
- ✅ Balanced updates

**High alpha** (α = 64+):
- ✅ Aggressive updates
- ❌ Risk of catastrophic forgetting

---

## Merging Weights for Inference

After training, merge LoRA weights back:

```python
def merge_lora_weights(model):
    """
    W_merged = W_0 + (α/r) BA

    Result: Single weight matrix, zero inference overhead
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Compute LoRA contribution
            lora_weight = module.lora_B.weight @ module.lora_A.weight
            lora_weight = lora_weight * module.scaling

            # Merge into base weight
            merged_weight = module.weight + lora_weight

            # Replace with standard linear layer
            new_linear = nn.Linear(module.in_features, module.out_features)
            new_linear.weight = nn.Parameter(merged_weight)
            new_linear.bias = module.bias

            # Replace in model
            parent = get_parent_module(model, name)
            setattr(parent, name.split('.')[-1], new_linear)

    return model
```

**Benefit**: No inference overhead! Merged model runs at full speed.

---

## Quantization: Compress Model Weights

**Goal**: Reduce memory footprint by using lower precision.

**Standard**: FP32 (32 bits = 4 bytes per parameter)
**Alternatives**: FP16 (2 bytes), INT8 (1 byte), INT4 (0.5 bytes)

### 8-bit Quantization

```python
def quantize_to_int8(tensor):
    """
    Map FP32 values to INT8 [-128, 127]

    Q = round(x / scale) + zero_point
    x ≈ (Q - zero_point) * scale
    """
    # Compute scale and zero point
    x_min, x_max = tensor.min(), tensor.max()
    scale = (x_max - x_min) / 255
    zero_point = round(-x_min / scale)

    # Quantize
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)

    return quantized, scale, zero_point

def dequantize(quantized, scale, zero_point):
    """Reconstruct FP32 values"""
    return (quantized.float() - zero_point) * scale
```

**Memory savings**: 4× reduction (4 bytes → 1 byte)

**Quality impact**: Minimal for well-trained models (~1% degradation)

### 4-bit Quantization

**Even more aggressive**:
```python
def quantize_to_int4(tensor):
    """
    Map FP32 to INT4 [-8, 7]

    Memory: 8× reduction!
    """
    x_min, x_max = tensor.min(), tensor.max()
    scale = (x_max - x_min) / 15  # 4-bit range: [-8, 7]
    zero_point = round(-x_min / scale)

    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, -8, 7)

    # Pack two 4-bit values into one byte
    # (Advanced: requires bit packing)
    return quantized, scale, zero_point
```

**Memory savings**: 8× reduction!

**Quality impact**: ~2-5% degradation (acceptable for many tasks)

### Quantized Linear Layer

```python
class QuantizedLinear(nn.Module):
    def __init__(self, weight, scale, zero_point):
        super().__init__()
        self.weight_quantized = nn.Parameter(weight, requires_grad=False)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        # Dequantize on-the-fly
        weight_fp = (self.weight_quantized.float() - self.zero_point) * self.scale

        # Standard matmul in FP32
        return F.linear(x, weight_fp)
```

**Trade-off**: Saves memory, but dequantization adds compute.

**Solution**: Use INT8 optimized kernels (bitsandbytes, llm.int8())

---

## Combining LoRA + Quantization: QLoRA

**The ultimate efficiency hack**: Quantize base model, add LoRA adapters.

```python
# 1. Load model in 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    load_in_4bit=True,           # 4-bit quantization
    device_map="auto"
)

# 2. Add LoRA adapters
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

# 3. Train!
trainer = Trainer(model=model, ...)
trainer.train()
```

**Memory savings**:
- 70B model in FP32: ~280 GB
- 70B model in 4-bit: ~35 GB
- LoRA adapters (r=16): ~100 MB
- **Total**: ~35 GB (fits on A100!)

**Result**: Fine-tune 70B models on single GPU! 🚀

---

## LoRA vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | LoRA (r=8) |
|--------|------------------|------------|
| **Trainable Params** | 100% | ~0.1% |
| **Memory (7B)** | ~112 GB | ~20 GB |
| **Training Speed** | Baseline | 1.5-2× faster |
| **Quality** | Highest | 95-99% of full |
| **Storage** | Full checkpoint | Tiny adapter |
| **Inference** | Standard | Merged = same |

---

## When to Use What?

### Use Full Fine-Tuning if:
- You have abundant compute (multiple A100s)
- You need absolute best quality
- You're making major domain shifts (e.g., medical LLM)

### Use LoRA if:
- Limited GPU memory (consumer GPUs)
- Need to train multiple adapters (multi-task)
- Want fast iteration (2× faster)
- Quality difference is acceptable (~2-5%)

### Use Quantization if:
- Inference memory is constrained
- You need to serve multiple models
- 1-5% quality loss is acceptable

### Use QLoRA if:
- You want to fine-tune huge models (70B+)
- You have limited hardware (single GPU)
- You're okay with ~5-10% quality loss

---

## Advanced: Multi-LoRA Serving

**Cool trick**: Serve multiple fine-tuned models with one base:

```python
# One base model (loaded once)
base_model = load_model("llama-2-7b")

# Multiple LoRA adapters (tiny!)
lora_customer_support = load_lora_adapter("customer-support.bin")  # 10 MB
lora_creative_writing = load_lora_adapter("creative-writing.bin")  # 10 MB
lora_coding = load_lora_adapter("coding.bin")                      # 10 MB

# Swap adapters dynamically
def serve_request(prompt, task):
    if task == "support":
        model.load_adapter(lora_customer_support)
    elif task == "creative":
        model.load_adapter(lora_creative_writing)
    elif task == "coding":
        model.load_adapter(lora_coding)

    return model.generate(prompt)
```

**Benefits**:
- One model in VRAM
- Swap adapters in milliseconds
- Serve 100s of specialized models!

---

## Implementation Tips

### LoRA Best Practices

**1. Choose layers wisely**:
```python
# Start minimal (Q, V only)
target_modules = ["q_proj", "v_proj"]

# Expand if quality lacking
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**2. Tune rank incrementally**:
```
r=4  → Too low? Try r=8
r=8  → Still lacking? Try r=16
r=16 → Sweet spot for most tasks
```

**3. Use α ≈ 2×r**:
```python
LoraConfig(r=8, lora_alpha=16)   # Common
LoraConfig(r=16, lora_alpha=32)  # Also good
```

### Quantization Best Practices

**1. Quantize after training** (QAT is complex):
```python
# Train in FP32/FP16
model.train()

# Quantize for inference
quantized_model = quantize(model, bits=8)
```

**2. Use symmetric quantization** (simpler, faster):
```python
# Symmetric: zero_point = 0
Q = round(x / scale)
```

**3. Calibrate on representative data**:
```python
# Pass calibration samples to determine scale
calibrate_quantization(model, calibration_dataloader)
```

---

## Measuring Efficiency

### Parameter Efficiency

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total:,}")
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

# Example output:
# Total params: 6,738,415,616
# Trainable: 8,388,608 (0.12%)  ← LoRA magic!
```

### Memory Footprint

```python
def get_model_size(model):
    """Estimate memory in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

# 7B model:
# FP32: ~28 GB
# FP16: ~14 GB
# INT8: ~7 GB
# INT4: ~3.5 GB
```

---

## Production Considerations

### Training Cost Reduction

**Full fine-tuning** (7B model):
- 4× A100 GPUs
- 2-3 days
- Cost: ~$1,000

**LoRA** (7B model):
- 1× A100 GPU
- 1 day
- Cost: ~$100

**Savings**: 10× cheaper!

### Deployment

**Multi-tenant serving**:
- Base model: 14 GB (FP16)
- LoRA adapter: 10 MB
- **100 adapters**: 15 GB total (vs 1.4 TB for 100 full models!)

---

## Key Takeaways

1. **LoRA**: Low-rank weight updates, 0.1% trainable params, 95%+ quality
2. **Rank (r)**: Start with 8, increase to 16 if needed
3. **Alpha**: Use α ≈ 2×r for balanced updates
4. **Quantization**: 8-bit (4× compression), 4-bit (8× compression)
5. **QLoRA**: Combine quantization + LoRA for extreme efficiency
6. **Weight merging**: Zero inference overhead after merging
7. **Multi-LoRA**: Serve 100s of models with one base
8. **Cost savings**: 10× cheaper training, 100× cheaper serving

---

## Further Reading

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers](https://arxiv.org/abs/2208.07339)
- [PEFT Library (Hugging Face)](https://github.com/huggingface/peft)
- [GitHub: Parameter-Efficient Fine-Tuning Implementation](https://github.com/DJ92/ai-research-portfolio/tree/main/12-parameter-efficient-tuning)

---

*Part of my [AI Research Portfolio](https://github.com/DJ92/ai-research-portfolio) - implementing PEFT techniques from scratch to understand efficient LLM fine-tuning.*
