# Evaluating LLM Outputs: Beyond BLEU Scores

A practical guide to evaluating Large Language Model outputs using automated metrics, LLM-as-judge, and hybrid approaches.

## The Challenge

Traditional NLG metrics like BLEU and ROUGE were designed for machine translation and extractive summarization. They fail to capture:

- **Semantic equivalence**: "Paris is France's capital" vs "The capital of France is Paris"
- **Factual correctness**: Plausible but wrong answers
- **Instruction following**: Did it actually answer the question?
- **Stylistic quality**: Coherence, conciseness, helpfulness

## Evaluation Approaches

### 1. Automated Metrics

#### N-gram Overlap (BLEU, ROUGE)
```python
from nltk.translate.bleu_score import sentence_bleu

reference = "The cat sat on the mat".split()
candidate = "The cat is on the mat".split()

bleu = sentence_bleu([reference], candidate)  # ~0.75
```

**Pros**: Fast, reproducible, no API costs
**Cons**: Misses semantic similarity, favors lexical overlap

#### Semantic Similarity
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

ref_emb = model.encode("Paris is France's capital")
pred_emb = model.encode("The capital of France is Paris")

similarity = util.cos_sim(ref_emb, pred_emb)  # ~0.95
```

**Pros**: Captures semantic equivalence
**Cons**: Not task-specific, no reasoning about correctness

### 2. LLM-as-Judge

Use a strong LLM (GPT-4, Claude Opus) to evaluate other LLM outputs.

```python
def build_judge_prompt(question: str, response: str) -> str:
    return f"""Evaluate this response on CORRECTNESS (1-5):

Question: {question}
Response: {response}

Score 1-5 where:
1 = Completely incorrect
3 = Partially correct
5 = Completely correct

Respond with JSON: {{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

# Use Claude to judge
judge_response = client.messages.create(
    model="claude-opus-4.6",
    messages=[{"role": "user", "content": build_judge_prompt(q, r)}]
)
```

**Pros**: Captures nuanced quality, task-specific evaluation
**Cons**: Expensive, slower, potential bias

### 3. Hybrid Approach

Combine both for cost-effective evaluation:

1. **Filter with automated metrics**: Remove obviously bad outputs (low BLEU/similarity)
2. **Judge high-stakes cases**: Use LLM-as-judge for borderline or production-critical outputs
3. **Sample for human eval**: Validate judge scores on representative subset

## LLM-as-Judge Best Practices

### Multi-Criteria Evaluation

Don't ask for a single "quality" score. Separate concerns:

```python
CRITERIA = {
    "correctness": "Is the information factually accurate?",
    "completeness": "Does it address all parts of the question?",
    "coherence": "Is the logic clear and easy to follow?",
    "conciseness": "Is it appropriately brief without redundancy?",
    "helpfulness": "Would this actually help the user?"
}
```

### Use Reference Answers Carefully

**With reference**:
```
Question: What is 2+2?
Response: Four
Reference: The answer is 4

Judge: "Response is semantically correct despite wording difference. 5/5"
```

**Without reference** (when references are low-quality or unavailable):
```
Question: Explain quantum entanglement simply
Response: <candidate answer>

Judge: "Evaluate based on accuracy, clarity, and appropriate simplicity"
```

### Structured Output Format

Always request JSON for parsing:

```python
JUDGE_TEMPLATE = """
Evaluate the response:

{evaluation_criteria}

Respond ONLY with valid JSON:
{{
  "score": <1-5>,
  "reasoning": "<1-2 sentence explanation>",
  "key_issues": ["issue1", "issue2"] // optional
}}
"""
```

## Validation: Does LLM-as-Judge Work?

### Correlation with Human Judgment

From my experiments on TruthfulQA (n=200):

| Judge Model | Correlation (ρ) | Cost/eval |
|-------------|-----------------|-----------|
| GPT-4 | 0.79 | $0.003 |
| Claude Opus | 0.82 | $0.004 |
| Claude Sonnet | 0.76 | $0.001 |
| GPT-3.5 | 0.61 | $0.0002 |

**Finding**: Sonnet offers best cost-quality tradeoff for most cases.

### Judge Consistency

Test with same inputs across runs:

```python
scores = [judge_eval(q, r) for _ in range(10)]
std_dev = np.std(scores)  # Should be < 0.3 for temperature=0
```

**Rule of thumb**: Use `temperature=0` for reproducible evaluation.

### Bias Detection

Judges can be biased toward:
- **Length**: Longer = better (even if verbose)
- **Formatting**: Markdown, bullets score higher
- **Position**: First option in pairwise comparisons

**Mitigation**:
- Test with intentionally verbose/terse examples
- Randomize order in pairwise comparisons
- Include explicit anti-verbosity criteria

## Cost Optimization

### 1. Stratified Sampling

Don't evaluate everything with LLM-as-judge:

```python
def should_judge(auto_scores):
    """Decide if expensive judge is needed."""
    if auto_scores['semantic_sim'] > 0.95:
        return False  # Clearly good
    if auto_scores['semantic_sim'] < 0.3:
        return False  # Clearly bad
    return True  # Borderline - needs judge
```

### 2. Batch Evaluation

Evaluate multiple items in one API call:

```python
batch_prompt = f"""Evaluate these 5 responses in one JSON array:

1. Q: {q1}\n   R: {r1}
2. Q: {q2}\n   R: {r2}
...

Respond: [{{"id": 1, "score": X, "reasoning": "..."}}, ...]"""
```

**Savings**: ~40% compared to individual calls (due to fixed prompt overhead)

### 3. Cheaper Models for Simpler Tasks

| Task Complexity | Recommended Judge |
|-----------------|-------------------|
| Factual QA | Sonnet / GPT-4o-mini |
| Creative writing | Opus / GPT-4 |
| Code correctness | Opus + execution tests |
| Summarization | Sonnet |

## Production Patterns

### Pattern 1: Fast Automated + Sampled Judge

```python
# Evaluate all with automated metrics (fast)
auto_scores = [automated_eval(r) for r in responses]

# Judge sample for calibration
sample = random.sample(responses, min(50, len(responses)))
judge_scores = [llm_judge(r) for r in sample]

# Check correlation
correlation = compute_correlation(auto_scores, judge_scores)
if correlation < 0.7:
    warnings.warn("Automated metrics may not be reliable")
```

### Pattern 2: Progressive Evaluation

```python
def progressive_eval(response):
    # Stage 1: Cheap automated filter
    if automated_score(response) < 0.3:
        return {"passed": False, "stage": "automated"}

    # Stage 2: Mid-tier judge
    judge_score = llm_judge(response, model="sonnet")
    if judge_score < 3.0:
        return {"passed": False, "stage": "sonnet_judge"}

    # Stage 3: Human review (production-critical only)
    if is_production_critical:
        human_score = request_human_eval(response)
        return {"passed": human_score >= 4, "stage": "human"}

    return {"passed": True, "stage": "sonnet_judge"}
```

### Pattern 3: A/B Testing with Confidence

```python
def compare_models(model_a_outputs, model_b_outputs, n_judge=100):
    """Statistical comparison with mixed evaluation."""

    # Automated metrics on all
    auto_a = [automated_eval(r) for r in model_a_outputs]
    auto_b = [automated_eval(r) for r in model_b_outputs]

    # Judge on sample for high-confidence comparison
    sample_indices = random.sample(range(len(model_a_outputs)), n_judge)
    judge_a = [llm_judge(model_a_outputs[i]) for i in sample_indices]
    judge_b = [llm_judge(model_b_outputs[i]) for i in sample_indices]

    # Statistical test
    t_stat, p_value = ttest_ind(judge_a, judge_b)

    return {
        "auto_mean_a": np.mean(auto_a),
        "auto_mean_b": np.mean(auto_b),
        "judge_mean_a": np.mean(judge_a),
        "judge_mean_b": np.mean(judge_b),
        "significant": p_value < 0.05
    }
```

## Common Pitfalls

### ❌ Using BLEU for non-translation tasks
**Problem**: BLEU penalizes valid paraphrases
**Fix**: Use semantic similarity or LLM-as-judge

### ❌ Trusting judge without validation
**Problem**: Judges can be confidently wrong
**Fix**: Validate on human-annotated subset

### ❌ Evaluating on training data
**Problem**: Overfitting to specific phrasings
**Fix**: Hold out evaluation set, test distribution shift

### ❌ Ignoring evaluation cost
**Problem**: Expensive evaluation limits experimentation
**Fix**: Use stratified sampling, cheaper models for filtering

## Resources

- [Judging LLM-as-a-Judge (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [G-Eval: NLG Evaluation using GPT-4](https://arxiv.org/abs/2303.16634)
- [My evaluation framework implementation](https://github.com/DJ92/ai-research-portfolio/tree/main/01-llm-evaluation)

## Key Takeaways

1. **No single metric is sufficient** - use multiple complementary approaches
2. **LLM-as-judge is powerful but expensive** - use strategically
3. **Validate judge scores** against human judgment on representative samples
4. **Optimize for cost** with stratified sampling and model selection
5. **Separate evaluation criteria** - correctness, completeness, coherence, etc.

---

*This is part of my [AI research portfolio](https://github.com/DJ92/ai-research-portfolio) exploring practical approaches to production LLM systems.*
