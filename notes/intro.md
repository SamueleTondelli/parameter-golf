# Parameter Golf — Baseline Architecture & Records Analysis

## To run
```bash
  RUN_ID=test ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 TTT_BATCH_SIZE=32 PYTORCH_ALLOC_CONF=expandable_segments:True python3 train_gpt.py
```

## Baseline Architecture (`train_gpt.py`)

### Overview

The baseline trains a compact causal language model under a **16MB compressed artifact** budget and a **10-minute wallclock** cap. It evaluates with **bits-per-byte (BPB)** — a tokenizer-agnostic compression metric — rather than standard perplexity, and includes **test-time training (TTT)** with per-document LoRA adapters.

### Model Shape (Defaults)

| Parameter | Value |
|---|---|
| Transformer blocks | 9 |
| Model width | 512 |
| Attention heads | 8 |
| KV heads (GQA) | 4 |
| MLP expansion | 2x |
| Vocab size | 1024 |
| Sequence length | 1024 |
| Embeddings | Tied input/output |

Small vocab and tied embeddings are deliberate: embeddings dominate the parameter budget in small models.

### What Makes It Unconventional

#### U-Net Skip Connections
The 9 blocks are split into an encoder half (first 4) and decoder half (last 5). The encoder stores intermediate activations; the decoder re-injects them in reverse order via learned `skip_weights`. All attention remains causal — it's a skip-connected symmetric stack, not a bidirectional encoder-decoder.

#### Residual Mixing with Original Embeddings
Each block has a learned `resid_mix` that blends the current hidden state with the original token embedding `x0`:
```python
x = mix[0] * x + mix[1] * x0
```
This gives every block a learned shortcut back to the original representation.

#### QK-RMSNorm + Per-Head Query Gain
Q and K are RMS-normalized before attention. A learned per-head `q_gain` scalar acts as a learned attention temperature. This stabilizes attention scores independently of projection magnitude.

#### ReLU² MLP
The MLP uses `ReLU(x)²` instead of GELU or SwiGLU — simpler, cheaper, and effective at this scale.

#### Zero-Initialized Output Projections
Attention output projections and MLP output projections start at zero, so blocks begin as near-identity and learn their residual contribution gradually.

#### Logit Softcap
All logits pass through `softcap * tanh(logits / softcap)` to prevent blowup, aid stability, and improve quantization robustness.

### Hybrid Optimizer

Parameters are split by type with different optimizers and learning rates:

| Parameter Type | Optimizer | Default LR |
|---|---|---|
| Token embedding | Adam | 0.6 (or 0.05 tied) |
| LM head (if untied) | Adam | 0.008 |
| Matrix params (transformer) | **Muon** | 0.04 |
| Scalars/vectors/control | Adam | 0.04 |

**Muon** orthogonalizes gradient matrices via Newton-Schulz iterations before applying updates. It's used only for 2D weight matrices; everything else uses Adam.

### Compile Priming (Not LR Warmup)

The "warmup" steps run forward/backward/optimizer to trigger `torch.compile` and prime CUDA kernels, then **rewind model and optimizer state** to their original values. No training occurs during warmup.

### BPB Evaluation

BPB (bits per byte) is used instead of token-level loss to fairly compare across different tokenizers:

```
BPB = (total_nll / ln(2)) / total_raw_text_bytes
```

Byte counts are estimated efficiently via precomputed SentencePiece lookup tables on GPU.

### Test-Time Training (TTT) with LoRA

At final evaluation, per-document low-rank adapters are applied:
- Fresh rank-8 LoRA on Q, V projections and LM head per document
- Independent weights per batch element (`BatchedLinearLoRA`)
- Score-before-update: chunk `i` is scored using adaptation from chunks `< i`
- Adapters reset between documents (no inter-document leakage)

### Post-Training Quantization

The final artifact uses int8 quantization + zlib compression:
- 2D tensors: per-row int8 with percentile clipping
- Vectors/scalars: per-tensor int8
- Small/control tensors: kept in fp16/fp32
- The script validates the **roundtripped compressed artifact**, not just the in-memory model

---

## Records Summary (`records/track_10min_16mb`)

### Leaderboard

| Rank | Submission | BPB | Δ vs Baseline | Key Techniques |
|---|---|---:|---:|---|
| 1 | SlidingWindow+FP16Embed+10L+MuonWD+OvertoneInit | **1.1748** | -0.0496 | Stacked: sliding eval, FP16 embed, 10L, Muon WD, spectral init |
| 2 | SlidingWindowEval | **1.1925** | -0.0319 | Eval-only: overlapping windows, stride 64 |
| 3 | LoRA TTT | **1.1929** | -0.0315 | Doc-isolated eval + strided eval + per-doc LoRA |
| 4 | TrainingOptSeq4096 | **1.2014** | -0.0229 | Train seq_len 4096, Muon momentum tuning |
| 5 | LongContextSeq2048 | **1.2058** | -0.0186 | Train seq_len 2048, tuned LRs |
| 6 | 10L MixedPrecision | **1.2147** | -0.0096 | 10 layers, mixed int8/int6 export |
| 7 | FP16Embed+WD3600 | **1.2197** | -0.0046 | FP16 tied embedding export, longer warmdown |
| 8 | LowerLR | **1.2230** | -0.0014 | Lower learning rates only |
| 9 | NaiveBaseline | **1.2244** | — | Baseline reference |
| 10 | WarmdownQuantization | ⚠️ | — | README/submission.json mismatch |

### Most Impactful Techniques

#### 1. Sliding / Strided Evaluation (~-0.032 BPB)
The single biggest isolated win. The baseline scores tokens with ~512 avg context; sliding window (stride=64) scores every token with 960+ context. Zero training changes, ~4x slower eval.

#### 2. Longer Training Context (~-0.019 to -0.023 BPB)
Training at seq_len 2048 or 4096 consistently helps despite fewer optimization steps per wallclock budget.

#### 3. FP16 Tied Embedding Export
Keeping the tied embedding in FP16 instead of int8 during export cuts quantization damage from ~0.007 BPB to ~0.0005 BPB.

#### 4. Extra Depth (10 Layers)
Adding a layer helps, but only if artifact size is managed via mixed-precision export (int8/int6) to stay under 16MB.

#### 5. LoRA TTT — Smaller Than Expected
The LoRA TTT submission's own ablation shows most of its gain comes from doc-isolation and strided eval (~-0.034 BPB), with LoRA adaptation adding only ~-0.003 BPB on top.

### Key Patterns

- **Eval tricks beat pure training tweaks** — sliding window alone outperforms all training-only changes.
- **Compression/export is part of the model design** — under a 16MB cap, how you quantize matters as much as what you train.
- **The best entries stack everything**: better eval context + better export + more capacity + optimizer tuning.
- **LR tuning alone is not the main story** — useful but never the biggest win.

