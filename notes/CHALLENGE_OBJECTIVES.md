# Parameter Golf Challenge

## Objective
- **Minimize**: Bits Per Byte (BPB) on FineWeb validation set
- **Constraint**: 16MB artifact (code + compressed model), train in under 10 minutes on 8xH100s
- **Metric**: Tokenizer-agnostic compression (BPB)

## Alternative Approaches Mentioned
- Test-time compute, parameter tying, depth recurrence, low-rank training
- Low precision / QAT / bitnets, novel tokenizers
- Test-time training, long context evaluation
- 1-bit quantization, ternary quantization
- JEPA, text diffusion, H-net tokenization
- Universal transformer, state-space models, E2E TTT
- Megakernels, learning adapters on random linear maps

---

## MLP Squeeze Experiments

### Hypothesis
Trading MLP width for depth: squeeze MLP dimensions while adding layers to maintain similar total parameter count. Depth often beats width for same parameter budget.

### Method
- Use V-shaped mlp_mults: 3.0 at edges, valley in middle
- Vary valley depth: 2.5 → 2.3 → 2.0 (more squeeze = fewer params)
- Add layers to reclaim parameter budget

### Test Matrix

#### Phase 1: 11 Layers (baseline squeeze)
| Config | Valley | mlp_mults | MLP Params |
|--------|--------|-----------|-------------|
| baseline_squeeze_11 | 2.5 | [3.0,2.9,2.8,2.7,2.6,2.5,2.6,2.7,2.8,2.9,3.0] | ~14M |
| squeeze_11_2.3 | 2.3 | [3.0,2.86,2.72,2.58,2.44,2.3,2.44,2.58,2.72,2.86,3.0] | ~13M |
| squeeze_11_2.0 | 2.0 | [3.0,2.8,2.6,2.4,2.2,2.0,2.2,2.4,2.6,2.8,3.0] | ~12M |

#### Phase 2: 12 Layers
| Config | Valley | mlp_mults | MLP Params |
|--------|--------|-----------|-------------|
| baseline_squeeze_12 | 2.5 | [3.0,2.91,2.82,2.73,2.64,2.55,2.55,2.64,2.73,2.82,2.91,3.0] | ~15M |
| squeeze_12_2.3 | 2.3 | [3.0,2.86,2.72,2.58,2.44,2.3,2.3,2.44,2.58,2.72,2.86,3.0] | ~14M |
| squeeze_12_2.0 | 2.0 | [3.0,2.8,2.6,2.4,2.2,2.0,2.0,2.2,2.4,2.6,2.8,3.0] | ~13M |

#### Phase 3: 13 Layers
| Config | Valley | mlp_mults | MLP Params |
|--------|--------|-----------|-------------|
| baseline_squeeze_13 | 2.5 | [3.0,2.92,2.83,2.75,2.67,2.58,2.5,2.58,2.67,2.75,2.83,2.92,3.0] | ~16M |
| squeeze_13_2.3 | 2.3 | [3.0,2.92,2.83,2.75,2.67,2.58,2.3,2.58,2.67,2.75,2.83,2.92,3.0] | ~15M |
| squeeze_13_2.0 | 2.0 | [3.0,2.83,2.67,2.5,2.33,2.17,2.0,2.17,2.33,2.5,2.67,2.83,3.0] | ~14M |

#### Phase 4: Aggressive (scale mlp_mult with layers)
Match total parameters by scaling mlp_mult up when adding layers:
- 12 layers: mlp_mult scaled to maintain ~17M MLP params
- 13 layers: mlp_mult scaled to maintain ~17M MLP params

### Metrics to Track
1. **Training stability** - any divergence, NaN, loss spikes
2. **Training time** - ms/step with extra layers
3. **Validation BPB** - final compression quality
4. **Parameter count** - total model params

### Expected Baseline (for comparison)
Current SOTA: **1.1147 BPB** with 11 layers, mlp_mult=3x all layers

---

## Env Variables for Testing
```bash
# Basic squeeze (11 layers, valley 2.5)
NUM_LAYERS=11 MLP_SQUEEZE_VALLEY=2.5 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Custom mlp_mults override
MLP_MULTS="3.0,2.8,2.6,2.4,2.5,2.4,2.6,2.8,3.0,2.8,2.6" torchrun ...

# 12 layers with valley 2.3
NUM_LAYERS=12 MLP_SQUEEZE_VALLEY=2.3 torchrun ...
```