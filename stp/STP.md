# Semantic Tube Prediction (STP) for LLM Training

Based on the paper *"Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA,"* this guide explains the theory behind the Semantic Tube Prediction (STP) regularizer and provides a practical guide on how to implement it in your own Large Language Model (LLM) training pipelines.

---

## 1. What is Semantic Tube Prediction (STP)?

Standard LLMs are trained using **Next Token Prediction (NTP)** (cross-entropy loss). While highly effective, NTP is a *local* objective. It tends to conflate global semantic signals with surface-level statistical noise. During inference, this noise accumulates (like Brownian motion), occasionally causing the model to drift off the ideal semantic path and suffer from mode collapse or hallucination.

**Semantic Tube Prediction (STP)** is a lightweight, JEPA-style (Joint-Embedding Predictive Architecture) regularization term added to the standard NTP loss. 

It dramatically improves the **Signal-to-Noise Ratio (SNR)** of the model's hidden states, allowing models to achieve baseline accuracy with **16x less training data**, effectively bypassing standard Chinchilla-style scaling laws.

### The Geodesic Hypothesis & The "Tube"
The method relies on the **Geodesic Hypothesis**: the idea that a logical, error-free sequence of tokens traces a smooth, locally linear path (a geodesic) through the model's high-dimensional representation space.

If the trajectory is locally linear, any three hidden states ($h_s, h_r, h_t$) in a sequence should form roughly a straight line. STP enforces a "Semantic Tube" around this straight line. It penalizes the hidden state $h_r$ if it deviates perpendicularly from the vector connecting $h_s$ and $h_t$.

---

## 2. How It Works (The Math)

The STP loss is remarkably simple and requires **no extra forward passes** and **no auxiliary predictor networks** (unlike standard JEPA models).

Given a sequence of tokens, let $s$, $r$, and $t$ be the indices of three tokens such that $s < r < t$. Let $h$ represent the hidden state of a token at the **last layer** of the network.

We define two vectors:
1.  **$h_r - h_s$**: The semantic evolution from token $s$ to $r$.
2.  **$h_t - h_r$**: The semantic evolution from token $r$ to $t$.

Because the path should be a straight line, these two vectors should point in the exact same direction. Therefore, we minimize the cosine distance between them:

$$ \mathcal{L}_{STP} = 1 - \cos(h_t - h_r, h_r - h_s) $$

The total training objective becomes:
$$ \mathcal{L}_{Total} = \mathcal{L}_{NTP} + \lambda \cdot \mathcal{L}_{STP} $$

Where $\lambda$ is a hyperparameter controlling the strength of the tube regularization.

---

## 3. How to Implement It

Implementing STP is incredibly straightforward, especially if you are using PyTorch and HuggingFace `transformers`. 

Because you already compute the hidden states during the forward pass to get your standard Cross-Entropy loss, calculating STP adds virtually zero computational overhead.

### Step 1: The Loss Function in PyTorch

Here is a drop-in PyTorch function to compute the STP loss given a batch of hidden states:

```python
import torch
import torch.nn.functional as F

def compute_stp_loss(hidden_states):
    """
    Computes the Semantic Tube Prediction loss.
    
    Args:
        hidden_states (torch.Tensor): Last layer hidden states of shape 
                                      [batch_size, sequence_length, hidden_dim]
    Returns:
        torch.Tensor: The scalar STP loss.
    """
    batch_size, seq_len, _ = hidden_states.shape
    
    # We need at least 3 tokens to pick s < r < t
    if seq_len < 3:
        return torch.tensor(0.0, device=hidden_states.device)
    
    # Randomly pick 3 indices
    # (Alternatively, you can sample s, r, t independently for each item in the batch)
    indices = torch.sort(torch.randperm(seq_len)[:3])[0]
    s, r, t = indices[0], indices[1], indices[2]
    
    # Extract the hidden states for these indices
    h_s = hidden_states[:, s, :]
    h_r = hidden_states[:, r, :]
    h_t = hidden_states[:, t, :]
    
    # Calculate the semantic evolution vectors
    vec1 = h_t - h_r
    vec2 = h_r - h_s
    
    # Compute Cosine Similarity
    # F.cosine_similarity returns values between -1 and 1
    cos_sim = F.cosine_similarity(vec1, vec2, dim=-1)
    
    # STP Loss is 1 - cosine_similarity (we want it to be as close to 1 as possible)
    l_stp = (1.0 - cos_sim).mean()
    
    return l_stp
```

### Step 2: Integrating with a HuggingFace Training Loop

When using a standard HuggingFace model, you just need to tell the model to return its hidden states, compute both losses, and add them together.

```python
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Hyperparameters
LAMBDA_STP = 0.02 # Recommended value from the paper

def training_step(batch, model, optimizer):
    optimizer.zero_grad()
    
    # 1. Forward pass: ensure output_hidden_states=True
    outputs = model(
        input_ids=batch['input_ids'], 
        labels=batch['labels'], 
        output_hidden_states=True
    )
    
    # 2. Get standard Next-Token Prediction (NTP) Loss
    l_ntp = outputs.loss
    
    # 3. Get last layer hidden states 
    # outputs.hidden_states is a tuple of all layers; [-1] gets the final layer
    last_layer_hidden_states = outputs.hidden_states[-1] 
    
    # 4. Compute STP Loss
    l_stp = compute_stp_loss(last_layer_hidden_states)
    
    # 5. Combine Losses
    total_loss = l_ntp + (LAMBDA_STP * l_stp)
    
    # 6. Backward and step
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
```

---

## 4. Best Practices & Paper Insights

If you are integrating STP into your training setup, keep the following empirical findings from the paper in mind:

*   **Tuning $\lambda$:** The paper states that because geodesics naturally have *some* curvature, you don't want to enforce a perfectly rigid straight line. **Keep $\lambda \ll 1$**. The optimal values across various models (Llama 3, Gemma 2, Qwen) consistently fell between **$0.01$ and $0.08$**. A good default starting point is **`0.02`**.
*   **Which Hidden States to Use:** Use the hidden states from the **last layer** of the transformer (right before the unembedding head).
*   **Choosing Indices ($s, r, t$):**
    *   **Random Selection:** Standard random selection (as shown in the code above) works exceptionally well and removes the need for manual data formatting.
    *   **Smart Anchoring (Optional):** If your data is formatted as `[System Prompt] [Query] [Answer]`, you can anchor $s$ after the system prompt to ignore static instruction embeddings, or pick $s$ and $t$ to span across the query and answer to skip over "noisy" reasoning steps in the middle.
*   **No Predictor Needed:** Unlike traditional JEPA architectures which require a trained projection head to map representations to one another, STP explicitly relies on the *identity function*. Do not add an MLP or linear layer to project $h_r$ to $h_t$—the paper's ablation studies show this degrades performance.
