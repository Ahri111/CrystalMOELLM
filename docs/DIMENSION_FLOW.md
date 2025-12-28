# Dimension Flow: Q-Former → Property Prediction

Complete dimension tracking from crystal graph input to final property prediction.

---

## 📊 Overview

```
Crystal Graph (DGL)
    ↓
[MoE Encoder] → [B, 256]
    ↓
[LayerNorm] → [B, 256]
    ↓
[Q-Former] → [B, 32, 768]
    ↓
[Pooling] → [B, 768]
    ↓
[Property Head] → [B]
```

---

## Step-by-Step Dimension Changes

### 1. Input: Crystal Graph

```python
# Input: DGL batched graph
graph_batch = dgl.batch([g1, g2, ..., gB])
# Nodes: variable per graph
# Edges: variable per graph
```

**Dimensions:**
- Not a fixed tensor shape (graph structure)
- Contains node features, edge features, line graph features

---

### 2. MoE Encoder: Property-Specific Routing

```python
# In blip2qformer.py forward()
batch_node, batch_mask = self.graph_encoder(
    graph_batch,
    property_info['property_name']  # e.g., 'band_gap'
)
```

**Inside FrozenMoEEncoder:**

```python
# Property routing
config = self.topk_config['band_gap']
# → {'indices': [0, 7, 2], 'probs': [0.48, 0.35, 0.17]}

# Extract from each expert
for idx in [0, 7, 2]:
    expert_features, _ = self.extractors[idx](g)  # [B, hidden_features]

# Each ALIGNN expert:
#   Input: DGL graph
#   → ALIGNN layers
#   → Global pooling
#   Output: [B, hidden_features]

# Weighted combination
combined = sum(expert_features * probs)  # [B, hidden_features]
```

**Output Dimensions:**
- `batch_node`: **[B, hidden_features]** (typically 256)
- `batch_mask`: **[B]** (all True)

---

### 3. Layer Normalization

```python
batch_node = self.ln_graph(batch_node)
```

**Dimensions:**
- Input: [B, 256]
- Output: **[B, 256]** (normalized)

---

### 4. Q-Former Input Preparation

#### 4a. Query Tokens (Learnable)

```python
query_tokens = self.query_tokens.expand(batch_size, -1, -1)
```

**Dimensions:**
- Learnable params: [32, 768]
- Expanded: **[B, 32, 768]**

Where:
- 32 = number of query tokens (default in BLIP2)
- 768 = hidden size of Q-Former (BERT-base)

---

#### 4b. Property Prompt Tokens

```python
prompt_text = self.property_prompts['band_gap']
# → "Calculate the band gap energy:"

prompt_tokens = self.tokenizer(
    [prompt_text] * batch_size,
    return_tensors='pt',
    ...
)
```

**Dimensions:**
- `prompt_tokens.input_ids`: **[B, prompt_len]**
- `prompt_tokens.attention_mask`: **[B, prompt_len]**

Where:
- `prompt_len` ≈ 6-10 tokens (depends on prompt text)

---

### 5. Q-Former Forward Pass

```python
query_output = self.Qformer.bert(
    input_ids=prompt_tokens.input_ids,           # [B, prompt_len]
    attention_mask=prompt_tokens.attention_mask, # [B, prompt_len]
    query_embeds=query_tokens,                   # [B, 32, 768]
    encoder_hidden_states=batch_node,            # [B, 256]
    encoder_attention_mask=batch_mask,           # [B, 256]
    return_dict=True,
)
```

**What happens inside Q-Former:**

1. **Input Embeddings:**
   - Prompt tokens → word embeddings: [B, prompt_len, 768]
   - Query embeds (already 768-dim): [B, 32, 768]
   - **Combined sequence**: [B, prompt_len + 32, 768]

2. **Cross-Attention to Graph Features:**
   - Query: Combined sequence [B, prompt_len + 32, 768]
   - Key/Value: `encoder_hidden_states` [B, 256, 768] (after projection)
   - Output: Attended features [B, prompt_len + 32, 768]

3. **Self-Attention:**
   - Query tokens attend to each other
   - Query tokens attend to prompt tokens
   - Property context spreads through attention

**Output:**
```python
query_output.last_hidden_state  # [B, prompt_len + 32, 768]
```

---

### 6. Extract Query-Only Features

```python
query_output_features = query_output.last_hidden_state[:, :query_tokens.size(1), :]
```

**Dimensions:**
- Input: [B, prompt_len + 32, 768]
- Slicing: `[:, :32, :]` (first 32 tokens are query tokens)
- Output: **[B, 32, 768]**

**Why slice?**
- Prompt tokens were used to condition the attention
- But we only need query token representations for property prediction
- Query tokens now contain property-specific information!

---

### 7. Property Prediction Head

#### 7a. Pooling

```python
query_mean = query_output_features.mean(dim=1)
```

**Dimensions:**
- Input: [B, 32, 768]
- Mean pooling over 32 query tokens
- Output: **[B, 768]**

---

#### 7b. MLP Head

```python
property_head = self.property_heads['band_gap']
# = nn.Sequential(
#     nn.Linear(768, 256),
#     nn.ReLU(),
#     nn.Dropout(0.1),
#     nn.Linear(256, 1)
# )

prop_pred = property_head(query_mean).squeeze()
```

**Step-by-step:**

1. **Linear(768, 256):**
   - Input: [B, 768]
   - Weight: [768, 256]
   - Output: [B, 256]

2. **ReLU:**
   - Input: [B, 256]
   - Output: [B, 256]

3. **Dropout(0.1):**
   - Input: [B, 256]
   - Output: [B, 256] (training: random zeros; inference: identity)

4. **Linear(256, 1):**
   - Input: [B, 256]
   - Weight: [256, 1]
   - Output: [B, 1]

5. **Squeeze:**
   - Input: [B, 1]
   - Output: **[B]**

**Final Output:**
- Predicted property values: **[B]** (scalar per sample)

---

## 🔍 Complete Dimension Summary

| Component | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| Crystal Graph | DGL graph | - | Variable structure |
| MoE Encoder | DGL graph | [B, 256] | Property routing |
| LayerNorm | [B, 256] | [B, 256] | Normalization |
| Query Tokens | [32, 768] | [B, 32, 768] | Expand batch |
| Property Prompt | Text | [B, prompt_len] | Tokenized |
| Q-Former Input | - | [B, prompt_len+32, 768] | Combined sequence |
| Q-Former Output | [B, prompt_len+32, 768] | [B, prompt_len+32, 768] | Attended features |
| Query Extraction | [B, prompt_len+32, 768] | [B, 32, 768] | Slice first 32 |
| Mean Pooling | [B, 32, 768] | [B, 768] | Average over queries |
| Linear 1 | [B, 768] | [B, 256] | Dimensionality reduction |
| ReLU + Dropout | [B, 256] | [B, 256] | Activation |
| Linear 2 | [B, 256] | [B, 1] | Scalar prediction |
| Squeeze | [B, 1] | [B] | Remove dim |

---

## 🎨 Example with Concrete Numbers

Assume:
- Batch size: B = 16
- MoE hidden: 256
- Q-Former queries: 32
- Q-Former hidden: 768
- Property prompt: "Calculate the band gap energy:" (6 tokens)

**Forward Pass:**

```
Crystal Graph (16 graphs)
    ↓
MoE Encoder → [16, 256]
    ↓
LayerNorm → [16, 256]
    ↓
Query Tokens (expanded) → [16, 32, 768]
Property Prompt (tokenized) → [16, 6]
    ↓
Q-Former Input: [16, 6+32=38, 768]
    ↓
Q-Former Output: [16, 38, 768]
    ↓
Slice [:, :32, :] → [16, 32, 768]
    ↓
Mean pooling → [16, 768]
    ↓
Linear(768→256) → [16, 256]
    ↓
ReLU + Dropout → [16, 256]
    ↓
Linear(256→1) → [16, 1]
    ↓
Squeeze → [16]
```

**Output:** 16 predicted band gap values (scalars)

---

## 💡 Key Insights

### 1. Why Property Prompts Work

```
Without prompt:
  Query tokens attend to graph features blindly
  ❌ No context about what to extract

With prompt "Calculate the band gap energy:":
  Query tokens attend to both:
    - Graph features (structure info)
    - Prompt tokens (semantic guidance)
  ✅ Property-specific feature extraction
```

### 2. Why Mean Pooling?

```python
query_output_features: [B, 32, 768]
# 32 query tokens, each captures different aspects

query_mean: [B, 768]
# Aggregate all aspects into single representation
# Property head expects fixed-size input
```

Alternative pooling options:
- **Max pooling**: Takes most salient features
- **Attention pooling**: Learned weighted combination
- **First token**: Use query_output_features[:, 0, :] (like [CLS] in BERT)

Current choice: **Mean pooling** (simple and effective)

### 3. Why Separate Property Heads?

```python
self.property_heads = nn.ModuleDict({
    'band_gap': MLP(),       # Specialized for band gap
    'formation_energy': MLP(),  # Specialized for formation energy
    ...
})
```

**Benefits:**
- Each property has different scale/distribution
- Separate heads learn property-specific transformations
- More flexible than shared head

---

## 🧪 Debugging Tips

### Check Intermediate Shapes

```python
# In blip2qformer.py forward()
print(f"batch_node: {batch_node.shape}")  # [B, 256]
print(f"query_tokens: {query_tokens.shape}")  # [B, 32, 768]
print(f"query_output: {query_output.last_hidden_state.shape}")  # [B, 38, 768]
print(f"query_features: {query_output_features.shape}")  # [B, 32, 768]
print(f"query_mean: {query_mean.shape}")  # [B, 768]
print(f"prop_pred: {prop_pred.shape}")  # [B]
```

### Verify Property Head Selection

```python
property_name = property_info['property_name']
print(f"Using property head: {property_name}")
print(f"Head architecture: {self.property_heads[property_name]}")
```

---

## 📚 References

- **BLIP2 Paper**: Query tokens = 32, hidden = 768
- **Q-Former**: BERT-base architecture
- **Property Prompts**: Inspired by instruction tuning (InstructBLIP)

---

**Last Updated:** 2025-12-28
