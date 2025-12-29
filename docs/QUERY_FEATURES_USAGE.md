# Query Features 사용 방식: ITC, ITM, LM Loss

Q-Former의 query output `[B, 32, 768]`이 각 loss에서 어떻게 사용되는지 상세 설명

---

## 📊 전체 Flow

```
Q-Former Output: [B, 32, 768]
    ├─ ITC Loss → graph_proj → [B, 32, 256] → max pooling → [B, 256]
    ├─ ITM Loss → gtm_head → [B, 32, 2] → mean pooling → [B, 2]
    └─ LM Loss → past_key_values → decoder attention
```

---

## 1. ITC Loss (Image-Text Contrastive)

### 목적
Graph representation과 text representation을 contrastive learning으로 정렬

### Dimension Flow

```python
# Q-Former output
query_output = self.Qformer.bert(
    query_embeds=query_tokens,
    encoder_hidden_states=batch_node,
    encoder_attention_mask=batch_mask,
    use_cache=True,
    return_dict=True,
)
# query_output.last_hidden_state: [B, 32, 768]

# Graph feature projection
graph_feats = self.graph_proj(query_output.last_hidden_state)
# graph_proj = nn.Linear(768, 256)
# Output: [B, 32, 256]

# Text feature (CLS token only)
text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True)
text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
# text_proj = nn.Linear(768, 256)
# Output: [B, 256]
```

**핵심:**
- **32개 query tokens 모두 사용**: [B, 32, 256]
- Text는 **[CLS] token 하나만** 사용: [B, 256]

### Contrastive Loss 계산

```python
def contrast(self, features_graph, features_text):
    '''
    features_graph: [B, 32, 256]
    features_text: [B, 256]
    '''
    # Normalize
    features_graph = F.normalize(features_graph, dim=-1)  # [B, 32, 256]
    features_text = F.normalize(features_text, dim=-1)    # [B, 256]

    # Cosine similarity
    sim_q2t = features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)
    # [B, 1, 32, 256] @ [B, 256, 1] = [B, B, 32]

    # Max pooling over 32 queries
    sim_g2t, _ = sim_q2t.max(-1)  # [B, B]

    # Temperature scaling
    logits_per_graph = sim_g2t / self.temperature  # [B, B]
    logits_per_text = logits_per_graph.t()         # [B, B]

    # Cross-entropy loss
    labels = torch.arange(B, device=device)  # [0, 1, 2, ..., B-1]
    loss = (F.cross_entropy(logits_per_graph, labels) +
            F.cross_entropy(logits_per_text, labels)) / 2
```

**핵심 아이디어:**
1. **32개 query token 각각**이 text와 similarity 계산
2. **Max pooling**으로 가장 유사한 query 선택
3. 이유: 각 query가 graph의 **다른 측면**을 포착하므로, 가장 관련있는 query 사용

**왜 max pooling?**
- 32개 query는 graph의 **complementary aspects** 포착
- Text description은 특정 aspect와만 관련될 수 있음
- 가장 관련있는 query 선택 = 가장 높은 similarity

---

## 2. ITM Loss (Image-Text Matching)

### 목적
Graph-text pair가 실제로 매칭되는지 binary classification

### Dimension Flow

```python
# Positive + Negative pairs
# text_ids_all: [3B, seq_len] (pos, pos, neg)
# graph_embeds_all: [3B, 256] (pos, neg, pos)

# Q-Former with both modalities
query_tokens_itm = self.query_tokens.expand(3*B, -1, -1)  # [3B, 32, 768]
query_atts_itm = torch.ones(query_tokens_itm.size()[:-1])  # [3B, 32]
attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)  # [3B, 32+seq_len]

output_itm = self.Qformer.bert(
    text_ids_all,                           # [3B, seq_len]
    query_embeds=query_tokens_itm,          # [3B, 32, 768]
    attention_mask=attention_mask_all,      # [3B, 32+seq_len]
    encoder_hidden_states=graph_embeds_all, # [3B, 256]
    encoder_attention_mask=graph_atts_all,  # [3B, 256]
    return_dict=True,
)
# output_itm.last_hidden_state: [3B, 32+seq_len, 768]
```

**핵심:**
- Query tokens과 text tokens이 **함께** attention
- Graph features는 cross-attention의 key/value로 사용

### Classification Head

```python
# Extract query tokens only
vl_embeddings = output_itm.last_hidden_state[:, :query_tokens_itm.size(1), :]
# vl_embeddings: [3B, 32, 768]

# Classification head
vl_output = self.gtm_head(vl_embeddings)
# gtm_head = nn.Linear(768, 2)
# vl_output: [3B, 32, 2]

# Mean pooling over 32 queries
logits = vl_output.mean(dim=1)  # [3B, 2]

# Binary classification
itm_labels = torch.cat([
    torch.ones(B),   # positive pairs
    torch.zeros(2*B) # negative pairs
])  # [3B]

loss_itm = F.cross_entropy(logits, itm_labels)
```

**왜 mean pooling?**
- ITC와 달리 여기서는 **전체적인 매칭** 판단
- 32개 query 모두가 graph-text alignment에 기여
- Mean = **모든 query의 consensus**

**차이점 정리:**
| Loss | Pooling | 이유 |
|------|---------|------|
| ITC | **Max** | 가장 관련있는 aspect 선택 |
| ITM | **Mean** | 전체적인 매칭 정도 종합 |

---

## 3. LM Loss (Language Modeling)

### 목적
Text를 autoregressive하게 생성 (caption generation)

### Dimension Flow

```python
# Use cached key-values from query_output
query_output = self.Qformer.bert(
    query_embeds=query_tokens,
    encoder_hidden_states=batch_node,
    encoder_attention_mask=batch_mask,
    use_cache=True,  # ← IMPORTANT!
    return_dict=True,
)
# query_output.past_key_values: tuple of (key, value) tensors

# Decoder input
decoder_input_ids = text.clone()
decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
labels = decoder_input_ids.masked_fill(
    decoder_input_ids == self.tokenizer.pad_token_id, -100
)

# Query attention mask
query_atts = torch.ones(query_tokens.size()[:-1])  # [B, 32]
attention_mask = torch.cat([query_atts, mask], dim=1)  # [B, 32+seq_len]

# Language modeling with past_key_values
lm_output = self.Qformer(
    decoder_input_ids,                    # [B, seq_len]
    attention_mask=attention_mask,        # [B, 32+seq_len]
    past_key_values=query_output.past_key_values,  # ← Pre-computed!
    return_dict=True,
    labels=labels,
)

loss_lm = lm_output.loss
```

### `past_key_values`란?

**구조:**
```python
past_key_values = (
    (layer0_key, layer0_value),  # Layer 0
    (layer1_key, layer1_value),  # Layer 1
    ...
    (layer11_key, layer11_value) # Layer 11 (BERT-base has 12 layers)
)

# Each key/value:
layer_key: [B, num_heads, 32, head_dim]
layer_value: [B, num_heads, 32, head_dim]
```

**역할:**
- Q-Former가 **이미 계산한** query tokens의 key/value를 캐싱
- Decoder가 text를 생성할 때 **재계산 없이** query attention 수행
- **효율성**: Query tokens을 한 번만 처리

**Decoder Attention Mechanism:**

```python
# Decoder의 각 layer에서
for layer_idx, layer in enumerate(decoder_layers):
    # Self-attention on text tokens
    text_self_attn = layer.self_attention(text_tokens)

    # Cross-attention to queries (using cached key/value!)
    query_cross_attn = layer.cross_attention(
        query=text_tokens,
        key=past_key_values[layer_idx][0],    # Pre-computed query keys
        value=past_key_values[layer_idx][1],  # Pre-computed query values
    )

    # Combine
    output = text_self_attn + query_cross_attn
```

**핵심:**
- **32개 query tokens의 정보가 key/value로 저장됨**
- Decoder가 각 text token을 생성할 때 **32개 query 모두 attend**
- 즉, 전체 graph information이 generation에 사용됨

---

## 🔍 비교 정리

| Loss | Query 사용 방식 | Pooling | Output Dim | 목적 |
|------|----------------|---------|------------|------|
| **ITC** | graph_proj → [B, 32, 256] | **Max** | [B, 256] | Contrastive alignment |
| **ITM** | gtm_head → [B, 32, 2] | **Mean** | [B, 2] | Binary matching |
| **LM** | past_key_values (cached) | **All 32** | [B, seq_len, vocab] | Text generation |
| **Property** | mean → [B, 768] → head | **Mean** | [B, 1] | Regression |

---

## 💡 핵심 Insight

### 1. 왜 32개 query tokens?

**각 query가 다른 aspect를 포착:**
- Query 1: Crystal structure symmetry
- Query 2: Chemical composition
- Query 3: Electronic properties
- ...
- Query 32: Long-range interactions

**Task에 따라 다른 query가 중요:**
- Band gap 예측: Electronic property queries 중요
- Formation energy: Chemical composition queries 중요
- Text matching: Context에 따라 다른 query 선택 (max pooling)

### 2. 왜 projection dimension = 256?

```python
graph_proj = nn.Linear(768, 256)
text_proj = nn.Linear(768, 256)
```

**이유:**
- 768은 Q-Former hidden size (BERT-base)
- 256은 **contrastive learning space**
- 낮은 차원 = 더 **compact representation**
- Contrastive learning에서 일반적 (CLIP도 512 사용)

### 3. ITC vs ITM: Max vs Mean

**ITC (Max pooling):**
```
Text: "This crystal has high band gap"
Query similarities: [0.3, 0.8, 0.2, 0.5, ...]
                            ↑
                    Most relevant query
Select: 0.8 (electronic property query)
```

**ITM (Mean pooling):**
```
Overall matching score needed
All queries contribute: (0.3 + 0.8 + 0.2 + ... + 0.5) / 32
Balanced judgment across all aspects
```

---

## 🎯 MoE에서의 차이점

MoE를 사용할 때 **Property-conditioned Q-Former**:

```python
# Property prompt 추가
prompt_text = "Calculate the band gap energy:"
prompt_tokens = tokenizer([prompt_text] * B, ...)

query_output = self.Qformer.bert(
    input_ids=prompt_tokens.input_ids,      # Property context!
    query_embeds=query_tokens,
    encoder_hidden_states=batch_node,
    ...
)

# Query extraction (첫 32개만)
query_features = query_output.last_hidden_state[:, :32, :]
```

**효과:**
- Property prompt가 **query attention을 guide**
- 32개 query 중 **relevant queries에 더 높은 weight**
- 예: "Calculate band gap" → electronic property queries 강조

---

## 📊 Dimension Flow 요약

```
Crystal Graph
    ↓
MoE Encoder: [B, 256]
    ↓
Q-Former: [B, 32, 768]
    ├─ ITC:  graph_proj → [B, 32, 256] → max → [B, 256] → contrastive loss
    ├─ ITM:  gtm_head → [B, 32, 2] → mean → [B, 2] → binary CE loss
    ├─ LM:   past_key_values → decoder attention → [B, seq_len, vocab] → NLL loss
    └─ Property: mean → [B, 768] → head → [B, 1] → L1 loss
```

---

**Last Updated:** 2025-12-28
