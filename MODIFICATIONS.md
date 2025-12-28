# 기존 파일 수정 가이드

이 문서는 기존 ZeroMAT 코드에 MoE 기능을 추가하기 위한 수정 사항을 설명합니다.

## 1. model/blip2qformer.py 수정

### 1.1. Import 추가

```python
from model.moe_alignn_encoder import FrozenMoEALIGNNEncoder
```

### 1.2. __init__ 메서드 수정

기존 `__init__` 메서드에 다음 파라미터 추가:

```python
def __init__(
    self,
    # 기존 파라미터들...

    # MoE 파라미터 추가
    use_moe=False,
    moe_extractor_dir=None,
    moe_topk_config=None,
    property_list=None,

    # 나머지 파라미터들...
):
```

Graph encoder 부분을 다음과 같이 수정:

```python
# Graph Encoder
if use_moe:
    # MoE Encoder 사용
    self.graph_encoder = FrozenMoEALIGNNEncoder(
        extractor_dir=moe_extractor_dir,
        topk_config_path=moe_topk_config,
        property_list=property_list
    )

    # Freeze
    self.graph_encoder.eval()
    for param in self.graph_encoder.parameters():
        param.requires_grad = False

    # LayerNorm for graph features (256 dim)
    self.ln_graph = nn.LayerNorm(256)

else:
    # 기존 코드 (단일 ALIGNN)
    self.graph_encoder = ALIGNN(...)  # 기존 코드 유지
    self.ln_graph = nn.LayerNorm(hidden_features)
```

Property prediction heads 추가:

```python
# Property prediction heads (12개)
if property_list is not None:
    self.property_heads = nn.ModuleDict({
        prop: nn.Sequential(
            nn.Linear(768, 256),  # Q-Former output → hidden
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Scalar prediction
        )
        for prop in property_list
    })
else:
    self.property_heads = None
```

### 1.3. forward 메서드 수정

기존 forward 메서드의 signature 수정:

```python
def forward(self, batch):
    """
    Args:
        batch: Dictionary containing:
            - graphs: DGL batched graph
            - texts: List of text strings
            - property_name: Target property name (str)
            - property_values: Property values [B]
    """
```

Graph encoding 부분 수정:

```python
# 1. Graph Encoding
if hasattr(self, 'graph_encoder') and isinstance(self.graph_encoder, FrozenMoEALIGNNEncoder):
    # MoE 모드
    graph_embeds, graph_mask = self.graph_encoder(
        batch['graphs'],
        property_name=batch['property_name']  # Property name 전달!
    )
else:
    # 기존 단일 ALIGNN 모드
    graph_embeds = self.graph_encoder(batch['graphs'])
    graph_mask = torch.ones(graph_embeds.size(0), dtype=torch.bool, device=graph_embeds.device)

# LayerNorm
graph_embeds = self.ln_graph(graph_embeds)
```

Q-Former 처리 (기존 코드 활용):

```python
# 2. Q-Former
# 기존 코드 그대로 사용
query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
query_output = self.Qformer.bert(
    query_embeds=query_tokens,
    encoder_hidden_states=graph_embeds.unsqueeze(1),
    encoder_attention_mask=graph_mask.unsqueeze(1),
    use_cache=True,
    return_dict=True,
)
```

Loss 계산:

```python
# 3. Loss 1: ITC (Graph-Text Contrastive)
# 기존 코드 활용
loss_itc = self.contrast_global(...)  # 기존 코드 그대로

# 4. Loss 2: LM (Language Modeling for robocrys generation)
# 기존 코드 활용
loss_lm = self.llm_model(...)  # 기존 코드 그대로

# 5. Loss 3: Property Prediction (NEW!)
loss_property = 0
if self.property_heads is not None and 'property_name' in batch:
    property_name = batch['property_name']
    property_head = self.property_heads[property_name]

    # Q-Former output의 평균을 사용
    query_feats = query_output.last_hidden_state.mean(dim=1)  # [B, 768]

    # Prediction
    property_pred = property_head(query_feats)  # [B, 1]
    property_target = batch['property_values']  # [B]

    # L1 Loss
    loss_property = F.l1_loss(property_pred.squeeze(), property_target)

# Total loss
total_loss = loss_itc + loss_lm + loss_property
```

Return 수정:

```python
return BlipOutput(
    loss=total_loss,
    loss_itc=loss_itc,
    loss_lm=loss_lm,
    loss_property=loss_property  # 추가
)
```

BlipOutput dataclass에도 추가:

```python
@dataclass
class BlipOutput:
    loss: torch.Tensor
    loss_itc: torch.Tensor
    loss_lm: torch.Tensor
    loss_property: torch.Tensor  # 추가
```

---

## 2. data/crystal_dataset.py 수정 (선택사항)

기존 dataset이 없다면 train_stage3_zeromat.py의 CrystalPropertyDataset을 사용하면 됩니다.

만약 기존 dataset 클래스를 수정하려면:

### 2.1. __getitem__ 수정

```python
def __getitem__(self, idx):
    sample = self.data[idx]

    # 기존 코드...
    g, _ = dgl.load_graphs(sample['graph_path'])
    g = g[0]
    text = sample['text']

    # 추가: Random property 선택
    import random
    available_props = [p for p in self.property_list if p in sample['properties']]
    property_name = random.choice(available_props)
    property_value = sample['properties'][property_name]

    return {
        'graph': g,
        'text': text,
        'property_name': property_name,       # 추가
        'property_value': property_value      # 추가
    }
```

### 2.2. collate_fn 수정

```python
def collate_fn(batch):
    # Batch 내 가장 많이 나온 property 선택
    from collections import Counter
    property_counts = Counter([item['property_name'] for item in batch])
    main_property = property_counts.most_common(1)[0][0]

    # Batch graphs
    graphs = [item['graph'] for item in batch]
    batched_graph = dgl.batch(graphs)

    # Batch texts
    texts = [item['text'] for item in batch]

    # Batch property values (main_property만)
    property_values = torch.tensor([
        item['properties'][main_property]
        for item in batch
    ])

    return {
        'graphs': batched_graph,
        'texts': texts,
        'property_name': main_property,
        'property_values': property_values
    }
```

---

## 3. 수정 체크리스트

- [ ] model/blip2qformer.py
  - [ ] Import FrozenMoEALIGNNEncoder
  - [ ] __init__에 MoE 파라미터 추가
  - [ ] Graph encoder를 MoE/단일 모드로 분기
  - [ ] Property heads 추가
  - [ ] forward에서 property_name 전달
  - [ ] Loss 3 (property prediction) 추가
  - [ ] BlipOutput에 loss_property 추가

- [ ] data/crystal_dataset.py (선택)
  - [ ] __getitem__에 property_name, property_value 추가
  - [ ] collate_fn에서 main_property 선택 로직 추가

---

## 4. 테스트

수정 후 다음 명령으로 테스트:

```bash
# 간단한 forward pass 테스트
python -c "
from model.blip2qformer import Blip2Qformer
model = Blip2Qformer(
    use_moe=True,
    moe_extractor_dir='checkpoints/extractors',
    moe_topk_config='config/moe_topk.json',
    property_list=['band_gap', 'formation_energy']
)
print('Model created successfully!')
"
```
