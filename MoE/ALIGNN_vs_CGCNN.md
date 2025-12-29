# ALIGNN vs CGCNN: 상세 비교

## 1. 그래프 구조

### CGCNN
```
입력: 결정 구조 (CIF/POSCAR)
     ↓
원자 그래프 생성:
  - 노드: 원자 (atomic features)
  - 엣지: 원자 간 거리 (Gaussian expansion)
     ↓
그래프 표현:
    O ----d1---- O
    |            |
   d2           d3
    |            |
    O ----d4---- O
```

### ALIGNN
```
입력: 결정 구조 (CIF/POSCAR)
     ↓
두 개의 그래프 생성:

1) Atom Graph (g):
    - 노드: 원자
    - 엣지: bond (원자 간 연결)

2) Line Graph (lg):
    - 노드: bond (atom graph의 엣지)
    - 엣지: angle (두 bond 사이의 각도)

예시:
Atom Graph:
    O ----b1---- O ----b2---- O

Line Graph:
    [b1] ----θ(b1,b2)---- [b2]

    여기서 θ는 두 bond 사이의 각도
```

## 2. Message Passing 메커니즘

### CGCNN
```python
for layer in conv_layers:
    # 각 원자 노드 업데이트
    for atom_i in atoms:
        # 이웃 원자들로부터 정보 수집
        messages = []
        for neighbor_j in neighbors(atom_i):
            edge_feature = gaussian_expand(distance(i, j))
            message = neighbor_j.feature * gate(edge_feature)
            messages.append(message)

        # 원자 i의 특성 업데이트
        atom_i.feature = aggregate(messages)

# Pooling
crystal_feature = mean(all_atom_features)
```

**특징:**
- 단방향 정보 흐름: 원자 → 원자
- 2-body interaction만 포착
- Edge features는 distance만 사용

### ALIGNN
```python
for alignn_layer in alignn_layers:

    # Step 1: Atom Graph 업데이트
    # EdgeGatedGraphConv on atom graph
    for edge_ij in atom_edges:
        edge_ij.feature = update_edge(
            node_i.feature,
            node_j.feature,
            edge_ij.feature
        )

    for node_i in atom_nodes:
        node_i.feature = update_node(
            node_i.feature,
            aggregate(neighbor_edges)
        )

    # Step 2: Line Graph 업데이트
    # EdgeGatedGraphConv on line graph
    for edge_angle in line_edges:
        # angle은 line graph의 엣지
        edge_angle.feature = update_edge(
            bond_i.feature,  # line graph의 노드 (atom graph의 엣지)
            bond_j.feature,
            edge_angle.feature
        )

    for bond in line_nodes:
        bond.feature = update_node(
            bond.feature,
            aggregate(neighbor_angles)
        )

    # Step 3: Line Graph → Atom Graph 피드백
    # Line graph의 업데이트된 bond features를
    # Atom graph의 edge features로 다시 전달

# Pooling
crystal_feature = mean(all_atom_features)
```

**특징:**
- 양방향 정보 흐름: atom ↔ bond ↔ angle
- 3-body interaction 포착 (bond angle)
- 더 풍부한 구조 정보

## 3. 정보 캡처 능력

### CGCNN이 포착하는 정보
```
원자 A와 B 사이 거리: 2.5 Å
원자 A와 C 사이 거리: 2.3 Å

    C
    |  2.3 Å
    A -------- B
      2.5 Å

포착 못하는 정보:
- C-A-B 각도 (∠CAB)
- Local geometry
```

### ALIGNN이 포착하는 정보
```
Bond AC: 2.3 Å
Bond AB: 2.5 Å
Angle CAB: 120°

    C
    |  2.3 Å
    A -------- B
 120°  2.5 Å

포착하는 정보:
- 거리 (bond length)
- 각도 (bond angle) ✓
- Local coordination geometry ✓
- 3-body correlation ✓
```

## 4. 코드 레벨 비교

### CGCNN Forward Pass
```python
def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
    """
    Args:
        atom_fea: (N_atoms, atom_fea_len)
        nbr_fea: (N_atoms, M_neighbors, nbr_fea_len)
        nbr_fea_idx: (N_atoms, M_neighbors)
        crystal_atom_idx: (N_crystals,)
    """
    # Atom embedding
    atom_fea = self.embedding(atom_fea)

    # Convolution layers
    for conv_func in self.convs:
        atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

    # Pooling
    crys_fea = self.pooling(atom_fea, crystal_atom_idx)

    # Prediction
    crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
    out = self.fc_out(crys_fea)

    return out
```

### ALIGNN Forward Pass
```python
def forward(self, g, lg):
    """
    Args:
        g: DGL graph (atom graph)
            - g.ndata['atom_features']: atom features
            - g.edata['r']: bond lengths
        lg: DGL graph (line graph)
            - lg.ndata: bond features (from g.edata)
            - lg.edata['h']: angle features
    """
    # Initial embeddings
    g, lg = self.atom_embedding(g), self.edge_embedding(g, lg)

    # ALIGNN layers (alternating atom & line graph updates)
    for alignn_layer in self.alignn_layers:
        g, lg = alignn_layer(g, lg)

    # Final GCN layers on atom graph
    for gcn_layer in self.gcn_layers:
        g = gcn_layer(g)

    # Readout (pooling)
    out = self.readout(g)

    # Prediction
    out = self.fc(out)

    return out
```

## 5. 장단점 비교

### CGCNN

**장점:**
- 단순하고 빠름
- 메모리 효율적
- 구현이 간단

**단점:**
- 2-body interaction만 포착
- Bond angle 정보 손실
- 복잡한 구조 패턴 포착 어려움

### ALIGNN

**장점:**
- 3-body interaction 명시적 모델링
- Bond angle 정보 포착 ✓
- 더 풍부한 구조 정보
- **성능이 일반적으로 더 좋음** (특히 band gap 같은 전자 구조 예측)

**단점:**
- 계산 비용 증가 (2배 그래프)
- 메모리 사용량 증가
- 구현 복잡도 증가

## 6. 성능 비교 (Band Gap 예측)

논문 결과:
```
데이터셋: Materials Project (약 70,000 결정)

CGCNN MAE:     0.388 eV
ALIGNN MAE:    0.218 eV  ← 43% 성능 향상!

이유:
- Band gap은 전자 구조에 민감
- 전자 구조는 local geometry (bond angle)에 크게 영향받음
- ALIGNN이 이를 명시적으로 모델링
```

## 7. 우리 MOE 프레임워크에서의 활용

### 데이터 흐름

```python
# 1. 데이터 로딩 (MoE/alignn/data.py)
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph

atoms = Atoms.from_cif('structure.cif')
g, lg = Graph.atom_dgl_multigraph(
    atoms,
    cutoff=8.0,
    max_neighbors=12,
    compute_line_graph=True  # ← Line graph 생성!
)

# 2. ALIGNN 모델 (MoE/alignn/model.py)
model = ALIGNNRegression(
    alignn_layers=4,
    gcn_layers=4,
    hidden_features=256,
)

# 3. Forward pass
features = model.get_features(g, lg)  # ← (g, lg) 두 개 입력!

# 4. MOE에서 사용 (MoE/moe/model.py)
extractors = [extractor1, extractor2, ...]
for extractor in extractors:
    features = extractor(g, lg)  # ← 각 extractor가 (g, lg) 처리
```

## 8. 핵심 차이 요약

| 항목 | CGCNN | ALIGNN |
|------|-------|--------|
| 입력 그래프 | 1개 (atom graph) | 2개 (atom + line graph) |
| Interaction | 2-body (distance) | 3-body (distance + angle) |
| Message Passing | 단방향 | 양방향 (atom ↔ line) |
| Bond Angle | ✗ | ✓ |
| 성능 (band gap) | MAE 0.388 eV | MAE 0.218 eV |
| 계산 비용 | 낮음 | 높음 |
| 메모리 | 적음 | 많음 |

## 9. 언제 어떤 모델을 쓸까?

**CGCNN 추천:**
- 빠른 스크리닝 필요
- 메모리 제약
- 간단한 property (density, volume 등)

**ALIGNN 추천:**
- 높은 정확도 필요
- 전자 구조 관련 property (band gap, formation energy)
- 복잡한 구조 패턴 중요
- **MOE 프레임워크** ← 우리 케이스!

## 10. 참고 자료

- CGCNN 논문: "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties" (2018)
- ALIGNN 논문: "Atomistic Line Graph Neural Network for Improved Materials Property Predictions" (2021)
- ALIGNN GitHub: https://github.com/usnistgov/alignn
