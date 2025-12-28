# ALIGNN MoE + ZeroMAT Pipeline

Materials Science MLOps 파이프라인: Mixture-of-Experts ALIGNN + ZeroMAT 통합

## 📋 Overview

이 프로젝트는 12개의 물성별 ALIGNN Expert를 학습하고, MoE(Mixture-of-Experts) 방식으로 각 물성에 최적화된 Expert 조합을 자동으로 선택하여 ZeroMAT과 통합합니다.

### 핵심 아이디어

1. **Phase 1**: 각 물성(band_gap, formation_energy 등)별로 개별 ALIGNN Expert 학습
2. **Phase 1.5** ⭐: MoE Downstream 학습으로 각 물성에 최적인 Top-k Expert 조합 결정
3. **Phase 2**: Top-k 정보 추출 및 Extractor 저장
4. **Phase 3**: Frozen MoE + Q-Former + LLM 학습 (3 Losses)

### Architecture

```
Input: Crystal Structure (CIF)
    ↓
[12 Frozen ALIGNN Extractors]
    ↓
[Property-specific Top-k Selection] ← config/moe_topk.json
    ↓
[Weighted Combination]
    ↓
[Q-Former] ← Trainable
    ↓
[LLM] ← Frozen
    ↓
Outputs:
  - ITC Loss (Graph-Text Contrastive)
  - LM Loss (Robocrys Generation)
  - Property Loss (Scalar Prediction)
```

---

## 🚀 Quick Start

### 1. 환경 설정

```bash
# 필수 패키지 설치
pip install torch dgl jarvis-tools alignn transformers pyyaml

# 또는
pip install -r requirements.txt
```

### 2. 데이터 준비

CSV 파일을 `data/raw_data.csv`에 배치:

```csv
material_id,cif,robocrys,band_gap,formation_energy,...
mp-1234,<CIF content>,<robocrys text>,3.2,-1.5,...
```

### 3. 전체 파이프라인 실행

```bash
# 한 번에 실행
./run_pipeline.sh

# 또는 단계별 실행 (아래 참조)
```

---

## 📂 Directory Structure

```
CrystalMOELLM/
├── data/
│   ├── raw_data.csv          # 입력 CSV
│   ├── graphs/               # DGL graph files
│   ├── train.json           # Train split
│   ├── val.json             # Val split
│   └── test.json            # Test split
│
├── checkpoints/
│   ├── experts/             # Phase 1: ALIGNN experts
│   │   ├── alignn_band_gap.pt
│   │   ├── alignn_formation_energy.pt
│   │   └── ...
│   ├── moe_downstream/      # Phase 1.5: MoE checkpoints
│   │   ├── moe_band_gap.pt
│   │   └── ...
│   ├── extractors/          # Phase 2: Frozen extractors
│   │   ├── extractor_band_gap.pt
│   │   └── ...
│   └── stage3/              # Phase 3: Final model
│       └── best_model.pt
│
├── config/
│   ├── moe_topk.json        # Top-k configuration
│   └── stage3_zeromat.yaml  # Training config
│
├── model/
│   ├── alignn_extractor.py  # ALIGNN backbone extractor
│   ├── moe_alignn_encoder.py # Frozen MoE encoder
│   ├── blip2qformer.py      # (수정 필요)
│   └── ...
│
├── preprocess_data.py       # Phase 0
├── train_alignn_expert.py   # Phase 1
├── train_moe_downstream.py  # Phase 1.5 ⭐
├── extract_topk_config.py   # Phase 2
├── extract_alignn_backbone.py # Phase 2
├── train_stage3_zeromat.py  # Phase 3
└── run_pipeline.sh          # Full pipeline
```

---

## 🔧 단계별 실행 가이드

### Phase 0: Preprocessing

```bash
python preprocess_data.py \
    --csv_path data/raw_data.csv \
    --output_dir data
```

**출력**:
- `data/graphs/*.bin`: DGL graph 파일
- `data/{train,val,test}.json`: Split 파일

---

### Phase 1: Train ALIGNN Experts

각 물성별로 개별 ALIGNN 모델 학습:

```bash
# 예: band_gap expert
python train_alignn_expert.py \
    --target_property band_gap \
    --epochs 300 \
    --batch_size 64

# 모든 물성에 대해 반복
for prop in band_gap formation_energy bulk_modulus ...; do
    python train_alignn_expert.py --target_property $prop
done
```

**출력**: `checkpoints/experts/alignn_{property}.pt`

**Checkpoint 내용**:
```python
{
    'model_state_dict': {...},  # Full ALIGNN weights
    'property_name': 'band_gap',
    'val_mae': 0.15,
    'model_config': {...}
}
```

---

### Phase 1.5: Train MoE Downstream ⭐ (핵심!)

각 물성별로 MoE 학습하여 **어떤 Expert 조합이 best인지 결정**:

```bash
python train_moe_downstream.py \
    --target_property band_gap \
    --k_experts 3 \
    --epochs 100 \
    --batch_size 64
```

**핵심 동작**:
1. 12개 Extractor 로드 (Frozen)
2. Learnable gating weights 초기화
3. Top-k selection + prediction head 학습
4. Best top-k indices 저장

**출력**: `checkpoints/moe_downstream/moe_{property}.pt`

**Checkpoint 내용** (중요!):
```python
{
    'property_name': 'band_gap',
    'gating_weights': tensor([3.2, 0.8, 2.5, ...]),  # 12개
    'top_k_indices': [0, 7, 2],       # ← 이 물성에 best인 Expert!
    'top_k_probs': [0.48, 0.35, 0.17], # ← 가중치!
    'head_state_dict': {...}
}
```

**예시 출력**:
```
Training MoE: band_gap
Epoch 50/100
Train Loss: 0.145, Train MAE: 0.145
Val Loss: 0.152, Val MAE: 0.152

Current Top-3 Experts:
  1. Expert 0 (band_gap): 0.4832
  2. Expert 7 (n_Egap): 0.3456
  3. Expert 2 (bulk_modulus): 0.1712
```

---

### Phase 2: Extract Configuration

#### 2.1. Extract Top-k Config

MoE checkpoint에서 top-k 정보만 추출:

```bash
python extract_topk_config.py \
    --moe_dir checkpoints/moe_downstream \
    --output config/moe_topk.json
```

**출력**: `config/moe_topk.json`

```json
{
  "band_gap": {
    "indices": [0, 7, 2],
    "probs": [0.48, 0.35, 0.17]
  },
  "formation_energy": {
    "indices": [1, 4, 5],
    "probs": [0.52, 0.30, 0.18]
  },
  ...
}
```

#### 2.2. Extract ALIGNN Backbones

ALIGNN Expert에서 Head 제거, Backbone만 추출:

```bash
python extract_alignn_backbone.py \
    --expert_dir checkpoints/experts \
    --output_dir checkpoints/extractors
```

**출력**: `checkpoints/extractors/extractor_{property}.pt`

---

### Phase 3: Train ZeroMAT with Frozen MoE

Frozen MoE + Q-Former + LLM 학습:

```bash
python train_stage3_zeromat.py \
    --config config/stage3_zeromat.yaml
```

**3 Losses**:
1. **ITC Loss**: Graph ↔ Text Contrastive (기존 ZeroMAT)
2. **LM Loss**: Graph → Robocrys Generation (기존 ZeroMAT)
3. **Property Loss**: Graph → Scalar Prediction (NEW!)

**Frozen**:
- MoE Encoder (12 Extractors)
- LLM

**Trainable**:
- Q-Former
- Layer Norms
- LLM Projection
- Property Heads (12개)

**출력**: `checkpoints/stage3/best_model.pt`

---

## ⚙️ Configuration

`config/stage3_zeromat.yaml`:

```yaml
model:
  use_moe: true
  moe_extractor_dir: checkpoints/extractors
  moe_topk_config: config/moe_topk.json
  property_list: [band_gap, formation_energy, ...]

training:
  batch_size: 32
  epochs: 50
  lr: 1e-4

  loss_weights:
    itc: 1.0
    lm: 1.0
    property: 1.0
```

---

## 🔬 How MoE Works

### Phase 1.5에서 결정

```
각 물성별로 best Expert 조합 학습:
  - band_gap → Extractors [0, 7, 2]
  - formation_energy → Extractors [1, 4, 5]
  - ...
```

### Phase 3에서 사용

```python
# Lookup (계산 없음!)
if property_name == "band_gap":
    selected_experts = [0, 7, 2]
    probs = [0.48, 0.35, 0.17]

# Extract features
features = []
for idx in selected_experts:
    features.append(extractors[idx](graph))

# Weighted sum
combined = sum(feat * prob for feat, prob in zip(features, probs))
```

**완전 Frozen!** - No learnable parameters in MoE

---

## 📊 12 Properties

```python
PROPERTIES = [
    'band_gap',              # Band gap (eV)
    'formation_energy',      # Formation energy (eV/atom)
    'bulk_modulus',          # Bulk modulus (GPa)
    'shear_modulus',         # Shear modulus (GPa)
    'elastic_anisotropy',    # Elastic anisotropy
    'poisson_ratio',         # Poisson ratio
    'total_magnetization',   # Total magnetization (μB)
    'n_Egap',                # n-type band gap (eV)
    'p_Egap',                # p-type band gap (eV)
    'n_mass',                # n-type effective mass
    'p_mass',                # p-type effective mass
    'eij_max'                # Max piezoelectric constant
]
```

---

## 🛠️ 기존 코드 수정

ZeroMAT 코드에 MoE 기능 추가:

### 1. `model/blip2qformer.py` 수정

자세한 내용은 `MODIFICATIONS.md` 참조

주요 수정:
- `__init__`: MoE 파라미터 추가, Property heads 추가
- `forward`: property_name 전달, Loss 3 추가

### 2. `data/crystal_dataset.py` 수정 (선택)

`train_stage3_zeromat.py`의 `CrystalPropertyDataset` 사용 가능

---

## 📈 Expected Results

### Phase 1: Expert Training

```
band_gap Expert:
  Best Val MAE: 0.145 eV
  Epochs: 87/300 (early stopping)

formation_energy Expert:
  Best Val MAE: 0.032 eV/atom
  Epochs: 112/300
```

### Phase 1.5: MoE Downstream

```
band_gap MoE:
  Best Val MAE: 0.138 eV (개선!)
  Top-3 Experts: [0, 7, 2]

formation_energy MoE:
  Best Val MAE: 0.028 eV/atom (개선!)
  Top-3 Experts: [1, 4, 5]
```

### Phase 3: ZeroMAT

```
Epoch 50/50
Train - Loss: 2.456, ITC: 0.832, LM: 1.425, Property: 0.199
Val   - Loss: 2.512, ITC: 0.845, LM: 1.458, Property: 0.209

Property Prediction MAE:
  band_gap: 0.142 eV
  formation_energy: 0.030 eV/atom
  ...
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train_alignn_expert.py --batch_size 32

# Or use gradient accumulation
```

### Missing Dependencies

```bash
pip install jarvis-tools alignn
```

### Graph Conversion Errors

일부 CIF 파일 변환 실패 시 자동으로 skip됨

---

## 📝 Citation

```bibtex
@article{alignn2021,
  title={Atomistic Line Graph Neural Network for improved materials property predictions},
  author={Choudhary, Kamal and DeCost, Brian},
  journal={npj Computational Materials},
  year={2021}
}

@article{zeromat2024,
  title={ZeroMAT: Zero-shot Materials Discovery},
  ...
}
```

---

## 📧 Contact

For issues or questions, please open an issue on GitHub.

---

## ✅ Checklist

실행 전 확인:

- [ ] `data/raw_data.csv` 준비됨
- [ ] 필수 패키지 설치 완료
- [ ] GPU 사용 가능 (권장)
- [ ] `model/blip2qformer.py` 수정 완료 (MODIFICATIONS.md 참조)
- [ ] Disk space 충분 (~50GB 권장)

실행 후 확인:

- [ ] `checkpoints/experts/` 에 12개 파일
- [ ] `checkpoints/moe_downstream/` 에 12개 파일
- [ ] `config/moe_topk.json` 생성됨
- [ ] `checkpoints/extractors/` 에 12개 파일
- [ ] `checkpoints/stage3/best_model.pt` 생성됨

---

**Happy Training! 🚀**
