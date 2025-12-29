# ALIGNN-based Mixture of Experts (MOE) for Band Gap Prediction

CGCNN 기반 MOE를 ALIGNN으로 변환한 프레임워크입니다. 여러 데이터셋에서 학습된 ALIGNN 모델들을 MOE로 통합하여 성능을 향상시킵니다.

## 📁 프로젝트 구조

```
MoE/
├── alignn/                    # ALIGNN 모듈
│   ├── model.py              # ALIGNN 모델 래퍼
│   ├── data.py               # CSV 기반 데이터 로더
│   ├── utils.py              # 학습 유틸리티
│   └── __init__.py
├── moe/                       # MOE 모듈
│   ├── model.py              # MOE 모델 (MixtureOfExtractors, MultiheadedMOE)
│   ├── utils.py              # MOE 유틸리티
│   └── __init__.py
├── single_train.py            # 단일 property 학습 스크립트
├── moe_train.py               # MOE 통합 학습 스크립트
└── README.md                  # 이 파일
```

## 🎯 지원 데이터셋 (Band Gap)

1. **MP_band_gap** - Materials Project DFT band gaps
2. **jarvis_dft_3d_optb88** - JARVIS 3D (OptB88vdW functional)
3. **jarvis_dft_3d_tbmbj** - JARVIS 3D (TBMBJ functional)
4. **jarvis_dft_2d_optb88** - JARVIS 2D (OptB88vdW functional)
5. **jarvis_dft_2d_tbmbj** - JARVIS 2D (TBMBJ functional)
6. **matminer_exp_bandgap** - Experimental band gaps

## 📊 데이터셋 형식

### CSV 파일 구조
```csv
id,structure_file,band_gap
mp-1234,structures/mp-1234.cif,1.23
mp-5678,structures/mp-5678.cif,2.45
jv-1000,structures/jv-1000.vasp,0.89
```

**필수 컬럼:**
- `id`: 고유 식별자
- `structure_file`: CIF 또는 POSCAR 파일 경로 (절대 경로 또는 CSV 기준 상대 경로)
- `band_gap`: Target 값 (eV)

### 디렉토리 구조 예시
```
data/
├── mp_bandgap.csv
├── jarvis_3d_optb88.csv
├── matminer_exp_bandgap.csv
└── structures/
    ├── mp-1234.cif
    ├── mp-5678.cif
    └── jv-1000.vasp
```

## 🚀 사용 방법

### Step 1: 단일 Property 학습

각 데이터셋에 대해 개별적으로 ALIGNN 모델을 학습합니다.

```bash
# Materials Project 데이터셋 학습
python single_train.py \
    --csv_file data/mp_bandgap.csv \
    --data_dir data/ \
    --dataset_name "Materials Project" \
    --output_dir checkpoints/mp_bandgap \
    --batch_size 32 \
    --epochs 1000 \
    --learning_rate 1e-3 \
    --patience 100

# JARVIS 3D OptB88 학습
python single_train.py \
    --csv_file data/jarvis_3d_optb88.csv \
    --data_dir data/ \
    --dataset_name "JARVIS 3D OptB88" \
    --output_dir checkpoints/jarvis_3d_optb88 \
    --batch_size 32 \
    --epochs 1000 \
    --learning_rate 1e-3

# MatMiner Experimental 학습
python single_train.py \
    --csv_file data/matminer_exp_bandgap.csv \
    --data_dir data/ \
    --dataset_name "MatMiner Experimental" \
    --output_dir checkpoints/matminer_exp \
    --batch_size 16 \
    --epochs 1000 \
    --learning_rate 1e-3
```

**학습 후 생성되는 파일:**
- `best_model.pt` - 최고 성능 모델 체크포인트
- `best_model_info.pkl` - 모델 정보 (데이터셋, 설정, 성능)
- `split_indices.pkl` - Train/Val/Test 분할 인덱스
- `training_log.csv` - 에포크별 학습 로그
- `test_results.txt` - 최종 테스트 결과

### Step 2: 사전 학습된 모델 업로드

이미 학습된 `.pt` 파일이 있다면 바로 사용 가능합니다.

```bash
# 체크포인트 디렉토리에 복사
mkdir -p checkpoints/pretrained
cp /path/to/your/model.pt checkpoints/pretrained/mp_bandgap.pt
cp /path/to/another/model.pt checkpoints/pretrained/jarvis_3d.pt
```

### Step 3: MOE 통합 학습

여러 사전 학습된 모델을 MOE로 통합합니다.

```bash
# 6개 expert 모델로 MOE 학습 (실험 데이터셋에 fine-tuning)
python moe_train.py \
    --checkpoint_paths \
        checkpoints/mp_bandgap/best_model.pt \
        checkpoints/jarvis_3d_optb88/best_model.pt \
        checkpoints/jarvis_3d_tbmbj/best_model.pt \
        checkpoints/jarvis_2d_optb88/best_model.pt \
        checkpoints/jarvis_2d_tbmbj/best_model.pt \
        checkpoints/matminer_exp/best_model.pt \
    --target_csv data/matminer_exp_bandgap.csv \
    --data_dir data/ \
    --dataset_name "MOE Experimental" \
    --output_dir checkpoints/moe_exp \
    --moe_option multiheaded \
    --num_heads 3 \
    --k_extractors 4 \
    --use_all_extractors \
    --batch_size 32 \
    --epochs 500 \
    --learning_rate 1e-4 \
    --patience 50
```

**MOE 옵션:**
- `multiheaded`: Multi-head MOE (권장)
  - `--num_heads`: MOE 헤드 개수 (default: 3)
  - `--k_extractors`: 각 헤드가 사용할 top-k expert 개수 (default: 4)
  - `--use_all_extractors`: 모든 expert를 soft weighting으로 사용
- `ensemble`: Simple weighted ensemble

**학습 전략:**
- `--finetune_extractors`: Expert 파라미터도 함께 fine-tune (기본: freeze)
- `--reg_weight`: Head diversity regularization 가중치 (default: 0.01)

## 🔧 주요 파라미터

### ALIGNN 모델 설정
```bash
--alignn_layers 4           # ALIGNN 레이어 개수
--gcn_layers 4              # GCN 레이어 개수
--hidden_features 256       # Hidden 차원
--embedding_features 64     # Embedding 차원
--dropout 0.1               # Dropout rate
```

### 그래프 구성
```bash
--max_neighbors 12          # 최대 이웃 원자 개수
--cutoff 8.0                # Cutoff radius (Angstrom)
```

### 학습 설정
```bash
--batch_size 32             # 배치 크기
--epochs 1000               # 최대 에포크
--learning_rate 1e-3        # 학습률
--weight_decay 1e-5         # Weight decay
--patience 100              # Early stopping patience
```

### 데이터 분할
```bash
--train_ratio 0.8           # 학습 데이터 비율
--val_ratio 0.1             # 검증 데이터 비율
--test_ratio 0.1            # 테스트 데이터 비율
--seed 42                   # Random seed
--split_file path.pkl       # 사전 정의된 분할 사용
```

## 📈 결과 확인

### 학습 로그
```python
import pandas as pd

# 학습 과정 확인
log = pd.read_csv('checkpoints/mp_bandgap/training_log.csv')
print(log.tail())

# 최고 성능 확인
best_epoch = log.loc[log['val_mae'].idxmin()]
print(f"Best epoch: {best_epoch['epoch']}")
print(f"Best val MAE: {best_epoch['val_mae']:.4f}")
```

### 모델 로딩 및 예측
```python
import torch
from alignn.model import create_alignn_model
from alignn.data import BandGapDataset
from alignn.utils import Normalizer

# 모델 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/mp_bandgap/best_model.pt', map_location=device)

model = create_alignn_model(config_dict=checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Normalizer 로딩
normalizer = Normalizer()
normalizer.load_state_dict(checkpoint['normalizer_state_dict'])

# 예측
# ... (데이터 로더에서 배치 가져오기)
with torch.no_grad():
    output = model(g, lg)
    prediction = normalizer.denorm(output)
```

## 🔬 MOE 작동 원리

### 1. MixtureOfExtractors
여러 ALIGNN extractor를 결합:
- **pairwise_TL**: 단일 extractor + 새로운 head
- **concat**: 모든 extractor 출력을 연결 + 학습 가능한 스케일
- **add_k**: Top-k gating으로 최적 expert 선택

### 2. MultiheadedMixtureOfExpertsModel
- 여러 MOE 헤드가 독립적으로 expert 선택
- 각 헤드의 출력을 결합하여 최종 예측
- Regularization으로 헤드 간 다양성 유지

### 3. Transfer Learning 전략
1. 대규모 DFT 데이터셋 (MP, JARVIS)으로 사전 학습
2. Expert로 고정 (freeze) 또는 fine-tune
3. 소규모 실험 데이터셋에 MOE 학습

## 📝 예제 워크플로우

```bash
# 1. 6개 데이터셋 각각 학습
for dataset in mp jarvis_3d_opt jarvis_3d_tbm jarvis_2d_opt jarvis_2d_tbm matminer_exp
do
    python single_train.py \
        --csv_file data/${dataset}_bandgap.csv \
        --output_dir checkpoints/${dataset} \
        --batch_size 32 \
        --epochs 1000
done

# 2. MOE 통합 (실험 데이터에 적용)
python moe_train.py \
    --checkpoint_paths checkpoints/*/best_model.pt \
    --target_csv data/matminer_exp_bandgap.csv \
    --output_dir checkpoints/moe_final \
    --moe_option multiheaded \
    --num_heads 3 \
    --use_all_extractors \
    --epochs 500

# 3. 결과 비교
echo "Single model (experimental only):"
cat checkpoints/matminer_exp/test_results.txt

echo "\nMOE model:"
cat checkpoints/moe_final/test_results.txt
```

## 💡 Tips

1. **데이터 크기 불균형**: 대규모 데이터셋 (MP, JARVIS)으로 먼저 학습하고, 소규모 데이터셋 (실험 데이터)에 MOE 적용

2. **메모리 부족 시**: `--batch_size` 줄이기, `--num_workers` 조정

3. **과적합 방지**: `--dropout` 증가, `--weight_decay` 증가, early stopping 사용

4. **MOE 헤드 수 조정**:
   - Expert 개수가 많으면 `--num_heads` 증가
   - `--k_extractors`는 보통 expert 수의 50-70%

5. **학습 속도**:
   - Single training: 학습률 1e-3, 긴 patience (100+)
   - MOE training: 학습률 1e-4, 짧은 patience (50)

## 🐛 문제 해결

### Import Error
```bash
# ALIGNN 설치 확인
pip install alignn

# DGL 설치 확인
pip install dgl-cu117  # CUDA 11.7
# 또는
pip install dgl        # CPU only
```

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
--batch_size 16

# 모델 크기 줄이기
--hidden_features 128
--alignn_layers 2
--gcn_layers 2
```

### 학습이 수렴하지 않음
```bash
# 학습률 조정
--learning_rate 5e-4

# Normalizer 확인
# 출력에서 "Target mean: X, std: Y" 확인
# std가 너무 작거나 크면 데이터 확인 필요
```

## 📚 참고 자료

- [ALIGNN Paper](https://www.nature.com/articles/s41524-021-00650-1)
- [CGCNN MOE Paper](https://arxiv.org/abs/2203.08372)
- [Original CGCNN MOE Repo](https://github.com/rees-c/MoE)

## 📧 문의

이슈가 있으면 GitHub Issues에 등록해주세요.
