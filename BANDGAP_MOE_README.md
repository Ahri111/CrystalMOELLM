# Band Gap Prediction with ALIGNN-based MOE

CGCNN → ALIGNN으로 교체한 3단계 MOE 프레임워크

## 📊 데이터 소스 (6개)

1. **MP_band_gap** - Materials Project
2. **JARVIS_DFT_3D_OptB88vdW**
3. **JARVIS_DFT_3D_TBMBJ**
4. **JARVIS_DFT_2D_OptB88vdW**
5. **JARVIS_DFT_2D_TBMBJ**
6. **MatMiner_Experimental**

## 🗂️ CSV 형식

```csv
cif_string,robocrys_text,band_gap,source_label
"{dict}","Silicon crystallizes in...",1.14,"mp"
"{dict}","GaAs has zinc blende...",1.42,"jarvis_3d_optb88"
```

**필수 컬럼:**
- `cif_string`: CIF dictionary 형태
- `robocrys_text`: Robocrys 설명
- `band_gap`: Band gap 값 (eV)
- `source_label`: 데이터 소스

## 🚀 사용법

### **Stage 1: Q-Former Pre-training**

Modified BLIP-2 with ALIGNN encoder

```bash
python train_stage1_qformer.py \
    --train_csv data/train_all_sources.csv \
    --val_csv data/val_all_sources.csv \
    --output_dir checkpoints/stage1 \
    --batch_size 32 \
    --max_epochs 100 \
    --gtm \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256
```

**핵심:**
- ✅ ALIGNN encoder (frozen)
- ✅ ITC + ITM loss
- ✅ Masked prediction loss
- ❌ ITG loss 제거!

**Template:**
```
"{source}_bandgap is [MASK] eV. {robocrys_with_numbers_masked}"
```

---

### **Stage 2: Single Expert Training**

각 source별로 expert 따로 학습

```bash
# MP expert
python train_stage2_single.py \
    --qformer_checkpoint checkpoints/stage1/best_model.ckpt \
    --train_csv data/train_all_sources.csv \
    --val_csv data/val_all_sources.csv \
    --source_label mp \
    --output_dir checkpoints/stage2 \
    --epochs 100

# JARVIS 3D OptB88 expert
python train_stage2_single.py \
    --qformer_checkpoint checkpoints/stage1/best_model.ckpt \
    --train_csv data/train_all_sources.csv \
    --val_csv data/val_all_sources.csv \
    --source_label jarvis_3d_optb88 \
    --output_dir checkpoints/stage2 \
    --epochs 100

# ... 나머지 4개 source도 동일하게
```

**기능:**
- ALIGNN + Q-Former frozen
- [MASK] embedding → expert head
- **.pt 파일 저장**
- **기존 .pt 업로드 지원** (`--expert_checkpoint`)

**생성 파일:**
```
checkpoints/stage2/
├── mp_expert.pt
├── jarvis_3d_optb88_expert.pt
├── jarvis_3d_tbmbj_expert.pt
├── jarvis_2d_optb88_expert.pt
├── jarvis_2d_tbmbj_expert.pt
└── matminer_exp_expert.pt
```

---

### **Stage 3: MOE 통합**

6개의 expert를 Multi-headed MOE로 통합

```bash
python train_stage3_moe.py \
    --qformer_checkpoint checkpoints/stage1/best_model.ckpt \
    --expert_checkpoints \
        checkpoints/stage2/mp_expert.pt \
        checkpoints/stage2/jarvis_3d_optb88_expert.pt \
        checkpoints/stage2/jarvis_3d_tbmbj_expert.pt \
        checkpoints/stage2/jarvis_2d_optb88_expert.pt \
        checkpoints/stage2/jarvis_2d_tbmbj_expert.pt \
        checkpoints/stage2/matminer_exp_expert.pt \
    --train_csv data/train_all_sources.csv \
    --val_csv data/val_all_sources.csv \
    --output_dir checkpoints/stage3_moe \
    --num_heads 3 \
    --k_extractors 4 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --reg_weight 0.01
```

**핵심:**
- ✅ Q-Former frozen (Stage 1에서 로드)
- ✅ Expert heads frozen (Stage 2에서 로드)
- ✅ Multi-headed gating mechanism 학습
- ✅ Orthogonality regularization
- ✅ Top-k expert selection per head

**작동 원리:**
1. Frozen Q-Former에서 [MASK] embeddings 추출
2. 각 MOE head가 독립적으로 top-k experts 선택
3. Head별로 weighted combination
4. 모든 head 결과를 final layer로 결합
5. Regularization으로 head 간 diversity 강제

## 📁 파일 구조

```
CrystalMOELLM/
├── model/
│   ├── blip2qformer_bandgap.py      # Modified BLIP-2 with ALIGNN
│   └── ...
├── data_provider/
│   ├── bandgap_dataset.py           # CSV 데이터 로더 + Robocrys processor
│   └── ...
├── MoE/                              # MOE 프레임워크
│   ├── alignn/                       # ALIGNN wrapper
│   │   ├── model.py                  # ALIGNNRegression, ALIGNNExtractor
│   │   ├── data.py                   # ALIGNN data loader
│   │   └── utils.py
│   ├── moe/                          # MOE models
│   │   ├── model.py                  # MixtureOfExtractors, MultiheadedMOE
│   │   └── utils.py                  # load_pretrained_extractors 등
│   ├── single_train.py               # ALIGNN single model training
│   └── moe_train.py                  # MOE training (standalone)
├── train_stage1_qformer.py          # Stage 1: Q-Former pre-training
├── train_stage2_single.py           # Stage 2: Single expert training
├── train_stage3_moe.py              # Stage 3: MOE integration ✓ NEW
└── BANDGAP_MOE_README.md            # 이 파일
```

## 🔑 핵심 변경사항

| 항목 | Original | Modified |
|------|----------|----------|
| Encoder | CGCNN/GIN | **ALIGNN** |
| Stage 1 Loss | ITC + ITM + ITG | **ITC + ITM + Masked** |
| Stage 2 | LLM fine-tuning | **Expert head training** |
| Stage 3 | Generalist LLM | **MOE** |
| Text | Raw description | **Masked template** |

## 💡 Template 설명

**원본 Robocrys:**
```
"Silicon crystallizes in a cubic structure with lattice parameter 5.43 angstrom."
```

**숫자 masking:**
```
"Silicon crystallizes in a cubic structure with lattice parameter <num> angstrom."
```

**최종 template:**
```
"mp_bandgap is [MASK] eV. Silicon crystallizes in a cubic structure with lattice parameter <num> angstrom."
```

**Q-Former가 학습:**
- [MASK] token에서 band gap 예측
- 구조 정보 (robocrys) 활용
- Source 정보 (mp, jarvis 등) 활용

## 📝 TODO

- [x] Stage 1 Q-Former training script (train_stage1_qformer.py) ✓
- [x] Stage 2 Single expert training script (train_stage2_single.py) ✓
- [x] Stage 3 MOE training script (train_stage3_moe.py) ✓
- [x] CSV 데이터 로더 + Robocrys processor ✓
- [ ] Inference 스크립트 (추론 전용)
- [ ] 6개 데이터 소스 CSV 생성
- [ ] 성능 평가 및 분석 도구

## 🐛 문제 해결

**Import Error:**
```bash
# ALIGNN 설치
pip install alignn

# DGL 설치
pip install dgl
```

**CUDA OOM:**
```bash
# 배치 크기 줄이기
--batch_size 16

# Precision 낮추기
--precision 16-mixed
```

## 📚 참고

- Original 3D-MoLM: https://github.com/lsh0520/3D-MoLM
- ALIGNN: https://github.com/usnistgov/alignn
- BLIP-2: https://github.com/salesforce/LAVIS
