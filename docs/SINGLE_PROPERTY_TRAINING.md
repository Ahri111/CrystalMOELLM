# Property-Specific Training Guide

## Single Property Training (권장!)

각 property별로 순차적으로 학습하는 방식입니다.

### Why Single Property?

**문제점 (Multi-property 동시 학습):**
- Batch에서 property filtering 발생 → 샘플 낭비
- Effective batch size 감소 (32 → 8-15개)
- Property간 균형 어려움

**장점 (Single property 학습):**
- ✅ Batch 효율 100% (filtering 불필요)
- ✅ Property-specific fine-tuning 가능
- ✅ Curriculum learning 가능 (쉬운 것 → 어려운 것)
- ✅ 안정적인 학습

---

## Training Examples

### 1. Band Gap (쉬움, 시작하기 좋음)

```bash
python scripts/train_moe_stage2.py \
    --filename moe_stage2_bandgap \
    --target_property band_gap \
    --use_moe \
    --moe_extractor_dir checkpoints/extractors \
    --moe_topk_config config/moe_topk.json \
    --batch_size 32 \
    --max_epochs 50 \
    --devices 0,1
```

**Expected Output:**
```
[CrystalDataset] Loaded 8,234 samples for 'band_gap' from data/splits/train.json
[CrystalDataset] Loaded 1,029 samples for 'band_gap' from data/splits/val.json

Epoch 50/50
Train - Loss: 1.856, ITC: 0.732, LM: 1.025, Property: 0.099
Val   - Loss: 1.912, ITC: 0.758, LM: 1.042, Property: 0.112

Best Val Property MAE: 0.112 eV
```

---

### 2. Formation Energy (중간 난이도)

```bash
python scripts/train_moe_stage2.py \
    --filename moe_stage2_formation \
    --target_property formation_energy \
    --init_checkpoint all_checkpoints/moe_stage2_bandgap/best.ckpt \  # Transfer learning!
    --use_moe \
    --batch_size 32 \
    --max_epochs 30 \
    --lr 5e-5  # Lower LR for fine-tuning
```

---

### 3. Mechanical Properties (그룹 학습)

여러 관련 property를 함께 학습할 수도 있습니다 (하지만 여전히 하나씩 추천):

```bash
# Option A: 하나씩 (권장)
for prop in bulk_modulus shear_modulus elastic_anisotropy poisson_ratio; do
    python scripts/train_moe_stage2.py \
        --filename moe_stage2_${prop} \
        --target_property ${prop} \
        --init_checkpoint all_checkpoints/moe_stage2_formation/best.ckpt \
        --batch_size 32 \
        --max_epochs 30
done

# Option B: Multi-property (덜 권장)
# target_property를 지정하지 않으면 여러 property 혼합
python scripts/train_moe_stage2.py \
    --filename moe_stage2_mechanical \
    --batch_size 32  # Effective size는 더 작음
```

---

## Curriculum Learning

쉬운 것부터 어려운 것 순서로:

```bash
# Stage 1: Band Gap (가장 쉬움, 데이터 많음)
python scripts/train_moe_stage2.py \
    --target_property band_gap \
    --max_epochs 50

# Stage 2: Mechanical Properties (중간)
python scripts/train_moe_stage2.py \
    --target_property bulk_modulus \
    --init_checkpoint all_checkpoints/moe_stage2_bandgap/best.ckpt \
    --max_epochs 30

# Stage 3: Formation Energy (어려움)
python scripts/train_moe_stage2.py \
    --target_property formation_energy \
    --init_checkpoint all_checkpoints/moe_stage2_bulk/best.ckpt \
    --max_epochs 30
```

---

## Property별 특성

| Property | Difficulty | Data Size | Recommended Epochs |
|----------|-----------|-----------|-------------------|
| band_gap | Easy | ~10k | 50 |
| n_Egap / p_Egap | Easy | ~8k | 50 |
| bulk_modulus | Medium | ~6k | 30 |
| shear_modulus | Medium | ~6k | 30 |
| formation_energy | Hard | ~12k | 30 |
| total_magnetization | Hard | ~4k | 40 |

---

## Transfer Learning

이전 property checkpoint에서 시작:

```bash
# Band gap 학습 후
python scripts/train_moe_stage2.py \
    --target_property formation_energy \
    --init_checkpoint all_checkpoints/moe_stage2_bandgap/last.ckpt \
    --lr 5e-5  # 더 낮은 LR
    --max_epochs 30
```

---

## Monitoring

### 학습 로그 확인

```bash
# Tensorboard (if using)
tensorboard --logdir all_checkpoints/moe_stage2_bandgap

# CSV logs
cat all_checkpoints/moe_stage2_bandgap/lightning_logs/version_0/metrics.csv
```

### Expected Metrics

```
epoch,train_loss,train_loss_itc,train_loss_lm,train_loss_property,val_loss
0,3.245,1.234,1.856,0.155,3.312
10,2.156,0.856,1.145,0.155,2.234
50,1.856,0.732,1.025,0.099,1.912
```

---

## Comparison: Single vs Multi-Property

### Multi-Property Mode (Not Recommended)

```bash
python scripts/train_moe_stage2.py \
    --batch_size 32  # Actual: ~10-15 samples per batch
    --max_epochs 100  # Needs more epochs
```

**Issues:**
- 24/32 samples filtered out per batch
- Slower convergence
- Imbalanced property learning

### Single-Property Mode (Recommended)

```bash
python scripts/train_moe_stage2.py \
    --target_property band_gap \
    --batch_size 32  # Actual: 32 samples!
    --max_epochs 50  # Faster convergence
```

**Benefits:**
- ✅ 100% batch efficiency
- ✅ Faster convergence
- ✅ Better property-specific learning

---

## Summary

**Best Practice:**
1. Train one property at a time with `--target_property`
2. Start with easy properties (band_gap)
3. Use transfer learning for harder properties
4. Monitor property-specific metrics

**Quick Start:**
```bash
# Just specify target_property!
python scripts/train_moe_stage2.py --target_property band_gap
```
