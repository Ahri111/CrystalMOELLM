#!/bin/bash
# ALIGNN MoE + ZeroMAT 전체 파이프라인 실행 스크립트

set -e  # Exit on error

echo "======================================"
echo "ALIGNN MoE + ZeroMAT Pipeline"
echo "======================================"

# Property names (12개)
PROPERTIES=(
    "band_gap"
    "formation_energy"
    "bulk_modulus"
    "shear_modulus"
    "elastic_anisotropy"
    "poisson_ratio"
    "total_magnetization"
    "n_Egap"
    "p_Egap"
    "n_mass"
    "p_mass"
    "eij_max"
)

# ======================================
# Phase 0: Preprocessing
# ======================================
echo ""
echo "Phase 0: CSV → Graph Conversion"
echo "======================================"
python preprocess_data.py \
    --csv_path data/raw_data.csv \
    --output_dir data

# ======================================
# Phase 1: Train ALIGNN Experts
# ======================================
echo ""
echo "Phase 1: Training 12 ALIGNN Experts"
echo "======================================"

for prop in "${PROPERTIES[@]}"; do
    echo ""
    echo "Training Expert: $prop"
    echo "--------------------------------------"
    python train_alignn_expert.py \
        --target_property $prop \
        --train_data data/train.json \
        --val_data data/val.json \
        --epochs 300 \
        --batch_size 64 \
        --output_dir checkpoints/experts
done

# ======================================
# Phase 1.5: Train MoE Downstream
# ======================================
echo ""
echo "Phase 1.5: Training MoE for each property"
echo "======================================"

for prop in "${PROPERTIES[@]}"; do
    echo ""
    echo "Training MoE: $prop"
    echo "--------------------------------------"
    python train_moe_downstream.py \
        --target_property $prop \
        --train_data data/train.json \
        --val_data data/val.json \
        --expert_dir checkpoints/experts \
        --k_experts 3 \
        --epochs 100 \
        --batch_size 64 \
        --output_dir checkpoints/moe_downstream
done

# ======================================
# Phase 2: Extract Top-k Config & Extractors
# ======================================
echo ""
echo "Phase 2: Extracting Top-k Configuration"
echo "======================================"

# Extract top-k config
python extract_topk_config.py \
    --moe_dir checkpoints/moe_downstream \
    --output config/moe_topk.json

# Extract ALIGNN backbones
echo ""
echo "Extracting ALIGNN Backbones"
echo "--------------------------------------"
python extract_alignn_backbone.py \
    --expert_dir checkpoints/experts \
    --output_dir checkpoints/extractors

# ======================================
# Phase 3: Train ZeroMAT with Frozen MoE
# ======================================
echo ""
echo "Phase 3: Training ZeroMAT with Frozen MoE"
echo "======================================"

python train_stage3_zeromat.py \
    --config config/stage3_zeromat.yaml

# ======================================
# Done
# ======================================
echo ""
echo "======================================"
echo "Pipeline Completed Successfully!"
echo "======================================"
echo ""
echo "Outputs:"
echo "  - Phase 1: checkpoints/experts/"
echo "  - Phase 1.5: checkpoints/moe_downstream/"
echo "  - Phase 2: config/moe_topk.json, checkpoints/extractors/"
echo "  - Phase 3: checkpoints/stage3/"
