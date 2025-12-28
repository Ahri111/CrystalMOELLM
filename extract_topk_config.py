#!/usr/bin/env python3
"""
Phase 2: Extract Top-k Configuration
MoE 결과에서 Top-k 정보만 추출하여 JSON으로 저장
"""

import argparse
import json
from pathlib import Path
import torch


def extract_topk_config(moe_dir, output_path):
    """
    Extract top-k expert configuration from MoE checkpoints

    Args:
        moe_dir: Directory containing moe_*.pt files
        output_path: Output JSON path
    """

    moe_dir = Path(moe_dir)

    # Property names
    property_names = [
        'band_gap', 'formation_energy', 'bulk_modulus', 'shear_modulus',
        'elastic_anisotropy', 'poisson_ratio', 'total_magnetization',
        'n_Egap', 'p_Egap', 'n_mass', 'p_mass', 'eij_max'
    ]

    topk_config = {}

    # Process each property
    for prop in property_names:
        checkpoint_path = moe_dir / f"moe_{prop}.pt"

        if not checkpoint_path.exists():
            print(f"Warning: {checkpoint_path} not found, skipping...")
            continue

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract top-k info
        top_k_indices = checkpoint['top_k_indices']
        top_k_probs = checkpoint['top_k_probs']

        topk_config[prop] = {
            'indices': top_k_indices,
            'probs': top_k_probs
        }

        print(f"{prop}:")
        print(f"  Indices: {top_k_indices}")
        print(f"  Probs: {[f'{p:.4f}' for p in top_k_probs]}")

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(topk_config, f, indent=2)

    print(f"\nSaved top-k config to {output_path}")
    print(f"Total properties: {len(topk_config)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract top-k configuration from MoE checkpoints')

    parser.add_argument('--moe_dir', type=str, default='checkpoints/moe_downstream',
                        help='Directory containing MoE checkpoints')
    parser.add_argument('--output', type=str, default='config/moe_topk.json',
                        help='Output JSON path')

    args = parser.parse_args()

    extract_topk_config(args.moe_dir, args.output)
