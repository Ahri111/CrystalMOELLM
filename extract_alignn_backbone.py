#!/usr/bin/env python3
"""
Phase 2: Extract ALIGNN Backbone
ALIGNN Expert에서 Head 제거, Backbone만 추출하여 Extractor로 저장
"""

import argparse
from pathlib import Path
import torch
from model.alignn_extractor import ALIGNNExtractor


def extract_backbone(expert_dir, output_dir):
    """
    Extract backbone from ALIGNN expert checkpoints

    Args:
        expert_dir: Directory containing alignn_*.pt files
        output_dir: Output directory for extractors
    """

    expert_dir = Path(expert_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Property names
    property_names = [
        'band_gap', 'formation_energy', 'bulk_modulus', 'shear_modulus',
        'elastic_anisotropy', 'poisson_ratio', 'total_magnetization',
        'n_Egap', 'p_Egap', 'n_mass', 'p_mass', 'eij_max'
    ]

    # Process each expert
    for prop in property_names:
        expert_path = expert_dir / f"alignn_{prop}.pt"

        if not expert_path.exists():
            print(f"Warning: {expert_path} not found, skipping...")
            continue

        print(f"Processing {prop}...")

        # Load expert checkpoint
        checkpoint = torch.load(expert_path, map_location='cpu')

        # Create extractor
        extractor = ALIGNNExtractor.from_pretrained(str(expert_path))

        # Save extractor
        extractor_checkpoint = {
            'model_state_dict': extractor.state_dict(),
            'property_name': prop,
            'hidden_features': extractor.hidden_features,
            'source_checkpoint': str(expert_path),
        }

        output_path = output_dir / f"extractor_{prop}.pt"
        torch.save(extractor_checkpoint, output_path)

        print(f"  Saved to {output_path}")

    print(f"\nExtraction completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ALIGNN backbones from expert checkpoints')

    parser.add_argument('--expert_dir', type=str, default='checkpoints/experts',
                        help='Directory containing expert checkpoints')
    parser.add_argument('--output_dir', type=str, default='checkpoints/extractors',
                        help='Output directory for extractors')

    args = parser.parse_args()

    extract_backbone(args.expert_dir, args.output_dir)
