#!/usr/bin/env python3
"""
Phase 0: CSV → Graph 변환
입력: CSV 파일 (material_id, cif, robocrys, 12개 물성)
출력: DGL graphs + train/val/test JSON splits
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import dgl
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from tqdm import tqdm


def cif_to_dgl_graph(cif_string):
    """Convert CIF string to DGL graph using JARVIS"""
    try:
        atoms = Atoms.from_cif(cif_string)
        jarvis_graph = Graph.atom_dgl_multigraph(atoms)

        # Convert to DGL graph
        g = dgl.graph((jarvis_graph.edges()[0], jarvis_graph.edges()[1]))

        # Add node features
        g.ndata['atom_features'] = jarvis_graph.ndata['atom_features']

        # Add edge features if available
        if 'r' in jarvis_graph.edata:
            g.edata['r'] = jarvis_graph.edata['r']

        return g
    except Exception as e:
        print(f"Error converting CIF: {e}")
        return None


def preprocess_data(csv_path, output_dir='data', train_ratio=0.8, val_ratio=0.1):
    """Main preprocessing function"""

    # Load CSV
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")

    # Property names (12개)
    property_names = [
        'band_gap', 'formation_energy', 'bulk_modulus', 'shear_modulus',
        'elastic_anisotropy', 'poisson_ratio', 'total_magnetization',
        'n_Egap', 'p_Egap', 'n_mass', 'p_mass', 'eij_max'
    ]

    # Create output directories
    graphs_dir = Path(output_dir) / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    processed_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting CIFs to graphs"):
        material_id = row['material_id']
        cif_string = row['cif']
        robocrys = row['robocrys']

        # Convert CIF to graph
        g = cif_to_dgl_graph(cif_string)
        if g is None:
            continue

        # Save graph
        graph_path = graphs_dir / f"{material_id}.bin"
        dgl.save_graphs(str(graph_path), [g])

        # Extract properties
        properties = {}
        for prop in property_names:
            if prop in row and pd.notna(row[prop]):
                properties[prop] = float(row[prop])

        # Skip if no valid properties
        if not properties:
            continue

        # Create data entry
        data_entry = {
            'id': material_id,
            'graph_path': f"data/graphs/{material_id}.bin",
            'properties': properties,
            'text': robocrys
        }
        processed_data.append(data_entry)

    print(f"Successfully processed {len(processed_data)} samples")

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(processed_data))

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_data = [processed_data[i] for i in train_indices]
    val_data = [processed_data[i] for i in val_indices]
    test_data = [processed_data[i] for i in test_indices]

    # Save splits
    output_path = Path(output_dir)

    with open(output_path / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_path / 'val.json', 'w') as f:
        json.dump(val_data, f, indent=2)

    with open(output_path / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\nSplit statistics:")
    print(f"Train: {len(train_data)} samples")
    print(f"Val: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")

    # Print property statistics
    print("\nProperty coverage:")
    for prop in property_names:
        count = sum(1 for d in processed_data if prop in d['properties'])
        print(f"{prop}: {count}/{len(processed_data)} ({100*count/len(processed_data):.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess crystal data to DGL graphs')
    parser.add_argument('--csv_path', type=str, default='data/raw_data.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for graphs and splits')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')

    args = parser.parse_args()

    preprocess_data(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
