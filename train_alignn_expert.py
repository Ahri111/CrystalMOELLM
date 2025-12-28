#!/usr/bin/env python3
"""
Phase 1: Train ALIGNN Expert for specific property
각 물성별로 개별 ALIGNN 모델 학습
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
from alignn.models.alignn import ALIGNN
from tqdm import tqdm
import numpy as np


class PropertyDataset(Dataset):
    """Dataset for single property prediction"""

    def __init__(self, data_path, target_property):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.target_property = target_property

        # Filter samples with target property
        self.data = [d for d in self.data if target_property in d['properties']]

        print(f"Loaded {len(self.data)} samples with {target_property}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load graph
        g, _ = dgl.load_graphs(sample['graph_path'])
        g = g[0]

        # Get property value
        target = sample['properties'][self.target_property]

        return g, torch.tensor([target], dtype=torch.float32)


def collate_fn(batch):
    """Collate function for graph batching"""
    graphs, targets = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batched_targets = torch.cat(targets)
    return batched_graph, batched_targets


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mae = 0
    count = 0

    for graphs, targets in tqdm(loader, desc="Training"):
        graphs = graphs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(graphs)

        # Loss
        loss = criterion(outputs.squeeze(), targets)

        # Backward
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item() * len(targets)
        total_mae += torch.abs(outputs.squeeze() - targets).sum().item()
        count += len(targets)

    return total_loss / count, total_mae / count


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_mae = 0
    count = 0

    with torch.no_grad():
        for graphs, targets in tqdm(loader, desc="Validating"):
            graphs = graphs.to(device)
            targets = targets.to(device)

            # Forward
            outputs = model(graphs)

            # Loss
            loss = criterion(outputs.squeeze(), targets)

            # Stats
            total_loss += loss.item() * len(targets)
            total_mae += torch.abs(outputs.squeeze() - targets).sum().item()
            count += len(targets)

    return total_loss / count, total_mae / count


def train_expert(args):
    """Main training function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = PropertyDataset(args.train_data, args.target_property)
    val_dataset = PropertyDataset(args.val_data, args.target_property)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    # Create model
    model = ALIGNN(
        alignn_layers=args.alignn_layers,
        gcn_layers=args.gcn_layers,
        atom_input_features=92,
        edge_input_features=128,
        triplet_input_features=40,
        embedding_features=args.hidden_features,
        hidden_features=args.hidden_features,
        output_features=1,
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = nn.L1Loss()

    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        # Scheduler
        scheduler.step(val_mae)

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'property_name': args.target_property,
                'val_mae': val_mae,
                'model_config': {
                    'alignn_layers': args.alignn_layers,
                    'gcn_layers': args.gcn_layers,
                    'hidden_features': args.hidden_features,
                }
            }

            output_path = Path(args.output_dir) / f"alignn_{args.target_property}.pt"
            torch.save(checkpoint, output_path)
            print(f"Saved best model: {output_path}")

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining completed. Best Val MAE: {best_val_mae:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ALIGNN expert for specific property')

    # Data
    parser.add_argument('--train_data', type=str, default='data/train.json')
    parser.add_argument('--val_data', type=str, default='data/val.json')
    parser.add_argument('--target_property', type=str, required=True,
                        help='Target property to predict')

    # Model
    parser.add_argument('--alignn_layers', type=int, default=4)
    parser.add_argument('--gcn_layers', type=int, default=4)
    parser.add_argument('--hidden_features', type=int, default=256)

    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/experts')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Train
    train_expert(args)
