#!/usr/bin/env python3
"""
Phase 1.5: MoE Downstream Training (핵심!)
목적: 각 물성별로 어떤 Expert 조합이 best인지 결정
- 12개 Extractor: Frozen
- Gating weights: Learnable
- Top-k selection
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
from model.alignn_extractor import ALIGNNExtractor
from tqdm import tqdm
import numpy as np


class PropertyDataset(Dataset):
    """Dataset for single property prediction"""

    def __init__(self, data_path, target_property):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.target_property = target_property
        self.data = [d for d in self.data if target_property in d['properties']]

        print(f"Loaded {len(self.data)} samples with {target_property}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        g, _ = dgl.load_graphs(sample['graph_path'])
        g = g[0]
        target = sample['properties'][self.target_property]
        return g, torch.tensor([target], dtype=torch.float32)


def collate_fn(batch):
    graphs, targets = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batched_targets = torch.cat(targets)
    return batched_graph, batched_targets


class MoEDownstream(nn.Module):
    """
    MoE for downstream property prediction
    - 12 frozen extractors
    - Learnable gating weights
    - Top-k selection
    - Prediction head
    """

    def __init__(self, extractor_paths, k_experts=3, hidden_dim=256):
        super().__init__()

        self.k_experts = k_experts
        self.num_experts = len(extractor_paths)
        self.hidden_dim = hidden_dim

        # Load 12 frozen extractors
        self.extractors = nn.ModuleList()
        for path in extractor_paths:
            extractor = ALIGNNExtractor.from_pretrained(path)
            extractor.eval()
            for param in extractor.parameters():
                param.requires_grad = False
            self.extractors.append(extractor)

        # Learnable gating weights (초기값: uniform)
        self.gating_weights = nn.Parameter(
            torch.ones(self.num_experts) / self.num_experts
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, g):
        """
        Args:
            g: batched DGL graph

        Returns:
            prediction: [B, 1]
            top_k_indices: selected expert indices
            top_k_probs: normalized probabilities
        """
        batch_size = g.batch_size

        # Get gating probabilities
        gating_probs = F.softmax(self.gating_weights, dim=0)  # [num_experts]

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gating_probs, self.k_experts)
        top_k_probs = top_k_probs / top_k_probs.sum()  # Renormalize

        # Extract features from selected experts
        expert_features = []
        for idx in top_k_indices:
            with torch.no_grad():
                feats, _ = self.extractors[idx](g)  # [B, hidden_dim]
            expert_features.append(feats)

        # Stack and weighted sum
        expert_features = torch.stack(expert_features, dim=1)  # [B, k, hidden_dim]
        top_k_probs = top_k_probs.view(1, -1, 1)  # [1, k, 1]

        # Weighted combination
        combined_features = (expert_features * top_k_probs).sum(dim=1)  # [B, hidden_dim]

        # Prediction
        prediction = self.head(combined_features)  # [B, 1]

        return prediction, top_k_indices, top_k_probs.squeeze()


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
        outputs, _, _ = model(graphs)

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
            outputs, _, _ = model(graphs)

            # Loss
            loss = criterion(outputs.squeeze(), targets)

            # Stats
            total_loss += loss.item() * len(targets)
            total_mae += torch.abs(outputs.squeeze() - targets).sum().item()
            count += len(targets)

    return total_loss / count, total_mae / count


def train_moe(args):
    """Main training function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get all expert paths
    expert_dir = Path(args.expert_dir)
    property_names = [
        'band_gap', 'formation_energy', 'bulk_modulus', 'shear_modulus',
        'elastic_anisotropy', 'poisson_ratio', 'total_magnetization',
        'n_Egap', 'p_Egap', 'n_mass', 'p_mass', 'eij_max'
    ]

    extractor_paths = [
        str(expert_dir / f"alignn_{prop}.pt") for prop in property_names
    ]

    # Check all experts exist
    for path in extractor_paths:
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping...")

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

    # Create MoE model
    model = MoEDownstream(
        extractor_paths=extractor_paths,
        k_experts=args.k_experts,
        hidden_dim=256
    ).to(device)

    # Optimizer (only gating weights and head are trainable)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
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

        # Print current top-k experts
        with torch.no_grad():
            gating_probs = F.softmax(model.gating_weights, dim=0)
            top_k_probs, top_k_indices = torch.topk(gating_probs, args.k_experts)
            top_k_probs = top_k_probs / top_k_probs.sum()

            print(f"Current Top-{args.k_experts} Experts:")
            for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
                print(f"  {i+1}. Expert {idx.item()} ({property_names[idx.item()]}): {prob.item():.4f}")

        # Scheduler
        scheduler.step(val_mae)

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0

            # Get final top-k
            with torch.no_grad():
                gating_probs = F.softmax(model.gating_weights, dim=0)
                top_k_probs, top_k_indices = torch.topk(gating_probs, args.k_experts)
                top_k_probs = top_k_probs / top_k_probs.sum()

            # Save checkpoint
            checkpoint = {
                'property_name': args.target_property,
                'gating_weights': model.gating_weights.detach().cpu(),
                'top_k_indices': top_k_indices.detach().cpu().tolist(),  # 중요!
                'top_k_probs': top_k_probs.detach().cpu().tolist(),  # 중요!
                'head_state_dict': model.head.state_dict(),
                'val_mae': val_mae,
                'k_experts': args.k_experts,
            }

            output_path = Path(args.output_dir) / f"moe_{args.target_property}.pt"
            torch.save(checkpoint, output_path)
            print(f"Saved best model: {output_path}")

        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining completed. Best Val MAE: {best_val_mae:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoE downstream for property prediction')

    # Data
    parser.add_argument('--train_data', type=str, default='data/train.json')
    parser.add_argument('--val_data', type=str, default='data/val.json')
    parser.add_argument('--target_property', type=str, required=True)

    # Model
    parser.add_argument('--expert_dir', type=str, default='checkpoints/experts')
    parser.add_argument('--k_experts', type=int, default=3)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/moe_downstream')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_moe(args)
