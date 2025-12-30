"""
Stage 3: MOE Training Script
Combines multiple expert heads trained in Stage 2 into a Mixture of Experts model
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

from model.blip2qformer_bandgap import Blip2QformerBandgap
from data_provider.bandgap_dataset import BandGapDataset, bandgap_collate_fn


class ExpertHead(nn.Module):
    """Single expert head for band gap prediction"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=1, dropout=0.1):
        super(ExpertHead, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        out = self.fc3(x)
        return out


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts for band gap prediction
    Combines multiple expert heads with multi-headed gating mechanism
    """

    def __init__(
        self,
        expert_heads,
        num_heads=3,
        k_experts=4,
        input_dim=768,
    ):
        """
        Args:
            expert_heads: List of expert head modules
            num_heads: Number of MOE pseudo-attention heads
            k_experts: Number of top experts to select per head
            input_dim: Input feature dimension (from Q-Former)
        """
        super(MixtureOfExperts, self).__init__()

        self.expert_heads = nn.ModuleList(expert_heads)
        self.num_experts = len(expert_heads)
        self.num_heads = num_heads
        self.k_experts = min(k_experts, self.num_experts)
        self.input_dim = input_dim

        # Freeze expert heads (only train gating mechanism)
        for expert in self.expert_heads:
            for param in expert.parameters():
                param.requires_grad = False

        # Create multiple MOE heads with independent gating
        self.expert_scores = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_experts))
            for _ in range(num_heads)
        ])

        # Final combination layer
        self.final_fc = nn.Linear(num_heads, 1)

    def forward(self, mask_embeddings):
        """
        Forward pass through MOE

        Args:
            mask_embeddings: [MASK] embeddings from Q-Former [batch_size, input_dim]

        Returns:
            predictions: Final predictions [batch_size, 1]
            loss_reg: Regularization loss
        """
        batch_size = mask_embeddings.size(0)

        # Get predictions from all experts [num_experts, batch_size, 1]
        expert_predictions = []
        for expert in self.expert_heads:
            pred = expert(mask_embeddings)
            expert_predictions.append(pred)
        expert_predictions = torch.stack(expert_predictions, dim=0)

        # Process each MOE head
        head_predictions = []
        head_weights = []

        for head_idx in range(self.num_heads):
            scores = self.expert_scores[head_idx]

            # Select top-k experts (batch-wide selection)
            top_k_values, top_k_indices = torch.topk(scores, self.k_experts)
            top_k_probabilities = F.softmax(top_k_values, dim=0)

            # Weighted combination of top-k experts
            head_pred = torch.zeros(batch_size, 1, device=mask_embeddings.device)
            for i in range(self.k_experts):
                prob = top_k_probabilities[i]
                idx = top_k_indices[i]
                head_pred = head_pred + prob * expert_predictions[idx]

            head_predictions.append(head_pred)

            # Create sparse weight vector for regularization
            sparse_weights = torch.zeros(
                self.num_experts,
                device=mask_embeddings.device
            ).scatter_(0, top_k_indices, top_k_probabilities)
            head_weights.append(sparse_weights)

        # Stack head predictions: [batch_size, num_heads]
        head_predictions = torch.cat(head_predictions, dim=1)

        # Final combination
        final_predictions = self.final_fc(head_predictions)

        # Compute regularization loss (encourage diversity)
        loss_reg = self.compute_regularization_loss(head_weights)

        return final_predictions, loss_reg

    def compute_regularization_loss(self, head_weights):
        """
        Orthogonality regularization: ||score.T @ score - I||^2
        Forces different heads to select different experts

        Args:
            head_weights: List of weight tensors [num_experts] for each head

        Returns:
            loss_reg: Regularization loss
        """
        if len(head_weights) == 0:
            return torch.tensor(0.0, device='cpu')

        # Stack: [num_experts, num_heads]
        score_matrix = torch.stack(head_weights, dim=-1)

        # Gram matrix: [num_heads, num_heads]
        gram_matrix = torch.transpose(score_matrix, 0, 1) @ score_matrix

        # Identity matrix
        identity = torch.eye(self.num_heads, device=score_matrix.device)

        # Orthogonality loss
        loss_reg = torch.pow(torch.norm(gram_matrix - identity), 2)

        return loss_reg


def extract_mask_embeddings(qformer_model, batch, device):
    """
    Extract [MASK] embeddings from frozen Q-Former

    Args:
        qformer_model: Frozen Blip2QformerBandgap model
        batch: Data batch
        device: Device

    Returns:
        mask_embeddings: [MASK] token embeddings [batch_size, hidden_dim]
        band_gaps: Ground truth band gaps [batch_size]
    """
    graphs, lg_graphs, texts, band_gaps, sources = batch

    graphs = graphs.to(device)
    lg_graphs = lg_graphs.to(device)
    band_gaps = band_gaps.to(device)

    with torch.no_grad():
        # Get graph embeddings from ALIGNN
        graph_embeds = qformer_model.ln_graph(
            qformer_model.alignn_encoder.get_features(graphs, lg_graphs)
        )
        graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(device)

        # Get query tokens
        query_tokens = qformer_model.query_tokens.expand(graph_embeds.shape[0], -1, -1)

        # Q-Former forward pass
        query_output = qformer_model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_atts,
            return_dict=True,
        )

        # Extract [MASK] embeddings (pool query tokens)
        mask_embeddings = query_output.last_hidden_state[:, :query_tokens.size(1), :].mean(dim=1)

    return mask_embeddings, band_gaps


def train_epoch(moe_model, qformer_model, dataloader, optimizer, device, reg_weight=0.01):
    """Train one epoch"""
    moe_model.train()
    qformer_model.eval()

    total_loss = 0
    total_mse_loss = 0
    total_reg_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Extract [MASK] embeddings
        mask_embeddings, band_gaps = extract_mask_embeddings(qformer_model, batch, device)

        # Forward through MOE
        predictions, loss_reg = moe_model(mask_embeddings)

        # MSE loss
        loss_mse = F.mse_loss(predictions.squeeze(), band_gaps)

        # Total loss
        loss = loss_mse + reg_weight * loss_reg

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse_loss += loss_mse.item()
        total_reg_loss += loss_reg.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{loss_mse.item():.4f}',
            'reg': f'{loss_reg.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse_loss / len(dataloader)
    avg_reg = total_reg_loss / len(dataloader)

    return avg_loss, avg_mse, avg_reg


@torch.no_grad()
def validate(moe_model, qformer_model, dataloader, device):
    """Validate model"""
    moe_model.eval()
    qformer_model.eval()

    total_mse = 0
    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc='Validating'):
        # Extract [MASK] embeddings
        mask_embeddings, band_gaps = extract_mask_embeddings(qformer_model, batch, device)

        # Forward through MOE
        predictions, _ = moe_model(mask_embeddings)

        # MSE loss
        mse = F.mse_loss(predictions.squeeze(), band_gaps)
        total_mse += mse.item()

        all_predictions.extend(predictions.squeeze().cpu().numpy())
        all_targets.extend(band_gaps.cpu().numpy())

    avg_mse = total_mse / len(dataloader)
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))

    return avg_mse, mae


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Q-Former model (frozen)
    print(f"\nLoading Q-Former from {args.qformer_checkpoint}")
    qformer_checkpoint = torch.load(args.qformer_checkpoint, map_location='cpu')
    qformer_model = Blip2QformerBandgap(args)
    qformer_model.load_state_dict(qformer_checkpoint['model'], strict=False)
    qformer_model.to(device)
    qformer_model.eval()

    # Freeze Q-Former
    for param in qformer_model.parameters():
        param.requires_grad = False

    print("Q-Former loaded and frozen")

    # Load expert heads
    print(f"\nLoading {len(args.expert_checkpoints)} expert heads:")
    expert_heads = []

    for i, expert_path in enumerate(args.expert_checkpoints):
        print(f"  {i+1}. {expert_path}")

        # Create expert head
        expert_head = ExpertHead(
            input_dim=args.hidden_dim,
            hidden_dim=128,
            output_dim=1,
            dropout=0.1
        )

        # Load checkpoint
        checkpoint = torch.load(expert_path, map_location='cpu')
        expert_head.load_state_dict(checkpoint['expert_state_dict'])
        expert_head.to(device)
        expert_head.eval()

        expert_heads.append(expert_head)

    print(f"\nLoaded {len(expert_heads)} experts")

    # Create MOE model
    print(f"\nCreating MOE model:")
    print(f"  Num heads: {args.num_heads}")
    print(f"  K experts per head: {args.k_extractors}")

    moe_model = MixtureOfExperts(
        expert_heads=expert_heads,
        num_heads=args.num_heads,
        k_experts=args.k_extractors,
        input_dim=args.hidden_dim,
    )
    moe_model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in moe_model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")

    # Load dataset
    print(f"\nLoading dataset from {args.train_csv}")
    train_dataset = BandGapDataset(
        csv_file=args.train_csv,
        alignn_config={
            'alignn_layers': args.alignn_layers,
            'gcn_layers': args.gcn_layers,
            'hidden_features': args.hidden_features,
        }
    )

    val_dataset = BandGapDataset(
        csv_file=args.val_csv,
        alignn_config={
            'alignn_layers': args.alignn_layers,
            'gcn_layers': args.gcn_layers,
            'hidden_features': args.hidden_features,
        }
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=bandgap_collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=bandgap_collate_fn,
        num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Optimizer (only MOE parameters)
    optimizer = torch.optim.Adam(
        [p for p in moe_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Regularization weight: {args.reg_weight}")

    best_val_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss, train_mse, train_reg = train_epoch(
            moe_model, qformer_model, train_loader, optimizer, device, args.reg_weight
        )

        # Validate
        val_mse, val_mae = validate(moe_model, qformer_model, val_loader, device)

        # Update scheduler
        scheduler.step(val_mae)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, Reg: {train_reg:.4f}")
        print(f"  Val   - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae

            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, 'best_moe_model.pt')

            torch.save({
                'epoch': epoch,
                'model_state_dict': moe_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_mse': val_mse,
                'args': vars(args),
            }, save_path)

            print(f"  ✓ Saved best model (MAE: {val_mae:.4f})")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(f"Model saved to: {os.path.join(args.output_dir, 'best_moe_model.pt')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 3: MOE Training')

    # Model paths
    parser.add_argument('--qformer_checkpoint', type=str, required=True,
                        help='Path to Stage 1 Q-Former checkpoint')
    parser.add_argument('--expert_checkpoints', nargs='+', required=True,
                        help='Paths to Stage 2 expert .pt files')

    # Data
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Training CSV file')
    parser.add_argument('--val_csv', type=str, required=True,
                        help='Validation CSV file')

    # MOE architecture
    parser.add_argument('--num_heads', type=int, default=3,
                        help='Number of MOE pseudo-attention heads')
    parser.add_argument('--k_extractors', type=int, default=4,
                        help='Number of top experts per head')

    # Q-Former config (must match Stage 1)
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='Hidden dimension (Q-Former output)')
    parser.add_argument('--num_query_token', type=int, default=32,
                        help='Number of query tokens')

    # ALIGNN config (must match Stage 1)
    parser.add_argument('--alignn_layers', type=int, default=4,
                        help='Number of ALIGNN layers')
    parser.add_argument('--gcn_layers', type=int, default=4,
                        help='Number of GCN layers')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Hidden features in ALIGNN')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--reg_weight', type=float, default=0.01,
                        help='Regularization weight')

    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage3_moe',
                        help='Output directory')

    # Data loading
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    main(args)
