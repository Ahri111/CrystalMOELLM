"""
MOE (Mixture of Experts) Training Script

Usage:
    python moe_train.py \
        --checkpoint_paths checkpoints/mp/best_model.pt checkpoints/jarvis_3d/best_model.pt \
        --target_csv data/matminer_exp_bandgap.csv \
        --output_dir checkpoints/moe_exp

This script:
1. Loads multiple pretrained ALIGNN models as experts
2. Trains an MOE model to combine them
3. Fine-tunes on a target dataset (typically smaller/experimental data)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv as csv_module

from alignn.data import get_train_val_test_loader, load_indices_from_file
from alignn.utils import Normalizer, AverageMeter, compute_metrics
from moe.model import MultiheadedMixtureOfExpertsModel, EnsembleModel
from moe.utils import (
    load_pretrained_extractors,
    create_dataset_splits,
    save_split_indices,
    save_pretrained_model_info,
)


def train_epoch_moe(model, train_loader, criterion, optimizer, device, normalizer=None):
    """
    Train MOE model for one epoch

    Args:
        model: MOE model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        normalizer: Target normalizer

    Returns:
        avg_loss: Average loss
        avg_mae: Average MAE
    """
    model.train()

    losses = AverageMeter()
    maes = AverageMeter()

    # Regularization weight (원본 CGCNN MOE에서는 0.01로 고정)
    reg_weight = 0.01

    for batch_idx, (g, lg, target, _) in enumerate(train_loader):
        # Move to device
        g = g.to(device)
        lg = lg.to(device)
        target = target.to(device)

        # Normalize target
        if normalizer is not None:
            target_norm = normalizer.norm(target)
        else:
            target_norm = target

        # Forward pass
        if isinstance(model, MultiheadedMixtureOfExpertsModel):
            output, loss_reg = model(g, lg)
            output = output.squeeze()
        else:
            output = model(g, lg).squeeze()
            loss_reg = 0.0

        # Compute loss
        loss_pred = criterion(output, target_norm)
        loss = loss_pred + reg_weight * loss_reg

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Denormalize for metrics
        if normalizer is not None:
            output_denorm = normalizer.denorm(output)
        else:
            output_denorm = output

        # Compute MAE
        mae = torch.mean(torch.abs(output_denorm - target)).item()

        # Update meters
        losses.update(loss.item(), target.size(0))
        maes.update(mae, target.size(0))

    return losses.avg, maes.avg


def validate_moe(model, val_loader, criterion, device, normalizer=None):
    """
    Validate MOE model

    Args:
        model: MOE model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        normalizer: Target normalizer

    Returns:
        avg_loss, avg_mae, predictions, targets
    """
    model.eval()

    losses = AverageMeter()
    maes = AverageMeter()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for g, lg, target, _ in val_loader:
            # Move to device
            g = g.to(device)
            lg = lg.to(device)
            target = target.to(device)

            # Normalize target
            if normalizer is not None:
                target_norm = normalizer.norm(target)
            else:
                target_norm = target

            # Forward pass
            if isinstance(model, MultiheadedMixtureOfExpertsModel):
                output, _ = model(g, lg)
                output = output.squeeze()
            else:
                output = model(g, lg).squeeze()

            loss = criterion(output, target_norm)

            # Denormalize
            if normalizer is not None:
                output_denorm = normalizer.denorm(output)
            else:
                output_denorm = output

            # Compute MAE
            mae = torch.mean(torch.abs(output_denorm - target)).item()

            # Update meters
            losses.update(loss.item(), target.size(0))
            maes.update(mae, target.size(0))

            # Store predictions
            all_predictions.extend(output_denorm.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    return losses.avg, maes.avg, all_predictions, all_targets


def main(args):
    """Main MOE training function"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained extractors
    print("\n=== Loading Pretrained Extractors ===")

    model_config = {
        'alignn_layers': args.alignn_layers,
        'gcn_layers': args.gcn_layers,
        'hidden_features': args.hidden_features,
        'embedding_features': args.embedding_features,
    }

    extractors, feature_dim = load_pretrained_extractors(
        checkpoint_paths=args.checkpoint_paths,
        config_dict=model_config,
        freeze_extractors=not args.finetune_extractors,
        device=device,
    )

    # Create MOE model
    print(f"\n=== Creating MOE Model (option: {args.moe_option}) ===")

    if args.moe_option == 'multiheaded':
        model = MultiheadedMixtureOfExpertsModel(
            extractors=extractors,
            feature_dim=feature_dim,
            output_dim=1,
            num_heads=args.num_heads,
            k_extractors=args.k_extractors,
            hidden_dim=args.moe_hidden_dim,
        )
    elif args.moe_option == 'ensemble':
        model = EnsembleModel(
            extractors=extractors,
            feature_dim=feature_dim,
            output_dim=1,
        )
    else:
        raise ValueError(f"Unknown MOE option: {args.moe_option}")

    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load or create dataset splits
    if args.split_file:
        print(f"\n=== Loading splits from {args.split_file} ===")
        train_indices, val_indices, test_indices = load_indices_from_file(args.split_file)
    else:
        print("\n=== Creating new dataset splits ===")
        from alignn.data import BandGapDataset
        temp_dataset = BandGapDataset(args.target_csv, root_dir=args.data_dir)
        train_indices, val_indices, test_indices = create_dataset_splits(
            temp_dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        split_save_path = os.path.join(args.output_dir, 'split_indices.pkl')
        save_split_indices(train_indices, val_indices, test_indices, split_save_path)

    # Create dataloaders
    print("\n=== Creating Dataloaders ===")
    train_loader, val_loader, test_loader, dataset = get_train_val_test_loader(
        csv_file=args.target_csv,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        max_neighbors=args.max_neighbors,
        cutoff=args.cutoff,
    )

    # Create normalizer
    print("\n=== Computing normalization statistics ===")
    train_targets = []
    for _, _, target, _ in train_loader:
        train_targets.append(target)
    train_targets_tensor = torch.cat(train_targets)
    normalizer = Normalizer(train_targets_tensor)
    print(f"Target mean: {normalizer.mean:.4f}, std: {normalizer.std:.4f}")

    # Loss and optimizer
    criterion = nn.MSELoss()

    # Only optimize MOE parameters (not frozen extractors)
    if isinstance(model, MultiheadedMixtureOfExpertsModel):
        # Use non_extractor_parameters() for multiheaded MOE
        optimizer = optim.AdamW(
            model.non_extractor_parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        # For ensemble, filter trainable parameters
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.patience // 2,
        verbose=True,
    )

    # Training loop
    print("\n=== Starting MOE Training ===")
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    epochs_no_improve = 0

    csv_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv_module.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_mae', 'val_loss', 'val_mae', 'lr'])

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_mae = train_epoch_moe(
            model, train_loader, criterion, optimizer, device, normalizer
        )

        # Validate
        val_loss, val_mae, val_preds, val_targets = validate_moe(
            model, val_loader, criterion, device, normalizer
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv_module.writer(f)
            writer.writerow([epoch, train_loss, train_mae, val_loss, val_mae, current_lr])

        # Print progress
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            epochs_no_improve = 0

            val_metrics = compute_metrics(val_preds, val_targets)
            print(f"  *** New best MOE model! Val MAE: {val_mae:.4f}, RMSE: {val_metrics['rmse']:.4f}, R2: {val_metrics['r2']:.4f}")

            checkpoint_path = os.path.join(args.output_dir, 'best_moe_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'normalizer_state_dict': normalizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_mae': best_val_mae,
                'val_metrics': val_metrics,
                'args': vars(args),
            }, checkpoint_path)

            save_pretrained_model_info(
                checkpoint_path=checkpoint_path,
                dataset_name=args.dataset_name or os.path.basename(args.target_csv),
                config={'moe_option': args.moe_option, 'num_experts': len(args.checkpoint_paths)},
                metrics=val_metrics,
            )

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        print()

    # Final test evaluation
    print("\n=== Final Test Evaluation ===")

    checkpoint_path = os.path.join(args.output_dir, 'best_moe_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    normalizer.load_state_dict(checkpoint['normalizer_state_dict'])

    test_loss, test_mae, test_preds, test_targets = validate_moe(
        model, test_loader, criterion, device, normalizer
    )
    test_metrics = compute_metrics(test_preds, test_targets)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")
    print(f"Test MAPE: {test_metrics['mape']:.2f}%")

    results_path = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"MOE Test Results\n")
        f.write(f"================\n")
        f.write(f"Number of experts: {len(args.checkpoint_paths)}\n")
        f.write(f"MOE option: {args.moe_option}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"MAE: {test_mae:.4f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"R2: {test_metrics['r2']:.4f}\n")
        f.write(f"MAPE: {test_metrics['mape']:.2f}%\n")

    print(f"\nMOE training complete! Best model saved to {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MOE model for band gap prediction')

    # Pretrained models
    parser.add_argument('--checkpoint_paths', nargs='+', required=True,
                        help='Paths to pretrained model checkpoints')
    parser.add_argument('--finetune_extractors', action='store_true',
                        help='Fine-tune extractor parameters (default: freeze)')

    # Target dataset
    parser.add_argument('--target_csv', type=str, required=True,
                        help='Path to target dataset CSV')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory for structure files')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Name of target dataset')

    # Data split
    parser.add_argument('--split_file', type=str, default=None,
                        help='Path to pre-defined split indices')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    # Base model config (should match pretrained models)
    parser.add_argument('--alignn_layers', type=int, default=4)
    parser.add_argument('--gcn_layers', type=int, default=4)
    parser.add_argument('--hidden_features', type=int, default=256)
    parser.add_argument('--embedding_features', type=int, default=64)

    # MOE architecture
    parser.add_argument('--moe_option', type=str, default='multiheaded',
                        choices=['multiheaded', 'ensemble'],
                        help='MOE architecture type')
    parser.add_argument('--num_heads', type=int, default=3,
                        help='Number of MOE heads (for multiheaded)')
    parser.add_argument('--k_extractors', type=int, default=4,
                        help='Number of top extractors per head')
    parser.add_argument('--moe_hidden_dim', type=int, default=128,
                        help='Hidden dimension for MOE MLP head')

    # Graph construction
    parser.add_argument('--max_neighbors', type=int, default=12)
    parser.add_argument('--cutoff', type=float, default=8.0)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500,
                        help='Maximum epochs (default: 500 for MOE)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4 for MOE)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')

    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    args = parser.parse_args()

    main(args)
