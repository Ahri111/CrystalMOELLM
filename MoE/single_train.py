"""
Single Property Training Script for ALIGNN Band Gap Prediction

Usage:
    python single_train.py --csv_file data/mp_bandgap.csv --output_dir checkpoints/mp_bandgap

This script trains a single ALIGNN model on one dataset.
The trained model can later be used as an expert in the MOE framework.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv as csv_module

from alignn.data import get_train_val_test_loader, load_indices_from_file
from alignn.model import create_alignn_model
from alignn.utils import (
    Normalizer,
    train_epoch,
    validate,
    compute_metrics,
    save_checkpoint,
)
from moe.utils import (
    create_dataset_splits,
    save_split_indices,
    save_pretrained_model_info,
)


def main(args):
    """Main training function"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Model configuration
    model_config = {
        'alignn_layers': args.alignn_layers,
        'gcn_layers': args.gcn_layers,
        'hidden_features': args.hidden_features,
        'embedding_features': args.embedding_features,
        'output_features': 1,
        'dropout': args.dropout,
        'use_batch_norm': True,
    }

    # Create model
    print("\n=== Creating Model ===")
    model = create_alignn_model(config_dict=model_config)
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
        # Create temporary dataset to get size
        from alignn.data import BandGapDataset
        temp_dataset = BandGapDataset(args.csv_file, root_dir=args.data_dir)
        train_indices, val_indices, test_indices = create_dataset_splits(
            temp_dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        # Save splits
        split_save_path = os.path.join(args.output_dir, 'split_indices.pkl')
        save_split_indices(train_indices, val_indices, test_indices, split_save_path)

    # Create dataloaders
    print("\n=== Creating Dataloaders ===")
    train_loader, val_loader, test_loader, dataset = get_train_val_test_loader(
        csv_file=args.csv_file,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        max_neighbors=args.max_neighbors,
        cutoff=args.cutoff,
    )

    # Create normalizer from training data
    print("\n=== Computing normalization statistics ===")
    train_targets = []
    for _, _, target, _ in train_loader:
        train_targets.append(target)
    train_targets_tensor = torch.cat(train_targets)
    normalizer = Normalizer(train_targets_tensor)
    print(f"Target mean: {normalizer.mean:.4f}, std: {normalizer.std:.4f}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.patience // 2,
        verbose=True,
    )

    # Training loop
    print("\n=== Starting Training ===")
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    epochs_no_improve = 0

    # CSV logger
    csv_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv_module.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_mae', 'val_loss', 'val_mae', 'lr'])

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, normalizer
        )

        # Validate
        val_loss, val_mae, val_preds, val_targets = validate(
            model, val_loader, criterion, device, normalizer
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Get current learning rate
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

            # Compute detailed metrics
            val_metrics = compute_metrics(val_preds, val_targets)
            print(f"  *** New best model! Val MAE: {val_mae:.4f}, RMSE: {val_metrics['rmse']:.4f}, R2: {val_metrics['r2']:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'normalizer_state_dict': normalizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_mae': best_val_mae,
                'val_metrics': val_metrics,
                'config': model_config,
            }, filename=checkpoint_path)

            # Save model info
            save_pretrained_model_info(
                checkpoint_path=checkpoint_path,
                dataset_name=args.dataset_name or os.path.basename(args.csv_file),
                config=model_config,
                metrics=val_metrics,
            )

        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        print()

    # Final test evaluation
    print("\n=== Final Test Evaluation ===")

    # Load best model
    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    normalizer.load_state_dict(checkpoint['normalizer_state_dict'])

    # Test
    test_loss, test_mae, test_preds, test_targets = validate(
        model, test_loader, criterion, device, normalizer
    )
    test_metrics = compute_metrics(test_preds, test_targets)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")
    print(f"Test MAPE: {test_metrics['mape']:.2f}%")

    # Save test results
    results_path = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Results\n")
        f.write(f"============\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"MAE: {test_mae:.4f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"R2: {test_metrics['r2']:.4f}\n")
        f.write(f"MAPE: {test_metrics['mape']:.2f}%\n")

    print(f"\nTraining complete! Best model saved to {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ALIGNN for band gap prediction')

    # Data
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Path to CSV file with dataset')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory for structure files (default: CSV dir)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Name of dataset (for logging)')

    # Data split
    parser.add_argument('--split_file', type=str, default=None,
                        help='Path to pre-defined split indices (pickle file)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting (default: 42)')

    # Model
    parser.add_argument('--alignn_layers', type=int, default=4,
                        help='Number of ALIGNN layers (default: 4)')
    parser.add_argument('--gcn_layers', type=int, default=4,
                        help='Number of GCN layers (default: 4)')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Hidden feature dimension (default: 256)')
    parser.add_argument('--embedding_features', type=int, default=64,
                        help='Embedding dimension (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Graph construction
    parser.add_argument('--max_neighbors', type=int, default=12,
                        help='Maximum number of neighbors (default: 12)')
    parser.add_argument('--cutoff', type=float, default=8.0,
                        help='Cutoff radius in Angstrom (default: 8.0)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of epochs (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience (default: 100)')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')

    args = parser.parse_args()

    main(args)
