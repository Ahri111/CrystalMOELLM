"""
Utility functions for ALIGNN training and evaluation
"""

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Normalizer:
    """
    Normalize and denormalize target values
    """

    def __init__(self, tensor=None):
        """
        Args:
            tensor: Tensor to compute mean and std from
        """
        if tensor is not None:
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)
        else:
            self.mean = 0.0
            self.std = 1.0

    def norm(self, tensor):
        """Normalize tensor"""
        return (tensor - self.mean) / self.std

    def denorm(self, tensor):
        """Denormalize tensor"""
        return tensor * self.std + self.mean

    def state_dict(self):
        """Save state"""
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        """Load state"""
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, train_loader, criterion, optimizer, device, normalizer=None):
    """
    Train for one epoch

    Args:
        model: ALIGNN model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cpu/cuda)
        normalizer: Target normalizer

    Returns:
        avg_loss: Average loss
        avg_mae: Average MAE
    """
    model.train()

    losses = AverageMeter()
    maes = AverageMeter()

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
        output = model(g, lg)
        loss = criterion(output.squeeze(), target_norm)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Denormalize for metrics
        if normalizer is not None:
            output_denorm = normalizer.denorm(output.squeeze())
        else:
            output_denorm = output.squeeze()

        # Compute MAE
        mae = torch.mean(torch.abs(output_denorm - target)).item()

        # Update meters
        losses.update(loss.item(), target.size(0))
        maes.update(mae, target.size(0))

    return losses.avg, maes.avg


def validate(model, val_loader, criterion, device, normalizer=None):
    """
    Validate model

    Args:
        model: ALIGNN model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        normalizer: Target normalizer

    Returns:
        avg_loss: Average loss
        avg_mae: Average MAE
        predictions: List of predictions
        targets: List of targets
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
            output = model(g, lg)
            loss = criterion(output.squeeze(), target_norm)

            # Denormalize
            if normalizer is not None:
                output_denorm = normalizer.denorm(output.squeeze())
            else:
                output_denorm = output.squeeze()

            # Compute MAE
            mae = torch.mean(torch.abs(output_denorm - target)).item()

            # Update meters
            losses.update(loss.item(), target.size(0))
            maes.update(mae, target.size(0))

            # Store predictions
            all_predictions.extend(output_denorm.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    return losses.avg, maes.avg, all_predictions, all_targets


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics

    Args:
        predictions: List of predictions
        targets: List of targets

    Returns:
        metrics: Dictionary of metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
    }

    return metrics


def save_checkpoint(state, filename='checkpoint.pt'):
    """
    Save checkpoint

    Args:
        state: State dict containing model, optimizer, etc.
        filename: Checkpoint filename
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, model, optimizer=None, device='cpu'):
    """
    Load checkpoint

    Args:
        filename: Checkpoint filename
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to

    Returns:
        start_epoch: Epoch to resume from
        best_val_loss: Best validation loss
    """
    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"Checkpoint loaded from {filename} (epoch {start_epoch})")

    return start_epoch, best_val_loss
