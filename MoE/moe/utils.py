"""
Utility functions for MOE training
"""

import torch
import os
import pickle
from ..alignn.model import ALIGNNRegression, ALIGNNExtractor


def get_alignn_parameters_to_finetune(extractor, num_layers_to_unfreeze):
    """
    Get parameters to fine-tune from ALIGNN extractor (unfreeze from the end)

    Args:
        extractor: ALIGNNExtractor
        num_layers_to_unfreeze: Number of layers to unfreeze from the end

    Returns:
        trainable_params: List of parameters to fine-tune
    """
    # First freeze all
    for p in extractor.parameters():
        p.requires_grad = False

    trainable_params = []
    layers_unfrozen = 0

    # Unfreeze from the end: GCN layers -> ALIGNN layers -> embeddings

    # 1. GCN layers (from the end)
    if hasattr(extractor.alignn, 'gcn_layers'):
        for i in range(len(extractor.alignn.gcn_layers) - 1, -1, -1):
            if layers_unfrozen >= num_layers_to_unfreeze:
                break

            gcn_layer = extractor.alignn.gcn_layers[i]
            trainable_params.extend(list(gcn_layer.parameters()))
            layers_unfrozen += 1
            print(f'    UNFROZE gcn_layers[{i}]')

    # 2. ALIGNN layers (from the end)
    if hasattr(extractor.alignn, 'alignn_layers'):
        for i in range(len(extractor.alignn.alignn_layers) - 1, -1, -1):
            if layers_unfrozen >= num_layers_to_unfreeze:
                break

            alignn_layer = extractor.alignn.alignn_layers[i]
            trainable_params.extend(list(alignn_layer.parameters()))
            layers_unfrozen += 1
            print(f'    UNFROZE alignn_layers[{i}]')

    # 3. Edge embedding
    if layers_unfrozen < num_layers_to_unfreeze:
        if hasattr(extractor.alignn, 'edge_embedding'):
            trainable_params.extend(list(extractor.alignn.edge_embedding.parameters()))
            layers_unfrozen += 1
            print(f'    UNFROZE edge_embedding')

    # 4. Atom embedding
    if layers_unfrozen < num_layers_to_unfreeze:
        if hasattr(extractor.alignn, 'atom_embedding'):
            trainable_params.extend(list(extractor.alignn.atom_embedding.parameters()))
            layers_unfrozen += 1
            print(f'    UNFROZE atom_embedding')

    # Set requires_grad
    for p in trainable_params:
        p.requires_grad = True

    return trainable_params


def load_pretrained_extractors(
    checkpoint_paths,
    config_dict=None,
    freeze_extractors=True,
    num_layers_to_unfreeze=0,
    device='cpu',
):
    """
    Load multiple pretrained ALIGNN models as extractors

    Args:
        checkpoint_paths: List of paths to pretrained .pt files
        config_dict: Configuration for ALIGNN model
        freeze_extractors: Whether to freeze extractor parameters
        num_layers_to_unfreeze: Number of layers to unfreeze from the end (if > 0)
        device: Device to load models to

    Returns:
        extractors: List of ALIGNNExtractor modules
        feature_dim: Feature dimension
    """
    extractors = []
    feature_dim = None

    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Loading extractor {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")

        # Create ALIGNN model
        config = config_dict or {}
        model = ALIGNNRegression(**config)

        # Load pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Create extractor (without final prediction layer)
        extractor = ALIGNNExtractor(model)
        extractor.to(device)

        if freeze_extractors:
            for param in extractor.parameters():
                param.requires_grad = False

            # Unfreeze some layers if requested
            if num_layers_to_unfreeze > 0:
                print(f"  Unfreezing {num_layers_to_unfreeze} layers:")
                get_alignn_parameters_to_finetune(extractor, num_layers_to_unfreeze)
            else:
                print(f"  Extractor {i+1} frozen")
        else:
            print(f"  Extractor {i+1} unfrozen (all layers trainable)")

        extractors.append(extractor)

        # Get feature dimension
        if feature_dim is None:
            feature_dim = extractor.feature_dim

    print(f"\nLoaded {len(extractors)} extractors with feature_dim={feature_dim}")

    return extractors, feature_dim


def save_pretrained_model_info(checkpoint_path, dataset_name, config, metrics):
    """
    Save information about a pretrained model

    Args:
        checkpoint_path: Path where checkpoint is saved
        dataset_name: Name of dataset used for training
        config: Model configuration
        metrics: Performance metrics
    """
    info = {
        'dataset': dataset_name,
        'config': config,
        'metrics': metrics,
        'checkpoint_path': checkpoint_path,
    }

    info_path = checkpoint_path.replace('.pt', '_info.pkl')

    with open(info_path, 'wb') as f:
        pickle.dump(info, f)

    print(f"Model info saved to {info_path}")


def load_pretrained_model_info(checkpoint_path):
    """
    Load information about a pretrained model

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        info: Dictionary with model information
    """
    info_path = checkpoint_path.replace('.pt', '_info.pkl')

    if not os.path.exists(info_path):
        print(f"Warning: No info file found at {info_path}")
        return None

    with open(info_path, 'rb') as f:
        info = pickle.load(f)

    return info


def create_dataset_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Create reproducible dataset splits

    Args:
        dataset: Dataset object
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        train_indices, val_indices, test_indices
    """
    import numpy as np

    total_size = len(dataset)
    indices = list(range(total_size))

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices


def save_split_indices(train_indices, val_indices, test_indices, save_path):
    """
    Save dataset split indices

    Args:
        train_indices: Training indices
        val_indices: Validation indices
        test_indices: Test indices
        save_path: Path to save pickle file
    """
    split_dict = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(split_dict, f)

    print(f"Split indices saved to {save_path}")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")


def load_split_indices(load_path):
    """
    Load dataset split indices

    Args:
        load_path: Path to pickle file

    Returns:
        train_indices, val_indices, test_indices
    """
    with open(load_path, 'rb') as f:
        split_dict = pickle.load(f)

    return split_dict['train'], split_dict['val'], split_dict['test']


def get_dataset_config():
    """
    Get configuration for the 6 band gap datasets

    Returns:
        dataset_configs: Dictionary of dataset configurations
    """
    dataset_configs = {
        'mp_bandgap': {
            'name': 'Materials Project Band Gap',
            'csv_file': 'data/mp_bandgap.csv',
            'property': 'band_gap',
            'description': 'DFT band gaps from Materials Project',
        },
        'jarvis_3d_optb88': {
            'name': 'JARVIS 3D OptB88vdW',
            'csv_file': 'data/jarvis_3d_optb88.csv',
            'property': 'band_gap',
            'description': 'DFT band gaps (OptB88vdW functional) for 3D materials',
        },
        'jarvis_3d_tbmbj': {
            'name': 'JARVIS 3D TBMBJ',
            'csv_file': 'data/jarvis_3d_tbmbj.csv',
            'property': 'band_gap',
            'description': 'DFT band gaps (TBMBJ functional) for 3D materials',
        },
        'jarvis_2d_optb88': {
            'name': 'JARVIS 2D OptB88vdW',
            'csv_file': 'data/jarvis_2d_optb88.csv',
            'property': 'band_gap',
            'description': 'DFT band gaps (OptB88vdW functional) for 2D materials',
        },
        'jarvis_2d_tbmbj': {
            'name': 'JARVIS 2D TBMBJ',
            'csv_file': 'data/jarvis_2d_tbmbj.csv',
            'property': 'band_gap',
            'description': 'DFT band gaps (TBMBJ functional) for 2D materials',
        },
        'matminer_exp': {
            'name': 'MatMiner Experimental',
            'csv_file': 'data/matminer_exp_bandgap.csv',
            'property': 'band_gap',
            'description': 'Experimental band gaps from literature',
        },
    }

    return dataset_configs
