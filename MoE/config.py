"""
Configuration file for ALIGNN MOE training

This file contains default configurations and dataset information
"""

# Dataset configurations for 6 band gap datasets
DATASET_CONFIGS = {
    'mp_bandgap': {
        'name': 'Materials Project Band Gap',
        'csv_file': 'data/mp_bandgap.csv',
        'property': 'band_gap',
        'description': 'DFT band gaps from Materials Project',
        'size': 'large',  # Expected dataset size
    },
    'jarvis_3d_optb88': {
        'name': 'JARVIS 3D OptB88vdW',
        'csv_file': 'data/jarvis_3d_optb88.csv',
        'property': 'band_gap',
        'description': 'DFT band gaps (OptB88vdW functional) for 3D materials',
        'size': 'large',
    },
    'jarvis_3d_tbmbj': {
        'name': 'JARVIS 3D TBMBJ',
        'csv_file': 'data/jarvis_3d_tbmbj.csv',
        'property': 'band_gap',
        'description': 'DFT band gaps (TBMBJ functional) for 3D materials',
        'size': 'large',
    },
    'jarvis_2d_optb88': {
        'name': 'JARVIS 2D OptB88vdW',
        'csv_file': 'data/jarvis_2d_optb88.csv',
        'property': 'band_gap',
        'description': 'DFT band gaps (OptB88vdW functional) for 2D materials',
        'size': 'medium',
    },
    'jarvis_2d_tbmbj': {
        'name': 'JARVIS 2D TBMBJ',
        'csv_file': 'data/jarvis_2d_tbmbj.csv',
        'property': 'band_gap',
        'description': 'DFT band gaps (TBMBJ functional) for 2D materials',
        'size': 'medium',
    },
    'matminer_exp': {
        'name': 'MatMiner Experimental',
        'csv_file': 'data/matminer_exp_bandgap.csv',
        'property': 'band_gap',
        'description': 'Experimental band gaps from literature',
        'size': 'small',
    },
}

# Default model configurations
DEFAULT_MODEL_CONFIG = {
    'alignn_layers': 4,
    'gcn_layers': 4,
    'atom_input_features': 92,
    'edge_features': 80,
    'triplet_input_features': 40,
    'embedding_features': 64,
    'hidden_features': 256,
    'output_features': 1,
    'dropout': 0.1,
    'use_batch_norm': True,
}

# Graph construction parameters
GRAPH_CONFIG = {
    'max_neighbors': 12,
    'cutoff': 8.0,
    'use_canonize': True,
}

# Single task training parameters
SINGLE_TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 1000,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 100,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'seed': 42,
    'num_workers': 4,
}

# MOE training parameters
MOE_TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 500,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'patience': 50,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'seed': 42,
    'num_workers': 4,
    # MOE specific
    'moe_option': 'multiheaded',
    'num_heads': 3,
    'k_extractors': 4,
    'use_all_extractors': True,
    'moe_hidden_dim': 128,
    'reg_weight': 0.01,
    'finetune_extractors': False,
}


def get_dataset_config(dataset_name):
    """
    Get configuration for a specific dataset

    Args:
        dataset_name: Name of dataset (key in DATASET_CONFIGS)

    Returns:
        config: Dataset configuration dictionary
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    return DATASET_CONFIGS[dataset_name].copy()


def get_all_dataset_checkpoints(checkpoint_root='checkpoints'):
    """
    Get paths to all 6 dataset checkpoints

    Args:
        checkpoint_root: Root directory for checkpoints

    Returns:
        checkpoint_paths: List of checkpoint paths
    """
    import os

    checkpoint_paths = []

    for dataset_key in DATASET_CONFIGS.keys():
        checkpoint_path = os.path.join(checkpoint_root, dataset_key, 'best_model.pt')
        if os.path.exists(checkpoint_path):
            checkpoint_paths.append(checkpoint_path)
        else:
            print(f"Warning: Checkpoint not found for {dataset_key}: {checkpoint_path}")

    return checkpoint_paths


def create_training_script(dataset_name, output_dir, **kwargs):
    """
    Generate a training command for a specific dataset

    Args:
        dataset_name: Name of dataset
        output_dir: Output directory for checkpoints
        **kwargs: Override default parameters

    Returns:
        cmd: Training command string
    """
    config = get_dataset_config(dataset_name)
    train_config = SINGLE_TRAIN_CONFIG.copy()
    train_config.update(kwargs)

    cmd = f"python single_train.py \\\n"
    cmd += f"    --csv_file {config['csv_file']} \\\n"
    cmd += f"    --dataset_name \"{config['name']}\" \\\n"
    cmd += f"    --output_dir {output_dir} \\\n"

    for key, value in train_config.items():
        if isinstance(value, bool):
            if value:
                cmd += f"    --{key} \\\n"
        elif isinstance(value, float):
            cmd += f"    --{key} {value} \\\n"
        else:
            cmd += f"    --{key} {value} \\\n"

    return cmd.rstrip(' \\\n')


def create_moe_training_script(checkpoint_paths, target_dataset, output_dir, **kwargs):
    """
    Generate an MOE training command

    Args:
        checkpoint_paths: List of checkpoint paths
        target_dataset: Target dataset name
        output_dir: Output directory
        **kwargs: Override default parameters

    Returns:
        cmd: MOE training command string
    """
    config = get_dataset_config(target_dataset)
    moe_config = MOE_TRAIN_CONFIG.copy()
    moe_config.update(kwargs)

    cmd = f"python moe_train.py \\\n"
    cmd += f"    --checkpoint_paths {' '.join(checkpoint_paths)} \\\n"
    cmd += f"    --target_csv {config['csv_file']} \\\n"
    cmd += f"    --dataset_name \"{config['name']} MOE\" \\\n"
    cmd += f"    --output_dir {output_dir} \\\n"

    for key, value in moe_config.items():
        if isinstance(value, bool):
            if value:
                cmd += f"    --{key} \\\n"
        elif isinstance(value, float):
            cmd += f"    --{key} {value} \\\n"
        else:
            cmd += f"    --{key} {value} \\\n"

    return cmd.rstrip(' \\\n')


if __name__ == '__main__':
    """Print example commands"""

    print("=" * 80)
    print("ALIGNN MOE Configuration")
    print("=" * 80)

    print("\n### Dataset Information ###\n")
    for key, config in DATASET_CONFIGS.items():
        print(f"{key}:")
        print(f"  Name: {config['name']}")
        print(f"  File: {config['csv_file']}")
        print(f"  Size: {config['size']}")
        print()

    print("\n### Example: Single Training Commands ###\n")
    for dataset_key in ['mp_bandgap', 'matminer_exp']:
        print(f"# {dataset_key}")
        cmd = create_training_script(
            dataset_name=dataset_key,
            output_dir=f'checkpoints/{dataset_key}'
        )
        print(cmd)
        print()

    print("\n### Example: MOE Training Command ###\n")
    checkpoint_paths = [
        f'checkpoints/{key}/best_model.pt'
        for key in DATASET_CONFIGS.keys()
    ]
    cmd = create_moe_training_script(
        checkpoint_paths=checkpoint_paths,
        target_dataset='matminer_exp',
        output_dir='checkpoints/moe_final'
    )
    print(cmd)
