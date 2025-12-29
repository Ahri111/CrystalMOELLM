"""
ALIGNN Data Loader for Band Gap Prediction with CSV format
Supports multiple datasets with CIF/POSCAR files
"""

import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph


class BandGapDataset(Dataset):
    """
    Dataset class for loading crystal structures and band gap values from CSV

    CSV format:
        id, structure_file, band_gap
        mp-1234, path/to/structure.cif, 1.23
    """

    def __init__(
        self,
        csv_file,
        root_dir=None,
        target_column='band_gap',
        structure_column='structure_file',
        max_neighbors=12,
        cutoff=8.0,
        use_canonize=True,
    ):
        """
        Args:
            csv_file: Path to CSV file with structure paths and targets
            root_dir: Root directory for structure files (if relative paths in CSV)
            target_column: Name of target column in CSV
            structure_column: Name of column containing structure file paths
            max_neighbors: Maximum number of neighbors for graph construction
            cutoff: Cutoff radius for neighbors (Angstrom)
            use_canonize: Whether to canonize the structure
        """
        self.csv_file = csv_file
        self.root_dir = root_dir or os.path.dirname(csv_file)
        self.target_column = target_column
        self.structure_column = structure_column
        self.max_neighbors = max_neighbors
        self.cutoff = cutoff
        self.use_canonize = use_canonize

        # Load CSV data
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

        print(f"Loaded {len(self.data)} samples from {csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            g: ALIGNN graph object
            target: Band gap value
            cif_id: Structure identifier
        """
        row = self.data[idx]

        # Get structure file path
        structure_path = row[self.structure_column]
        if not os.path.isabs(structure_path):
            structure_path = os.path.join(self.root_dir, structure_path)

        # Load structure
        atoms = Atoms.from_cif(structure_path)

        if self.use_canonize:
            atoms = atoms.get_primitive_atoms()

        # Convert to ALIGNN graph
        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            compute_line_graph=True,
            use_canonize=self.use_canonize,
        )

        # Get target value
        target = float(row[self.target_column])

        # Get ID
        cif_id = row.get('id', str(idx))

        return g, lg, torch.tensor(target, dtype=torch.float32), cif_id


def collate_fn(batch):
    """
    Collate function for batching graphs

    Args:
        batch: List of (g, lg, target, cif_id) tuples

    Returns:
        batched graphs, targets, and ids
    """
    import dgl

    graphs, line_graphs, targets, ids = zip(*batch)

    # Batch graphs
    batched_graph = dgl.batch(graphs)
    batched_line_graph = dgl.batch(line_graphs)
    batched_targets = torch.stack(targets)

    return batched_graph, batched_line_graph, batched_targets, list(ids)


def get_train_val_test_loader(
    csv_file,
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    num_workers=4,
    pin_memory=True,
    split_seed=42,
    train_indices=None,
    val_indices=None,
    test_indices=None,
    **kwargs
):
    """
    Create train/val/test dataloaders from CSV file

    Args:
        csv_file: Path to CSV file
        batch_size: Batch size
        train_ratio: Training set ratio (if indices not provided)
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        num_workers: Number of workers for dataloader
        pin_memory: Pin memory for GPU
        split_seed: Random seed for splitting
        train_indices: Pre-defined training indices
        val_indices: Pre-defined validation indices
        test_indices: Pre-defined test indices
        **kwargs: Additional arguments for BandGapDataset

    Returns:
        train_loader, val_loader, test_loader, dataset
    """

    dataset = BandGapDataset(csv_file, **kwargs)

    total_size = len(dataset)
    indices = list(range(total_size))

    # Use provided indices or create split
    if train_indices is None:
        np.random.seed(split_seed)
        np.random.shuffle(indices)

        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    return train_loader, val_loader, test_loader, dataset


def load_indices_from_file(indices_file):
    """
    Load pre-defined train/val/test indices from file

    File format (pickle):
        {
            'train': [idx1, idx2, ...],
            'val': [idx3, idx4, ...],
            'test': [idx5, idx6, ...]
        }
    """
    import pickle

    with open(indices_file, 'rb') as f:
        indices_dict = pickle.load(f)

    return indices_dict['train'], indices_dict['val'], indices_dict['test']
