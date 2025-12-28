"""
Crystal DataModule for PyTorch Lightning
Follows the pattern of stage3_dm.py
"""

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import dgl
from .crystal_dataset import CrystalPropertyDataset, CrystalInferenceDataset


class CrystalCollater:
    """Collate function for crystal graphs with property info"""

    def __init__(self, tokenizer, text_max_len=128):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def __call__(self, batch):
        """
        Collate batch of crystal data

        Returns:
            graph_batch: Batched DGL graph
            text_batch: Tokenized texts
            property_info: {'property_name': str, 'property_values': tensor}
        """
        graphs = [item['graph'] for item in batch]
        texts = [item['text'] for item in batch]
        property_names = [item['property_name'] for item in batch]
        property_values = [item['property_value'] for item in batch]

        # Batch graphs
        batched_graph = dgl.batch(graphs)

        # Find most common property in batch
        from collections import Counter
        property_counts = Counter(property_names)
        main_property = property_counts.most_common(1)[0][0]

        # Filter to use only main property
        filtered_indices = [i for i, pname in enumerate(property_names) if pname == main_property]

        # Re-batch with filtered samples
        if len(filtered_indices) < len(batch):
            filtered_graphs = [graphs[i] for i in filtered_indices]
            filtered_texts = [texts[i] for i in filtered_indices]
            filtered_values = [property_values[i] for i in filtered_indices]

            batched_graph = dgl.batch(filtered_graphs)
            texts = filtered_texts
            property_values = filtered_values

        # Tokenize texts
        text_batch = self.tokenizer(
            texts,
            truncation=True,
            padding='longest',
            max_length=self.text_max_len,
            return_tensors='pt',
            return_attention_mask=True
        )

        # Property info
        property_info = {
            'property_name': main_property,
            'property_values': torch.tensor(property_values, dtype=torch.float32)
        }

        return batched_graph, text_batch, property_info


class CrystalDM(LightningDataModule):
    """
    Crystal DataModule for MoE training
    """

    def __init__(
        self,
        mode='train',
        num_workers=4,
        batch_size=32,
        train_path='data/splits/train.json',
        val_path='data/splits/val.json',
        test_path='data/splits/test.json',
        text_max_len=128,
        tokenizer=None,
        property_list=None,
        args=None,
    ):
        super().__init__()
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer

        if property_list is None:
            property_list = [
                'band_gap', 'formation_energy', 'bulk_modulus', 'shear_modulus',
                'elastic_anisotropy', 'poisson_ratio', 'total_magnetization',
                'n_Egap', 'p_Egap', 'n_mass', 'p_mass', 'eij_max'
            ]
        self.property_list = property_list

        # Create datasets
        if 'train' in mode:
            self.train_dataset = CrystalPropertyDataset(
                train_path, property_list, tokenizer, text_max_len
            )
            self.val_dataset = CrystalPropertyDataset(
                val_path, property_list, tokenizer, text_max_len
            )
        elif 'eval' in mode or 'test' in mode:
            self.test_dataset = CrystalPropertyDataset(
                test_path, property_list, tokenizer, text_max_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=CrystalCollater(self.tokenizer, self.text_max_len),
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=CrystalCollater(self.tokenizer, self.text_max_len),
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=CrystalCollater(self.tokenizer, self.text_max_len),
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Crystal Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--train_path', type=str, default='data/splits/train.json')
        parser.add_argument('--val_path', type=str, default='data/splits/val.json')
        parser.add_argument('--test_path', type=str, default='data/splits/test.json')
        parser.add_argument('--text_max_len', type=int, default=128)
        return parent_parser
