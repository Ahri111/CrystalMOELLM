"""
Crystal Dataset for ALIGNN MoE
Loads DGL graphs from preprocessed crystal data
"""

import json
import random
import torch
import dgl
from torch.utils.data import Dataset


class CrystalPropertyDataset(Dataset):
    """
    Dataset for crystal property prediction with MoE
    Can target specific property or use random selection
    """

    def __init__(self, data_path, property_list, tokenizer=None, max_txt_len=128, target_property=None):
        """
        Args:
            data_path: Path to JSON file (train.json, val.json, test.json)
            property_list: List of property names
            tokenizer: Text tokenizer (for Robocrys)
            max_txt_len: Max text length
            target_property: Specific property to use (if None, random selection)
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.property_list = property_list if isinstance(property_list, list) else [property_list]
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.target_property = target_property

        # Filter samples
        if target_property:
            # Single property mode: only samples with target property
            self.data = [
                d for d in self.data
                if target_property in d.get('properties', {})
            ]
            print(f"[CrystalDataset] Loaded {len(self.data)} samples for '{target_property}' from {data_path}")
        else:
            # Multi-property mode: samples with at least one property
            self.data = [
                d for d in self.data
                if any(prop in d.get('properties', {}) for prop in self.property_list)
            ]
            print(f"[CrystalDataset] Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load DGL graph
        graph_path = sample['graph_path']
        graphs, _ = dgl.load_graphs(graph_path)
        g = graphs[0]

        # Property selection
        if self.target_property:
            # Single property mode: use target property
            property_name = self.target_property
            property_value = sample['properties'][property_name]
        else:
            # Multi-property mode: random selection
            available_props = [
                p for p in self.property_list
                if p in sample.get('properties', {})
            ]

            if not available_props:
                available_props = [self.property_list[0]]

            property_name = random.choice(available_props)
            property_value = sample.get('properties', {}).get(property_name, 0.0)

        # Robocrys text
        text = sample.get('text', sample.get('robocrys', ''))

        return {
            'graph': g,
            'text': text,
            'property_name': property_name,
            'property_value': float(property_value),
            'material_id': sample.get('id', 'unknown')
        }


class CrystalInferenceDataset(Dataset):
    """
    Dataset for inference - uses specific property
    """

    def __init__(self, data_path, target_property, tokenizer=None, max_txt_len=128):
        """
        Args:
            data_path: Path to JSON file
            target_property: Specific property to predict
            tokenizer: Text tokenizer
            max_txt_len: Max text length
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.target_property = target_property
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len

        # Filter samples with target property
        self.data = [
            d for d in self.data
            if target_property in d.get('properties', {})
        ]

        print(f"[CrystalInferenceDataset] Loaded {len(self.data)} samples for {target_property}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load graph
        graph_path = sample['graph_path']
        graphs, _ = dgl.load_graphs(graph_path)
        g = graphs[0]

        # Target property
        property_value = sample['properties'][self.target_property]

        # Text
        text = sample.get('text', sample.get('robocrys', ''))

        return {
            'graph': g,
            'text': text,
            'property_name': self.target_property,
            'property_value': float(property_value),
            'material_id': sample.get('id', 'unknown')
        }
