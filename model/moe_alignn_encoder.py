"""
Frozen MoE ALIGNN Encoder
Phase 3에서 사용되는 완전 Frozen MoE Encoder
Property별로 다른 Expert 조합 사용 (Lookup만)
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from model.alignn_extractor import ALIGNNExtractor


class FrozenMoEALIGNNEncoder(nn.Module):
    """
    Frozen Mixture-of-Experts ALIGNN Encoder

    - 12개 frozen extractors
    - Property별 top-k lookup (no learnable params!)
    - Weighted combination
    """

    def __init__(self, extractor_dir, topk_config_path, property_list):
        """
        Args:
            extractor_dir: Path to extractors directory
            topk_config_path: Path to moe_topk.json
            property_list: List of property names (12개)
        """
        super().__init__()

        self.extractor_dir = Path(extractor_dir)
        self.property_list = property_list

        # Load top-k config
        with open(topk_config_path, 'r') as f:
            self.topk_config = json.load(f)

        print(f"Loading {len(property_list)} extractors...")

        # Load 12 frozen extractors
        self.extractors = nn.ModuleList()
        for prop in property_list:
            extractor_path = self.extractor_dir / f"extractor_{prop}.pt"

            if not extractor_path.exists():
                raise FileNotFoundError(f"Extractor not found: {extractor_path}")

            # Load extractor
            extractor = self._load_extractor(str(extractor_path))

            # Freeze
            extractor.eval()
            for param in extractor.parameters():
                param.requires_grad = False

            self.extractors.append(extractor)
            print(f"  Loaded {prop}")

        # Freeze entire module
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        print("MoE Encoder: All parameters frozen")

    def _load_extractor(self, checkpoint_path):
        """Load extractor from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        hidden_features = checkpoint.get('hidden_features', 256)

        extractor = ALIGNNExtractor(hidden_features=hidden_features)
        extractor.load_state_dict(checkpoint['model_state_dict'])

        return extractor

    def forward(self, g, property_name):
        """
        Args:
            g: Batched DGL graph
            property_name: Target property name (str)

        Returns:
            features: [B, 256] tensor
            mask: [B] bool tensor
        """

        if property_name not in self.topk_config:
            raise ValueError(f"Property {property_name} not found in topk_config")

        # Lookup top-k configuration
        config = self.topk_config[property_name]
        top_k_indices = config['indices']  # List of expert indices
        top_k_probs = config['probs']  # List of probabilities

        # Extract features from selected experts
        expert_features = []

        with torch.no_grad():
            for idx in top_k_indices:
                feats, mask = self.extractors[idx](g)  # [B, 256]
                expert_features.append(feats)

        # Stack: [B, k, 256]
        expert_features = torch.stack(expert_features, dim=1)

        # Probabilities: [1, k, 1]
        probs = torch.tensor(top_k_probs, device=expert_features.device)
        probs = probs.view(1, -1, 1)

        # Weighted combination: [B, 256]
        combined_features = (expert_features * probs).sum(dim=1)

        return combined_features, mask

    def train(self, mode=True):
        """Disable training mode"""
        return self

    def eval(self):
        """Always in eval mode"""
        return super().eval()


# Disable training for safety
def disabled_train(self, mode=True):
    """Override train() to prevent accidental training"""
    return self


FrozenMoEALIGNNEncoder.train = disabled_train
