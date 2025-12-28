"""
ALIGNN Extractor: ALIGNN backbone without prediction head
Extracts 256-dim features from crystal graphs
"""

import torch
import torch.nn as nn
from alignn.models.alignn import ALIGNN


class ALIGNNExtractor(nn.Module):
    """
    ALIGNN Feature Extractor (Frozen)
    Removes prediction head from full ALIGNN model
    Output: (features [B, 256], mask [B])
    """

    def __init__(self, alignn_layers=4, gcn_layers=4, hidden_features=256):
        super().__init__()

        self.hidden_features = hidden_features

        # Create full ALIGNN model
        self.alignn = ALIGNN(
            alignn_layers=alignn_layers,
            gcn_layers=gcn_layers,
            atom_input_features=92,  # Atomic number features
            edge_input_features=128,
            triplet_input_features=40,
            embedding_features=hidden_features,
            hidden_features=hidden_features,
            output_features=1,  # Dummy, will be removed
        )

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, g):
        """
        Args:
            g: DGL graph with node/edge features

        Returns:
            features: [B, hidden_features] tensor
            mask: [B] bool tensor (all True for valid graphs)
        """
        # Get embeddings before final prediction head
        node_features = g.ndata['atom_features']
        edge_features = g.edata.get('r', None)

        # Process through ALIGNN layers
        h = self.alignn.atom_embedding(node_features)

        # ALIGNN layers
        for alignn_layer in self.alignn.alignn_layers:
            h = alignn_layer(g, h, edge_features)

        # GCN layers
        for gcn_layer in self.alignn.gcn_layers:
            h = gcn_layer(g, h)

        # Readout (graph-level pooling)
        # Mean pooling over nodes
        batch_size = g.batch_size
        features = []

        for i in range(batch_size):
            node_mask = (g.batch_num_nodes() == i)
            graph_feats = h[node_mask].mean(dim=0)  # [hidden_features]
            features.append(graph_feats)

        features = torch.stack(features)  # [B, hidden_features]
        mask = torch.ones(batch_size, dtype=torch.bool, device=features.device)

        return features, mask

    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """Load from ALIGNN expert checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get model config
        config = checkpoint.get('model_config', {})
        alignn_layers = config.get('alignn_layers', 4)
        gcn_layers = config.get('gcn_layers', 4)
        hidden_features = config.get('hidden_features', 256)

        # Create extractor
        extractor = cls(
            alignn_layers=alignn_layers,
            gcn_layers=gcn_layers,
            hidden_features=hidden_features
        )

        # Load state dict (only backbone weights)
        model_state = checkpoint['model_state_dict']

        # Filter out prediction head weights
        backbone_state = {
            k: v for k, v in model_state.items()
            if not k.startswith('fc') and not k.startswith('out_norm')
        }

        extractor.alignn.load_state_dict(backbone_state, strict=False)

        # Freeze
        for param in extractor.parameters():
            param.requires_grad = False

        extractor.eval()

        return extractor


def disabled_train(self, mode=True):
    """Disable training mode"""
    return self


# Monkey patch to prevent training
ALIGNNExtractor.train = disabled_train
