"""
ALIGNN Model Wrapper for Band Gap Prediction
Compatible with MOE framework
"""

import torch
import torch.nn as nn
from alignn.models.alignn import ALIGNN, ALIGNNConfig


class ALIGNNRegression(nn.Module):
    """
    ALIGNN model wrapper for regression tasks (band gap prediction)

    This wrapper provides a consistent interface for:
    - Single task training
    - Transfer learning
    - MOE integration
    """

    def __init__(
        self,
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,
        edge_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=256,
        output_features=1,
        link='identity',
        zero_inflated=False,
        classification=False,
        num_classes=2,
        use_batch_norm=True,
        dropout=0.0,
        **kwargs
    ):
        """
        Args:
            alignn_layers: Number of ALIGNN layers
            gcn_layers: Number of GCN layers
            atom_input_features: Input features per atom
            edge_features: Edge feature dimension
            triplet_input_features: Triplet feature dimension
            embedding_features: Embedding dimension
            hidden_features: Hidden layer dimension
            output_features: Output dimension (1 for regression)
            link: Output activation ('identity', 'log', 'logit')
            zero_inflated: Use zero-inflated model
            classification: Classification mode
            num_classes: Number of classes for classification
            use_batch_norm: Use batch normalization
            dropout: Dropout rate
        """
        super(ALIGNNRegression, self).__init__()

        config = ALIGNNConfig(
            name="alignn_bandgap",
            alignn_layers=alignn_layers,
            gcn_layers=gcn_layers,
            atom_input_features=atom_input_features,
            edge_features=edge_features,
            triplet_input_features=triplet_input_features,
            embedding_features=embedding_features,
            hidden_features=hidden_features,
            output_features=output_features,
            link=link,
            zero_inflated=zero_inflated,
            classification=classification,
            num_classes=num_classes,
        )

        # ALIGNN encoder
        self.alignn = ALIGNN(config)

        # Feature dimension after ALIGNN encoding
        self.feature_dim = hidden_features

        # Additional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_features)

        # Output head
        self.fc_out = nn.Linear(hidden_features, output_features)

        self.classification = classification
        if classification:
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, g, lg, return_features=False):
        """
        Forward pass

        Args:
            g: Atom graph (DGL graph)
            lg: Line graph (DGL graph)
            return_features: If True, return features before final layer

        Returns:
            out: Predictions
            features: Encoded features (if return_features=True)
        """
        # ALIGNN encoding
        features = self.alignn.forward(g, lg)

        # Apply batch norm
        if self.use_batch_norm:
            features = self.batch_norm(features)

        # Apply dropout
        features = self.dropout(features)

        if return_features:
            return features

        # Final prediction
        out = self.fc_out(features)

        if self.classification:
            out = self.softmax(out)

        return out

    def get_features(self, g, lg):
        """
        Extract features (for MOE)

        Args:
            g: Atom graph
            lg: Line graph

        Returns:
            features: Encoded features
        """
        return self.forward(g, lg, return_features=True)

    def load_pretrained(self, checkpoint_path, strict=True):
        """
        Load pretrained weights

        Args:
            checkpoint_path: Path to .pt file
            strict: Strict loading (default: True)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)

        if not strict:
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

        print(f"Loaded pretrained model from {checkpoint_path}")

    def freeze_backbone(self):
        """
        Freeze ALIGNN backbone for transfer learning
        """
        for param in self.alignn.parameters():
            param.requires_grad = False
        print("ALIGNN backbone frozen")

    def unfreeze_backbone(self):
        """
        Unfreeze ALIGNN backbone
        """
        for param in self.alignn.parameters():
            param.requires_grad = True
        print("ALIGNN backbone unfrozen")


class ALIGNNExtractor(nn.Module):
    """
    ALIGNN feature extractor (for MOE)

    This class only extracts features without the final prediction layer
    """

    def __init__(self, model):
        """
        Args:
            model: ALIGNNRegression model
        """
        super(ALIGNNExtractor, self).__init__()

        self.alignn = model.alignn
        self.batch_norm = model.batch_norm if model.use_batch_norm else nn.Identity()
        self.dropout = model.dropout
        self.feature_dim = model.feature_dim

    def forward(self, g, lg):
        """
        Extract features

        Args:
            g: Atom graph
            lg: Line graph

        Returns:
            features: Encoded features
        """
        features = self.alignn.forward(g, lg)

        if isinstance(self.batch_norm, nn.BatchNorm1d):
            features = self.batch_norm(features)

        features = self.dropout(features)

        return features


def create_alignn_model(config_dict=None, pretrained_path=None):
    """
    Factory function to create ALIGNN model

    Args:
        config_dict: Configuration dictionary
        pretrained_path: Path to pretrained weights

    Returns:
        model: ALIGNNRegression model
    """
    config = config_dict or {}

    model = ALIGNNRegression(**config)

    if pretrained_path:
        model.load_pretrained(pretrained_path)

    return model
