"""
Mixture of Experts (MOE) Model for ALIGNN
Adapted from CGCNN MOE framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MixtureOfExtractors(nn.Module):
    """
    Mixture of Extractors combining multiple pretrained ALIGNN models

    Supports multiple combination strategies:
    - pairwise_TL: Single extractor with new head
    - concat: Concatenate all extractor outputs with learned scales
    - add_k: Top-k gating with learned importance scores
    """

    def __init__(
        self,
        extractors,
        feature_dim,
        option='add_k',
        k_extractors=4,
    ):
        """
        Args:
            extractors: List of ALIGNN extractors (ALIGNNExtractor)
            feature_dim: Feature dimension from each extractor
            option: Combination strategy ('pairwise_TL', 'concat', 'add_k')
            k_extractors: Number of top extractors to use (for add_k)
        """
        super(MixtureOfExtractors, self).__init__()

        self.extractors = nn.ModuleList(extractors)
        self.num_extractors = len(extractors)
        self.feature_dim = feature_dim
        self.option = option
        self.k_extractors = min(k_extractors, self.num_extractors)

        if self.option == 'add_k':
            assert 0 < self.k_extractors <= len(extractors)

        # Learnable scaling parameters for each extractor
        self.extractor_scores = nn.Parameter(
            torch.ones(self.num_extractors),
            requires_grad=(option != 'pairwise_TL' and option != 'ensemble')
        )

        self.softmax = nn.Softmax(dim=0)

        # Output dimension depends on strategy
        if option == 'concat':
            self.output_dim = feature_dim * self.num_extractors
        else:
            self.output_dim = feature_dim

    def forward(self, g, lg):
        """
        Forward pass through mixture of extractors

        Args:
            g: Atom graph
            lg: Line graph

        Returns:
            mixed_features: Combined features
            extractor_weights: Weights assigned to each extractor (for add_k)
        """
        # Extract features from all extractors
        features_list = []
        for extractor in self.extractors:
            with torch.set_grad_enabled(extractor.training):
                features = extractor(g, lg)
                features_list.append(features)

        # Stack features: [num_extractors, batch_size, feature_dim]
        stacked_features = torch.stack(features_list, dim=0)

        if self.option == 'pairwise_TL':
            # Use only the first extractor
            mixed_features = stacked_features[0]
            extractor_weights = None

        elif self.option == 'concat':
            # Concatenate all features with learned scales
            scaled_features = []
            for i in range(self.num_extractors):
                scaled = stacked_features[i] * self.extractor_scores[i]
                scaled_features.append(scaled)
            mixed_features = torch.cat(scaled_features, dim=1)
            extractor_weights = self.extractor_scores

        elif self.option == 'add_k':
            # Top-k gating mechanism (원본 CGCNN MOE 방식)
            # 전체 배치에 동일한 top-k extractor를 선택

            # Select top-k extractors based on scaling parameters
            top_k_values, top_k_indices = torch.topk(
                self.extractor_scores, self.k_extractors)
            # Normalize top-k scores to probabilities
            top_k_probabilities = F.softmax(top_k_values, dim=0)

            # Initialize with first top extractor
            top_prob = top_k_probabilities[0]
            top_idx = top_k_indices[0]
            mixed_features = top_prob * stacked_features[top_idx]

            # Add remaining k-1 extractors
            for i in range(1, self.k_extractors):
                prob = top_k_probabilities[i]
                idx = top_k_indices[i]
                mixed_features = mixed_features + prob * stacked_features[idx]

            # Create sparse backbone scores (only selected extractors have non-zero scores)
            extractor_weights = torch.zeros(
                self.num_extractors,
                device=self.extractor_scores.device
            ).scatter_(0, top_k_indices, top_k_probabilities)

        else:
            raise ValueError(f"Unknown option: {self.option}")

        return mixed_features, extractor_weights


class MultiheadedMixtureOfExpertsModel(nn.Module):
    """
    Multi-headed MOE model with multiple MixtureOfExtractors heads

    Each head independently selects and combines extractors,
    then all heads are combined for final prediction
    """

    def __init__(
        self,
        extractors,
        feature_dim,
        output_dim=1,
        num_heads=3,
        k_extractors=4,
        hidden_dim=128,
    ):
        """
        Args:
            extractors: List of ALIGNN extractors
            feature_dim: Feature dimension from each extractor
            output_dim: Output dimension (1 for regression)
            num_heads: Number of MOE heads (pseudo-attention heads)
            k_extractors: Number of top extractors per head
            hidden_dim: Hidden dimension for MLP head
        """
        super(MultiheadedMixtureOfExpertsModel, self).__init__()

        self.num_heads = num_heads
        self.k_extractors = k_extractors
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # Create multiple MOE heads (pseudo-attention heads)
        self.moe_heads = nn.ModuleList([
            MixtureOfExtractors(
                extractors=extractors,
                feature_dim=feature_dim,
                option='add_k',
                k_extractors=k_extractors,
            )
            for _ in range(num_heads)
        ])

        # MLP head for final prediction
        combined_dim = feature_dim * num_heads
        self.mlp_head = MultilayerPerceptronHead(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, g, lg):
        """
        Forward pass

        Args:
            g: Atom graph
            lg: Line graph

        Returns:
            output: Predictions
            loss_reg: Regularization loss (for training)
        """
        # Get features from each head
        head_features = []
        head_weights = []

        for head in self.moe_heads:
            features, weights = head(g, lg)
            head_features.append(features)
            if weights is not None:
                head_weights.append(weights)

        # Concatenate all head features
        combined_features = torch.cat(head_features, dim=1)

        # Final prediction
        output = self.mlp_head(combined_features)

        # Compute regularization loss (encourage diversity among heads)
        loss_reg = self.compute_regularization_loss(head_weights)

        return output, loss_reg

    def compute_regularization_loss(self, head_weights):
        """
        Compute orthogonality regularization loss (원본 CGCNN MOE 방식)

        Forces the score matrix to be orthogonal, encouraging different heads
        to select different extractors

        Args:
            head_weights: List of weight tensors from each head
                         Each tensor has shape [num_extractors]

        Returns:
            loss_reg: Regularization loss
        """
        if len(head_weights) == 0:
            return torch.tensor(0.0, device='cpu')

        # Stack weights: [num_extractors, num_heads]
        score_matrix = torch.stack(head_weights, dim=-1)

        # Compute: score_matrix.T @ score_matrix
        # Shape: (num_heads, num_extractors) @ (num_extractors, num_heads) = (num_heads, num_heads)
        gram_matrix = torch.transpose(score_matrix, 0, 1) @ score_matrix

        # Identity matrix
        identity = torch.eye(self.num_heads, device=score_matrix.device)

        # Loss: ||gram_matrix - I||^2
        # This forces the heads to be orthogonal (select different extractors)
        loss_reg = torch.pow(torch.norm(gram_matrix - identity), 2)

        return loss_reg

    def non_extractor_parameters(self):
        """
        Yield all parameters except extractor parameters

        This is useful for freezing extractors while training MOE components
        """
        for n, p in self.named_parameters():
            if p.requires_grad and 'extractors' not in n:
                yield p


class MultilayerPerceptronHead(nn.Module):
    """
    MLP head for final prediction
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=1, dropout=0.1):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(MultilayerPerceptronHead, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features

        Returns:
            out: Predictions
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)

        out = self.fc3(x)

        return out


class EnsembleModel(nn.Module):
    """
    Simple ensemble of multiple ALIGNN models

    Each model makes independent predictions, then weighted average
    """

    def __init__(self, extractors, feature_dim, output_dim=1):
        """
        Args:
            extractors: List of ALIGNN extractors
            feature_dim: Feature dimension
            output_dim: Output dimension
        """
        super(EnsembleModel, self).__init__()

        self.extractors = nn.ModuleList(extractors)
        self.num_extractors = len(extractors)

        # Prediction head for each extractor
        self.heads = nn.ModuleList([
            nn.Linear(feature_dim, output_dim)
            for _ in range(self.num_extractors)
        ])

        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(
            torch.ones(self.num_extractors) / self.num_extractors
        )

    def forward(self, g, lg):
        """
        Forward pass

        Args:
            g: Atom graph
            lg: Line graph

        Returns:
            output: Ensemble prediction
        """
        predictions = []

        for extractor, head in zip(self.extractors, self.heads):
            features = extractor(g, lg)
            pred = head(features)
            predictions.append(pred)

        # Stack predictions: [num_extractors, batch_size, 1]
        stacked_preds = torch.stack(predictions, dim=0)

        # Weighted average
        weights = F.softmax(self.ensemble_weights, dim=0)
        weights_expanded = weights.view(-1, 1, 1)

        output = torch.sum(weights_expanded * stacked_preds, dim=0)

        return output
