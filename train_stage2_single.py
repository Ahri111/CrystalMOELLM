"""
Stage 2: Single Expert Training
- Train expert head on [MASK] embeddings from Q-Former
- ALIGNN + Q-Former frozen
- Each source gets separate expert (.pt file)
- Support loading pretrained .pt files
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from model.blip2qformer_bandgap import Blip2QformerBandGap
from data_provider.bandgap_dataset import BandGapDataset, collate_fn


class ExpertHead(nn.Module):
    """Expert head for band gap prediction from [MASK] embeddings"""

    def __init__(self, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, mask_embeddings):
        """
        Args:
            mask_embeddings: [MASK] token embeddings from Q-Former [B, D]

        Returns:
            predictions: Band gap predictions [B]
        """
        return self.head(mask_embeddings).squeeze()


def extract_mask_embeddings(qformer_model, batch, device):
    """
    Extract [MASK] token embeddings from Q-Former

    Args:
        qformer_model: Trained Q-Former model (Stage 1)
        batch: Data batch
        device: Device

    Returns:
        mask_embeddings: [MASK] embeddings [B, D]
        band_gaps: Ground truth [B]
    """
    graphs, lg_graphs, texts, band_gaps, sources = batch

    graphs = graphs.to(device)
    lg_graphs = lg_graphs.to(device)
    band_gaps = band_gaps.to(device)

    with torch.no_grad():
        # ALIGNN encoding
        graph_embeds = qformer_model.alignn_encoder.get_features(graphs, lg_graphs)
        graph_embeds = qformer_model.ln_graph(graph_embeds).unsqueeze(1)

        # Tokenize text
        text_tokens = qformer_model.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        # Q-Former
        query_tokens = qformer_model.query_tokens.expand(graph_embeds.shape[0], -1, -1)

        mask_output = qformer_model.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(device),
            return_dict=True,
        )

        # Extract [MASK] embeddings (pool query tokens)
        mask_embeddings = mask_output.last_hidden_state[:, :query_tokens.size(1), :].mean(dim=1)

    return mask_embeddings, band_gaps


def train_expert(qformer_model, expert_head, train_loader, val_loader, args, device):
    """Train expert head"""

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(expert_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        expert_head.train()
        train_loss = 0.0

        for batch in train_loader:
            mask_embeddings, band_gaps = extract_mask_embeddings(qformer_model, batch, device)

            predictions = expert_head(mask_embeddings)
            loss = criterion(predictions, band_gaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        expert_head.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                mask_embeddings, band_gaps = extract_mask_embeddings(qformer_model, batch, device)
                predictions = expert_head(mask_embeddings)
                loss = criterion(predictions, band_gaps)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'expert_state_dict': expert_head.state_dict(),
                'val_loss': val_loss,
                'source': args.source_label,
            }, os.path.join(args.output_dir, f'{args.source_label}_expert.pt'))
            print(f"  Saved best model for {args.source_label}")

    return best_val_loss


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Stage 1 Q-Former model
    print("Loading Stage 1 Q-Former model...")
    qformer_model = Blip2QformerBandGap(
        gtm=args.gtm,
        bert_name=args.bert_name,
        temperature=0.07,
        num_query_token=args.num_query_token,
        embed_dim=256,
        args=args,
    ).to(device)

    # Load checkpoint
    if args.qformer_checkpoint:
        checkpoint = torch.load(args.qformer_checkpoint, map_location=device)
        qformer_model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded Q-Former from {args.qformer_checkpoint}")

    # Freeze Q-Former
    for param in qformer_model.parameters():
        param.requires_grad = False
    qformer_model.eval()

    # Create expert head
    expert_head = ExpertHead(hidden_dim=qformer_model.Qformer.config.hidden_size).to(device)

    # Load pretrained expert if provided
    if args.expert_checkpoint:
        checkpoint = torch.load(args.expert_checkpoint, map_location=device)
        expert_head.load_state_dict(checkpoint['expert_state_dict'])
        print(f"Loaded expert from {args.expert_checkpoint}")

    # Dataset (filter by source)
    train_dataset = BandGapDataset(args.train_csv)
    val_dataset = BandGapDataset(args.val_csv)

    # Filter by source if specified
    if args.source_label:
        train_dataset.df = train_dataset.df[train_dataset.df['source_label'] == args.source_label]
        val_dataset.df = val_dataset.df[val_dataset.df['source_label'] == args.source_label]
        print(f"Filtered dataset for source: {args.source_label}")
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Train expert
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = train_expert(qformer_model, expert_head, train_loader, val_loader, args, device)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Expert saved to: {args.output_dir}/{args.source_label}_expert.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--source_label', type=str, required=True,
                        help='Source to train expert for (mp, jarvis_3d_optb88, etc.)')

    # Model
    parser.add_argument('--qformer_checkpoint', type=str, required=True,
                        help='Path to Stage 1 Q-Former checkpoint')
    parser.add_argument('--expert_checkpoint', type=str, default=None,
                        help='Path to pretrained expert .pt file (for fine-tuning)')

    # Q-Former config (should match Stage 1)
    parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
    parser.add_argument('--num_query_token', type=int, default=32)
    parser.add_argument('--gtm', action='store_true')
    parser.add_argument('--alignn_layers', type=int, default=4)
    parser.add_argument('--gcn_layers', type=int, default=4)
    parser.add_argument('--hidden_features', type=int, default=256)
    parser.add_argument('--embedding_features', type=int, default=64)
    parser.add_argument('--tune_gnn', action='store_true')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage2')

    args = parser.parse_args()

    main(args)
