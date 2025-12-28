#!/usr/bin/env python3
"""
Phase 3: Train Frozen MoE + Q-Former + LLM
3 Losses: ITC + LM + Property Prediction
"""

import argparse
import yaml
import json
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
from tqdm import tqdm
from model.blip2qformer import Blip2Qformer  # 기존 ZeroMAT 모델
from transformers import AutoTokenizer


class CrystalPropertyDataset(Dataset):
    """
    Dataset for Stage 3 training
    Randomly selects one property per sample
    """

    def __init__(self, data_path, property_list, tokenizer, max_txt_len=128):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.property_list = property_list
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len

        # Filter samples with at least one property
        self.data = [
            d for d in self.data
            if any(prop in d['properties'] for prop in property_list)
        ]

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load graph
        g, _ = dgl.load_graphs(sample['graph_path'])
        g = g[0]

        # Random property selection
        available_props = [p for p in self.property_list if p in sample['properties']]
        property_name = random.choice(available_props)
        property_value = sample['properties'][property_name]

        # Text (robocrys)
        text = sample['text']

        return {
            'graph': g,
            'text': text,
            'property_name': property_name,
            'property_value': torch.tensor([property_value], dtype=torch.float32),
            'material_id': sample['id']
        }


def collate_fn(batch):
    """
    Collate function
    Batch 내 가장 많이 나온 property 선택
    """
    # Find most common property in batch
    property_counts = {}
    for item in batch:
        prop = item['property_name']
        property_counts[prop] = property_counts.get(prop, 0) + 1

    main_property = max(property_counts, key=property_counts.get)

    # Filter batch to only use main_property
    filtered_batch = [
        item for item in batch
        if item['property_name'] == main_property
    ]

    # If no samples with main_property, use first item
    if not filtered_batch:
        filtered_batch = [batch[0]]
        main_property = batch[0]['property_name']

    # Batch graphs
    graphs = [item['graph'] for item in filtered_batch]
    batched_graph = dgl.batch(graphs)

    # Batch texts
    texts = [item['text'] for item in filtered_batch]

    # Batch property values
    property_values = torch.cat([item['property_value'] for item in filtered_batch])

    # Material IDs
    material_ids = [item['material_id'] for item in filtered_batch]

    return {
        'graphs': batched_graph,
        'texts': texts,
        'property_name': main_property,
        'property_values': property_values,
        'material_ids': material_ids
    }


def train_epoch(model, loader, optimizer, device, loss_weights):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_itc = 0
    total_lm = 0
    total_property = 0
    count = 0

    for batch in tqdm(loader, desc="Training"):
        batch['graphs'] = batch['graphs'].to(device)
        batch['property_values'] = batch['property_values'].to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(batch)

        # Weighted loss
        loss = (
            loss_weights['itc'] * outputs.loss_itc +
            loss_weights['lm'] * outputs.loss_lm +
            loss_weights['property'] * outputs.loss_property
        )

        # Backward
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        total_itc += outputs.loss_itc.item()
        total_lm += outputs.loss_lm.item()
        total_property += outputs.loss_property.item()
        count += 1

    return {
        'loss': total_loss / count,
        'itc': total_itc / count,
        'lm': total_lm / count,
        'property': total_property / count
    }


def validate(model, loader, device, loss_weights):
    """Validate model"""
    model.eval()

    total_loss = 0
    total_itc = 0
    total_lm = 0
    total_property = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch['graphs'] = batch['graphs'].to(device)
            batch['property_values'] = batch['property_values'].to(device)

            # Forward
            outputs = model(batch)

            # Weighted loss
            loss = (
                loss_weights['itc'] * outputs.loss_itc +
                loss_weights['lm'] * outputs.loss_lm +
                loss_weights['property'] * outputs.loss_property
            )

            # Stats
            total_loss += loss.item()
            total_itc += outputs.loss_itc.item()
            total_lm += outputs.loss_lm.item()
            total_property += outputs.loss_property.item()
            count += 1

    return {
        'loss': total_loss / count,
        'itc': total_itc / count,
        'lm': total_lm / count,
        'property': total_property / count
    }


def train_stage3(config_path):
    """Main training function"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['llm_model'])

    # Datasets
    train_dataset = CrystalPropertyDataset(
        config['data']['train_file'],
        config['model']['property_list'],
        tokenizer,
        config['data']['max_txt_len']
    )

    val_dataset = CrystalPropertyDataset(
        config['data']['val_file'],
        config['model']['property_list'],
        tokenizer,
        config['data']['max_txt_len']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers']
    )

    # Model (기존 blip2qformer를 MoE 설정으로)
    model = Blip2Qformer(
        use_moe=config['model']['use_moe'],
        moe_extractor_dir=config['model']['moe_extractor_dir'],
        moe_topk_config=config['model']['moe_topk_config'],
        property_list=config['model']['property_list'],
        num_query_token=config['model']['num_query_token'],
        cross_attention_freq=config['model']['cross_attention_freq'],
        llm_model=config['model']['llm_model'],
        freeze_llm=config['model']['freeze_llm'],
    ).to(device)

    # Optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = config['training']['warmup_steps']

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config['training']['min_lr']
    )

    # Loss weights
    loss_weights = config['training']['loss_weights']

    # Training loop
    best_val_loss = float('inf')
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, loss_weights)
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"ITC: {train_metrics['itc']:.4f}, "
              f"LM: {train_metrics['lm']:.4f}, "
              f"Property: {train_metrics['property']:.4f}")

        # Validate
        if (epoch + 1) % config['training']['eval_freq'] == 0:
            val_metrics = validate(model, val_loader, device, loss_weights)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"ITC: {val_metrics['itc']:.4f}, "
                  f"LM: {val_metrics['lm']:.4f}, "
                  f"Property: {val_metrics['property']:.4f}")

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': config
                }, checkpoint_path)
                print(f"Saved best model: {checkpoint_path}")

        # Save checkpoint
        if (epoch + 1) % config['training']['save_freq'] == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)

        # Scheduler step
        scheduler.step()

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stage 3 ZeroMAT with MoE')

    parser.add_argument('--config', type=str, default='config/stage3_zeromat.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    train_stage3(args.config)
