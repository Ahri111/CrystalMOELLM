#!/usr/bin/env python3
"""
Train BLIP2 MoE Stage 2
Frozen MoE Encoder + Trainable Q-Former
"""

import os
import argparse
import warnings
import yaml
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger

from model.blip2_moe_stage2 import Blip2MoEStage2
from data_provider.crystal_dm import CrystalDM

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
torch.set_float32_matmul_precision('medium')


def load_property_list(config_path='configs/property_list.yaml'):
    """Load property list from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['properties']


def main(args):
    pl.seed_everything(args.seed)

    # Load property list
    args.property_list = load_property_list()
    print(f"Loaded {len(args.property_list)} properties")

    # Model
    if args.init_checkpoint:
        model = Blip2MoEStage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"Loaded from checkpoint: {args.init_checkpoint}")
    else:
        model = Blip2MoEStage2(args)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Data module
    tokenizer = model.blip2qformer.tokenizer
    dm = CrystalDM(
        mode=args.mode,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        text_max_len=args.text_max_len,
        tokenizer=tokenizer,
        property_list=args.property_list,
        target_property=args.target_property,  # NEW: single property mode
        args=args
    )

    # Callbacks
    callbacks = []
    callbacks.append(plc.ModelCheckpoint(
        dirpath=f'all_checkpoints/{args.filename}/',
        filename='{epoch:02d}',
        every_n_epochs=args.save_every_n_epochs,
        save_last=True,
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    ))

    # Strategy
    if len(args.devices.split(',')) > 1:
        strategy = strategies.DDPStrategy(start_method='spawn')
    else:
        strategy = 'auto'

    # Logger
    logger = CSVLogger(save_dir=f'all_checkpoints/{args.filename}/')

    # Trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )

    # Train
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.test(model, datamodule=dm)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def get_args():
    parser = argparse.ArgumentParser(description='Train BLIP2 MoE Stage 2')

    # General
    parser.add_argument('--filename', type=str, default='moe_stage2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])

    # Hardware
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--precision', type=str, default='bf16-mixed')

    # Training
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    parser.add_argument('--save_every_n_epochs', type=int, default=5)

    # Checkpoint
    parser.add_argument('--init_checkpoint', type=str, default='')

    # Add model and data specific args
    parser = Blip2MoEStage2.add_model_specific_args(parser)
    parser = CrystalDM.add_model_specific_args(parser)

    args = parser.parse_args()

    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"{k:30s} = {v}")
    print("=" * 60)

    return args


if __name__ == '__main__':
    import torch
    main(get_args())
