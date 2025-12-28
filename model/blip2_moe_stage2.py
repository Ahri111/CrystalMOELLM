"""
BLIP2 MoE Stage 2: Frozen MoE + Q-Former Training
Follows the pattern of blip2_stage1.py
"""

import torch
import contextlib
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler
from model.blip2qformer import Blip2Qformer
from model.help_funcs import AttrDict
from typing import Any, Dict


def precision2dtype(precision):
    if precision == '16':
        return torch.float16
    elif precision == '32':
        return torch.float32
    elif precision.find('bf16') >= 0:
        return torch.bfloat16
    else:
        raise NotImplementedError()


class Blip2MoEStage2(pl.LightningModule):
    """
    BLIP2 MoE Stage 2 Training
    - Frozen MoE Encoder
    - Trainable Q-Former
    - 4 Losses: ITC, ITM (optional), LM, Property
    """

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args

        # Create Blip2Qformer with MoE
        self.blip2qformer = Blip2Qformer(
            gtm=args.gtm,
            lm=args.lm,
            bert_name=args.bert_name,
            temperature=args.temperature,
            gin_num_layers=args.gin_num_layers,
            gin_hidden_dim=args.gin_hidden_dim,
            gin_drop_ratio=args.drop_ratio,
            tune_gnn=False,  # MoE is frozen
            num_query_token=args.num_query_token,
            cross_attention_freq=args.cross_attention_freq,
            embed_dim=args.projection_dim,
            args=args
        )

        self.save_hyperparameters(args)

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.args.init_lr,
            weight_decay=self.args.weight_decay
        )

        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                self.args.max_epochs,
                self.args.min_lr,
                self.args.init_lr,
                warmup_steps,
                self.args.warmup_lr
            )
        else:
            self.scheduler = None

        return optimizer

    def training_step(self, batch, batch_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[0].batch_size  # DGL graph batch size

        blip2_loss = self.blip2qformer(batch)

        # Log losses
        self.log("train_loss_itc", float(blip2_loss.loss_itc), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_itm", float(blip2_loss.loss_itm), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_lm", float(blip2_loss.loss_lm), batch_size=batch_size, sync_dist=True)

        # Property loss (MoE specific)
        if hasattr(self.blip2qformer, 'use_moe') and self.blip2qformer.use_moe:
            # Calculate property loss from total loss
            loss_property = blip2_loss.loss - blip2_loss.loss_itc - blip2_loss.loss_itm - blip2_loss.loss_lm
            self.log("train_loss_property", float(loss_property), batch_size=batch_size, sync_dist=True)

        self.log("train_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)

        return blip2_loss.loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = batch[0].batch_size

        blip2_loss = self.blip2qformer(batch)

        # Log losses
        self.log("val_loss_itc", float(blip2_loss.loss_itc), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_itm", float(blip2_loss.loss_itm), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_lm", float(blip2_loss.loss_lm), batch_size=batch_size, sync_dist=True)

        if hasattr(self.blip2qformer, 'use_moe') and self.blip2qformer.use_moe:
            loss_property = blip2_loss.loss - blip2_loss.loss_itc - blip2_loss.loss_itm - blip2_loss.loss_lm
            self.log("val_loss_property", float(loss_property), batch_size=batch_size, sync_dist=True)

        self.log("val_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Blip2MoEStage2")

        # Training
        parser.add_argument('--temperature', type=float, default=0.1)
        parser.add_argument('--gtm', action='store_true', default=False)
        parser.add_argument('--lm', action='store_true', default=True)

        # Model
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--projection_dim', type=int, default=256)
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=32)

        # MoE specific
        parser.add_argument('--use_moe', action='store_true', default=True)
        parser.add_argument('--moe_extractor_dir', type=str, default='checkpoints/extractors')
        parser.add_argument('--moe_topk_config', type=str, default='config/moe_topk.json')

        # Optimization
        parser.add_argument('--weight_decay', type=float, default=0.05)
        parser.add_argument('--init_lr', type=float, default=1e-4)
        parser.add_argument('--min_lr', type=float, default=1e-6)
        parser.add_argument('--warmup_lr', type=float, default=1e-6)
        parser.add_argument('--warmup_steps', type=int, default=1000)
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr')

        return parent_parser

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Remove optimizer states to save space
        checkpoint.pop('optimizer_states', None)

        # Remove frozen parameters
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)

        for key in to_be_removed:
            checkpoint['state_dict'].pop(key, None)
