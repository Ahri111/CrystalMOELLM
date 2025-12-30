"""
Stage 1: Modified BLIP-2 Q-Former Training
- ALIGNN encoder + Q-Former
- ITC + ITM + Masked Prediction Loss
- No ITG loss
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import argparse

from model.blip2qformer_bandgap import Blip2QformerBandGap
from data_provider.bandgap_dataset import BandGapDataset, collate_fn


class Stage1Module(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Blip2QformerBandGap(
            gtm=args.gtm,
            bert_name=args.bert_name,
            temperature=args.temperature,
            num_query_token=args.num_query_token,
            cross_attention_freq=args.cross_attention_freq,
            embed_dim=args.embed_dim,
            args=args,
        )

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        self.log('train_loss', output.loss, prog_bar=True)
        self.log('train_itc', output.loss_itc, prog_bar=True)
        self.log('train_itm', output.loss_itm)
        self.log('train_masked', output.loss_lm, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        self.log('val_loss', output.loss, prog_bar=True, sync_dist=True)
        self.log('val_itc', output.loss_itc, sync_dist=True)
        self.log('val_itm', output.loss_itm, sync_dist=True)
        self.log('val_masked', output.loss_lm, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.max_epochs,
            eta_min=self.args.min_lr
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def main(args):
    # Dataset
    train_dataset = BandGapDataset(
        csv_file=args.train_csv,
        max_text_len=args.max_text_len,
    )
    val_dataset = BandGapDataset(
        csv_file=args.val_csv,
        max_text_len=args.max_text_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Model
    model = Stage1Module(args)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='stage1-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator='gpu' if args.devices > 0 else 'cpu',
        strategy='ddp' if args.devices > 1 else 'auto',
        callbacks=[checkpoint_callback, lr_monitor],
        precision=args.precision,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--max_text_len', type=int, default=512)

    # Model
    parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
    parser.add_argument('--num_query_token', type=int, default=32)
    parser.add_argument('--cross_attention_freq', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--gtm', action='store_true', help='Use Graph-Text Matching')

    # ALIGNN
    parser.add_argument('--alignn_layers', type=int, default=4)
    parser.add_argument('--gcn_layers', type=int, default=4)
    parser.add_argument('--hidden_features', type=int, default=256)
    parser.add_argument('--embedding_features', type=int, default=64)
    parser.add_argument('--tune_gnn', action='store_true', help='Fine-tune ALIGNN')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage1')

    args = parser.parse_args()

    main(args)
