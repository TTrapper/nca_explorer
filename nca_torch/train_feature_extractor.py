"""
Training script for the 3D perceptual loss feature extractor.

Trains a 3D autoencoder on sequence data to learn spatiotemporal features
that capture motion/dynamics. The trained encoder is then used as a frozen
feature extractor for perceptual loss during dynamics training.

Usage:
    python train_feature_extractor.py --data ./data/boids/boids_32x32.npy --epochs 20
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for training
import matplotlib.pyplot as plt
import numpy as np

from perceptual_loss import SimpleFeatureAutoencoder3D
from datasets import SequenceDataset


def get_args():
    parser = argparse.ArgumentParser(description="Train 3D Feature Extractor for Perceptual Loss")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to sequence data (.npy file or directory)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--seq-length", type=int, default=4,
                        help="Number of frames per sequence for 3D conv")

    # Model
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[32, 64, 128],
                        help="Hidden dimensions for encoder layers")
    parser.add_argument("--temporal-kernel", type=int, default=3,
                        help="Temporal kernel size for 3D convolutions")

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints_feature_extractor")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--visualize-interval", type=int, default=1)

    return parser.parse_args()


class FeatureExtractorTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Load dataset - use seq_length frames as context, 1 future frame
        # We'll use context_frames to get sequences
        print(f"Loading data from {args.data}...")
        full_dataset = SequenceDataset(
            args.data,
            context_frames=args.seq_length,
            future_frames=1,
        )

        # Infer dimensions from data
        self.in_channels = full_dataset.channels
        self.height = full_dataset.height
        self.width = full_dataset.width
        self.seq_length = args.seq_length
        print(f"Data: {self.in_channels} channels, {self.height}x{self.width} resolution")
        print(f"Sequence length: {self.seq_length} frames")

        # Split train/val
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"Train: {train_size}, Val: {val_size}")

        # Model - 3D autoencoder
        self.model = SimpleFeatureAutoencoder3D(
            in_channels=self.in_channels,
            hidden_dims=args.hidden_dims,
            temporal_kernel=args.temporal_kernel,
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Hidden dims: {args.hidden_dims}")
        print(f"Temporal kernel: {args.temporal_kernel}")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

        # Logging
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

        # Fixed samples for visualization
        self.fixed_samples = None

    def _prepare_sequence(self, context_stacked: torch.Tensor) -> torch.Tensor:
        """
        Convert stacked context frames to sequence format for 3D conv.

        Args:
            context_stacked: [B, C*T, H, W] stacked frames

        Returns:
            sequence: [B, C, T, H, W] for 3D conv
        """
        B, CT, H, W = context_stacked.shape
        C = self.in_channels
        T = CT // C

        # Reshape: [B, C*T, H, W] -> [B, T, C, H, W] -> [B, C, T, H, W]
        sequence = context_stacked.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return sequence

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Dataset returns (context_stacked, init_frame, target_frame)
            context_stacked, _, _ = batch
            context_stacked = context_stacked.to(self.device)

            # Convert to 3D sequence format: [B, C, T, H, W]
            sequence = self._prepare_sequence(context_stacked)

            self.optimizer.zero_grad()

            # Reconstruct the sequence
            recon, _ = self.model(sequence)

            # MSE reconstruction loss
            loss = F.mse_loss(recon, sequence)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.args.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            context_stacked, _, _ = batch
            context_stacked = context_stacked.to(self.device)

            sequence = self._prepare_sequence(context_stacked)

            recon, _ = self.model(sequence)
            loss = F.mse_loss(recon, sequence)

            total_loss += loss.item()
            num_batches += 1

            # Store fixed samples for visualization
            if self.fixed_samples is None:
                self.fixed_samples = sequence[:8].clone()

        return total_loss / num_batches

    @torch.no_grad()
    def visualize(self, epoch: int):
        """Visualize sequence reconstructions."""
        self.model.eval()

        if self.fixed_samples is None:
            return

        n_samples = min(4, self.fixed_samples.shape[0])
        n_frames = self.fixed_samples.shape[2]  # T dimension

        recon, _ = self.model(self.fixed_samples[:n_samples])

        # Create figure: 2 rows per sample (original, reconstruction), n_frames columns
        fig, axes = plt.subplots(n_samples * 2, n_frames, figsize=(n_frames * 2, n_samples * 4))

        for sample_idx in range(n_samples):
            for frame_idx in range(n_frames):
                # Original
                row = sample_idx * 2
                img = self.fixed_samples[sample_idx, :, frame_idx].permute(1, 2, 0).cpu().numpy()
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                    axes[row, frame_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
                else:
                    axes[row, frame_idx].imshow(np.clip(img, 0, 1))
                axes[row, frame_idx].axis('off')
                if frame_idx == 0:
                    axes[row, frame_idx].set_ylabel(f"Orig {sample_idx}", fontsize=8)

                # Reconstruction
                row = sample_idx * 2 + 1
                img = recon[sample_idx, :, frame_idx].permute(1, 2, 0).cpu().numpy()
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                    axes[row, frame_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
                else:
                    axes[row, frame_idx].imshow(np.clip(img, 0, 1))
                axes[row, frame_idx].axis('off')
                if frame_idx == 0:
                    axes[row, frame_idx].set_ylabel(f"Recon {sample_idx}", fontsize=8)

        plt.suptitle(f'3D Feature Extractor Reconstruction (Epoch {epoch})')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"viz_epoch_{epoch:03d}.png", dpi=150)
        plt.close()

    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "args": {
                **vars(self.args),
                "in_channels": self.in_channels,
                "height": self.height,
                "width": self.width,
                "temporal_kernel": self.args.temporal_kernel,
            },
            "in_channels": self.in_channels,
            "hidden_dims": self.args.hidden_dims,
            "temporal_kernel": self.args.temporal_kernel,
        }
        torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, self.save_dir / "checkpoint_latest.pt")

    def save_history(self):
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f)

    def train(self):
        print(f"\nTraining 3D Feature Extractor on {self.device}")
        print(f"Epochs: {self.args.epochs}, LR: {self.args.lr}")

        best_loss = float("inf")

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print(f"  New best model saved!")

            if epoch % self.args.visualize_interval == 0:
                self.visualize(epoch)

        # Always save final checkpoint
        self.save_checkpoint(self.args.epochs, val_loss)
        self.save_history()
        print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
        print(f"Checkpoint saved to: {self.save_dir / 'checkpoint_latest.pt'}")


def main():
    args = get_args()
    trainer = FeatureExtractorTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
