"""
Training script for NCA dynamics prediction.

Trains the NCA to predict future frames from current frames.
One NCA step = one simulation time step.

Key features:
- Multi-frame context: Encode multiple frames to capture velocity
- Fixed k=1: Each NCA step produces the next frame
- Scheduled sampling: Train on model predictions to reduce autoregressive error accumulation

Usage:
    python train_dynamics.py --data ./data/boids/boids_32x32.npy --context-frames 4
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from autoencoder import NCAAutoencoder, vae_loss
from datasets import SequenceDataset


def get_args():
    parser = argparse.ArgumentParser(description="Train NCA Dynamics Predictor")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to sequence data (.npy file or directory)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation")

    # Model
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--grid-channels", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--no-vae", action="store_true",
                        help="Use regular autoencoder instead of VAE")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.0001)

    # Context
    parser.add_argument("--context-frames", type=int, default=4,
                        help="Number of context frames for velocity encoding")

    # Scheduled sampling
    parser.add_argument("--ss-steps", type=int, default=8,
                        help="Number of autoregressive steps during scheduled sampling training")
    parser.add_argument("--ss-start-prob", type=float, default=0.0,
                        help="Initial probability of using model predictions (0 = always use ground truth)")
    parser.add_argument("--ss-end-prob", type=float, default=0.5,
                        help="Final probability of using model predictions")
    parser.add_argument("--ss-warmup-epochs", type=int, default=10,
                        help="Epochs to linearly increase scheduled sampling probability")

    # Init noise (helps NCA learn to start from noisy conditions)
    parser.add_argument("--init-noise-std", type=float, default=0.1,
                        help="Max noise std added to init grid (sampled uniformly from [0, this])")

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="checkpoints_dynamics")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--visualize-interval", type=int, default=1)

    return parser.parse_args()


class DynamicsTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Load dataset with multiple future frames for scheduled sampling
        print(f"Loading data from {args.data}...")
        self.context_frames = args.context_frames
        self.ss_steps = args.ss_steps

        full_dataset = SequenceDataset(
            args.data,
            context_frames=args.context_frames,
            future_frames=args.ss_steps,  # Get multiple future frames
        )

        # Infer dimensions from data
        self.in_channels = full_dataset.channels
        self.grid_size = (full_dataset.height, full_dataset.width)
        print(f"Data: {full_dataset.channels} channels, {self.grid_size} resolution")
        print(f"Context frames: {args.context_frames} (encoder input: {full_dataset.channels * args.context_frames} channels)")
        print(f"Scheduled sampling steps: {args.ss_steps}")

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

        # Model
        self.model = NCAAutoencoder(
            latent_dim=args.latent_dim,
            grid_channels=args.grid_channels,
            hidden_dim=args.hidden_dim,
            use_vae=not args.no_vae,
            in_channels=self.in_channels,
            grid_size=self.grid_size,
            context_frames=args.context_frames,
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

        # Logging
        self.history = {
            "train_loss": [], "train_recon": [], "train_kl": [],
            "val_loss": [], "val_recon": [], "val_kl": [],
            "ss_prob": [],
        }

        # Fixed samples for visualization
        self.fixed_samples = None

    def get_ss_prob(self, epoch: int) -> float:
        """Get scheduled sampling probability for current epoch."""
        if epoch <= self.args.ss_warmup_epochs:
            # Linear warmup from start_prob to end_prob
            progress = epoch / self.args.ss_warmup_epochs
            return self.args.ss_start_prob + progress * (self.args.ss_end_prob - self.args.ss_start_prob)
        return self.args.ss_end_prob

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        ss_prob = self.get_ss_prob(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Dataset returns: (context_stacked, init_frame, future_targets)
            # future_targets shape: [B, M, C, H, W] where M = ss_steps
            context_stacked, init_frame, future_targets = batch

            context_stacked = context_stacked.to(self.device)
            init_frame = init_frame.to(self.device)
            future_targets = future_targets.to(self.device)

            B = context_stacked.shape[0]
            C = self.in_channels
            N = self.context_frames
            H, W = self.grid_size
            M = future_targets.shape[1]  # Number of future frames

            self.optimizer.zero_grad()

            # Reshape context for sliding window updates: [B, N, C, H, W]
            context_window = context_stacked.view(B, N, C, H, W)

            total_step_loss = 0
            total_step_recon = 0
            total_step_kl = 0

            current_frame = init_frame

            for step in range(M):
                # Stack context: [B, N*C, H, W]
                encoder_input = context_window.view(B, N * C, H, W)

                # Sample random init noise (helps NCA learn to start from noisy conditions)
                init_noise = torch.rand(1).item() * self.args.init_noise_std

                # Encode and predict next frame (k=1)
                z, mu, logvar = self.model.encode(encoder_input)

                # Sample noise uniformly from [0, max_noise] to prevent identity shortcut.
                # At low noise: learns to continue from clean frames
                # At high noise: forced to rely on latent encoding
                decode_init_noise_std = torch.rand(1).item() * self.args.init_noise_std

                pred = self.model.decode(
                    z,
                    num_steps=1,  # Fixed k=1
                    init_mode="image_noisy",
                    init_images=current_frame,
                    init_noise_std=decode_init_noise_std,
                )

                # Loss against ground truth target
                target = future_targets[:, step]  # [B, C, H, W]
                loss, recon_loss, kl_loss = vae_loss(
                    pred, target, mu, logvar, self.args.kl_weight
                )

                total_step_loss += loss
                total_step_recon += recon_loss.item()
                total_step_kl += kl_loss.item()

                # Scheduled sampling: decide whether to use prediction or ground truth
                # for the next step's context window
                if step < M - 1:  # Don't need to update for last step
                    use_prediction = torch.rand(1).item() < ss_prob

                    if use_prediction:
                        next_frame = pred.detach()  # Use model's prediction
                    else:
                        next_frame = target  # Use ground truth

                    # Update context window: drop oldest, add newest
                    context_window = torch.cat([
                        context_window[:, 1:],  # Drop first frame
                        next_frame.unsqueeze(1)  # Add new frame
                    ], dim=1)

                    current_frame = next_frame

            # Average loss over steps
            loss = total_step_loss / M

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += total_step_recon / M
            total_kl += total_step_kl / M
            num_batches += 1

            if batch_idx % self.args.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{total_step_recon / M:.4f}",
                    "ss_prob": f"{ss_prob:.2f}",
                })

        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_kl / num_batches,
        )

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        for batch in self.val_loader:
            context_stacked, init_frame, future_targets = batch

            context_stacked = context_stacked.to(self.device)
            init_frame = init_frame.to(self.device)
            future_targets = future_targets.to(self.device)

            B = context_stacked.shape[0]
            C = self.in_channels
            N = self.context_frames
            H, W = self.grid_size

            # For validation, just evaluate single-step prediction (no noise, no scheduled sampling)
            z, mu, logvar = self.model.encode(context_stacked)
            pred = self.model.decode(
                z,
                num_steps=1,
                init_mode="image",  # Clean init for evaluation
                init_images=init_frame,
            )

            # Loss against first future frame
            target = future_targets[:, 0]
            loss, recon_loss, kl_loss = vae_loss(
                pred, target, mu, logvar, self.args.kl_weight
            )

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

            # Store fixed samples for visualization
            if self.fixed_samples is None:
                self.fixed_samples = {
                    "context_stacked": context_stacked[:8].clone(),
                    "init_frame": init_frame[:8].clone(),
                    "future_targets": future_targets[:8].clone(),
                }

        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_kl / num_batches,
        )

    @torch.no_grad()
    def visualize(self, epoch: int):
        """Visualize single-step predictions."""
        self.model.eval()

        if self.fixed_samples is None:
            return

        context_stacked = self.fixed_samples["context_stacked"]
        init_frame = self.fixed_samples["init_frame"]
        future_targets = self.fixed_samples["future_targets"]

        n_samples = min(8, init_frame.shape[0])
        n_rows = 3  # input, target, prediction

        fig, axes = plt.subplots(n_rows, n_samples, figsize=(n_samples * 2, n_rows * 2))

        z, _, _ = self.model.encode(context_stacked[:n_samples])

        for col in range(n_samples):
            # Row 0: Init frame (last context frame)
            img = init_frame[col].permute(1, 2, 0).cpu().numpy()
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                axes[0, col].imshow(img, cmap='gray')
            else:
                axes[0, col].imshow(img)
            axes[0, col].axis('off')
            if col == 0:
                axes[0, col].set_ylabel("Input (t)", fontsize=10)

            # Row 1: Target frame (t+1)
            img = future_targets[col, 0].permute(1, 2, 0).cpu().numpy()
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                axes[1, col].imshow(img, cmap='gray')
            else:
                axes[1, col].imshow(img)
            axes[1, col].axis('off')
            if col == 0:
                axes[1, col].set_ylabel("Target (t+1)", fontsize=10)

            # Row 2: Prediction (t+1)
            pred = self.model.decode(
                z[col:col+1],
                num_steps=1,
                init_mode="image",
                init_images=init_frame[col:col+1],
            )
            img = pred[0].permute(1, 2, 0).cpu().numpy()
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                axes[2, col].imshow(img, cmap='gray')
            else:
                axes[2, col].imshow(img)
            axes[2, col].axis('off')
            if col == 0:
                axes[2, col].set_ylabel("Pred (t+1)", fontsize=10)

        plt.suptitle(f'Single-Step Prediction (Epoch {epoch})')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"viz_epoch_{epoch:03d}.png", dpi=150)
        plt.close()

    @torch.no_grad()
    def visualize_rollout(self, epoch: int):
        """Visualize autoregressive rollout without re-encoding.

        This shows the intended inference mode: encode once, then let the NCA
        run autonomously for many steps.
        """
        self.model.eval()

        if self.fixed_samples is None:
            return

        context_stacked = self.fixed_samples["context_stacked"][0:1].clone()
        init_frame = self.fixed_samples["init_frame"][0:1].clone()
        future_targets = self.fixed_samples["future_targets"][0:1].clone()

        rollout_steps = 32

        # Encode ONCE from initial context
        z, _, _ = self.model.encode(context_stacked)

        # Generate NCA parameters once
        layer1_w, layer1_b, layer2_w, layer2_b = self.model.decoder.generate_params(z)

        # Initialize NCA grid from init_frame
        grid = self.model.decoder.init_grid(
            batch_size=1,
            grid_size=self.grid_size,
            device=self.device,
            init_mode="image",
            init_images=init_frame,
        )

        frames = [torch.sigmoid(grid[:, :self.in_channels]).clone()]

        # Run NCA autonomously (no re-encoding!)
        for _ in range(rollout_steps):
            grid = self.model.decoder.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)
            frame = torch.sigmoid(grid[:, :self.in_channels])
            frames.append(frame.clone())

        # Plot: ground truth on top, predictions on bottom
        n_frames = min(16, len(frames))
        n_gt = min(n_frames, future_targets.shape[1] + 1)  # +1 for init frame

        fig, axes = plt.subplots(2, n_frames, figsize=(n_frames * 1.5, 3))

        for i in range(n_frames):
            # Row 0: Ground truth (where available)
            if i == 0:
                img = init_frame[0].permute(1, 2, 0).cpu().numpy()
            elif i <= future_targets.shape[1]:
                img = future_targets[0, i-1].permute(1, 2, 0).cpu().numpy()
            else:
                img = np.ones_like(img) * 0.5  # Gray for unavailable

            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                axes[0, i].imshow(np.clip(img, 0, 1))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel("GT", fontsize=10)

            # Row 1: NCA rollout
            idx = i * len(frames) // n_frames
            img = frames[idx][0].permute(1, 2, 0).cpu().numpy()
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                axes[1, i].imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                axes[1, i].imshow(np.clip(img, 0, 1))
            axes[1, i].axis('off')
            axes[1, i].set_title(f"t={idx}", fontsize=8)
            if i == 0:
                axes[1, i].set_ylabel("NCA", fontsize=10)

        plt.suptitle(f'Autonomous NCA Rollout (Epoch {epoch})')
        plt.tight_layout()
        plt.savefig(self.save_dir / f"rollout_epoch_{epoch:03d}.png", dpi=150)
        plt.close()

    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "args": vars(self.args),
            "in_channels": self.in_channels,
            "grid_size": self.grid_size,
            "context_frames": self.context_frames,
        }
        torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, self.save_dir / "checkpoint_latest.pt")

    def save_history(self):
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f)

    def train(self):
        print(f"\nTraining NCA Dynamics Predictor on {self.device}")
        print(f"VAE mode: {not self.args.no_vae}")
        print(f"Context frames: {self.context_frames}")
        print(f"Scheduled sampling: {self.args.ss_steps} steps, "
              f"prob {self.args.ss_start_prob} -> {self.args.ss_end_prob} "
              f"over {self.args.ss_warmup_epochs} epochs")

        best_loss = float("inf")

        for epoch in range(1, self.args.epochs + 1):
            ss_prob = self.get_ss_prob(epoch)
            train_loss, train_recon, train_kl = self.train_epoch(epoch)
            val_loss, val_recon, val_kl = self.evaluate()

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_recon"].append(train_recon)
            self.history["train_kl"].append(train_kl)
            self.history["val_loss"].append(val_loss)
            self.history["val_recon"].append(val_recon)
            self.history["val_kl"].append(val_kl)
            self.history["ss_prob"].append(ss_prob)

            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, ss_prob={ss_prob:.2f}")

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print(f"  New best model saved!")

            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch, val_loss)
                self.save_history()

            if epoch % self.args.visualize_interval == 0:
                self.visualize(epoch)
                self.visualize_rollout(epoch)

        self.save_history()
        print(f"\nTraining complete. Best val loss: {best_loss:.4f}")


def main():
    args = get_args()
    trainer = DynamicsTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
