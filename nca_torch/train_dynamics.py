"""
Training script for NCA dynamics prediction.

Trains the NCA to predict future frames from current frames.
One NCA step = one simulation time step.

Key features:
- Multi-frame context: Encode multiple frames to capture velocity
- Fixed k=1: Each NCA step produces the next frame
- Autoregressive: Always uses own predictions for next step

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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for training
import matplotlib.pyplot as plt
import numpy as np

from autoencoder import NCAAutoencoder, vae_loss
from datasets import SequenceDataset
from perceptual_loss import load_perceptual_loss


def get_args():
    parser = argparse.ArgumentParser(description="Train NCA Dynamics Predictor")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to sequence data (.npy file or directory)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of training samples (for quick experiments)")
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

    # NCA steps per frame
    parser.add_argument("--num-steps", type=int, default=1,
                        help="Starting number of NCA steps (doubles each epoch up to --max-num-steps)")
    parser.add_argument("--max-num-steps", type=int, default=1,
                        help="Maximum NCA steps per frame (schedule: 1, 2, 4, ... up to this)")
    parser.add_argument("--detach-steps", action="store_true",
                        help="Detach grid between steps (each step learns independently)")

    # Latent noise (helps fill in latent space)
    parser.add_argument("--latent-noise-std", type=float, default=0.0,
                        help="Noise std added to latent z during training (helps fill latent space)")

    # Perceptual loss
    parser.add_argument("--perceptual-checkpoint", type=str, default=None,
                        help="Path to pretrained feature extractor checkpoint")
    parser.add_argument("--perceptual-weight", type=float, default=1.0,
                        help="Weight for perceptual loss term")
    parser.add_argument("--recon-weight", type=float, default=0.0,
                        help="Weight for reconstruction loss (0.0 = perceptual only)")

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

        # Load dataset to get dimensions and max future frames
        print(f"Loading data from {args.data}...")
        self.context_frames = args.context_frames
        self.data_path = args.data

        # First load to get sequence info
        temp_dataset = SequenceDataset(
            args.data,
            context_frames=args.context_frames,
            future_frames=1,
        )
        self.max_future_frames = temp_dataset.seq_length - args.context_frames
        self.num_sequences = temp_dataset.num_sequences

        # Infer dimensions from data
        self.in_channels = temp_dataset.channels
        self.grid_size = (temp_dataset.height, temp_dataset.width)
        print(f"Data: {temp_dataset.channels} channels, {self.grid_size} resolution")
        print(f"Context frames: {args.context_frames} (encoder input: {temp_dataset.channels * args.context_frames} channels)")
        print(f"Max future frames: {self.max_future_frames} (sequence length warmup)")

        # Will be set per epoch
        self.train_loader = None
        self.val_loader = None

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

        # Perceptual loss module (optional)
        self.perceptual_loss = None
        if args.perceptual_checkpoint:
            print(f"Loading perceptual loss from {args.perceptual_checkpoint}")
            self.perceptual_loss = load_perceptual_loss(
                checkpoint_path=args.perceptual_checkpoint,
                data_in_channels=self.in_channels,
                device=self.device,
            )
            print(f"Perceptual loss weight: {args.perceptual_weight}")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )

        # Logging
        self.history = {
            "train_loss": [], "train_recon": [], "train_kl": [], "train_perceptual": [],
            "val_loss": [], "val_recon": [], "val_kl": [], "val_perceptual": [],
        }

        # Fixed samples for visualization
        self.fixed_samples = None

    def create_dataloaders(self, future_frames: int):
        """Create train/val dataloaders for a given number of future frames."""
        dataset = SequenceDataset(
            self.data_path,
            context_frames=self.context_frames,
            future_frames=future_frames,
        )

        # Limit samples if requested
        if self.args.max_samples and self.args.max_samples < len(dataset):
            indices = list(range(self.args.max_samples))
            dataset = torch.utils.data.Subset(dataset, indices)

        # Split train/val
        val_size = int(len(dataset) * self.args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        return train_size, val_size

    def train_epoch(self, epoch: int):
        self.model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_perceptual = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Dataset returns: (context_stacked, init_frame, future_targets)
            # future_targets shape: [B, M, C, H, W] where M = future_frames
            context_stacked, init_frame, future_targets = batch

            context_stacked = context_stacked.to(self.device)
            init_frame = init_frame.to(self.device)
            future_targets = future_targets.to(self.device)

            B = context_stacked.shape[0]
            C = self.in_channels
            N = self.context_frames
            H, W = self.grid_size
            M = future_targets.shape[1]  # Number of future frames

            # Keep context for perceptual loss (ground truth, not updated)
            context_window = context_stacked.view(B, N, C, H, W)

            total_step_loss = 0
            total_step_recon = 0
            total_step_kl = 0
            total_step_perceptual = 0

            # ============ Encode ONCE from ground truth context ============
            self.optimizer.zero_grad()

            z, mu, logvar = self.model.encode(context_stacked)

            # Add noise to latent to fill in latent space
            if self.args.latent_noise_std > 0:
                z = z + torch.randn_like(z) * self.args.latent_noise_std

            # Generate NCA params once
            layer1_w, layer1_b, layer2_w, layer2_b = self.model.decoder.generate_params(z)

            # Initialize grid from FirstFrameDecoder (full grid: RGB + hidden channels)
            grid = self.model.first_frame_decoder(z)  # [B, grid_channels, H, W]

            # Total NCA steps = num_steps * M (target advances every num_steps)
            num_steps = self.current_num_steps
            total_nca_steps = num_steps * M
            target_idx = 0

            for step in range(total_nca_steps):
                # Run one NCA step
                grid = self.model.decoder.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)

                # Squash RGB (sigmoid) and hidden (tanh) - applied once per step
                grid = torch.cat([torch.sigmoid(grid[:, :C]), torch.tanh(grid[:, C:])], dim=1)

                # Extract RGB for loss (already in [0,1] from squash)
                pred = grid[:, :C]

                # Loss against current target
                target = future_targets[:, target_idx]  # [B, C, H, W]
                _, recon_loss, kl_loss = vae_loss(
                    pred, target, mu, logvar, self.args.kl_weight
                )

                loss = self.args.recon_weight * recon_loss + self.args.kl_weight * kl_loss

                perceptual_loss_val = 0.0
                if self.perceptual_loss is not None:
                    perceptual_loss_val = self.perceptual_loss(pred, target, context=context_window)
                    loss = loss + self.args.perceptual_weight * perceptual_loss_val

                total_step_loss += loss
                total_step_recon += recon_loss.item()
                total_step_kl += kl_loss.item()
                total_step_perceptual += perceptual_loss_val.item() if torch.is_tensor(perceptual_loss_val) else 0.0

                # Detach grid to cut gradient flow between steps
                if self.args.detach_steps:
                    grid = grid.detach()

                # Advance target every num_steps
                if (step + 1) % num_steps == 0 and target_idx < M - 1:
                    target_idx += 1

            # Average loss over all NCA steps
            avg_loss = total_step_loss / total_nca_steps

            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += avg_loss.item()
            total_recon += total_step_recon / M
            total_kl += total_step_kl / M
            total_perceptual += total_step_perceptual / M
            num_batches += 1

            if batch_idx % self.args.log_interval == 0:
                pbar.set_postfix({"loss": f"{avg_loss.item():.4f}"})

        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_kl / num_batches,
            total_perceptual / num_batches,
        )

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_perceptual = 0
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
            M = future_targets.shape[1]

            context_window = context_stacked.view(B, N, C, H, W)

            # Encode once from ground truth context (matches training)
            z, mu, logvar = self.model.encode(context_stacked)

            # Generate NCA params once
            layer1_w, layer1_b, layer2_w, layer2_b = self.model.decoder.generate_params(z)

            # Initialize grid from FirstFrameDecoder (full grid: RGB + hidden channels)
            grid = self.model.first_frame_decoder(z)  # [B, grid_channels, H, W]

            # Total NCA steps = num_steps * M (target advances every num_steps)
            num_steps = self.current_num_steps
            total_nca_steps = num_steps * M
            target_idx = 0
            batch_loss = 0

            for step in range(total_nca_steps):
                grid = self.model.decoder.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)

                # Squash first, then extract for loss
                grid = torch.cat([torch.sigmoid(grid[:, :C]), torch.tanh(grid[:, C:])], dim=1)
                pred = grid[:, :C]

                target = future_targets[:, target_idx]
                _, recon_loss, kl_loss = vae_loss(
                    pred, target, mu, logvar, self.args.kl_weight
                )

                loss = self.args.recon_weight * recon_loss + self.args.kl_weight * kl_loss

                perceptual_loss_val = 0.0
                if self.perceptual_loss is not None:
                    perceptual_loss_val = self.perceptual_loss(pred, target, context=context_window)
                    loss = loss + self.args.perceptual_weight * perceptual_loss_val

                batch_loss += loss.item()

                # Advance target every num_steps
                if (step + 1) % num_steps == 0 and target_idx < M - 1:
                    target_idx += 1

            total_loss += batch_loss / total_nca_steps
            total_recon += batch_loss / total_nca_steps
            total_kl += kl_loss.item()
            total_perceptual += perceptual_loss_val.item() if torch.is_tensor(perceptual_loss_val) else 0.0
            num_batches += 1

            # Store fixed samples for visualization (update each epoch as sequence grows)
            if self.fixed_samples is None or future_targets.shape[1] > self.fixed_samples["future_targets"].shape[1]:
                self.fixed_samples = {
                    "context_stacked": context_stacked[:8].clone(),
                    "init_frame": init_frame[:8].clone(),
                    "future_targets": future_targets[:8].clone(),
                }

        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_kl / num_batches,
            total_perceptual / num_batches,
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

            # Row 2: Prediction (t+1) using FirstFrameDecoder → NCA
            pred = self.model.decode(
                z[col:col+1],
                num_steps=self.current_num_steps,
                init_mode="first_frame",
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
        C = self.in_channels
        H, W = self.grid_size

        # Encode ONCE from initial context
        z, _, _ = self.model.encode(context_stacked)

        # Generate NCA parameters for dynamics
        layer1_w, layer1_b, layer2_w, layer2_b = self.model.decoder.generate_params(z)

        # Initialize grid from FirstFrameDecoder (full grid: RGB + hidden channels)
        grid = self.model.first_frame_decoder(z)  # [1, grid_channels, H, W]

        frames = [grid[:, :C].clone()]  # First frame is RGB portion

        # Run NCA steps, capturing frame every num_steps
        num_steps = self.current_num_steps
        total_nca_steps = rollout_steps * num_steps
        for step in range(total_nca_steps):
            grid = self.model.decoder.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)

            # Squash RGB (sigmoid) and hidden (tanh)
            grid = torch.cat([torch.sigmoid(grid[:, :C]), torch.tanh(grid[:, C:])], dim=1)

            # Capture frame every num_steps
            if (step + 1) % num_steps == 0:
                frames.append(grid[:, :C].clone())

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
            "num_steps": self.args.max_num_steps,
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
        print(f"NCA steps schedule: 1 → {self.args.max_num_steps} (doubles each epoch)")
        print(f"Detach steps: {self.args.detach_steps}")
        print(f"Max future frames: {self.max_future_frames} (sequence warmup: 1 → max)")
        print(f"First frame: FirstFrameDecoder, dynamics: NCA (up to {self.args.max_num_steps} steps)")
        print(f"Loss weights: recon={self.args.recon_weight}, kl={self.args.kl_weight}", end="")
        if self.perceptual_loss is not None:
            print(f", perceptual={self.args.perceptual_weight}")
        else:
            print()

        best_loss = float("inf")
        self.current_num_steps = 1  # Will be updated each epoch

        for epoch in range(1, self.args.epochs + 1):
            # Sequence length warmup: epoch 1 = 1 future frame, epoch 2 = 2, etc.
            future_frames = min(epoch, self.max_future_frames)
            train_size, val_size = self.create_dataloaders(future_frames)

            # NCA steps warmup: doubles each epoch (1, 2, 4, 8, ...) up to max
            self.current_num_steps = min(2 ** (epoch - 1), self.args.max_num_steps)
            print(f"\nEpoch {epoch}: future_frames={future_frames}, num_steps={self.current_num_steps}, samples={train_size}")

            train_loss, train_recon, train_kl, train_perceptual = self.train_epoch(epoch)
            val_loss, val_recon, val_kl, val_perceptual = self.evaluate()

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_recon"].append(train_recon)
            self.history["train_kl"].append(train_kl)
            self.history["train_perceptual"].append(train_perceptual)
            self.history["val_loss"].append(val_loss)
            self.history["val_recon"].append(val_recon)
            self.history["val_kl"].append(val_kl)
            self.history["val_perceptual"].append(val_perceptual)

            if self.perceptual_loss is not None:
                print(f"Epoch {epoch}: train={train_loss:.4f} (perc={train_perceptual:.4f}), val={val_loss:.4f}")
            else:
                print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

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
