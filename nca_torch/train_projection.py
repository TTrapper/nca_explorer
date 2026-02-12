#!/usr/bin/env python3
"""
Train a projection between two model latent spaces.

Given two trained models with potentially different latent spaces, this script
learns a mapping from the source latent space to the target latent space.

Modes:
- distribution: Match overall statistics (whiten source, color with target)
- nearest: Train linear projection on nearest-neighbor pairs
- codebook: Distribution matching + k-means codebook for snapping to valid points
- ot: Optimal Transport matching + MLP projection (nonlinear, structure-preserving)

Usage:
    python train_projection.py \
        --source-checkpoint checkpoints_audio/checkpoint_latest.pt \
        --source-data data/audio_visual.npy \
        --target-checkpoint checkpoints_emoji/checkpoint_latest.pt \
        --target-data data/emoji_32x32.npy \
        --output projection.pt \
        --mode ot
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

from autoencoder import NCAAutoencoder
from datasets import SequenceDataset


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})

    model = NCAAutoencoder(
        latent_dim=args.get("latent_dim", 64),
        grid_channels=args.get("grid_channels", 16),
        hidden_dim=args.get("hidden_dim", 256),
        use_vae=not args.get("no_vae", False),
        in_channels=checkpoint.get("in_channels", 3),
        grid_size=checkpoint.get("grid_size", (32, 32)),
        context_frames=checkpoint.get("context_frames", 1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def encode_dataset(model, dataset, device, batch_size=64, num_workers=4):
    """Encode all samples in a dataset to latent vectors."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_z = []
    for batch in tqdm(loader, desc="Encoding"):
        context_stacked, _, _ = batch
        context_stacked = context_stacked.to(device)

        z, mu, _ = model.encode(context_stacked)
        # Use mu for VAE (deterministic), z for non-VAE
        latent = mu if mu is not None else z
        all_z.append(latent.cpu())

    return torch.cat(all_z, dim=0)


def find_nearest_neighbors(source_z: torch.Tensor, target_z: torch.Tensor, batch_size=1000):
    """Find nearest neighbor in target_z for each point in source_z.

    Returns indices into target_z for each source point.
    """
    n_source = source_z.shape[0]
    n_target = target_z.shape[0]

    # Normalize for cosine similarity (often works better than L2)
    source_norm = F.normalize(source_z, dim=1)
    target_norm = F.normalize(target_z, dim=1)

    indices = []
    for i in tqdm(range(0, n_source, batch_size), desc="Finding neighbors"):
        batch = source_norm[i:i+batch_size]
        # Cosine similarity
        sim = batch @ target_norm.T  # [batch, n_target]
        nearest = sim.argmax(dim=1)
        indices.append(nearest)

    return torch.cat(indices, dim=0)


def compute_whitening_transform(z: torch.Tensor):
    """Compute whitening transform (zero mean, identity covariance)."""
    mean = z.mean(dim=0)
    centered = z - mean
    cov = (centered.T @ centered) / (z.shape[0] - 1)
    # Eigendecomposition for whitening
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # Clamp small eigenvalues for numerical stability
    eigvals = eigvals.clamp(min=1e-6)
    # Whitening matrix: V @ diag(1/sqrt(λ)) @ V^T
    whiten = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    return mean, whiten


def compute_coloring_transform(z: torch.Tensor):
    """Compute coloring transform (applies target covariance structure)."""
    mean = z.mean(dim=0)
    centered = z - mean
    cov = (centered.T @ centered) / (z.shape[0] - 1)
    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=1e-6)
    # Coloring matrix: V @ diag(sqrt(λ)) @ V^T
    color = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    return mean, color


def compute_codebook(z: torch.Tensor, n_clusters: int, random_state: int = 42):
    """Cluster latent vectors using k-means and return centroids."""
    print(f"  Running k-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(z.numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    labels = torch.from_numpy(kmeans.labels_)
    return centroids, labels


def analyze_codebook_spread(
    source_z: torch.Tensor,
    projection: nn.Module,
    codebook: torch.Tensor,
    device: torch.device,
):
    """Analyze how source samples distribute across codebook entries."""
    with torch.no_grad():
        # Project source to target space
        projected = projection(source_z.to(device)).cpu()

        # Find nearest codebook entry for each projected sample
        # Use L2 distance
        dists = torch.cdist(projected, codebook)  # [n_source, n_clusters]
        assignments = dists.argmin(dim=1)  # [n_source]

        # Count assignments per cluster
        n_clusters = codebook.shape[0]
        counts = torch.bincount(assignments, minlength=n_clusters)

        # Compute statistics
        n_used = (counts > 0).sum().item()
        n_source = source_z.shape[0]
        entropy = -sum(
            (c / n_source) * np.log(c / n_source + 1e-10)
            for c in counts.numpy() if c > 0
        )
        max_entropy = np.log(n_clusters)
        normalized_entropy = entropy / max_entropy

        print(f"\n  Codebook spread analysis:")
        print(f"    Clusters used: {n_used}/{n_clusters} ({100*n_used/n_clusters:.1f}%)")
        print(f"    Entropy: {entropy:.2f} / {max_entropy:.2f} ({100*normalized_entropy:.1f}%)")
        print(f"    Top 5 clusters: {counts.topk(5).values.tolist()}")
        print(f"    Bottom 5 non-zero: {sorted([c.item() for c in counts if c > 0])[:5]}")

        return {
            "n_used": n_used,
            "n_clusters": n_clusters,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized_entropy,
            "counts": counts.tolist(),
        }


class LinearProjection(nn.Module):
    """Linear projection with optional bias."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class MLPProjection(nn.Module):
    """Nonlinear MLP projection."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_ot_matching(source_z: torch.Tensor, target_z: torch.Tensor, reg: float = 0.05):
    """Compute optimal transport matching between source and target using Sinkhorn.

    Returns soft assignment matrix P where P[i,j] is the transport from source[i] to target[j].
    For hard matching, we take argmax per source sample.
    """
    try:
        import ot
    except ImportError:
        raise ImportError("POT library required for OT mode. Install with: pip install POT")

    n_source = source_z.shape[0]
    n_target = target_z.shape[0]

    # Normalize to unit sphere for better distance computation
    source_norm = F.normalize(source_z, dim=1).numpy()
    target_norm = F.normalize(target_z, dim=1).numpy()

    # Cost matrix: squared Euclidean distance
    M = ot.dist(source_norm, target_norm, metric='sqeuclidean')

    # Uniform marginals
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    print(f"  Computing Sinkhorn OT (reg={reg})...")
    # Sinkhorn algorithm
    P = ot.sinkhorn(a, b, M, reg, numItermax=1000, stopThr=1e-9)

    # Convert to hard matching: for each source, pick the target with highest transport
    # Scale by n_target to get proper probability-like values
    P_scaled = P * n_target
    hard_matches = P_scaled.argmax(axis=1)

    # Report matching statistics
    unique_targets = len(np.unique(hard_matches))
    print(f"  Matched {len(hard_matches)} sources to {unique_targets} unique targets ({100*unique_targets/len(hard_matches):.1f}% diversity)")

    return torch.from_numpy(hard_matches).long(), torch.from_numpy(P).float()


def compute_gw_matching(source_z: torch.Tensor, target_z: torch.Tensor):
    """Compute Gromov-Wasserstein matching between source and target.

    GW matches based on *relative* distances within each space, not absolute distances.
    This preserves structure: if A is close to B in source, their matches should be
    close in target. Better for unrelated domains with different scales.
    """
    try:
        import ot
    except ImportError:
        raise ImportError("POT library required for GW mode. Install with: pip install POT")

    n_source = source_z.shape[0]
    n_target = target_z.shape[0]

    # Compute intra-domain distance matrices
    print(f"  Computing distance matrices...")
    source_np = source_z.numpy()
    target_np = target_z.numpy()

    C1 = ot.dist(source_np, source_np, metric='sqeuclidean')
    C2 = ot.dist(target_np, target_np, metric='sqeuclidean')

    # Normalize distance matrices to [0, 1] for numerical stability
    C1 = C1 / C1.max() if C1.max() > 0 else C1
    C2 = C2 / C2.max() if C2.max() > 0 else C2

    # Uniform marginals
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    print(f"  Computing Gromov-Wasserstein transport...")
    # GW optimal transport
    P = ot.gromov.gromov_wasserstein(
        C1, C2, a, b, loss_fun='square_loss', verbose=True
    )

    # Convert to hard matching
    P_scaled = P * n_target
    hard_matches = P_scaled.argmax(axis=1)

    # Report matching statistics
    unique_targets = len(np.unique(hard_matches))
    print(f"  Matched {len(hard_matches)} sources to {unique_targets} unique targets ({100*unique_targets/len(hard_matches):.1f}% diversity)")

    return torch.from_numpy(hard_matches).long(), torch.from_numpy(P).float()


def train_mlp_projection(
    source_z: torch.Tensor,
    target_z: torch.Tensor,
    target_indices: torch.Tensor,
    latent_dim_source: int,
    latent_dim_target: int,
    device: torch.device,
    hidden_dim: int = 256,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Train MLP projection from source to target latent space."""

    projection = MLPProjection(latent_dim_source, latent_dim_target, hidden_dim).to(device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create paired dataset
    n_samples = source_z.shape[0]
    target_matched = target_z[target_indices]  # [n_samples, latent_dim_target]

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Shuffle
        perm = torch.randperm(n_samples)
        source_shuffled = source_z[perm]
        target_shuffled = target_matched[perm]

        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            src_batch = source_shuffled[i:i+batch_size].to(device)
            tgt_batch = target_shuffled[i:i+batch_size].to(device)

            optimizer.zero_grad()
            pred = projection(src_batch)
            loss = F.mse_loss(pred, tgt_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = projection.state_dict()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch}: loss={avg_loss:.6f}")

    projection.load_state_dict(best_state)
    return projection, best_loss


def train_projection(
    source_z: torch.Tensor,
    target_z: torch.Tensor,
    target_indices: torch.Tensor,
    latent_dim_source: int,
    latent_dim_target: int,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Train linear projection from source to target latent space."""

    projection = LinearProjection(latent_dim_source, latent_dim_target).to(device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create paired dataset
    n_samples = source_z.shape[0]
    target_matched = target_z[target_indices]  # [n_samples, latent_dim_target]

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # Shuffle
        perm = torch.randperm(n_samples)
        source_shuffled = source_z[perm]
        target_shuffled = target_matched[perm]

        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            src_batch = source_shuffled[i:i+batch_size].to(device)
            tgt_batch = target_shuffled[i:i+batch_size].to(device)

            optimizer.zero_grad()
            pred = projection(src_batch)
            loss = F.mse_loss(pred, tgt_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = projection.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: loss={avg_loss:.6f}")

    projection.load_state_dict(best_state)
    return projection, best_loss


def main():
    parser = argparse.ArgumentParser(description="Train latent space projection")

    # Source model (encoder we want to use)
    parser.add_argument("--source-checkpoint", type=str, required=True,
                        help="Checkpoint for source model (encoder)")
    parser.add_argument("--source-data", type=str, required=True,
                        help="Dataset for source model")
    parser.add_argument("--source-context-frames", type=int, default=None,
                        help="Context frames for source (default: from checkpoint)")

    # Target model (decoder/hypernetwork we want to drive)
    parser.add_argument("--target-checkpoint", type=str, required=True,
                        help="Checkpoint for target model (decoder)")
    parser.add_argument("--target-data", type=str, required=True,
                        help="Dataset for target model")
    parser.add_argument("--target-context-frames", type=int, default=None,
                        help="Context frames for target (default: from checkpoint)")

    # Mode
    parser.add_argument("--mode", type=str, default="gw",
                        choices=["distribution", "nearest", "codebook", "ot", "gw"],
                        help="distribution: match overall statistics, "
                             "nearest: match nearest neighbors, "
                             "codebook: distribution + k-means snapping, "
                             "ot: optimal transport + MLP, "
                             "gw: Gromov-Wasserstein + MLP (recommended, preserves structure)")

    # Codebook options
    parser.add_argument("--codebook-size", type=int, default=128,
                        help="Number of codebook entries (k-means clusters)")
    parser.add_argument("--codebook-seed", type=int, default=42,
                        help="Random seed for k-means")

    # OT options
    parser.add_argument("--ot-reg", type=float, default=0.05,
                        help="Sinkhorn regularization (lower = sharper matching)")
    parser.add_argument("--mlp-hidden", type=int, default=256,
                        help="Hidden dimension for MLP projection")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for faster experiments")
    parser.add_argument("--one-per-seq", action="store_true",
                        help="Only take first window from each sequence (reduces redundancy)")

    # Output
    parser.add_argument("--output", type=str, default="projection.pt",
                        help="Output path for projection weights")

    # System
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("Latent Space Projection Training")
    print("=" * 60)

    # Load models
    print(f"\nLoading source model from {args.source_checkpoint}")
    source_model, source_ckpt = load_model_from_checkpoint(args.source_checkpoint, device)
    source_latent_dim = source_model.latent_dim
    source_context = args.source_context_frames or source_ckpt.get("context_frames", 1)
    print(f"  Latent dim: {source_latent_dim}, context frames: {source_context}")

    print(f"\nLoading target model from {args.target_checkpoint}")
    target_model, target_ckpt = load_model_from_checkpoint(args.target_checkpoint, device)
    target_latent_dim = target_model.latent_dim
    target_context = args.target_context_frames or target_ckpt.get("context_frames", 1)
    print(f"  Latent dim: {target_latent_dim}, context frames: {target_context}")

    # Load datasets
    print(f"\nLoading source dataset from {args.source_data}")
    source_dataset = SequenceDataset(
        args.source_data,
        context_frames=source_context,
        future_frames=1,
    )
    print(f"  {len(source_dataset)} samples")

    print(f"\nLoading target dataset from {args.target_data}")
    target_dataset = SequenceDataset(
        args.target_data,
        context_frames=target_context,
        future_frames=1,
    )
    print(f"  {len(target_dataset)} samples")

    # Take only first window from each sequence (reduces redundant sliding windows)
    if args.one_per_seq:
        # Get one sample per sequence by stepping by samples_per_seq
        src_samples_per_seq = getattr(source_dataset, 'samples_per_seq', 1)
        tgt_samples_per_seq = getattr(target_dataset, 'samples_per_seq', 1)
        source_indices = list(range(0, len(source_dataset), src_samples_per_seq))
        target_indices = list(range(0, len(target_dataset), tgt_samples_per_seq))
        source_dataset = torch.utils.data.Subset(source_dataset, source_indices)
        target_dataset = torch.utils.data.Subset(target_dataset, target_indices)
        print(f"\nOne per sequence: {len(source_dataset)} source, {len(target_dataset)} target samples")

    # Limit samples if requested
    if args.max_samples:
        source_indices = list(range(min(args.max_samples, len(source_dataset))))
        target_indices = list(range(min(args.max_samples, len(target_dataset))))
        source_dataset = torch.utils.data.Subset(source_dataset, source_indices)
        target_dataset = torch.utils.data.Subset(target_dataset, target_indices)
        print(f"\nLimited to {len(source_dataset)} source, {len(target_dataset)} target samples")

    # Encode datasets
    print("\nEncoding source dataset...")
    source_z = encode_dataset(source_model, source_dataset, device)
    print(f"  Source embeddings: {source_z.shape}")

    print("\nEncoding target dataset...")
    target_z = encode_dataset(target_model, target_dataset, device)
    print(f"  Target embeddings: {target_z.shape}")

    codebook = None
    codebook_stats = None

    if args.mode in ("distribution", "codebook"):
        # Distribution matching: whiten source, color with target statistics
        # This is a closed-form solution - no training needed
        print("\nComputing distribution-matching projection...")
        print("  (whitens source distribution, applies target covariance structure)")

        source_mean, source_whiten = compute_whitening_transform(source_z)
        target_mean, target_color = compute_coloring_transform(target_z)

        # Combined transform: z_target = (z_source - source_mean) @ W + target_mean
        # where W = source_whiten @ target_color
        W = source_whiten @ target_color
        b = target_mean - source_mean @ W

        # Create projection module with computed weights
        projection = LinearProjection(source_latent_dim, target_latent_dim).to(device)
        projection.linear.weight.data = W.T.to(device)  # Linear expects [out, in]
        projection.linear.bias.data = b.to(device)

        # Compute final loss for reporting
        with torch.no_grad():
            projected = projection(source_z.to(device))
            # Loss is variance of target not explained (should be ~0 for matching distributions)
            final_loss = F.mse_loss(projected.mean(dim=0), target_z.to(device).mean(dim=0)).item()
        print(f"  Mean alignment error: {final_loss:.6f}")

        # For codebook mode, cluster target embeddings and analyze spread
        if args.mode == "codebook":
            print(f"\nBuilding codebook from target embeddings...")
            codebook, target_labels = compute_codebook(
                target_z, args.codebook_size, args.codebook_seed
            )
            print(f"  Codebook shape: {codebook.shape}")

            # Analyze how projected source samples distribute
            codebook_stats = analyze_codebook_spread(
                source_z, projection, codebook, device
            )

            # Warn if spread is poor
            if codebook_stats["normalized_entropy"] < 0.5:
                print("\n  WARNING: Poor spread! Most samples map to few clusters.")
                print("  Consider: increasing --codebook-size or using different data.")

    elif args.mode == "nearest":
        # Nearest neighbor matching (original approach)
        print("\nFinding nearest neighbors (cosine similarity)...")
        nn_indices = find_nearest_neighbors(source_z, target_z)
        print(f"  Matched {len(nn_indices)} source points to target points")

        # Train projection
        print(f"\nTraining linear projection: {source_latent_dim} → {target_latent_dim}")
        projection, final_loss = train_projection(
            source_z, target_z, nn_indices,
            source_latent_dim, target_latent_dim,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    elif args.mode == "ot":
        # Optimal Transport matching with MLP projection
        print("\nComputing Optimal Transport matching...")
        ot_indices, transport_matrix = compute_ot_matching(
            source_z, target_z, reg=args.ot_reg
        )

        # Train MLP projection on OT-matched pairs
        print(f"\nTraining MLP projection: {source_latent_dim} → {target_latent_dim}")
        print(f"  Hidden dim: {args.mlp_hidden}")
        projection, final_loss = train_mlp_projection(
            source_z, target_z, ot_indices,
            source_latent_dim, target_latent_dim,
            device,
            hidden_dim=args.mlp_hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    elif args.mode == "gw":
        # Gromov-Wasserstein matching with MLP projection
        # GW preserves relative distances - better for unrelated domains
        print("\nComputing Gromov-Wasserstein matching...")
        gw_indices, transport_matrix = compute_gw_matching(source_z, target_z)

        # Train MLP projection on GW-matched pairs
        print(f"\nTraining MLP projection: {source_latent_dim} → {target_latent_dim}")
        print(f"  Hidden dim: {args.mlp_hidden}")
        projection, final_loss = train_mlp_projection(
            source_z, target_z, gw_indices,
            source_latent_dim, target_latent_dim,
            device,
            hidden_dim=args.mlp_hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "projection_state_dict": projection.state_dict(),
        "source_latent_dim": source_latent_dim,
        "target_latent_dim": target_latent_dim,
        "source_checkpoint": args.source_checkpoint,
        "target_checkpoint": args.target_checkpoint,
        "mode": args.mode,
        "final_loss": final_loss,
    }

    # Add codebook if present
    if codebook is not None:
        save_dict["codebook"] = codebook  # [n_clusters, latent_dim]
        save_dict["codebook_size"] = args.codebook_size
        save_dict["codebook_stats"] = codebook_stats
        print(f"\n  Codebook saved ({args.codebook_size} centroids)")

    torch.save(save_dict, output_path)

    print(f"\n" + "=" * 60)
    print(f"Projection saved to {output_path}")
    print(f"Final MSE loss: {final_loss:.6f}")
    if codebook is not None:
        print(f"Codebook: {args.codebook_size} entries, {codebook_stats['n_used']} used by source")
    print("=" * 60)

    # Print usage example
    print(f"\nTo use this projection:")
    print(f"  ckpt = torch.load('{output_path}')")
    print(f"  projection = nn.Linear({source_latent_dim}, {target_latent_dim})")
    print(f"  projection.load_state_dict(ckpt['projection_state_dict'])")
    print(f"  z_target = projection(z_source)")
    if codebook is not None:
        print(f"\n  # Snap to nearest codebook entry:")
        print(f"  codebook = ckpt['codebook']  # [{args.codebook_size}, {target_latent_dim}]")
        print(f"  dists = torch.cdist(z_target.unsqueeze(0), codebook)")
        print(f"  z_snapped = codebook[dists.argmin()]")


if __name__ == "__main__":
    main()
