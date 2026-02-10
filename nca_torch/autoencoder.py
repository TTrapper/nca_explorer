"""
NCA-based Autoencoder with HyperNetwork decoder.

Architecture:
    Input Image → Encoder (CNN) → Latent → HyperNetwork → NCA Params → Run NCA → Reconstructed Image

The latent space captures not just class but style, thickness, angle, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """CNN encoder that maps images to latent embeddings."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        hidden_dims: list[int] = [32, 64, 128],
        context_frames: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels  # Channels per frame (e.g., 3 for RGB)
        self.context_frames = context_frames

        # Total input channels = in_channels * context_frames
        # e.g., 4 RGB frames = 12 input channels
        total_input_channels = in_channels * context_frames

        # Build encoder layers
        layers = []
        prev_channels = total_input_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_channels = h_dim

        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened size after convolutions
        # For 28x28 input with 3 stride-2 convs: 28 -> 14 -> 7 -> 4
        self.flat_size = hidden_dims[-1] * 4 * 4

        # Project to latent space
        self.fc = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 1, 28, 28]
        Returns:
            Latent embeddings [B, latent_dim]
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class VAEEncoder(nn.Module):
    """
    Variational encoder - outputs mean and log variance for reparameterization.
    Produces a more regularized, smoother latent space.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        hidden_dims: list[int] = [32, 64, 128],
        context_frames: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels  # Channels per frame (e.g., 3 for RGB)
        self.context_frames = context_frames

        # Total input channels = in_channels * context_frames
        # e.g., 4 RGB frames = 12 input channels
        total_input_channels = in_channels * context_frames

        # Build encoder layers
        layers = []
        prev_channels = total_input_channels
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_channels = h_dim

        self.conv_layers = nn.Sequential(*layers)
        self.flat_size = hidden_dims[-1] * 4 * 4

        # Separate heads for mean and log variance
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images [B, 1, 28, 28]
        Returns:
            z: Sampled latent [B, latent_dim]
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Use mean during inference

        return z, mu, logvar


class HyperNetworkDecoder(nn.Module):
    """
    HyperNetwork that generates NCA parameters from latent embeddings.
    Same as before but designed to work with the encoder.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        grid_channels: int = 12,
        hidden_dim: int = 256,
        out_channels: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_channels = grid_channels
        self.out_channels = out_channels  # 1 for grayscale, 3 for RGB

        # Calculate NCA parameter sizes
        self.layer1_weight_size = grid_channels * grid_channels * 3 * 3
        self.layer1_bias_size = grid_channels
        self.layer2_weight_size = grid_channels * grid_channels * 3 * 3
        self.layer2_bias_size = grid_channels

        total_params = (
            self.layer1_weight_size + self.layer1_bias_size +
            self.layer2_weight_size + self.layer2_bias_size
        )

        # HyperNetwork
        self.hypernet = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_params),
        )

        # Initialize to reasonable values (not too small)
        nn.init.normal_(self.hypernet[-1].weight, std=0.02)
        nn.init.zeros_(self.hypernet[-1].bias)

        # Learnable scale
        self.weight_scale = nn.Parameter(torch.tensor(1.0))

        # Learnable initial pattern
        self.initial_pattern = nn.Parameter(torch.randn(1, grid_channels, 7, 7) * 0.1)

    def generate_params(self, z: torch.Tensor):
        """Generate NCA parameters from latent."""
        params = self.hypernet(z)
        params = params * self.weight_scale  # No tanh - allow unbounded weights

        idx = 0
        layer1_w = params[:, idx:idx + self.layer1_weight_size]
        idx += self.layer1_weight_size
        layer1_b = params[:, idx:idx + self.layer1_bias_size]
        idx += self.layer1_bias_size
        layer2_w = params[:, idx:idx + self.layer2_weight_size]
        idx += self.layer2_weight_size
        layer2_b = params[:, idx:idx + self.layer2_bias_size]

        B = z.shape[0]
        layer1_w = layer1_w.view(B, self.grid_channels, self.grid_channels, 3, 3)
        layer2_w = layer2_w.view(B, self.grid_channels, self.grid_channels, 3, 3)

        return layer1_w, layer1_b, layer2_w, layer2_b

    def nca_step(self, grid, layer1_w, layer1_b, layer2_w, layer2_b):
        """Single NCA step with generated parameters."""
        B, C, H, W = grid.shape

        outputs = []
        for b in range(B):
            h = F.pad(grid[b:b+1], (1, 1, 1, 1), mode='circular')
            h = F.conv2d(h, layer1_w[b], bias=layer1_b[b], padding=0)
            h = F.relu(h)

            h = F.pad(h, (1, 1, 1, 1), mode='circular')
            h = F.conv2d(h, layer2_w[b], bias=layer2_b[b], padding=0)
            outputs.append(h)

        update = torch.cat(outputs, dim=0)
        return grid + update

    def init_grid(
        self,
        batch_size: int,
        grid_size: tuple[int, int],
        device: torch.device,
        init_mode: str = "learned",
        init_images: torch.Tensor | None = None,
        noise_std: float = 0.5,
    ) -> torch.Tensor:
        """
        Initialize the NCA grid with various strategies.

        Args:
            batch_size: Number of grids to create
            grid_size: (H, W) of grid
            device: torch device
            init_mode: One of:
                - "learned": Use learned initial pattern (default)
                - "noise": Random Gaussian noise
                - "zeros": All zeros (tests if NCA can create from nothing)
                - "image": Start from provided images
                - "image_noisy": Start from images with added noise
            init_images: Images to use for "image" or "image_noisy" modes [B, 1, H, W]
            noise_std: Std of noise for "noise" and "image_noisy" modes

        Returns:
            grid: Initialized grid [B, grid_channels, H, W]
        """
        H, W = grid_size
        grid = torch.zeros(batch_size, self.grid_channels, H, W, device=device)

        if init_mode == "learned":
            # Place learned pattern in center
            ph, pw = self.initial_pattern.shape[2:]
            start_h, start_w = (H - ph) // 2, (W - pw) // 2
            grid[:, :, start_h:start_h+ph, start_w:start_w+pw] = self.initial_pattern

        elif init_mode == "noise":
            # Full random noise
            grid = torch.randn_like(grid) * noise_std

        elif init_mode == "zeros":
            # Already zeros
            pass

        elif init_mode == "image":
            # Use provided images as first channels (1 for grayscale, 3 for RGB)
            if init_images is not None:
                C_img = init_images.shape[1]  # Number of image channels
                grid[:, :C_img] = init_images
                # Add small noise to remaining channels
                if self.grid_channels > C_img:
                    grid[:, C_img:] = torch.randn(
                        batch_size, self.grid_channels - C_img, H, W, device=device
                    ) * 0.1

        elif init_mode == "image_noisy":
            # Use noisy version of images
            if init_images is not None:
                C_img = init_images.shape[1]
                noisy = init_images + torch.randn_like(init_images) * noise_std
                grid[:, :C_img] = noisy.clamp(0, 1)
                if self.grid_channels > C_img:
                    grid[:, C_img:] = torch.randn(
                        batch_size, self.grid_channels - C_img, H, W, device=device
                    ) * noise_std

        return grid

    def forward(
        self,
        z: torch.Tensor,
        grid_size: tuple[int, int] = (28, 28),
        num_steps: int = 32,
        init_mode: str = "learned",
        init_images: torch.Tensor | None = None,
        init_noise_std: float = 0.5,
        step_noise_std: float = 0.0,
    ) -> torch.Tensor:
        """
        Run NCA with parameters generated from latent z.

        Args:
            z: Latent embeddings [B, latent_dim]
            grid_size: Output size
            num_steps: Number of NCA iterations
            init_mode: Grid initialization mode (see init_grid)
            init_images: Images for image-based initialization
            init_noise_std: Noise std for initialization
            step_noise_std: Noise std added at each NCA step (for robustness training)

        Returns:
            output: Reconstructed images [B, 1, H, W]
        """
        B = z.shape[0]
        H, W = grid_size
        device = z.device

        # Generate NCA parameters
        layer1_w, layer1_b, layer2_w, layer2_b = self.generate_params(z)

        # Initialize grid
        grid = self.init_grid(
            B, grid_size, device,
            init_mode=init_mode,
            init_images=init_images,
            noise_std=init_noise_std,
        )

        # Run NCA
        for _ in range(num_steps):
            grid = self.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)
            # Add perturbation noise during evolution (for robustness)
            if step_noise_std > 0 and self.training:
                grid = grid + torch.randn_like(grid) * step_noise_std

        # Extract output channels, sigmoid to [0, 1]
        output = torch.sigmoid(grid[:, :self.out_channels])
        return output


class FirstFrameDecoder(nn.Module):
    """
    CNN decoder for generating the initial grid (RGB + hidden channels) from latent.

    Used in dynamics training where:
    - Step 0: FirstFrameDecoder(z) → grid_0 (full grid: RGB sigmoided, hidden channels as memory)
    - Steps 1+: NCA(grid_n) → grid_{n+1} (dynamics with frozen hidden channels)

    The hidden channels serve as frozen read-only memory of z that the NCA can
    read from but never modifies. This allows gradients to flow from the NCA
    output back through the hidden channels to the encoder.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        out_channels: int = 3,
        grid_channels: int = 16,
        hidden_dims: list[int] = [128, 64, 32],
        grid_size: tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels  # RGB channels
        self.grid_channels = grid_channels  # Total grid channels (RGB + hidden)
        self.grid_size = grid_size

        # Initial projection: latent → spatial feature map
        # Start at 4x4 and upsample to target size
        self.init_size = 4
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.init_size * self.init_size)

        # Transposed conv layers for upsampling
        layers = []
        prev_channels = hidden_dims[0]

        for h_dim in hidden_dims[1:]:
            layers.extend([
                nn.ConvTranspose2d(prev_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ])
            prev_channels = h_dim

        # Final layer outputs full grid_channels (no activation - applied separately)
        layers.append(
            nn.ConvTranspose2d(prev_channels, grid_channels, kernel_size=4, stride=2, padding=1)
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, grid_size: tuple[int, int] | None = None) -> torch.Tensor:
        """
        Decode latent to initial grid (RGB + frozen hidden channels).

        Args:
            z: Latent embeddings [B, latent_dim]
            grid_size: Optional target size (will resize if different from native)

        Returns:
            grid: Full grid [B, grid_channels, H, W]
                  - First out_channels: sigmoid (RGB in [0,1])
                  - Remaining channels: tanh * 0.1 (hidden memory, small values)
        """
        if grid_size is None:
            grid_size = self.grid_size

        B = z.shape[0]

        # Project and reshape to spatial
        h = self.fc(z)
        h = h.view(B, -1, self.init_size, self.init_size)

        # Upsample through decoder
        out = self.decoder(h)

        # Resize to exact target size if needed
        if out.shape[2:] != grid_size:
            out = F.interpolate(out, size=grid_size, mode='bilinear', align_corners=False)

        # Apply activations: sigmoid for RGB, tanh * 0.1 for hidden
        rgb = torch.sigmoid(out[:, :self.out_channels])
        hidden = torch.tanh(out[:, self.out_channels:]) * 0.1

        return torch.cat([rgb, hidden], dim=1)


class NCAAutoencoder(nn.Module):
    """
    Complete NCA-based autoencoder.
    Encoder → Latent → HyperNetwork → NCA → Reconstruction

    For dynamics training with multi-frame context:
    - Encoder takes stacked context frames (context_frames * in_channels input channels)
    - Decoder outputs single frame (in_channels output channels)
    - FirstFrameDecoder generates initial frame from latent
    - NCA handles subsequent dynamics steps
    """

    def __init__(
        self,
        latent_dim: int = 64,
        grid_channels: int = 12,
        hidden_dim: int = 256,
        use_vae: bool = True,
        in_channels: int = 1,
        grid_size: tuple[int, int] = (28, 28),
        context_frames: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_vae = use_vae
        self.in_channels = in_channels  # Channels per frame: 1 for grayscale, 3 for RGB
        self.context_frames = context_frames  # Number of context frames for encoder
        self.grid_size = grid_size

        # Encoder takes stacked context frames
        # Input channels = in_channels * context_frames (e.g., 4 RGB frames = 12 channels)
        if use_vae:
            self.encoder = VAEEncoder(
                in_channels=in_channels,
                latent_dim=latent_dim,
                context_frames=context_frames,
            )
        else:
            self.encoder = Encoder(
                in_channels=in_channels,
                latent_dim=latent_dim,
                context_frames=context_frames,
            )

        # NCA decoder for dynamics (outputs single frame)
        self.decoder = HyperNetworkDecoder(
            latent_dim=latent_dim,
            grid_channels=grid_channels,
            hidden_dim=hidden_dim,
            out_channels=in_channels,  # Output single frame
        )

        # First frame decoder (generates full grid: RGB + hidden channels)
        self.first_frame_decoder = FirstFrameDecoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
            grid_channels=grid_channels,
            grid_size=grid_size,
        )

    def encode(self, x: torch.Tensor):
        """Encode images to latent space."""
        if self.use_vae:
            z, mu, logvar = self.encoder(x)
            return z, mu, logvar
        else:
            z = self.encoder(x)
            return z, None, None

    def decode(
        self,
        z: torch.Tensor,
        num_steps: int = 32,
        init_mode: str = "learned",
        init_images: torch.Tensor | None = None,
        init_noise_std: float = 0.5,
        step_noise_std: float = 0.0,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode latent to images via NCA (for dynamics/refinement).

        init_mode options:
            - "first_frame": Use FirstFrameDecoder output as initial RGB
            - "image": Use init_images as initial RGB
            - "learned", "noise", "zeros": Other grid init modes
        """
        if grid_size is None:
            grid_size = self.grid_size

        # Special mode: use FirstFrameDecoder to initialize grid RGB
        if init_mode == "first_frame":
            init_images = self.first_frame_decoder(z, grid_size=grid_size)
            init_mode = "image"

        return self.decoder(
            z,
            grid_size=grid_size,
            num_steps=num_steps,
            init_mode=init_mode,
            init_images=init_images,
            init_noise_std=init_noise_std,
            step_noise_std=step_noise_std,
        )

    def decode_first_frame(
        self,
        z: torch.Tensor,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode latent to first frame via CNN (direct, no NCA)."""
        if grid_size is None:
            grid_size = self.grid_size
        return self.first_frame_decoder(z, grid_size=grid_size)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → decode.

        Args:
            x: Input images [B, 1, 28, 28]
            num_steps: NCA steps

        Returns:
            recon: Reconstructed images [B, 1, 28, 28]
            z: Latent embeddings [B, latent_dim]
            mu: Mean (if VAE) [B, latent_dim]
            logvar: Log variance (if VAE) [B, latent_dim]
        """
        z, mu, logvar = self.encode(x)
        recon = self.decode(z, num_steps)
        return recon, z, mu, logvar

    def sample(
        self,
        num_samples: int,
        device: torch.device,
        num_steps: int = 32,
    ) -> torch.Tensor:
        """Sample random images from the latent space."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, num_steps)

    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_interp: int = 10,
        num_steps: int = 32,
    ) -> torch.Tensor:
        """Interpolate between two images in latent space."""
        z1, _, _ = self.encode(x1)
        z2, _, _ = self.encode(x2)

        alphas = torch.linspace(0, 1, num_interp, device=x1.device)
        zs = torch.stack([(1 - a) * z1 + a * z2 for a in alphas]).squeeze(1)

        return self.decode(zs, num_steps)

    def morph(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        num_steps: int = 32,
    ) -> torch.Tensor:
        """
        Morph from source image to target image.

        The NCA starts with the source image as initial state,
        but uses parameters generated from the target's latent code.
        This tests whether the NCA can transform one digit into another.

        Args:
            source: Starting image [1, 1, H, W]
            target: Target image [1, 1, H, W]
            num_steps: NCA steps to evolve

        Returns:
            Morphed image [1, 1, H, W]
        """
        z_target, _, _ = self.encode(target)
        return self.decode(
            z_target,
            num_steps=num_steps,
            init_mode="image",
            init_images=source,
        )

    def morph_sequence(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        num_steps: int = 32,
        capture_every: int = 1,
    ) -> torch.Tensor:
        """
        Capture the morphing sequence from source to target.

        Args:
            source: Starting image [1, 1, H, W]
            target: Target image [1, 1, H, W]
            num_steps: Total NCA steps
            capture_every: Capture frame every N steps

        Returns:
            Sequence of frames [num_frames, 1, H, W]
        """
        z_target, _, _ = self.encode(target)

        # Initialize from source
        B = 1
        H, W = self.grid_size
        device = source.device
        out_c = self.decoder.out_channels

        layer1_w, layer1_b, layer2_w, layer2_b = self.decoder.generate_params(z_target)
        grid = self.decoder.init_grid(
            B, (H, W), device,
            init_mode="image",
            init_images=source,
        )

        frames = [torch.sigmoid(grid[:, :out_c]).clone()]

        for step in range(num_steps):
            grid = self.decoder.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)
            if (step + 1) % capture_every == 0:
                frames.append(torch.sigmoid(grid[:, :out_c]).clone())

        return torch.cat(frames, dim=0)


def vae_loss(recon, target, mu, logvar, kl_weight: float = 0.001):
    """
    VAE loss = Reconstruction + KL divergence.

    Args:
        recon: Reconstructed images
        target: Target images
        mu: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL term (beta-VAE style)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if mu is not None and logvar is not None:
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        kl_loss = torch.tensor(0.0, device=recon.device)

    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
