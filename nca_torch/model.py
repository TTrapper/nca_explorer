"""
Neural Cellular Automata with FiLM conditioning for conditional image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""

    def __init__(self, num_features: int, conditioning_dim: int):
        super().__init__()
        self.num_features = num_features
        # Generate gamma and beta from conditioning
        self.film_generator = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.ReLU(),
            nn.Linear(conditioning_dim, num_features * 2)  # gamma and beta
        )
        # Initialize to identity transform (gamma=1, beta=0)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        self.film_generator[-1].bias.data[:num_features] = 1.0  # gamma = 1

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features [B, C, H, W]
            conditioning: Conditioning vector [B, conditioning_dim]
        Returns:
            Modulated features [B, C, H, W]
        """
        film_params = self.film_generator(conditioning)  # [B, C*2]
        gamma = film_params[:, :self.num_features].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = film_params[:, self.num_features:].unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        return gamma * x + beta


class NCALayer(nn.Module):
    """
    Single NCA layer: Convolution over Moore neighborhood + activation with FiLM.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditioning_dim: int,
        activation: str = "relu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 3x3 convolution for Moore neighborhood
        # We use groups=1 for full channel mixing (like our original design)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, padding_mode='circular',  # Wrap around (toroidal)
            bias=False
        )

        # FiLM conditioning
        self.film = FiLMLayer(out_channels, conditioning_dim)

        # Activation
        self.activation_type = activation

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Grid state [B, C, H, W]
            conditioning: Conditioning vector [B, conditioning_dim]
        Returns:
            Updated state [B, out_channels, H, W]
        """
        # Convolution
        h = self.conv(x)

        # FiLM modulation
        h = self.film(h, conditioning)

        # Activation
        if self.activation_type == "relu":
            h = F.relu(h)
        elif self.activation_type == "tanh":
            h = torch.tanh(h)
        elif self.activation_type == "sigmoid":
            h = torch.sigmoid(h)
        elif self.activation_type == "gelu":
            h = F.gelu(h)
        elif self.activation_type == "sin":
            h = torch.sin(h)
        else:
            pass  # identity

        return h


class ConditionalNCA(nn.Module):
    """
    Two-layer NCA with FiLM conditioning for conditional image generation.

    Architecture:
        Conditioning → Encoder → Latent
                                   ↓
        Grid → NCALayer1 (+ FiLM) → NCALayer2 (+ FiLM) → Output
                         ↑                    ↑
                      Latent               Latent
    """

    def __init__(
        self,
        grid_channels: int = 16,        # Hidden state channels (includes RGB)
        rgb_channels: int = 3,          # Visible output channels
        conditioning_dim: int = 64,     # Latent/conditioning dimension
        num_classes: int = 10,          # For class-conditional generation
        hidden_dim: int = 64,           # Hidden dimension in NCA layers
        activation: str = "relu",
    ):
        super().__init__()
        self.grid_channels = grid_channels
        self.rgb_channels = rgb_channels
        self.conditioning_dim = conditioning_dim

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, conditioning_dim)

        # Optional: encode more info into conditioning
        self.conditioning_encoder = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.ReLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )

        # NCA layers
        self.layer1 = NCALayer(grid_channels, hidden_dim, conditioning_dim, activation)
        self.layer2 = NCALayer(hidden_dim, grid_channels, conditioning_dim, activation)

        # Blend factor (learnable or fixed)
        self.blend = nn.Parameter(torch.tensor(0.5))

    def get_conditioning(self, class_labels: torch.Tensor) -> torch.Tensor:
        """Get conditioning vector from class labels."""
        emb = self.class_embedding(class_labels)  # [B, conditioning_dim]
        return self.conditioning_encoder(emb)

    def step(self, grid: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Single NCA update step.

        Args:
            grid: Current state [B, grid_channels, H, W]
            conditioning: Conditioning vector [B, conditioning_dim]
        Returns:
            Updated state [B, grid_channels, H, W]
        """
        # Two-layer update
        h = self.layer1(grid, conditioning)
        update = self.layer2(h, conditioning)

        # Residual blend
        blend = torch.sigmoid(self.blend)  # Keep in [0, 1]
        new_grid = (1 - blend) * grid + blend * update

        return new_grid

    def forward(
        self,
        class_labels: torch.Tensor,
        grid_size: tuple[int, int] = (28, 28),
        num_steps: int = 32,
        initial_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Run NCA for multiple steps.

        Args:
            class_labels: Class labels [B]
            grid_size: (H, W) of the grid
            num_steps: Number of NCA steps
            initial_grid: Optional initial state [B, grid_channels, H, W]

        Returns:
            final_rgb: Final RGB output [B, 3, H, W]
            trajectory: List of RGB states at each step
        """
        B = class_labels.shape[0]
        H, W = grid_size
        device = class_labels.device

        # Get conditioning
        conditioning = self.get_conditioning(class_labels)

        # Initialize grid
        if initial_grid is None:
            # Random initialization for hidden channels, zeros for RGB
            grid = torch.randn(B, self.grid_channels, H, W, device=device) * 0.1
        else:
            grid = initial_grid

        # Run NCA steps
        trajectory = []
        for _ in range(num_steps):
            grid = self.step(grid, conditioning)
            # Extract RGB (first 3 channels) and apply sigmoid to keep in [0, 1]
            rgb = torch.sigmoid(grid[:, :self.rgb_channels])
            trajectory.append(rgb)

        return trajectory[-1], trajectory

    def get_rgb(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract RGB channels from grid state."""
        return torch.sigmoid(grid[:, :self.rgb_channels])


class HyperNetworkNCA(nn.Module):
    """
    Pure HyperNetwork approach: a network generates ALL NCA parameters from class embedding.
    No FiLM - the class conditioning is entirely through generated weights.

    This maps: class_label → embedding → NCA parameters → run NCA → image

    The embedding space IS the NCA parameter space (compressed).
    """

    def __init__(
        self,
        grid_channels: int = 12,
        rgb_channels: int = 1,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        self.grid_channels = grid_channels
        self.rgb_channels = rgb_channels
        self.embedding_dim = embedding_dim

        # Class embedding - this IS our latent space
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)

        # Calculate how many parameters the NCA needs
        # Layer 1: grid_channels -> grid_channels (3x3 kernel) + bias
        self.layer1_weight_size = grid_channels * grid_channels * 3 * 3
        self.layer1_bias_size = grid_channels
        # Layer 2: grid_channels -> grid_channels (3x3 kernel) + bias
        self.layer2_weight_size = grid_channels * grid_channels * 3 * 3
        self.layer2_bias_size = grid_channels

        total_params = (
            self.layer1_weight_size + self.layer1_bias_size +
            self.layer2_weight_size + self.layer2_bias_size
        )

        # HyperNetwork: embedding → NCA parameters
        self.hypernet = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_params),
        )

        # Initialize hypernet output to very small values for stability
        nn.init.normal_(self.hypernet[-1].weight, std=0.001)
        nn.init.zeros_(self.hypernet[-1].bias)

        # Learnable scale for generated weights (start small)
        self.weight_scale = nn.Parameter(torch.tensor(0.1))

        # Learnable initial grid pattern (shared, but could also be generated)
        self.initial_pattern = nn.Parameter(torch.randn(1, grid_channels, 7, 7) * 0.1)

    def generate_nca_params(self, embedding: torch.Tensor):
        """Generate NCA parameters from embedding."""
        params = self.hypernet(embedding)

        # Apply tanh to bound outputs, then scale
        params = torch.tanh(params) * self.weight_scale

        # Split into layer1 weights, layer1 bias, layer2 weights, layer2 bias
        idx = 0
        layer1_w = params[:, idx:idx + self.layer1_weight_size]
        idx += self.layer1_weight_size
        layer1_b = params[:, idx:idx + self.layer1_bias_size]
        idx += self.layer1_bias_size
        layer2_w = params[:, idx:idx + self.layer2_weight_size]
        idx += self.layer2_weight_size
        layer2_b = params[:, idx:idx + self.layer2_bias_size]

        # Reshape weights
        B = embedding.shape[0]
        layer1_w = layer1_w.view(B, self.grid_channels, self.grid_channels, 3, 3)
        layer2_w = layer2_w.view(B, self.grid_channels, self.grid_channels, 3, 3)

        return layer1_w, layer1_b, layer2_w, layer2_b

    def nca_step(self, grid: torch.Tensor, layer1_w: torch.Tensor, layer1_b: torch.Tensor,
                 layer2_w: torch.Tensor, layer2_b: torch.Tensor) -> torch.Tensor:
        """
        Single NCA step with per-sample weights.

        Args:
            grid: [B, C, H, W]
            layer1_w: [B, C_out, C_in, 3, 3]
            layer1_b: [B, C_out]
            layer2_w: [B, C_out, C_in, 3, 3]
            layer2_b: [B, C_out]
        """
        B, C, H, W = grid.shape

        # Apply convolutions per sample (unfortunately need loop for per-sample weights)
        outputs = []
        for b in range(B):
            # Circular padding for toroidal topology
            h = F.pad(grid[b:b+1], (1, 1, 1, 1), mode='circular')

            # Layer 1
            h = F.conv2d(h, layer1_w[b], bias=layer1_b[b], padding=0)
            h = F.relu(h)

            # Circular padding again for layer 2
            h = F.pad(h, (1, 1, 1, 1), mode='circular')

            # Layer 2
            h = F.conv2d(h, layer2_w[b], bias=layer2_b[b], padding=0)
            outputs.append(h)

        update = torch.cat(outputs, dim=0)

        # Residual connection with small step size for stability
        # Also clamp to prevent explosion
        new_grid = grid + 0.1 * torch.tanh(update)
        return new_grid

    def forward(
        self,
        class_labels: torch.Tensor,
        grid_size: tuple[int, int] = (28, 28),
        num_steps: int = 32,
        initial_grid: torch.Tensor | None = None,
        return_params: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate image for given class labels.

        Args:
            class_labels: [B] class indices
            grid_size: (H, W) output size
            num_steps: number of NCA iterations
            initial_grid: optional starting grid
            return_params: if True, also return generated NCA params
        """
        B = class_labels.shape[0]
        H, W = grid_size
        device = class_labels.device

        # Get embedding and generate NCA parameters
        embedding = self.class_embedding(class_labels)  # [B, embedding_dim]
        layer1_w, layer1_b, layer2_w, layer2_b = self.generate_nca_params(embedding)

        # Initialize grid
        if initial_grid is None:
            grid = torch.zeros(B, self.grid_channels, H, W, device=device)
            # Place initial pattern in center
            ph, pw = self.initial_pattern.shape[2:]
            start_h, start_w = (H - ph) // 2, (W - pw) // 2
            grid[:, :, start_h:start_h+ph, start_w:start_w+pw] = self.initial_pattern
        else:
            grid = initial_grid

        # Run NCA steps
        trajectory = []
        for _ in range(num_steps):
            grid = self.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)
            rgb = torch.sigmoid(grid[:, :self.rgb_channels])
            trajectory.append(rgb)

        result = (trajectory[-1], trajectory)
        if return_params:
            result = (*result, (layer1_w, layer1_b, layer2_w, layer2_b))
        return result

    def forward_from_embedding(
        self,
        embedding: torch.Tensor,
        grid_size: tuple[int, int] = (28, 28),
        num_steps: int = 32,
    ) -> torch.Tensor:
        """
        Generate image directly from embedding (for interpolation/exploration).
        """
        B = embedding.shape[0]
        H, W = grid_size
        device = embedding.device

        layer1_w, layer1_b, layer2_w, layer2_b = self.generate_nca_params(embedding)

        grid = torch.zeros(B, self.grid_channels, H, W, device=device)
        ph, pw = self.initial_pattern.shape[2:]
        start_h, start_w = (H - ph) // 2, (W - pw) // 2
        grid[:, :, start_h:start_h+ph, start_w:start_w+pw] = self.initial_pattern

        for _ in range(num_steps):
            grid = self.nca_step(grid, layer1_w, layer1_b, layer2_w, layer2_b)

        return torch.sigmoid(grid[:, :self.rgb_channels])


class NCAWithHyperNetwork(nn.Module):
    """
    NCA where the convolution weights themselves are generated by a HyperNetwork.
    Also includes FiLM for additional modulation.
    (Legacy version - prefer HyperNetworkNCA for pure hypernetwork approach)
    """

    def __init__(
        self,
        grid_channels: int = 16,
        rgb_channels: int = 3,
        conditioning_dim: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        self.grid_channels = grid_channels
        self.rgb_channels = rgb_channels
        self.conditioning_dim = conditioning_dim

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, conditioning_dim)

        # HyperNetwork: generates conv weights from conditioning
        # Layer 1: grid_channels -> grid_channels (3x3 kernel)
        layer1_weights = grid_channels * grid_channels * 3 * 3
        # Layer 2: grid_channels -> grid_channels (3x3 kernel)
        layer2_weights = grid_channels * grid_channels * 3 * 3

        self.hyper_net = nn.Sequential(
            nn.Linear(conditioning_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, layer1_weights + layer2_weights),
        )

        self.layer1_weight_shape = (grid_channels, grid_channels, 3, 3)
        self.layer2_weight_shape = (grid_channels, grid_channels, 3, 3)
        self.total_layer1 = layer1_weights

        # FiLM for additional modulation
        self.film1 = FiLMLayer(grid_channels, conditioning_dim)
        self.film2 = FiLMLayer(grid_channels, conditioning_dim)

        self.blend = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        class_labels: torch.Tensor,
        grid_size: tuple[int, int] = (28, 28),
        num_steps: int = 32,
        initial_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B = class_labels.shape[0]
        H, W = grid_size
        device = class_labels.device

        # Get conditioning and generate weights
        conditioning = self.class_embedding(class_labels)
        weights = self.hyper_net(conditioning)

        # Split into layer weights
        w1 = weights[:, :self.total_layer1].view(B, *self.layer1_weight_shape)
        w2 = weights[:, self.total_layer1:].view(B, *self.layer2_weight_shape)

        # Initialize grid
        if initial_grid is None:
            grid = torch.randn(B, self.grid_channels, H, W, device=device) * 0.1
        else:
            grid = initial_grid

        # Run NCA steps
        trajectory = []
        blend = torch.sigmoid(self.blend)

        for _ in range(num_steps):
            # Apply convolutions with per-sample weights (using group conv trick)
            # For simplicity, we'll use a loop over batch
            # In production, could use batch-specific conv implementations
            new_grids = []
            for b in range(B):
                h = F.conv2d(
                    grid[b:b+1], w1[b],
                    padding=1, padding_mode='circular'
                )
                h = self.film1(h, conditioning[b:b+1])
                h = F.relu(h)

                h = F.conv2d(
                    h, w2[b],
                    padding=1, padding_mode='circular'
                )
                h = self.film2(h, conditioning[b:b+1])
                h = F.relu(h)

                new_grids.append(h)

            update = torch.cat(new_grids, dim=0)
            grid = (1 - blend) * grid + blend * update

            rgb = torch.sigmoid(grid[:, :self.rgb_channels])
            trajectory.append(rgb)

        return trajectory[-1], trajectory


# Simpler model for faster experimentation
class SimpleConditionalNCA(nn.Module):
    """
    Simplified NCA for faster training and experimentation.
    Uses shared weights with FiLM conditioning only.
    """

    def __init__(
        self,
        grid_channels: int = 12,
        rgb_channels: int = 1,  # 1 for MNIST grayscale
        conditioning_dim: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        self.grid_channels = grid_channels
        self.rgb_channels = rgb_channels

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, conditioning_dim)

        # Perception: what each cell sees (Sobel-like filters + identity)
        self.perception = nn.Conv2d(
            grid_channels, grid_channels * 3,  # identity + grad_x + grad_y
            kernel_size=3, padding=1, padding_mode='circular',
            groups=grid_channels, bias=False
        )
        # Initialize with Sobel-like kernels
        self._init_perception()

        # Update network
        self.update_net = nn.Sequential(
            nn.Conv2d(grid_channels * 3, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, grid_channels, 1, bias=False),
        )
        # Initialize last layer to near-zero for stable start
        nn.init.zeros_(self.update_net[-1].weight)

        # FiLM conditioning on the update
        self.film = FiLMLayer(grid_channels, conditioning_dim)

    def _init_perception(self):
        """Initialize perception kernels with identity + Sobel filters."""
        with torch.no_grad():
            # Identity, Sobel X, Sobel Y for each channel
            identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8

            kernel = torch.stack([identity, sobel_x, sobel_y])  # [3, 3, 3]
            kernel = kernel.unsqueeze(1)  # [3, 1, 3, 3]
            kernel = kernel.repeat(self.grid_channels, 1, 1, 1)  # [C*3, 1, 3, 3]
            self.perception.weight.data = kernel

    def step(self, grid: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Single NCA step."""
        # Perceive neighborhood
        perceived = self.perception(grid)  # [B, C*3, H, W]

        # Compute update
        update = self.update_net(perceived)  # [B, C, H, W]

        # Apply FiLM conditioning
        update = self.film(update, conditioning)

        # Stochastic update (helps with stability)
        if self.training:
            mask = (torch.rand(grid.shape[0], 1, grid.shape[2], grid.shape[3],
                              device=grid.device) < 0.5).float()
            update = update * mask

        return grid + update

    def forward(
        self,
        class_labels: torch.Tensor,
        grid_size: tuple[int, int] = (28, 28),
        num_steps: int = 32,
        initial_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B = class_labels.shape[0]
        H, W = grid_size
        device = class_labels.device

        conditioning = self.class_embedding(class_labels)

        if initial_grid is None:
            grid = torch.zeros(B, self.grid_channels, H, W, device=device)
            # Seed in center
            grid[:, 3:6, H//2-1:H//2+1, W//2-1:W//2+1] = 1.0
        else:
            grid = initial_grid

        trajectory = []
        for _ in range(num_steps):
            grid = self.step(grid, conditioning)
            rgb = torch.sigmoid(grid[:, :self.rgb_channels])
            trajectory.append(rgb)

        return trajectory[-1], trajectory
