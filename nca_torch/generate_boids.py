"""
Generate diverse boid simulation sequences for training.

Features:
- Multiple behavior types (flocking, predator-prey, particle life)
- Varied colors, shapes, sizes
- Randomized parameters for diverse dataset

Usage:
    python generate_boids.py --num-sequences 2000 --output-dir ./data/boids
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Literal
import cv2


@dataclass
class BoidParams:
    """Parameters for a boid simulation."""
    # World
    width: int = 32
    height: int = 32
    wrap: bool = True  # Wrap around edges

    # Boids
    num_boids: int = 10
    boid_size: float = 1.5  # Radius in pixels
    boid_shape: Literal["circle", "triangle", "square"] = "circle"
    max_speed: float = 2.0
    max_force: float = 0.1

    # Behavior weights
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    separation_radius: float = 4.0
    perception_radius: float = 8.0

    # Colors (RGB, 0-255)
    background_color: tuple = (0, 0, 0)
    boid_color: tuple = (255, 255, 255)

    # Special behaviors
    behavior_type: Literal["flock", "predator_prey", "particle_life"] = "flock"

    # For predator_prey
    num_predators: int = 1
    predator_color: tuple = (255, 0, 0)
    predator_speed: float = 1.5

    # For particle_life (attraction/repulsion between types)
    num_types: int = 3
    type_colors: list = None


class Boid:
    def __init__(self, x, y, vx, vy, boid_type=0):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([vx, vy], dtype=np.float32)
        self.acc = np.array([0.0, 0.0], dtype=np.float32)
        self.boid_type = boid_type

    def apply_force(self, force):
        self.acc += force

    def update(self, max_speed, width, height, wrap=True, friction=0.0):
        self.vel += self.acc
        # Apply friction (damping)
        if friction > 0:
            self.vel *= (1.0 - friction)
        # Limit speed
        speed = np.linalg.norm(self.vel)
        if speed > max_speed:
            self.vel = self.vel / speed * max_speed
        self.pos += self.vel
        self.acc *= 0  # Reset acceleration

        # Wrap or bounce
        if wrap:
            self.pos[0] = self.pos[0] % width
            self.pos[1] = self.pos[1] % height
        else:
            if self.pos[0] < 0 or self.pos[0] >= width:
                self.vel[0] *= -1
                self.pos[0] = np.clip(self.pos[0], 0, width - 1)
            if self.pos[1] < 0 or self.pos[1] >= height:
                self.vel[1] *= -1
                self.pos[1] = np.clip(self.pos[1], 0, height - 1)


class BoidSimulation:
    def __init__(self, params: BoidParams):
        self.params = params
        self.boids = []
        self.predators = []
        self.attraction_matrix = None
        self._step_count = 0

        self._init_boids()

    def _init_boids(self):
        p = self.params

        # Initialize regular boids
        for i in range(p.num_boids):
            x = np.random.uniform(0, p.width)
            y = np.random.uniform(0, p.height)
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, p.max_speed)
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed

            if p.behavior_type == "particle_life":
                boid_type = i % p.num_types
            else:
                boid_type = 0

            self.boids.append(Boid(x, y, vx, vy, boid_type))

        # Initialize predators
        if p.behavior_type == "predator_prey":
            for _ in range(p.num_predators):
                x = np.random.uniform(0, p.width)
                y = np.random.uniform(0, p.height)
                angle = np.random.uniform(0, 2 * np.pi)
                vx = np.cos(angle) * p.predator_speed
                vy = np.sin(angle) * p.predator_speed
                self.predators.append(Boid(x, y, vx, vy, boid_type=-1))

        # Initialize attraction matrix for particle_life
        if p.behavior_type == "particle_life":
            # Random attraction/repulsion between types
            # Use stronger values for more pronounced behavior
            self.attraction_matrix = np.random.uniform(-1, 1, (p.num_types, p.num_types))
            # Make some interactions stronger
            self.attraction_matrix *= np.random.uniform(0.5, 1.5, (p.num_types, p.num_types))
            # Clip to valid range
            self.attraction_matrix = np.clip(self.attraction_matrix, -1, 1)

    def _distance(self, b1, b2):
        """Calculate distance with optional wrapping."""
        diff = b2.pos - b1.pos
        if self.params.wrap:
            # Handle wraparound
            if abs(diff[0]) > self.params.width / 2:
                diff[0] -= np.sign(diff[0]) * self.params.width
            if abs(diff[1]) > self.params.height / 2:
                diff[1] -= np.sign(diff[1]) * self.params.height
        return diff, np.linalg.norm(diff)

    def _separation(self, boid, neighbors):
        """Steer away from nearby boids."""
        steer = np.array([0.0, 0.0])
        count = 0
        for other in neighbors:
            diff, dist = self._distance(boid, other)
            if 0 < dist < self.params.separation_radius:
                steer -= diff / (dist + 0.001)
                count += 1
        if count > 0:
            steer /= count
        return steer * self.params.separation_weight

    def _alignment(self, boid, neighbors):
        """Steer towards average heading of neighbors."""
        avg_vel = np.array([0.0, 0.0])
        count = 0
        for other in neighbors:
            _, dist = self._distance(boid, other)
            if 0 < dist < self.params.perception_radius:
                avg_vel += other.vel
                count += 1
        if count > 0:
            avg_vel /= count
            steer = avg_vel - boid.vel
            return steer * self.params.alignment_weight
        return np.array([0.0, 0.0])

    def _cohesion(self, boid, neighbors):
        """Steer towards center of mass of neighbors."""
        center = np.array([0.0, 0.0])
        count = 0
        for other in neighbors:
            diff, dist = self._distance(boid, other)
            if 0 < dist < self.params.perception_radius:
                center += boid.pos + diff  # Account for wrapping
                count += 1
        if count > 0:
            center /= count
            desired = center - boid.pos
            return desired * self.params.cohesion_weight
        return np.array([0.0, 0.0])

    def _flee_predators(self, boid):
        """Flee from predators."""
        steer = np.array([0.0, 0.0])
        for pred in self.predators:
            diff, dist = self._distance(boid, pred)
            if dist < self.params.perception_radius * 1.5:
                steer -= diff / (dist + 0.001) * 3.0  # Strong flee response
        return steer

    def _chase_prey(self, predator):
        """Predator chases nearest prey."""
        nearest = None
        nearest_dist = float('inf')
        for boid in self.boids:
            diff, dist = self._distance(predator, boid)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = diff
        if nearest is not None:
            return nearest * 0.5
        return np.array([0.0, 0.0])

    def _particle_life_force(self, boid):
        """
        Particle Life force: attraction/repulsion based on particle type.

        Uses a distance-dependent force curve:
        - Very close (< beta): Always repel (prevents overlap)
        - Medium distance (beta to radius): Attract or repel based on matrix
        - Far (> radius): No force
        """
        force = np.array([0.0, 0.0])
        beta = 3.0  # Repulsion zone radius
        radius = self.params.perception_radius

        for other in self.boids:
            if other is boid:
                continue
            diff, dist = self._distance(boid, other)

            if dist <= 0 or dist >= radius:
                continue

            # Get attraction strength from matrix (-1 to 1)
            attraction = self.attraction_matrix[boid.boid_type, other.boid_type]

            # Normalize direction
            direction = diff / dist

            if dist < beta:
                # Repulsion zone - always push apart
                # Linear ramp from -1 at dist=0 to 0 at dist=beta
                f = dist / beta - 1.0
            else:
                # Attraction/repulsion zone based on matrix
                # Smooth force curve that peaks in the middle
                normalized_dist = (dist - beta) / (radius - beta)
                # Bell-shaped curve
                f = attraction * (1.0 - abs(2.0 * normalized_dist - 1.0))

            force += direction * f * 0.5

        return force

    def step(self):
        """Advance simulation by one step."""
        p = self.params

        # Update boids
        for boid in self.boids:
            if p.behavior_type == "flock":
                sep = self._separation(boid, self.boids)
                ali = self._alignment(boid, self.boids)
                coh = self._cohesion(boid, self.boids)
                boid.apply_force(sep + ali + coh)

            elif p.behavior_type == "predator_prey":
                sep = self._separation(boid, self.boids)
                ali = self._alignment(boid, self.boids)
                coh = self._cohesion(boid, self.boids)
                flee = self._flee_predators(boid)
                boid.apply_force(sep + ali + coh + flee)

            elif p.behavior_type == "particle_life":
                force = self._particle_life_force(boid)
                boid.apply_force(force)

            # Limit force
            force_mag = np.linalg.norm(boid.acc)
            if force_mag > p.max_force:
                boid.acc = boid.acc / force_mag * p.max_force

            # Use friction for particle life to create stable structures
            friction = 0.05 if p.behavior_type == "particle_life" else 0.0
            boid.update(p.max_speed, p.width, p.height, p.wrap, friction)

        # Update predators
        for pred in self.predators:
            chase = self._chase_prey(pred)
            pred.apply_force(chase)
            pred.update(p.predator_speed, p.width, p.height, p.wrap, friction=0.0)

        # Check for predator-prey collisions
        if p.behavior_type == "predator_prey":
            catch_radius = p.boid_size + p.boid_size * 1.5  # prey + predator size
            eaten = []
            for pred in self.predators:
                for boid in self.boids:
                    _, dist = self._distance(pred, boid)
                    if dist < catch_radius:
                        eaten.append(boid)

            # Remove eaten prey
            for boid in eaten:
                if boid in self.boids:
                    self.boids.remove(boid)

        self._step_count += 1

    def render(self) -> np.ndarray:
        """Render current state to RGB image."""
        p = self.params
        img = np.full((p.height, p.width, 3), p.background_color, dtype=np.uint8)

        # Draw boids
        for boid in self.boids:
            x, y = int(boid.pos[0]) % p.width, int(boid.pos[1]) % p.height

            if p.behavior_type == "particle_life":
                # Particle life: single pixels with type-based colors
                color = p.type_colors[boid.boid_type] if p.type_colors else p.boid_color
                img[y, x] = color
            else:
                # Flock and predator_prey: use shapes
                color = p.boid_color

                if p.boid_shape == "circle":
                    cv2.circle(img, (x, y), int(p.boid_size), color, -1)
                elif p.boid_shape == "square":
                    s = int(p.boid_size)
                    cv2.rectangle(img, (x - s, y - s), (x + s, y + s), color, -1)
                elif p.boid_shape == "triangle":
                    # Triangle pointing in velocity direction
                    angle = np.arctan2(boid.vel[1], boid.vel[0])
                    s = p.boid_size * 1.5
                    pts = np.array([
                        [x + np.cos(angle) * s, y + np.sin(angle) * s],
                        [x + np.cos(angle + 2.5) * s * 0.7, y + np.sin(angle + 2.5) * s * 0.7],
                        [x + np.cos(angle - 2.5) * s * 0.7, y + np.sin(angle - 2.5) * s * 0.7],
                    ], dtype=np.int32)
                    cv2.fillPoly(img, [pts], color)

        # Draw predators (slightly larger, distinct color)
        for pred in self.predators:
            x, y = int(pred.pos[0]) % p.width, int(pred.pos[1]) % p.height
            # Draw predators as slightly larger filled circles
            size = int(p.boid_size * 1.5)
            cv2.circle(img, (x, y), size, p.predator_color, -1)

        return img

    def generate_sequence(self, num_frames: int, warmup: int = 20) -> np.ndarray:
        """Generate a sequence of frames."""
        self._step_count = 0

        # Warmup
        for _ in range(warmup):
            self.step()

        # Generate frames
        frames = []
        for _ in range(num_frames):
            frames.append(self.render())
            self.step()

        return np.stack(frames)


def random_color():
    """Generate a random vibrant color."""
    # Use HSV for better colors, then convert
    h = np.random.uniform(0, 180)
    s = np.random.uniform(150, 255)
    v = np.random.uniform(180, 255)
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(c) for c in rgb)


def random_params(size: int = 32, behavior: str = None) -> BoidParams:
    """Generate random simulation parameters."""
    if behavior is None:
        behavior = np.random.choice(["flock", "predator_prey", "particle_life"])

    # Random colors
    bg_brightness = np.random.choice(["dark", "light", "colored"])
    if bg_brightness == "dark":
        bg_color = tuple(np.random.randint(0, 50, 3).tolist())
    elif bg_brightness == "light":
        bg_color = tuple(np.random.randint(200, 255, 3).tolist())
    else:
        bg_color = random_color()

    boid_color = random_color()
    # Ensure contrast
    while np.sum(np.abs(np.array(boid_color) - np.array(bg_color))) < 200:
        boid_color = random_color()

    # Behavior-specific parameters
    if behavior == "flock":
        params = BoidParams(
            width=size,
            height=size,
            wrap=np.random.random() > 0.3,
            num_boids=np.random.randint(8, 20),
            boid_size=np.random.uniform(1.0, 2.0),
            boid_shape=np.random.choice(["circle", "triangle", "square"]),
            max_speed=np.random.uniform(1.5, 3.0),
            max_force=np.random.uniform(0.1, 0.3),
            # Stronger separation to prevent clumping
            separation_weight=np.random.uniform(2.0, 4.0),
            alignment_weight=np.random.uniform(0.5, 1.5),
            cohesion_weight=np.random.uniform(0.3, 1.0),
            separation_radius=np.random.uniform(3.0, 6.0),
            perception_radius=np.random.uniform(6.0, 12.0),
            background_color=bg_color,
            boid_color=boid_color,
            behavior_type=behavior,
        )

    elif behavior == "predator_prey":
        params = BoidParams(
            width=size,
            height=size,
            wrap=True,  # Always wrap for predator_prey
            num_boids=np.random.randint(15, 30),
            boid_size=np.random.uniform(1.0, 1.5),
            boid_shape=np.random.choice(["circle", "triangle"]),
            max_speed=np.random.uniform(2.0, 3.0),
            max_force=np.random.uniform(0.15, 0.3),
            separation_weight=np.random.uniform(1.5, 3.0),
            alignment_weight=np.random.uniform(0.5, 1.5),
            cohesion_weight=np.random.uniform(0.5, 1.5),
            separation_radius=np.random.uniform(2.0, 4.0),
            perception_radius=np.random.uniform(6.0, 10.0),
            background_color=bg_color,
            boid_color=boid_color,
            behavior_type=behavior,
            num_predators=np.random.randint(1, 2),
            predator_color=(255, np.random.randint(0, 80), np.random.randint(0, 80)),
            predator_speed=np.random.uniform(1.2, 1.8),  # Slower than prey
        )

    elif behavior == "particle_life":
        num_types = np.random.randint(3, 6)
        # Generate distinct colors for each type
        type_colors = []
        hues = np.linspace(0, 160, num_types, endpoint=False)  # Spread across hue spectrum
        np.random.shuffle(hues)
        for h in hues:
            hsv = np.array([[[h, 255, 255]]], dtype=np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
            type_colors.append(tuple(int(c) for c in rgb))

        params = BoidParams(
            width=size,
            height=size,
            wrap=True,
            num_boids=np.random.randint(40, 80),  # More particles for richer dynamics
            boid_size=1.0,  # Single pixel
            boid_shape="circle",  # Doesn't matter for single pixel
            max_speed=np.random.uniform(1.5, 3.0),
            max_force=np.random.uniform(0.2, 0.5),  # Stronger forces
            separation_weight=0.0,  # Handled by particle life force
            alignment_weight=0.0,
            cohesion_weight=0.0,
            separation_radius=1.5,
            perception_radius=np.random.uniform(10.0, 16.0),  # Larger perception
            background_color=bg_color,
            boid_color=(255, 255, 255),  # Not used
            behavior_type=behavior,
            num_types=num_types,
            type_colors=type_colors,
        )

    return params


def generate_dataset(
    num_sequences: int,
    frames_per_seq: int = 64,
    size: int = 32,
    output_dir: Path = None,
) -> np.ndarray:
    """Generate a dataset of boid sequences."""
    sequences = []

    for i in tqdm(range(num_sequences), desc="Generating sequences"):
        params = random_params(size)
        sim = BoidSimulation(params)
        seq = sim.generate_sequence(frames_per_seq)
        sequences.append(seq)

    return np.stack(sequences)


def generate_moving_circle_sequence(
    size: int = 32,
    frames: int = 64,
    radius: int = 3,
    speed: float = 1.0,
    start_x: int = None,
    start_y: int = None,
    bg_color: tuple = (0, 0, 0),
    circle_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """
    Generate a single sequence of a circle moving upward with toroidal wrapping.

    This is a simple toy dataset for sanity checking dynamics learning.

    Args:
        size: Grid size (width and height)
        frames: Number of frames to generate
        radius: Circle radius in pixels
        speed: Pixels to move up per frame
        start_x: Starting x position (default: center)
        start_y: Starting y position (default: center)
        bg_color: Background color (RGB)
        circle_color: Circle color (RGB)

    Returns:
        Sequence of frames [frames, size, size, 3]
    """
    if start_x is None:
        start_x = size // 2
    if start_y is None:
        start_y = size // 2

    sequence = []
    y_pos = float(start_y)

    for _ in range(frames):
        # Create frame
        img = np.full((size, size, 3), bg_color, dtype=np.uint8)

        # Draw circle at current position (moving UP means decreasing y)
        x = int(start_x)
        y = int(y_pos) % size  # Toroidal wrap
        cv2.circle(img, (x, y), radius, circle_color, -1)

        sequence.append(img)

        # Move up (decrease y, with wrapping)
        y_pos = (y_pos - speed) % size

    return np.stack(sequence)


def generate_toy_dataset(
    num_sequences: int = 100,
    frames_per_seq: int = 64,
    size: int = 32,
    variation: str = "fixed",
) -> np.ndarray:
    """
    Generate a toy dataset of circles moving upward.

    Args:
        num_sequences: Number of sequences to generate
        frames_per_seq: Frames per sequence
        size: Grid size
        variation: Type of variation:
            - "fixed": All sequences identical (single circle, center start, speed=1)
            - "position": Vary starting position
            - "speed": Vary speed (1-3 pixels/frame)
            - "all": Vary position, speed, radius, and colors

    Returns:
        Dataset array [num_sequences, frames_per_seq, size, size, 3]
    """
    sequences = []

    for i in tqdm(range(num_sequences), desc="Generating toy sequences"):
        if variation == "fixed":
            # All identical - simplest possible test
            seq = generate_moving_circle_sequence(
                size=size,
                frames=frames_per_seq,
                radius=3,
                speed=1.0,
            )
        elif variation == "position":
            # Vary starting position only
            seq = generate_moving_circle_sequence(
                size=size,
                frames=frames_per_seq,
                radius=3,
                speed=1.0,
                start_x=np.random.randint(4, size - 4),
                start_y=np.random.randint(0, size),
            )
        elif variation == "speed":
            # Vary speed only
            seq = generate_moving_circle_sequence(
                size=size,
                frames=frames_per_seq,
                radius=3,
                speed=np.random.uniform(0.5, 2.0),
            )
        elif variation == "all":
            # Vary everything
            bg_color = tuple(np.random.randint(0, 50, 3).tolist())
            circle_color = random_color()
            seq = generate_moving_circle_sequence(
                size=size,
                frames=frames_per_seq,
                radius=np.random.randint(2, 5),
                speed=np.random.uniform(0.5, 2.0),
                start_x=np.random.randint(4, size - 4),
                start_y=np.random.randint(0, size),
                bg_color=bg_color,
                circle_color=circle_color,
            )
        else:
            raise ValueError(f"Unknown variation: {variation}")

        sequences.append(seq)

    return np.stack(sequences)


def main():
    parser = argparse.ArgumentParser(description="Generate boid simulation data")
    parser.add_argument("--num-sequences", type=int, default=2000,
                        help="Number of sequences to generate")
    parser.add_argument("--frames-per-seq", type=int, default=64,
                        help="Frames per sequence")
    parser.add_argument("--size", type=int, default=32,
                        help="Grid size")
    parser.add_argument("--output-dir", type=str, default="./data/boids",
                        help="Output directory")
    parser.add_argument("--preview", action="store_true",
                        help="Preview a few simulations before generating")
    parser.add_argument("--toy", type=str, default=None,
                        choices=["fixed", "position", "speed", "all"],
                        help="Generate toy dataset (circle moving up) instead of boids. "
                             "Variations: fixed (identical), position (vary start), "
                             "speed (vary velocity), all (vary everything)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        print("Previewing simulations...")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if args.toy:
            # Preview toy dataset
            seq = generate_moving_circle_sequence(
                size=args.size,
                frames=90,
                radius=3,
                speed=1.0,
            )
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_title("Toy: Circle moving up")
            ax.axis('off')
            im = ax.imshow(seq[0])

            def update(i):
                im.set_array(seq[i])
                return [im]

            anim = FuncAnimation(fig, update, frames=len(seq), interval=50)
            plt.show()
        else:
            for behavior in ["flock", "predator_prey", "particle_life"]:
                params = random_params(args.size, behavior=behavior)
                sim = BoidSimulation(params)

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set_title(f"Behavior: {behavior}")
                ax.axis('off')

                # Warmup (same as generation)
                for _ in range(30):
                    sim.step()

                # Record frames
                frames = []
                for _ in range(90):
                    frames.append(sim.render())
                    sim.step()

                im = ax.imshow(frames[0])

                def update(i):
                    im.set_array(frames[i])
                    return [im]

                anim = FuncAnimation(fig, update, frames=len(frames), interval=50)
                plt.show()

        return

    # Generate dataset
    if args.toy:
        print(f"Generating {args.num_sequences} toy sequences (variation: {args.toy})...")
        data = generate_toy_dataset(
            num_sequences=args.num_sequences,
            frames_per_seq=args.frames_per_seq,
            size=args.size,
            variation=args.toy,
        )
        output_file = output_dir / f"toy_circle_{args.toy}_{args.size}x{args.size}.npy"
    else:
        print(f"Generating {args.num_sequences} boid sequences...")
        data = generate_dataset(
            args.num_sequences,
            frames_per_seq=args.frames_per_seq,
            size=args.size,
        )
        output_file = output_dir / f"boids_{args.size}x{args.size}.npy"

    np.save(output_file, data.astype(np.uint8))

    print(f"\nSaved to {output_file}")
    print(f"Shape: {data.shape}")
    print(f"Size: {data.nbytes / 1e9:.2f} GB")
    print(f"\nTo visualize:")
    print(f"  python visualize_data.py {output_file}")
    print(f"\nTo train:")
    print(f"  python train_dynamics.py --data {output_file}")


if __name__ == "__main__":
    main()
