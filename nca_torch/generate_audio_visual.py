"""
Generate audio-visual paired sequences for NCA training.

Pipeline:
1. Load audio samples (e.g., instrument notes, sounds)
2. Compute spectrograms (encoder input)
3. Generate deterministic visuals from audio (decoder target)
4. Output as sequences: spectrogram frames → visual frames

The encoder learns to map spectrograms to latent space.
The NCA learns to generate corresponding visuals.

When you play audio through the trained model, you get
visuals that correspond to the sound characteristics.
"""

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from typing import Optional
import warnings


def load_audio(path: Path, sr: int = 22050, duration: Optional[float] = None):
    """Load audio file, return mono waveform."""
    try:
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
        return y, sr
    except ImportError:
        raise ImportError("Install librosa: pip install librosa")


def compute_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
) -> np.ndarray:
    """
    Compute mel spectrogram from audio.

    Returns: [T, n_mels] array, normalized to [0, 1]
    """
    import librosa

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Convert to dB and normalize
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    return S_norm.T  # [T, n_mels]


def audio_to_oscilloscope(
    y: np.ndarray,
    sr: int,
    frame_samples: int,
    size: int = 32,
    color: bool = False,
) -> np.ndarray:
    """
    Generate oscilloscope-style visualization from audio.

    Creates a 2D pattern by plotting waveform as XY coordinates,
    using consecutive samples or Hilbert transform for stereo-like effect.

    Returns: [H, W] grayscale or [H, W, 3] RGB image
    """
    import colorsys
    from scipy.signal import hilbert

    # Use Hilbert transform to get analytic signal (creates pseudo-stereo)
    analytic = hilbert(y[:frame_samples])
    x = np.real(analytic)
    y_coord = np.imag(analytic)

    # Normalize to [-1, 1]
    x = x / (np.abs(x).max() + 1e-8)
    y_coord = y_coord / (np.abs(y_coord).max() + 1e-8)

    # Convert to pixel coordinates
    px = ((x + 1) / 2 * (size - 1)).astype(int)
    py = ((y_coord + 1) / 2 * (size - 1)).astype(int)

    # Clip to valid range
    px = np.clip(px, 0, size - 1)
    py = np.clip(py, 0, size - 1)

    if color:
        # Compute velocity (derivative of position)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y_coord, prepend=y_coord[0])
        velocity = np.sqrt(dx**2 + dy**2)

        # Normalize velocity to [0, 1]
        velocity = velocity / (velocity.max() + 1e-8)

        # RGB image with velocity-based coloring
        img = np.zeros((size, size, 3), dtype=np.float32)

        # Draw points with color based on velocity
        # Fast = warm (red/orange), slow = cool (blue/purple)
        for i in range(len(px)):
            vel = velocity[i]
            # Hue: 0.7 (blue) for slow → 0 (red) for fast
            hue = 0.7 * (1 - vel)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.8 + vel * 0.2)

            img[py[i], px[i], 0] += r
            img[py[i], px[i], 1] += g
            img[py[i], px[i], 2] += b

        # Apply blur to each channel
        for c in range(3):
            img[:, :, c] = cv2.GaussianBlur(img[:, :, c], (3, 3), 0.5)

        # Normalize
        if img.max() > 0:
            img = img / img.max()

        return img
    else:
        # Grayscale
        img = np.zeros((size, size), dtype=np.float32)

        for i in range(len(px)):
            img[py[i], px[i]] += 1.0

        img = cv2.GaussianBlur(img, (3, 3), 0.5)

        if img.max() > 0:
            img = img / img.max()

        return img


def audio_to_radial_spectrum(
    y: np.ndarray,
    sr: int,
    frame_samples: int,
    size: int = 32,
    n_bins: int = 32,
    color: bool = False,
) -> np.ndarray:
    """
    Generate radial frequency visualization.

    Frequency content is displayed as rings emanating from center.
    Low frequencies = center (warm colors), high frequencies = outer (cool colors).

    Returns: [H, W] grayscale or [H, W, 3] RGB image
    """
    import colorsys

    # Compute FFT
    fft = np.abs(np.fft.rfft(y[:frame_samples]))
    fft = fft[:n_bins]

    # Normalize
    fft = fft / (fft.max() + 1e-8)

    center = size // 2

    # Create coordinate grids
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    max_dist = center

    if color:
        # RGB output with frequency-based coloring
        img = np.zeros((size, size, 3), dtype=np.float32)

        # Map each frequency bin to a colored ring
        for i, magnitude in enumerate(fft):
            inner_r = (i / n_bins) * max_dist
            outer_r = ((i + 1) / n_bins) * max_dist

            mask = (dist >= inner_r) & (dist < outer_r)

            # Hue: 0 (red) for low freq → 0.7 (blue) for high freq
            hue = (i / n_bins) * 0.7
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, magnitude)

            img[mask, 0] = r
            img[mask, 1] = g
            img[mask, 2] = b

        return img
    else:
        # Grayscale output
        img = np.zeros((size, size), dtype=np.float32)

        for i, magnitude in enumerate(fft):
            inner_r = (i / n_bins) * max_dist
            outer_r = ((i + 1) / n_bins) * max_dist

            mask = (dist >= inner_r) & (dist < outer_r)
            img[mask] = magnitude

        return img


def audio_to_waveform_grid(
    y: np.ndarray,
    sr: int,
    frame_samples: int,
    size: int = 32,
) -> np.ndarray:
    """
    Generate waveform visualization as horizontal lines.

    Each row shows a portion of the waveform.

    Returns: [H, W] grayscale image
    """
    samples_per_row = frame_samples // size
    img = np.zeros((size, size), dtype=np.float32)

    for row in range(size):
        start = row * samples_per_row
        end = start + samples_per_row

        if end <= len(y):
            segment = y[start:end]
            # Resample to size columns
            resampled = np.interp(
                np.linspace(0, len(segment) - 1, size),
                np.arange(len(segment)),
                segment,
            )
            # Map amplitude to brightness
            img[row, :] = (resampled + 1) / 2

    return img


def audio_to_cymatics(
    y: np.ndarray,
    sr: int,
    frame_samples: int,
    size: int = 32,
) -> np.ndarray:
    """
    Generate cymatics-inspired pattern based on dominant frequencies.

    Uses detected frequencies to create Chladni-like patterns.

    Returns: [H, W] grayscale image
    """
    # Get dominant frequencies via FFT
    fft = np.abs(np.fft.rfft(y[:frame_samples]))
    freqs = np.fft.rfftfreq(frame_samples, 1 / sr)

    # Find top 3 frequency peaks
    peak_indices = np.argsort(fft)[-3:]
    peak_freqs = freqs[peak_indices]
    peak_mags = fft[peak_indices]
    peak_mags = peak_mags / (peak_mags.sum() + 1e-8)

    # Create coordinate grid
    coords = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(coords, coords)

    # Generate pattern as sum of standing waves
    img = np.zeros((size, size), dtype=np.float32)

    for freq, mag in zip(peak_freqs, peak_mags):
        # Map frequency to mode numbers (scaled to reasonable range)
        n = int((freq / sr) * 20) + 1
        m = n + 1

        # Chladni-like pattern
        pattern = (
            np.cos(n * np.pi * X) * np.cos(m * np.pi * Y)
            - np.cos(m * np.pi * X) * np.cos(n * np.pi * Y)
        )
        img += mag * np.abs(pattern)

    # Normalize
    if img.max() > 0:
        img = img / img.max()

    return img


def extract_audio_features_for_circles(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> dict:
    """Extract audio features for circle visualization."""
    import librosa

    # RMS energy (loudness) per frame
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms = rms / (rms.max() + 1e-8)

    # Spectral centroid (brightness/pitch)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid = centroid / (centroid.max() + 1e-8)

    # Onset strength (for speed/movement)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset = onset / (onset.max() + 1e-8)

    # Detect harmonics from FFT of the whole signal
    # Use a longer FFT for better frequency resolution
    n_fft = min(len(y), 4096)
    fft = np.abs(np.fft.rfft(y[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)

    # Find peaks (harmonics) - local maxima above threshold
    fft_norm = fft / (fft.max() + 1e-8)
    threshold = 0.1  # Minimum relative magnitude for a harmonic

    # Simple peak detection
    harmonics = []
    for i in range(1, len(fft_norm) - 1):
        if (fft_norm[i] > fft_norm[i-1] and
            fft_norm[i] > fft_norm[i+1] and
            fft_norm[i] > threshold and
            freqs[i] > 50):  # Ignore very low frequencies
            harmonics.append({
                'freq': freqs[i],
                'magnitude': fft_norm[i],
                'freq_normalized': min(freqs[i] / 2000, 1.0),  # Normalize to ~2kHz max
            })

    # Sort by magnitude and keep top harmonics (max 6)
    harmonics = sorted(harmonics, key=lambda h: h['magnitude'], reverse=True)[:6]

    # Sort by frequency for consistent ordering
    harmonics = sorted(harmonics, key=lambda h: h['freq'])

    return {
        'rms': rms,
        'centroid': centroid,
        'onset': onset,
        'harmonics': harmonics,
        'num_harmonics': len(harmonics),
    }


def audio_to_circles(
    features: dict,
    frame_idx: int,
    size: int = 32,
    base_direction: float = 0.0,
) -> np.ndarray:
    """
    Generate circle visualization based on audio features.

    Each detected harmonic becomes a circle:
    - Color: based on harmonic frequency (low = red, high = blue)
    - Size: based on harmonic magnitude × RMS
    - Position: orbiting, with fundamental at center

    Args:
        features: Dict with 'rms', 'centroid', 'onset', 'harmonics' arrays
        frame_idx: Current frame index
        size: Image size
        base_direction: Base movement direction in radians

    Returns: [H, W, 3] RGB image
    """
    import colorsys

    img = np.zeros((size, size, 3), dtype=np.float32)

    harmonics = features['harmonics']
    if not harmonics:
        return img  # No harmonics detected, return black

    # Get features for this frame (clamp to valid range)
    T = len(features['rms'])
    idx = min(frame_idx, T - 1)

    rms = features['rms'][idx]
    onset = features['onset'][idx]

    # Accumulate velocity for smooth motion
    speed = 0.3 + onset * 0.7
    dx = np.cos(base_direction) * speed * frame_idx * 0.4
    dy = np.sin(base_direction) * speed * frame_idx * 0.4

    num_harmonics = len(harmonics)

    for i, harmonic in enumerate(harmonics):
        freq_norm = harmonic['freq_normalized']
        magnitude = harmonic['magnitude']

        # Position: harmonics orbit around center
        # Fundamental (first harmonic) near center, higher harmonics further out
        orbit_radius = 3 + i * 3
        orbit_angle = i * (2 * np.pi / max(num_harmonics, 1)) + frame_idx * 0.02 * (i + 1)

        cx = size // 2 + np.cos(orbit_angle) * orbit_radius + dx
        cy = size // 2 + np.sin(orbit_angle) * orbit_radius + dy

        # Wrap around (toroidal)
        cx = cx % size
        cy = cy % size

        # Color based on frequency (low = red, high = blue)
        hue = freq_norm * 0.7
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.7 + magnitude * 0.3)

        # Size based on magnitude and RMS
        # Fundamental (i=0) is largest, higher harmonics smaller
        circle_radius = (2 + magnitude * 4 + rms * 2) * (1.0 - i * 0.1)
        circle_radius = max(circle_radius, 1.5)

        # Draw circle with soft edges
        Y, X = np.ogrid[:size, :size]

        # Handle wrapping for distance calculation
        dx_grid = np.minimum(np.abs(X - cx), size - np.abs(X - cx))
        dy_grid = np.minimum(np.abs(Y - cy), size - np.abs(Y - cy))
        dist = np.sqrt(dx_grid**2 + dy_grid**2)

        # Soft circle mask
        mask = np.clip(1.0 - (dist - circle_radius) / 2, 0, 1)

        img[:, :, 0] = np.maximum(img[:, :, 0], mask * r)
        img[:, :, 1] = np.maximum(img[:, :, 1], mask * g)
        img[:, :, 2] = np.maximum(img[:, :, 2], mask * b)

    return np.clip(img, 0, 1)


def generate_sequence(
    audio_path: Path,
    size: int = 32,
    seq_length: int = 16,
    sr: int = 22050,
    hop_length: int = 512,
    visual_mode: str = "oscilloscope",
    color: bool = False,
) -> tuple:
    """
    Generate a single audio-visual sequence.

    Returns:
        spectrogram_frames: [T, H, W, C] - encoder input (C=1 or 3)
        visual_frames: [T, H, W, C] - decoder target (C=1 or 3)
    """
    # Load audio
    y, sr = load_audio(audio_path, sr=sr)

    # Calculate frame parameters
    frame_samples = hop_length * 4  # Samples per visual frame
    total_samples = seq_length * hop_length

    if len(y) < total_samples + frame_samples:
        # Pad if too short
        y = np.pad(y, (0, total_samples + frame_samples - len(y)))

    # Compute full spectrogram
    spec = compute_spectrogram(y, sr, hop_length=hop_length, n_mels=size)

    # Generate frames
    spec_frames = []
    visual_frames = []

    visual_funcs = {
        "oscilloscope": audio_to_oscilloscope,
        "radial": audio_to_radial_spectrum,
        "waveform": audio_to_waveform_grid,
        "cymatics": audio_to_cymatics,
    }
    visual_func = visual_funcs.get(visual_mode, audio_to_oscilloscope)

    # Frame 0: Full 2D spectrogram (time x frequency) as encoder input
    # Use enough of the spectrogram to cover the sequence duration
    spec_frames_needed = min(len(spec), seq_length * 4)  # More temporal resolution
    spec_2d = spec[:spec_frames_needed, :]  # [T_spec, n_mels]
    # Resize to square: time on X-axis, frequency on Y-axis (low freq at bottom)
    spec_2d = np.flipud(spec_2d.T)  # [n_mels, T_spec] with low freq at bottom
    spec_img = cv2.resize(spec_2d, (size, size), interpolation=cv2.INTER_LINEAR)
    spec_frames.append(spec_img)

    # For circles mode, extract features once for the whole clip
    circle_features = None
    base_direction = 0.0
    if visual_mode == "circles":
        circle_features = extract_audio_features_for_circles(y, sr, hop_length)
        # Direction based on average centroid (deterministic per note)
        avg_centroid = circle_features['centroid'].mean()
        base_direction = avg_centroid * 2 * np.pi  # Map centroid to direction

    # Frames 1+: Visual representations
    for t in range(1, seq_length):
        # Visual frame
        audio_start = t * hop_length
        audio_chunk = y[audio_start : audio_start + frame_samples]
        if len(audio_chunk) < frame_samples:
            audio_chunk = np.pad(audio_chunk, (0, frame_samples - len(audio_chunk)))

        if visual_mode == "circles":
            visual_img = audio_to_circles(
                circle_features, t, size,
                base_direction=base_direction,
            )
        elif color and visual_mode in ("radial", "oscilloscope"):
            if visual_mode == "radial":
                visual_img = audio_to_radial_spectrum(audio_chunk, sr, frame_samples, size, color=True)
            else:
                visual_img = audio_to_oscilloscope(audio_chunk, sr, frame_samples, size, color=True)
        else:
            visual_img = visual_func(audio_chunk, sr, frame_samples, size)
        visual_frames.append(visual_img)

    # Add placeholder for frame 0 visual (not used, but keeps array aligned)
    if visual_mode == "circles":
        visual_frames.insert(0, visual_frames[0] if visual_frames else np.zeros((size, size, 3)))
    else:
        visual_frames.insert(0, visual_frames[0] if visual_frames else np.zeros((size, size)))

    # Apply temporal smoothing (exponential moving average)
    smoothing = 0.7  # Higher = smoother but more lag
    smoothed_frames = [visual_frames[0]]
    for t in range(1, len(visual_frames)):
        smoothed = smoothing * smoothed_frames[-1] + (1 - smoothing) * visual_frames[t]
        smoothed_frames.append(smoothed)
    visual_frames = smoothed_frames

    visual_frames = np.array(visual_frames)

    # Handle channel dimension
    if visual_mode == "circles" or (color and visual_mode in ("radial", "oscilloscope")):
        # RGB: visual_frames is [T, H, W, 3], spec needs to match
        # Convert grayscale spectrogram to RGB by replicating channels
        spec_frames = np.array(spec_frames)[:, :, :, np.newaxis]
        spec_frames = np.repeat(spec_frames, 3, axis=-1)
    else:
        # Grayscale: add channel dimension
        spec_frames = np.array(spec_frames)[:, :, :, np.newaxis]
        visual_frames = visual_frames[:, :, :, np.newaxis]

    # Normalize to uint8
    spec_frames = (spec_frames * 255).astype(np.uint8)
    visual_frames = (visual_frames * 255).astype(np.uint8)

    return spec_frames, visual_frames


def generate_instrument_spectrograms(
    audio_dir: Path,
    output_dir: Path,
    size: int = 32,
    sr: int = 22050,
):
    """
    Generate spectrograms for all instruments and notes (F4-B5).

    Handles two directory structures:
    1. Hierarchical: audio_dir/instrument/note.wav
    2. Flattened: audio_dir/instrument_note.wav

    Missing notes are filled by pitch-shifting from nearest available note.
    Output: output_dir/spectrograms/{instrument}/{note}.png
    """
    import librosa

    NOTE_ORDER = ['F4', 'Fs4', 'G4', 'Gs4', 'A4', 'As4', 'B4',
                  'C5', 'Cs5', 'D5', 'Ds5', 'E5', 'F5', 'Fs5',
                  'G5', 'Gs5', 'A5', 'As5', 'B5']
    NOTE_TO_SEMITONE = {note: i for i, note in enumerate(NOTE_ORDER)}

    def audio_to_spectrogram(audio_path: Path, pitch_shift: float = 0) -> np.ndarray:
        """Load audio and generate spectrogram image."""
        y, _ = librosa.load(str(audio_path), sr=sr, duration=2.0)
        if pitch_shift != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

        hop_length = max(1, len(y) // size)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=size, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        img = cv2.resize(S_norm, (size, size))
        img = np.flipud(img)  # Low freq at bottom
        img = (img * 255).astype(np.uint8)
        # Stack to 3 channels (RGB)
        return np.stack([img, img, img], axis=-1)

    def find_nearest_sample(available: dict, target_note: str):
        """Find nearest available sample and return (path, pitch_shift)."""
        if target_note in available:
            return available[target_note], 0

        target_idx = NOTE_TO_SEMITONE.get(target_note)
        if target_idx is None:
            return None, 0

        best_path, best_shift = None, float('inf')
        for note, path in available.items():
            if note in NOTE_TO_SEMITONE:
                shift = target_idx - NOTE_TO_SEMITONE[note]
                if abs(shift) < abs(best_shift):
                    best_shift = shift
                    best_path = path
        return best_path, best_shift if best_path else 0

    # Detect directory structure
    subdirs = [p for p in audio_dir.iterdir() if p.is_dir()]
    audio_files = [p for p in audio_dir.iterdir() if p.is_file() and p.suffix in ['.wav', '.mp3', '.ogg']]

    # Collect available samples per instrument
    instrument_samples = {}  # instrument -> {note: path}

    if subdirs:
        # Hierarchical: audio_dir/instrument/note.ext
        for instrument_dir in subdirs:
            instrument = instrument_dir.name
            instrument_samples[instrument] = {}
            for f in instrument_dir.iterdir():
                if f.suffix in ['.wav', '.mp3', '.ogg']:
                    instrument_samples[instrument][f.stem] = f
    elif audio_files:
        # Flattened: audio_dir/instrument_note.ext
        for f in audio_files:
            if '_' not in f.stem:
                continue
            parts = f.stem.rsplit('_', 1)
            if len(parts) != 2:
                continue
            instrument, note = parts
            if instrument not in instrument_samples:
                instrument_samples[instrument] = {}
            instrument_samples[instrument][note] = f

    if not instrument_samples:
        print("No audio samples found!")
        return

    # Generate spectrograms
    spec_dir = output_dir / "spectrograms"
    total_generated = 0

    for instrument in sorted(instrument_samples.keys()):
        available = instrument_samples[instrument]
        if not available:
            continue

        inst_dir = spec_dir / instrument
        inst_dir.mkdir(parents=True, exist_ok=True)

        notes_generated = 0
        for note in NOTE_ORDER:
            sample_path, pitch_shift = find_nearest_sample(available, note)
            if sample_path is None:
                continue

            try:
                spec_img = audio_to_spectrogram(sample_path, pitch_shift)
                out_path = inst_dir / f"{note}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(spec_img, cv2.COLOR_RGB2BGR))
                notes_generated += 1
            except Exception as e:
                warnings.warn(f"Failed to process {sample_path} for {note}: {e}")

        if notes_generated > 0:
            print(f"  {instrument}: {notes_generated} spectrograms")
            total_generated += notes_generated

    print(f"Generated {total_generated} spectrograms in {spec_dir}")


def build_complete_sample_set(audio_dir: Path):
    """
    Build complete sample set for all instruments and notes (F4-B5).

    Returns dict: {instrument: {note: (audio_path, pitch_shift_semitones)}}
    Missing notes are filled by finding nearest available note with pitch shift.
    """
    NOTE_ORDER = ['F4', 'Fs4', 'G4', 'Gs4', 'A4', 'As4', 'B4',
                  'C5', 'Cs5', 'D5', 'Ds5', 'E5', 'F5', 'Fs5',
                  'G5', 'Gs5', 'A5', 'As5', 'B5']
    NOTE_TO_SEMITONE = {note: i for i, note in enumerate(NOTE_ORDER)}

    def find_nearest_sample(available: dict, target_note: str):
        """Find nearest available sample and return (path, pitch_shift)."""
        if target_note in available:
            return available[target_note], 0

        target_idx = NOTE_TO_SEMITONE.get(target_note)
        if target_idx is None:
            return None, 0

        best_path, best_shift = None, float('inf')
        for note, path in available.items():
            if note in NOTE_TO_SEMITONE:
                shift = target_idx - NOTE_TO_SEMITONE[note]
                if abs(shift) < abs(best_shift):
                    best_shift = shift
                    best_path = path
        return best_path, best_shift if best_path else 0

    # Detect directory structure and collect available samples
    subdirs = [p for p in audio_dir.iterdir() if p.is_dir()]
    audio_files = [p for p in audio_dir.iterdir() if p.is_file() and p.suffix in ['.wav', '.mp3', '.ogg']]

    instrument_samples = {}  # instrument -> {note: path}

    if subdirs:
        # Hierarchical: audio_dir/instrument/note.ext
        for instrument_dir in subdirs:
            instrument = instrument_dir.name
            instrument_samples[instrument] = {}
            for f in instrument_dir.iterdir():
                if f.suffix in ['.wav', '.mp3', '.ogg']:
                    instrument_samples[instrument][f.stem] = f
    elif audio_files:
        # Flattened: audio_dir/instrument_note.ext
        for f in audio_files:
            if '_' not in f.stem:
                continue
            parts = f.stem.rsplit('_', 1)
            if len(parts) != 2:
                continue
            instrument, note = parts
            if instrument not in instrument_samples:
                instrument_samples[instrument] = {}
            instrument_samples[instrument][note] = f

    # Build complete set with pitch shifting for missing notes
    complete_set = {}
    for instrument, available in instrument_samples.items():
        if not available:
            continue
        complete_set[instrument] = {}
        for note in NOTE_ORDER:
            sample_path, pitch_shift = find_nearest_sample(available, note)
            if sample_path is not None:
                complete_set[instrument][note] = (sample_path, pitch_shift)

    return complete_set


def generate_sequence_with_pitch_shift(
    audio_path: Path,
    pitch_shift: float = 0,
    size: int = 32,
    seq_length: int = 16,
    sr: int = 22050,
    hop_length: int = 512,
    visual_mode: str = "oscilloscope",
    color: bool = False,
) -> tuple:
    """
    Generate a single audio-visual sequence with optional pitch shifting.

    Returns:
        spectrogram_frames: [T, H, W, C] - encoder input (C=1 or 3)
        visual_frames: [T, H, W, C] - decoder target (C=1 or 3)
    """
    import librosa

    # Load audio
    y, sr = load_audio(audio_path, sr=sr)

    # Apply pitch shift if needed
    if pitch_shift != 0:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

    # Calculate frame parameters
    frame_samples = hop_length * 4  # Samples per visual frame
    total_samples = seq_length * hop_length

    if len(y) < total_samples + frame_samples:
        # Pad if too short
        y = np.pad(y, (0, total_samples + frame_samples - len(y)))

    # Compute full spectrogram
    spec = compute_spectrogram(y, sr, hop_length=hop_length, n_mels=size)

    # Generate frames
    spec_frames = []
    visual_frames = []

    visual_funcs = {
        "oscilloscope": audio_to_oscilloscope,
        "radial": audio_to_radial_spectrum,
        "waveform": audio_to_waveform_grid,
        "cymatics": audio_to_cymatics,
    }
    visual_func = visual_funcs.get(visual_mode, audio_to_oscilloscope)

    # Frame 0: Full 2D spectrogram (time x frequency) as encoder input
    spec_frames_needed = min(len(spec), seq_length * 4)
    spec_2d = spec[:spec_frames_needed, :]
    spec_2d = np.flipud(spec_2d.T)  # [n_mels, T_spec] with low freq at bottom
    spec_img = cv2.resize(spec_2d, (size, size), interpolation=cv2.INTER_LINEAR)
    spec_frames.append(spec_img)

    # For circles mode, extract features once for the whole clip
    circle_features = None
    base_direction = 0.0
    if visual_mode == "circles":
        circle_features = extract_audio_features_for_circles(y, sr, hop_length)
        avg_centroid = circle_features['centroid'].mean()
        base_direction = avg_centroid * 2 * np.pi

    # Frames 1+: Visual representations
    for t in range(1, seq_length):
        audio_start = t * hop_length
        audio_chunk = y[audio_start : audio_start + frame_samples]
        if len(audio_chunk) < frame_samples:
            audio_chunk = np.pad(audio_chunk, (0, frame_samples - len(audio_chunk)))

        if visual_mode == "circles":
            visual_img = audio_to_circles(
                circle_features, t, size,
                base_direction=base_direction,
            )
        elif color and visual_mode in ("radial", "oscilloscope"):
            if visual_mode == "radial":
                visual_img = audio_to_radial_spectrum(audio_chunk, sr, frame_samples, size, color=True)
            else:
                visual_img = audio_to_oscilloscope(audio_chunk, sr, frame_samples, size, color=True)
        else:
            visual_img = visual_func(audio_chunk, sr, frame_samples, size)
        visual_frames.append(visual_img)

    # Add placeholder for frame 0 visual
    if visual_mode == "circles":
        visual_frames.insert(0, visual_frames[0] if visual_frames else np.zeros((size, size, 3)))
    else:
        visual_frames.insert(0, visual_frames[0] if visual_frames else np.zeros((size, size)))

    # Apply temporal smoothing
    smoothing = 0.7
    smoothed_frames = [visual_frames[0]]
    for t in range(1, len(visual_frames)):
        smoothed = smoothing * smoothed_frames[-1] + (1 - smoothing) * visual_frames[t]
        smoothed_frames.append(smoothed)
    visual_frames = smoothed_frames

    visual_frames = np.array(visual_frames)

    # Handle channel dimension
    if visual_mode == "circles" or (color and visual_mode in ("radial", "oscilloscope")):
        spec_frames = np.array(spec_frames)[:, :, :, np.newaxis]
        spec_frames = np.repeat(spec_frames, 3, axis=-1)
    else:
        spec_frames = np.array(spec_frames)[:, :, :, np.newaxis]
        visual_frames = visual_frames[:, :, :, np.newaxis]

    # Normalize to uint8
    spec_frames = (spec_frames * 255).astype(np.uint8)
    visual_frames = (visual_frames * 255).astype(np.uint8)

    return spec_frames, visual_frames


def generate_dataset(
    audio_dir: Path,
    output_path: Path,
    size: int = 32,
    seq_length: int = 16,
    visual_mode: str = "oscilloscope",
    samples_per_file: int = 4,
    color: bool = False,
):
    """
    Generate dataset from directory of audio files.

    Each audio file produces multiple sequences (different starting points).
    Output format: [N, T, H, W, C] where frame 0 is spectrogram (encoder input)
    and frames 1+ are visual targets.

    Includes pitch-shifted notes to fill in missing notes (F4-B5 range).
    Also generates spectrograms for all instruments/notes in output_dir/spectrograms/.
    """
    # Build complete sample set including pitch-shifted notes
    complete_samples = build_complete_sample_set(audio_dir)

    if not complete_samples:
        print("No audio files found!")
        return

    total_notes = sum(len(notes) for notes in complete_samples.values())
    print(f"Found {len(complete_samples)} instruments, {total_notes} notes (including pitch-shifted)")

    all_sequences = []

    for instrument in tqdm(sorted(complete_samples.keys()), desc="Processing instruments"):
        notes = complete_samples[instrument]
        for note, (audio_path, pitch_shift) in notes.items():
            try:
                for _ in range(samples_per_file):
                    spec_frames, visual_frames = generate_sequence_with_pitch_shift(
                        audio_path,
                        pitch_shift=pitch_shift,
                        size=size,
                        seq_length=seq_length,
                        visual_mode=visual_mode,
                        color=color,
                    )

                    # Combine: frame 0 = spectrogram, frames 1+ = visuals
                    sequence = np.concatenate(
                        [spec_frames[:1], visual_frames[1:]], axis=0
                    )
                    all_sequences.append(sequence)

            except Exception as e:
                warnings.warn(f"Error processing {audio_path} for {note}: {e}")
                continue

    if len(all_sequences) == 0:
        print("No sequences generated!")
        return

    sequences = np.array(all_sequences)
    print(f"Output shape: {sequences.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sequences)
    print(f"Saved to {output_path}")

    # Also generate spectrograms for all instruments/notes (for build_static.py)
    print("\nGenerating instrument spectrograms...")
    generate_instrument_spectrograms(audio_dir, output_path.parent, size=size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate audio-visual sequences")
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing audio files (.wav, .mp3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/audio_visual/audio_visual_32x32.npy",
        help="Output .npy file",
    )
    parser.add_argument("--size", type=int, default=32, help="Image size")
    parser.add_argument("--seq-length", type=int, default=128, help="Frames per sequence")
    parser.add_argument(
        "--visual-mode",
        type=str,
        default="cymatics",
        choices=["oscilloscope", "radial", "waveform", "cymatics", "circles"],
        help="Visual generation mode",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=1,
        help="Sequences to generate per audio file",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Use color output (RGB) for radial mode - maps frequency to hue",
    )

    args = parser.parse_args()

    generate_dataset(
        audio_dir=Path(args.audio_dir),
        output_path=Path(args.output),
        size=args.size,
        seq_length=args.seq_length,
        visual_mode=args.visual_mode,
        samples_per_file=args.samples_per_file,
        color=args.color,
    )
