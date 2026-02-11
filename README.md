# Neural Cellular Automata

The demo above *generates* a unique Neural Cellular Automaton for each sound you play. The NCA runs in real-time, creating evolving patterns based on the audio's spectral fingerprint, and facilitating the exploratin of an embeddings space of NCAs through music.


## How to Play

- **Piano keys**: Click or use keyboard (A-L for white keys, W/E/R/Y/U/O/P for black keys). Each note loads a different latent vector that generates a unique NCA.
- **Chords**: Hold multiple keys to blend their latents together, creating hybrid dynamics.
- **Instruments**: Use the dropdown to switch between piano, violin, guitar, and more. Each instrument produces distinct visual behaviors.
- **Perturbation**: Defaults to max. Press number keys 1-9 to reduce, or 0 for max. Higher values create more chaotic, unpredictable patterns. Press the same key twice to toggle off.
- **Random Latent**: Click "Random Latent" to explore completely random points in the learned space.

## Gallery

Here are some examples of intersting stable patters you might find the in the space of Neuromusical Cellular Automata. Set the **Perturbation** level high or use to the **Random Latent** button to discover more. You'll see interesting smokey fluid dynamics, shifting colors, and even shapes that appear to be swirling in 3D!

| | | |
|:---:|:---:|:---:|
| ![Dancing Vines\|256x256](assets/gallery/dancingvines.gif) | ![Eel Fire\|256x256](assets/gallery/eelfire.gif) | ![Flower Dolphins\|256x256](assets/gallery/flowerdolphins.gif) |
| ![Lilly Pads\|256x256](assets/gallery/lillypads.gif) | ![Organic Swirl 3D\|256x256](assets/gallery/organicswirl3D.gif) | ![Paraglider\|256x256](assets/gallery/paraglider.gif) |
| ![Rebirth\|256x256](assets/gallery/rebirth.gif) | ![Red Grid\|256x256](assets/gallery/redgrid.gif) | ![Water Channels\|256x256](assets/gallery/water_channels.gif) |


### Random NCAs

In contrast to the above NCA's sampled from a learned latent space (so that we are navigating a particularly interesting and beautiful region of NCA parameter space), most of NCA space is more boring and tends to collapse to blank or static patterns:

| | | |
|:---:|:---:|:---:|
| ![rand1\|128x128](assets/random_nca1.gif) | ![rand2\|128x128](assets/random_nca2.gif) | ![rand3\|128x128](assets/random_nca3.gif) |

---

## Why This Project

I have always been fascinated by neural cellular automata (and CAs in general) as they are like little physics engines for tiny virtual universes. So when I started this project I spun up a little NCA editor where I could adjust the architecture and parameter weights of NCAs to explore the space. I found some pretty neat settings, but predicatbly most setups were duds.

This got me thinking about the embeddings space of NCA parameters, and how we could explore it. To get a smoother and hopefully more interesting embedding space, I decided to train a neural network to *generate* NCA parameters from a conditioned latent space.

In this way, I've ended up with a neural network that generates neural networks from a latent space, how cool! Being a musician, I eventually came to the idea of constraining the encoder to the space of musical notes from various instruments, which allows us to use music to explore the embedding space, which is very fun.

## Cellular Automata

Cellular automata are systems of simple cells on a grid, each updating its state based on neighboring values. Classic examples like Conway's Game of Life show how complex global behavior can emerge from purely local rules.

Whereas Conway's Game of Life is the application of one possible update rule, we can think of the space of all possible rules as the set of all state transition functions for a given neighborhood, which could be listed in a (beyond astronomically massive) lookup table. For example a binary CA where the update rule depends on the Moore neighborhood has 2^1024 possible rules. There are 9 cells in a neighborhood and the 1 center cell's new state ,which gives us 2^10 (1024) entries per rule. Each entry maps the 9-cell states at time *t* to the center cell's state at time *t+1*. The rule is then convolved over the entire grid to get the new state at each time step.

![Rainbow Gliders|512x512](assets/rainbow_gliders.gif)

**Neural Cellular Automata (NCA)** replace the discrete update rules of CAs with a small neural network, allowing the system to learn its own dynamics from data. A neural network reads local values and outputs updated values for the next time step, which is fed back into the network at time *t+1*. This makes an NCA effectively a recurrent CNN! NCAs can learn to grow, regenerate, and sustain surprisingly complex dynamical patterns.

The space of all possible NCAs is infinite, even for a constrained neighborhood, because there are an infinite number of neural network architectures we could apply. For a fixed architecture, we can think of the "embedding space" of all possible parameter values. Typically, this space is pretty sparse and boring, producing mostly noise or blank outputs (although I did randomly stumble upon the "rainbow gliders" shown above - stable little colorful blobs that move!).


---

## Hardware Constraints

I landed on the following architecture, which is extremely tiny because I want this to run in realtime in the browser on a CPU, and I needed a fast turn-around time for experiments.

![Architecture](assets/architecture.png)

### Architecture Details

**VAE Encoder** (Context Frames → Latent)
- Input: context frames, 32×32
- Conv layers: 12→32→64→128, each 3×3 kernel, stride 2, BatchNorm, LeakyReLU(0.2)
- Flatten: 128 × 4 × 4 = 2,048 features
- Two linear heads (μ and σ): 2,048 → 64 each
- Output: 64-dimensional latent vector z

**HyperNetwork** (Latent → NCA Weights)
- Input: 64-dim latent
- MLP: 64 → 256 → 256 → 4,640, with LayerNorm + ReLU
- Output: weights for 2-layer NCA (2×[16×16×3×3] + 2×[16] = 4,640 params)

**NCA** (Grid Evolution)
- Grid: 16 channels (3 RGB visible + 13 hidden state)
- Layer 1: 16→16 channels, 3×3 conv, circular padding, ReLU
- Layer 2: 16→16 channels, 3×3 conv, circular padding, residual add
- Parameters generated per-sample by HyperNetwork


---

## Realtime Exploration

This demo combines two separately trained models:

- **Encoder:** From the spectrogram-to-NCA model used in the piano demo above
- **Hypernetwork:** From a new model trained to generate emoji (or rather to *generate NCAs* which generate emoji

The idea is to explore whether we can use music and sound to navigate the space of possible NCAs, even when the hypernetwork wasn't jointly trained with spectrograms. Your microphone audio gets converted to a spectrogram, encoded to a latent vector, and that latent drives the emoji NCA's dynamics in real-time.

<!-- REALTIME_DEMO -->

---


## Learned Latent Space

Instead of exploring the embedding space of random parameter values, we can learn a latent space and use that to *generate* NCAs. And that's what the Neuromusical Cellular Automata does:

This architecture uses a **variational autoencoder (VAE)** to map context frames into a low-dimensional latent space. A **hypernetwork** then transforms each latent vector into a unique set of NCA weights. This means every point in the latent space corresponds to a different cellular automaton with its own dynamics.

Because this latent space is learned it can be much smoother and more semantically meaningful than exploring the raw space of NCA parameters. Of course, what the latent space actually learns is entirely dependent on the data used to train the model.

---

## Dynamics

Because NCAs are recurrent, they can learn dynamics and movement. Here is an example of a model trained on sequences of frames from dynamic simulations. Given a few context frames as input, the model learns to predict how the system evolves over time. Each training sequence captures a different behavior, forcing the model to internalize the underlying rules of motion rather than memorizing individual frames.

The ground truth is shown on the right with the model predictions on the left:

![Ground Truth Example 1|512x256](assets/groundtruth-1.gif)
![Ground Truth Example 2|512x256](assets/groundtruth-2.gif)
![Ground Truth Example 3|512x256](assets/groundtruth-3.gif)

As you can see, the NCA does capture the general size, color, and motion of the objects, although it tends to blur over time as it struggles to guess the exact next frame. This is a classic problem for sequence prediction in continuous space, and could probably be rectictfied with diffusion or other techniques. But remember this is a *tiny* little network because it needs to run in realtime on a CPU and train in reasonable time. And anyway, I kind of like how the objects smear out over time and generate interesting patterns - it makes the latent space more surprising.

---

## Music

The training data for the Neuromusical Cellular Automata and the demo at the top of this page is generated by pairing audio with deterministic visualizations.

![Ground Truth Example 1|512x256](assets/spectrogram_piano.gif)

For each instrument sample (piano, violin, etc.), we extract audio features: detected harmonics, loudness (RMS), and spectral brightness. These features drive the "circles" visualization where each harmonic becomes a colored circle - low frequencies appear warm (red/orange) near the center, high frequencies appear cool (blue) toward the edges. Circle size pulses with loudness, and positions orbit based on harmonic relationships.

![Ground Truth Example 2|512x256](assets/spectrogram_xylophone.gif)

The first frame of each training sequence is a **mel spectrogram** (a 2D image of frequency vs. time), which the encoder compresses into a 64-dimensional latent vector. The decoder, a hypernetwork, transforms this latent into NCA weights that generate the subsequent circle animation frames.

![Ground Truth Example 3|512x256](assets/spectrogram_clarinet.gif)

For the piano interface, we pre-generate spectrograms for all 19 notes (F4 through B5) across every instrument. Each spectrogram is encoded into its latent vector and stored in a manifest. When you press a piano key, the corresponding latent is loaded; pressing multiple keys (a chord) averages their latents together. This blended latent then generates the NCA in real-time, creating visualizations that interpolate between the learned behaviors of each note.

---

## Future Potential

This approach opens several interesting directions:

- The latent space could be conditioned on higher-level descriptions, allowing **natural language control** over the generated dynamics
- Larger grids and deeper NCA architectures could capture more complex phenomena
- The ability for NCAs to track/produce **agentic behavior** on the grid is particularly fascinating

### Agentic behavior
I did some early experiments with boids exhibiting various behaviors such as flocking or predator-prey dynamics, but this proved too challenging for my tiny 2-layer conv nets. It would be fascinating to see if an NCA could learn theory of mind to generate the actions of human players in Atari games, or even more complex environments and life-like behaviors.

### Sound as Navigation
Broadly, music offers a compelling interface for exploring high-dimensional latent spaces. The simple circle visualizations here map audio features to color and motion, but richer correspondences are possible. We could learn a shared embedding where the space of complex music aligns with the space of natural images or video, not through superficial features, but through deeper semantic structure: tension and resolution in a symphony mapping to dramatic arcs in film, the texture of a jazz improvisation corresponding to the organic chaos of a forest canopy, or the emotional trajectory of a song finding its visual analogue in shifting landscapes.

Contrastive learning on large audio-visual datasets could discover these cross-modal correspondences, letting music become a navigation tool for exploring generative models of images, video, or even text, playing a melody to traverse a space of scenes that feel emotionally consonant with the sound.

### Endless Possibility
The combination of neural cellular automata with learned latent spaces suggests a fascinating paradigm: compact, local update rules that are themselves generated by a learned model, producing an open-ended family of emergent systems from a single trained network.

---

## Bonus: Game of Life

As an experiment, I trained the same NCA architecture on Conway's Game of Life—a cellular automaton that is famously Turing complete. The NCA successfully learned to emulate the Game of Life rules, correctly simulating gliders, oscillators, and chaotic patterns.

![GoL Example 2|512x256](assets/gol2.gif)
![GoL Example 3|512x256](assets/gol3.gif)

*Left: NCA prediction. Right: Ground truth.*

This result demonstrates that our tiny 2-layer NCA architecture has sufficient expressive power to encode the complete Game of Life transition function. Since the Game of Life is Turing complete, this means our NCA setup is capable of universal computation, and it's *learnable*, neat!
