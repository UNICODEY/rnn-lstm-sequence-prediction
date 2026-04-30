# Sequence Prediction: RNN vs LSTM

Exploring how recurrent neural networks learn to predict time series,
and how architecture choice affects long-horizon rollout stability.

## What this project does

Trains three models to predict a composite sine wave (`sin(t) + 0.5·sin(3t)`)
and evaluates how well each can **autoregressively roll out 100 steps**
feeding its own predictions back as input. This is the same core idea behind
world models like Dreamer, just with a much simpler signal.

## Models compared

| Model    | Hidden size | Params |
|----------|-------------|--------|
| RNN-32   | 32          | ~1K    |
| RNN-128  | 128         | ~17K   |
| LSTM-32  | 32          | ~4K    |

## Key finding

More parameters (RNN-128) did not help. The larger model was harder to
train and showed worse rollout stability. LSTM's gating mechanism didn't
consistently outperform vanilla RNN on this task either. The main takeaway:
**architecture choice matters less than training stability on simple signals;
compounding error in autoregressive rollout is the real challenge.**

This connects directly to why world models (RSSM, Dreamer) invest heavily
in structured latent spaces rather than simply scaling up hidden size.

## How to run

Open `architecture_comparison.py` in PyCharm or any Python environment.
Requires: `torch`, `numpy`, `matplotlib`.

## Results

![comparison](comparison.png)

## Method 2: Training Strategies & Latent Space

Building on the compounding error finding, this project was extended to explore
two directions:

**Training strategies** compare free rollout, teacher forcing, and scheduled
sampling on a harder signal (`sin(t) + 0.5·sin(3t) + 0.3·sin(7t) + noise`).
Key finding: on a capable model (LSTM), all three strategies converge. Differences
emerge under noise and longer horizons.

![methods](three_methods_comparison.png)

**Latent space prediction (RSSM-inspired)** encodes each timestep into a 16-dim
latent vector, predicts in that compressed space, and reconstructs the output via
a decoder. Training required gradient clipping to stabilise. Without it, the model
collapsed to predicting the mean, a common failure mode in encoder-decoder
architectures.

Result: latent predictor tracks the noisy signal more accurately than raw-space
rollout, demonstrating the core advantage of structured latent representations
used in RSSM/Dreamer.

![latent](latent_vs_raw.png)

**Method 3 (coming soon):** stochastic latent space. The encoder will output
(mu, sigma) instead of a fixed vector, completing the VAE-style design central
to RSSM.

![full comparison](full_comparison.png)

## Background

Built as a first step toward understanding world models and recurrent
state-space models (RSSM). Inspired by DreamerV3 (Hafner et al., 2023).
