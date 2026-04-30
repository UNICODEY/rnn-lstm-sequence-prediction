# Sequence Prediction: RNN vs LSTM

Exploring how recurrent neural networks learn to predict time series,
and how architecture choice affects long-horizon rollout stability.

## What this project does

Trains three models to predict a composite sine wave (`sin(t) + 0.5·sin(3t)`)
and evaluates how well each can **autoregressively roll out 100 steps** —
feeding its own predictions back as input. This is the same core idea behind
world models like Dreamer, just with a much simpler signal.

## Models compared

| Model    | Hidden size | Params |
|----------|-------------|--------|
| RNN-32   | 32          | ~1K    |
| RNN-128  | 128         | ~17K   |
| LSTM-32  | 32          | ~4K    |

## Key finding

More parameters (RNN-128) did not help — the larger model was harder to
train and showed worse rollout stability. LSTM's gating mechanism didn't
consistently outperform vanilla RNN on this task either. The main takeaway:
**architecture choice matters less than training stability on simple signals;
compounding error in autoregressive rollout is the real challenge.**

This connects directly to why world models (RSSM, Dreamer) invest heavily
in structured latent spaces rather than simply scaling up hidden size.

## How to run

Open `sequence_prediction.ipynb` in [Google Colab](https://colab.research.google.com)
and run all cells. No local installation needed.

## Results

![comparison](comparison.png)

## Background

Built as a first step toward understanding world models and recurrent
state-space models (RSSM). Inspired by DreamerV3 (Hafner et al., 2023).
