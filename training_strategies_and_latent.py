import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ── Data ──────────────────────────────────────────
t = np.linspace(0, 100, 1000)
data = (np.sin(t) + 0.5 * np.sin(3*t) + 0.3 * np.sin(7*t) + 0.1 * np.random.randn(len(t))).astype(np.float32)

def make_sequences(data, seq_len=20):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return torch.tensor(X).unsqueeze(-1), torch.tensor(Y).unsqueeze(-1)

X, Y = make_sequences(data)

# ── Models ────────────────────────────────────────
class SeqPredictor(nn.Module):
    """Vanilla LSTM predicting directly in raw input space."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LatentPredictor(nn.Module):
    """
    Encoder -> latent z -> LSTM -> Decoder.
    Predicts in a compressed latent space rather than raw values.
    Inspired by the deterministic path of RSSM (Dreamer).
    """
    def __init__(self, latent_dim=16, hidden_size=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, latent_dim), nn.Tanh())
        self.rnn     = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(self.encoder(x))
        return self.decoder(out[:, -1, :])

class StochasticLatentPredictor(nn.Module):
    """
    Encoder -> (mu, log_var) -> sample z -> LSTM -> Decoder.
    Adds stochasticity: predictions are sampled from a distribution
    rather than being a fixed point. This is the core of RSSM (Dreamer).
    """
    def __init__(self, latent_dim=16, hidden_size=64):
        super().__init__()
        self.encoder_mu     = nn.Linear(1, latent_dim)
        self.encoder_logvar = nn.Linear(1, latent_dim)
        self.rnn            = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size, batch_first=True)
        self.decoder        = nn.Linear(hidden_size, 1)

    def reparameterize(self, mu, log_var):
        """Sample from N(mu, sigma) while keeping gradients flowing."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, deterministic=False):
        mu      = self.encoder_mu(x)
        log_var = self.encoder_logvar(x)
        z       = mu if deterministic else self.reparameterize(mu, log_var)
        out, _  = self.rnn(z)
        return self.decoder(out[:, -1, :]).squeeze(-1), mu, log_var

# ── Training functions ────────────────────────────
def train_free(model, epochs=300):
    """Standard training — model never sees its own predictions during training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn   = nn.MSELoss()
    for _ in range(epochs):
        loss = loss_fn(model(X), Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

def train_teacher(model, epochs=300):
    """Teacher forcing — ground truth fed as input at every step."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn   = nn.MSELoss()
    for _ in range(epochs):
        out, _ = model.rnn(X)
        loss   = loss_fn(model.fc(out[:, -1, :]), Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

def train_scheduled(model, epochs=300):
    """Scheduled sampling — gradually shifts from teacher forcing to free rollout."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn   = nn.MSELoss()
    for epoch in range(epochs):
        teacher_ratio = 1.0 - (epoch / epochs)
        current = X.clone()
        for step in range(X.shape[1] - 1):
            out, _ = model.rnn(current)
            pred   = model.fc(out[:, -1:, :])
            use_teacher = torch.rand(1).item() < teacher_ratio
            next_input  = X[:, step+1:step+2, :] if use_teacher else pred.detach()
            current = torch.cat([current[:, 1:, :], next_input], dim=1)
        loss = loss_fn(model(X), Y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

def train_latent(model, epochs=1000):
    """
    Latent space training with gradient clipping.
    clip_grad_norm_ is essential — without it the model collapses to predicting the mean.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn   = nn.MSELoss()
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 200 == 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.6f}")

def train_stochastic(model, epochs=1000):
    """
    VAE-style training: reconstruction loss + KL divergence.
    KL weight 0.01 prevents the latent space from collapsing.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mse_loss  = nn.MSELoss()
    for epoch in range(epochs):
        pred, mu, log_var = model(X)
        recon = mse_loss(pred, Y.squeeze(-1))
        kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss  = recon + 0.01 * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 200 == 0:
            print(f"  Epoch {epoch} | Recon: {recon.item():.4f} | KL: {kl.item():.4f}")

# ── Rollout functions ─────────────────────────────
def rollout(model, steps=200):
    """Autoregressive rollout — feed predictions back as inputs."""
    model.eval()
    with torch.no_grad():
        current = X[0:1].clone()
        preds   = []
        for _ in range(steps):
            val = model(current)
            preds.append(val.item())
            current = torch.cat([current[:, 1:, :], val.unsqueeze(1)], dim=1)
    return preds

def rollout_latent(model, steps=200):
    """Rollout for LatentPredictor."""
    model.eval()
    with torch.no_grad():
        current = X[0:1].clone()
        preds   = []
        for _ in range(steps):
            val = model(current)
            preds.append(val.item())
            current = torch.cat([current[:, 1:, :], torch.tensor([[[val.item()]]])], dim=1)
    return preds

def rollout_stochastic(model, steps=200, deterministic=True):
    """Rollout for StochasticLatentPredictor."""
    model.eval()
    with torch.no_grad():
        current = X[0:1].clone()
        preds   = []
        for _ in range(steps):
            val, _, _ = model(current, deterministic=deterministic)
            preds.append(val.item())
            current = torch.cat([current[:, 1:, :], torch.tensor([[[val.item()]]])], dim=1)
    return preds

def compute_horizon_error(preds, ground_truth):
    preds = np.array(preds)
    truth = np.array(ground_truth[:len(preds)])
    return [(preds[i] - truth[i])**2 for i in range(len(preds))]

def smooth(errors, window=10):
    return np.convolve(errors, np.ones(window)/window, mode='valid')

# ── Train ─────────────────────────────────────────
print("Training free rollout...")
model_free = SeqPredictor(); train_free(model_free)

print("Training teacher forcing...")
model_teacher = SeqPredictor(); train_teacher(model_teacher)

print("Training scheduled sampling...")
model_scheduled = SeqPredictor(); train_scheduled(model_scheduled)

print("Training latent predictor...")
model_latent = LatentPredictor(); train_latent(model_latent)

print("Training stochastic latent predictor...")
torch.manual_seed(42)
model_stochastic = StochasticLatentPredictor(); train_stochastic(model_stochastic)

# ── Rollout ───────────────────────────────────────
ground_truth     = data[20:220]
preds_free       = rollout(model_free)
preds_teacher    = rollout(model_teacher)
preds_scheduled  = rollout(model_scheduled)
preds_latent     = rollout_latent(model_latent)
preds_stochastic = rollout_stochastic(model_stochastic)

errors_free       = compute_horizon_error(preds_free, ground_truth)
errors_teacher    = compute_horizon_error(preds_teacher, ground_truth)
errors_scheduled  = compute_horizon_error(preds_scheduled, ground_truth)
errors_latent     = compute_horizon_error(preds_latent, ground_truth)
errors_stochastic = compute_horizon_error(preds_stochastic, ground_truth)

# ── Plot ──────────────────────────────────────────
fig, axes = plt.subplots(6, 1, figsize=(12, 22))

axes[0].plot(ground_truth, color='steelblue')
axes[0].set_title('Ground Truth (noisy: sin + harmonics + noise)')

axes[1].plot(ground_truth, color='steelblue', alpha=0.4)
axes[1].plot(preds_free, color='coral', linestyle='--', label='Free rollout')
axes[1].set_title('Method 1a: Free Rollout Training')
axes[1].legend()

axes[2].plot(ground_truth, color='steelblue', alpha=0.4)
axes[2].plot(preds_teacher, color='green', linestyle='--', label='Teacher forcing')
axes[2].set_title('Method 1b: Teacher Forcing')
axes[2].legend()

axes[3].plot(ground_truth, color='steelblue', alpha=0.4)
axes[3].plot(preds_scheduled, color='purple', linestyle='--', label='Scheduled sampling')
axes[3].set_title('Method 1c: Scheduled Sampling')
axes[3].legend()

axes[4].plot(ground_truth, color='steelblue', alpha=0.4)
axes[4].plot(preds_latent, color='red', linestyle='--', label='Latent space (RSSM-inspired)')
axes[4].set_title('Method 2: Latent Space Predictor')
axes[4].legend()

axes[5].plot(ground_truth, color='steelblue', alpha=0.4)
axes[5].plot(preds_stochastic, color='orange', linestyle='--', label='Stochastic latent (RSSM)')
axes[5].set_title('Method 3: Stochastic Latent Space (VAE-style)')
axes[5].legend()

plt.tight_layout()
plt.savefig('full_comparison.png', dpi=150)
plt.show()

fig2, ax = plt.subplots(figsize=(12, 4))
ax.plot(smooth(errors_free),       color='coral',  label='Free rollout')
ax.plot(smooth(errors_teacher),    color='green',  label='Teacher forcing')
ax.plot(smooth(errors_scheduled),  color='purple', label='Scheduled sampling')
ax.plot(smooth(errors_latent),     color='red',    label='Latent space')
ax.plot(smooth(errors_stochastic), color='orange', label='Stochastic latent')
ax.set_xlabel('Prediction horizon (steps)')
ax.set_ylabel('MSE (smoothed)')
ax.set_title('Prediction Error vs Horizon (10-step moving average)')
ax.legend()
plt.tight_layout()
plt.savefig('error_vs_horizon.png', dpi=150)
plt.show()
