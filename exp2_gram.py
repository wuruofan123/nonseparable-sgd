"""
Experiment 2: Gram objective with sample-coupled gradients.
Reproduces Figures 2 and 3 in the paper.
"""

import copy
import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import sem
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Hyperparameters matching the paper, Section 5.2
# ------------------------------------------------------------------
device = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)

T_REF = 15000          # full-batch GD steps to find theta_ref
LR_REF = 3e-6          # learning rate for full-batch GD

N = 200                # number of samples
d = 8                  # row dimension of Y
INPUT_DIM = 32         # MLP input dimension
HIDDEN = 128           # MLP hidden dimension
BATCH_SIZE = 16        # mini-batch size
EPOCHS = 50000         # training epochs
ETA = 2e-6             # step size for SGD
RECORD_EVERY = 100     # recording interval
N_REPEATS = 5          # number of independent runs

PERTURB_SIGMA = 1e-3   # initialization perturbation scale

DATA_DIR = 'data'
FIG_DIR = 'fig'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class SmallMLP(nn.Module):
    """Single-hidden-layer MLP: input -> hidden -> out_dim."""
    def __init__(self, input_dim, hidden=128, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Data generation and objective helpers
# ------------------------------------------------------------------
def generate_data_and_S(N, input_dim, d, device):
    """Generate random inputs X and target Gram matrix S."""
    X = torch.randn(N, input_dim, device=device)
    k = min(d, 8)
    Ystar = torch.randn(N, k, device=device) @ torch.randn(k, d, device=device)
    S = (Ystar @ Ystar.t()) / float(d)
    return X, S


def compute_full_F_and_gradY(Y, S, d_factor=None):
    """Evaluate the Gram objective and its gradient w.r.t. Y."""
    if d_factor is None:
        d_factor = float(Y.shape[1])
    R = (Y @ Y.t()) / d_factor - S
    Fval = 0.5 * torch.sum(R * R)
    symR = R + R.t()
    gradY = (symR @ Y) / d_factor
    return Fval, gradY, R


def compute_Fstar_via_eig_numpy(S, d, tol=1e-12):
    """Compute F* via eigendecomposition of S."""
    S = np.asarray(S)
    S_sym = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S_sym)
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    vals[np.abs(vals) <= tol] = 0.0
    top_k = min(d, S.shape[0])
    vals_top = vals[:top_k].copy()
    vals_top = np.maximum(vals[:top_k], 0.0)
    U_d = vecs[:, :top_k]
    Lambda_d = np.diag(vals_top)
    Gd = U_d @ Lambda_d @ U_d.T
    Fstar = 0.5 * np.linalg.norm(Gd - S_sym, ord='fro')**2
    return float(Fstar)


# ------------------------------------------------------------------
# Reference initialization via full-batch GD
# ------------------------------------------------------------------
def find_theta_ref(seed=0):
    """Run full-batch GD to obtain a reference initialization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, S = generate_data_and_S(N, INPUT_DIM, d, device)
    model = SmallMLP(INPUT_DIM, HIDDEN, d).to(device)
    opt = optim.SGD(model.parameters(), lr=LR_REF)

    rec = {"iter": [], "F": []}
    for t in range(T_REF):
        Y = model(X)
        Fval, gradY, _ = compute_full_F_and_gradY(Y, S, float(d))
        opt.zero_grad()
        Y.backward(gradY)
        opt.step()
        if t % 50 == 0:
            with torch.no_grad():
                Fy, _, _ = compute_full_F_and_gradY(model(X).detach(), S, float(d))
            rec["iter"].append(t)
            rec["F"].append(Fy.item())

    F_ref = rec["F"][-1]
    theta_ref = copy.deepcopy(model.state_dict())
    return theta_ref, F_ref, X, S


def perturb_state_dict(state_dict, sigma):
    """Add Gaussian perturbation to a state dict."""
    new_sd = {}
    for k, v in state_dict.items():
        noise = torch.randn_like(v) * (sigma * (v.std().item() if v.numel() > 1 else 1.0))
        new_sd[k] = (v + noise).clone()
    return new_sd


# ------------------------------------------------------------------
# Single training run comparing three gradient estimators
# ------------------------------------------------------------------
def run_single(init_state_dict, X, S, seed=0):
    """Train with naive, H, and G estimators from the same init."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_naive = SmallMLP(INPUT_DIM, HIDDEN, d).to(device)
    model_H = SmallMLP(INPUT_DIM, HIDDEN, d).to(device)
    model_G = SmallMLP(INPUT_DIM, HIDDEN, d).to(device)

    model_naive.load_state_dict(init_state_dict)
    model_H.load_state_dict(init_state_dict)
    model_G.load_state_dict(init_state_dict)

    opt_naive = optim.SGD(model_naive.parameters(), lr=14 * ETA)
    opt_H = optim.SGD(model_H.parameters(), lr=ETA)
    opt_G = optim.SGD(model_G.parameters(), lr=ETA)

    with torch.no_grad():
        Y_init = model_H(X)
        Y_cache = Y_init.detach().clone()
        _, _, R_init = compute_full_F_and_gradY(Y_cache, S, d_factor=float(d))
        Gram_old = (Y_cache @ Y_cache.t()) / float(d)
        V_cache = Gram_old - S

    records = {
        "iter": [], "F_naive": [], "F_H": [], "F_G": [],
        "grad_norm_naive": [], "grad_norm_H": [], "grad_norm_G": []
    }

    step = 0

    def get_batches(N, batch_size):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            yield perm[i:i + batch_size]

    for ep in range(EPOCHS):
        for batch_idx in get_batches(N, BATCH_SIZE):
            batch_idx = np.array(batch_idx, dtype=np.int64)
            b = len(batch_idx)

            # Naive estimator
            Xb = X[batch_idx]
            Yb_naive = model_naive(Xb)
            with torch.no_grad():
                R_bb = (Yb_naive @ Yb_naive.t()) / float(d) - S[batch_idx][:, batch_idx]
            symR_bb = R_bb + R_bb.t()
            dL_dYb_naive = (symR_bb @ Yb_naive) / float(d)
            opt_naive.zero_grad()
            Yb_naive.backward(dL_dYb_naive)
            opt_naive.step()

            # Cached surrogate H
            Xb_H = X[batch_idx]
            Yb_H = model_H(Xb_H)
            with torch.no_grad():
                symV = V_cache + V_cache.t()
                dL_dYb_H = (symV[batch_idx] @ Y_cache) / float(d)
            opt_H.zero_grad()
            Yb_H.backward(dL_dYb_H)
            opt_H.step()

            with torch.no_grad():
                idx = torch.tensor(batch_idx, dtype=torch.long, device=Y_cache.device)
                old_rows = Y_cache[idx].clone()
                new_rows = Yb_H.detach().clone()
                delta = new_rows - old_rows
                row_update = (delta @ Y_cache.t()) / float(d)
                Gram_old[idx, :] = Gram_old[idx, :] + row_update
                Gram_old[:, idx] = Gram_old[idx, :].t()
                Y_cache[idx] = new_rows
                V_cache = Gram_old - S

            # Ideal per-batch G
            with torch.no_grad():
                Y_full_for_gradY = model_G(X)
                _, gradY_full, _ = compute_full_F_and_gradY(Y_full_for_gradY, S, d_factor=float(d))
                gradY_batch = gradY_full[batch_idx]
            Xb_G = X[batch_idx]
            Yb_G = model_G(Xb_G)
            opt_G.zero_grad()
            Yb_G.backward(gradY_batch.to(dtype=Yb_G.dtype, device=Yb_G.device))
            opt_G.step()

            # Record full metrics
            if step % RECORD_EVERY == 0:
                with torch.no_grad():
                    F_naive_val, gradY_naive, _ = compute_full_F_and_gradY(
                        model_naive(X).detach(), S, d_factor=float(d))
                    F_H_val, gradY_H, _ = compute_full_F_and_gradY(
                        model_H(X).detach(), S, d_factor=float(d))
                    F_G_val, gradY_G, _ = compute_full_F_and_gradY(
                        model_G(X).detach(), S, d_factor=float(d))
                    records["iter"].append(step)
                    records["F_naive"].append(F_naive_val.item())
                    records["F_H"].append(F_H_val.item())
                    records["F_G"].append(F_G_val.item())
                    records["grad_norm_naive"].append(float(torch.norm(gradY_naive).item()))
                    records["grad_norm_H"].append(float(torch.norm(gradY_H).item()))
                    records["grad_norm_G"].append(float(torch.norm(gradY_G).item()))
            step += 1

    return records


def run_repeats_and_summarize(theta_ref, X_ref, S_ref, sigma=1e-3, n_repeats=5):
    """Run multiple repeats and aggregate raw records."""
    all_records = {"F_naive": [], "F_H": [], "F_G": []}
    all_grad = {"F_naive": [], "F_H": [], "F_G": []}
    for r in range(n_repeats):
        sd_pert = perturb_state_dict(theta_ref, sigma)
        rec = run_single(sd_pert, X_ref, S_ref, seed=r)
        all_records["F_naive"].append(rec["F_naive"])
        all_records["F_H"].append(rec["F_H"])
        all_records["F_G"].append(rec["F_G"])
        all_grad["F_naive"].append(rec["grad_norm_naive"])
        all_grad["F_H"].append(rec["grad_norm_H"])
        all_grad["F_G"].append(rec["grad_norm_G"])
    minF = min(min(min(l) for l in all_records[m]) for m in all_records)
    return all_records, all_grad, minF


# ------------------------------------------------------------------
# Data persistence
# ------------------------------------------------------------------
def save_run_data(path_prefix, all_records, all_grads, meta=None):
    """Save records and metadata to disk."""
    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    save_dict = {}
    lengths = {}
    for method, runs in all_records.items():
        maxlen = max(len(r) for r in runs)
        arr = np.full((len(runs), maxlen), np.nan, dtype=float)
        for i, r in enumerate(runs):
            arr[i, :len(r)] = r
        save_dict[f"records_{method}"] = arr
        lengths[method] = [len(r) for r in runs]
    for method, runs in all_grads.items():
        maxlen = max(len(r) for r in runs)
        arr = np.full((len(runs), maxlen), np.nan, dtype=float)
        for i, r in enumerate(runs):
            arr[i, :len(r)] = r
        save_dict[f"grads_{method}"] = arr
        lengths[method] = [len(r) for r in runs]
    npz_path = path_prefix + ".npz"
    np.savez_compressed(npz_path, **save_dict)
    meta_out = meta.copy() if meta is not None else {}
    meta_out.setdefault("saved_at", time.asctime())
    meta_out["lengths_per_method"] = lengths
    meta_path = path_prefix + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Saved run data to {npz_path}")


def load_run_data(path_prefix):
    """Load records from disk."""
    npz_path = path_prefix + ".npz"
    meta_path = path_prefix + ".meta.json"
    npz = np.load(npz_path, allow_pickle=True)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    all_records = {}
    all_grads = {}
    for k in npz.files:
        if k.startswith("records_"):
            method = k[len("records_"):]
            arr = npz[k]
            lens = meta.get("lengths_per_method", {}).get(method, None)
            runs = []
            if lens is not None and len(lens) == arr.shape[0]:
                for i, L in enumerate(lens):
                    runs.append(arr[i, :L].tolist())
            else:
                for i in range(arr.shape[0]):
                    row = arr[i]
                    valid = ~np.isnan(row)
                    runs.append(row[valid].tolist())
            all_records[method] = runs
        elif k.startswith("grads_"):
            method = k[len("grads_"):]
            arr = npz[k]
            lens = meta.get("lengths_per_method", {}).get(method, None)
            runs = []
            if lens is not None and len(lens) == arr.shape[0]:
                for i, L in enumerate(lens):
                    runs.append(arr[i, :L].tolist())
            else:
                for i in range(arr.shape[0]):
                    row = arr[i]
                    valid = ~np.isnan(row)
                    runs.append(row[valid].tolist())
            all_grads[method] = runs
    return all_records, all_grads, meta


# ------------------------------------------------------------------
# Plotting: Figure 2 (objective gap + gradient norm)
# ------------------------------------------------------------------
def summarize_and_plot(all_records, all_grad, record_every, save_path=None):
    """Plot full objective gap and gradient norm for G/H/naive."""
    methods = ["F_G", "F_H", "F_naive"]
    T = min(len(all_records[m][0]) for m in methods)
    iters = np.arange(T) * record_every

    stats = {}
    for m in methods:
        arr = np.vstack([np.array(run)[:T] for run in all_records[m]])
        garr = np.vstack([np.array(run)[:T] for run in all_grad[m]])
        n_runs = arr.shape[0]
        stats[m] = {
            "F_mean": np.nanmean(arr, axis=0),
            "F_se": sem(arr, axis=0, nan_policy='omit') if n_runs > 1 else np.zeros(arr.shape[1]),
            "G_mean": np.nanmean(garr, axis=0),
            "G_se": sem(garr, axis=0, nan_policy='omit') if n_runs > 1 else np.zeros(garr.shape[1]),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 3.8)
    colors = {"F_G": "#1f77b4", "F_H": "#ff7f0e", "F_naive": "#2ca02c"}
    eps = 1e-30

    # (a) Full objective gap
    for m in methods:
        y = stats[m]["F_mean"]
        se = stats[m]["F_se"]
        lower = y - 1.96 * se
        lower = np.where(np.isnan(lower), y, lower)
        lower = np.maximum(lower, eps)
        upper = y + 1.96 * se
        ax1.plot(iters, y, label=m, color=colors[m], linewidth=2.0)
        ax1.fill_between(iters, lower, upper, color=colors[m], alpha=0.2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Step', fontsize=16)
    ax1.set_ylabel(r'$F - F^{*}$', fontsize=18)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_title('(a) Full objective gap', fontsize=18)
    ax1.legend(fontsize=16)

    def format_step(x, pos):
        if x >= 1e6:
            return f'{x / 1e6:.0f}M'
        elif x >= 1e3:
            return f'{x / 1e3:.0f}K'
        else:
            return f'{x:.0f}'
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_step))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_step))

    # (b) Gradient norm
    for m in methods:
        y = stats[m]["G_mean"]
        se = stats[m]["G_se"]
        lower = y - 1.96 * se
        lower = np.where(np.isnan(lower), y, lower)
        ax2.plot(iters, y, label=m, color=colors[m], linewidth=2.0)
        ax2.fill_between(iters, lower, y + 1.96 * se, color=colors[m], alpha=0.2)
    ax2.set_xlabel('Step', fontsize=16)
    ax2.set_ylabel('Gradient norm', fontsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_title('(b) Gradient norm', fontsize=18)
    ax2.legend(fontsize=16)

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.90, wspace=0.45)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------------------------------
# Plotting: Figure 3 (log-log diagnostics for G and H)
# ------------------------------------------------------------------
def plot_GH_combined_loglog(all_records, record_every,
                            warmup_fraction=0.1,
                            fit_fraction=0.6,
                            save_path=None):
    """Combined log-log plot with late-stage fits for methods G and H."""
    methods = ["F_G", "F_H"]
    T = min(len(all_records[m][0]) for m in methods)
    k = np.arange(1, T + 1)
    eps = 1e-12

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 4.0)

    axes = [ax1, ax2]
    titles = ['(a) Method $G$', '(b) Method $H$']

    for ax, method, title in zip(axes, methods, titles):
        runs = all_records[method]
        mean_run = np.mean([np.array(run[:T], dtype=float) for run in runs], axis=0)

        valid_mean = (mean_run > 1e-10) & (k > 0)
        ax.loglog(k[valid_mean], mean_run[valid_mean], 'g-', linewidth=2, label='Mean Loss')

        # Late-stage log-log fit
        start = int(np.ceil(T * warmup_fraction))
        tail_len = int(np.floor(T * fit_fraction))
        if tail_len < 5:
            tail_len = max(5, T - start)
        fit_start = max(start, T - tail_len)
        fit_end = T

        x_fit = np.log(k[fit_start:fit_end] + eps).reshape(-1, 1)
        y_fit = np.log(mean_run[fit_start:fit_end] + eps)
        lin = LinearRegression().fit(x_fit, y_fit)
        m = float(lin.coef_[0])
        b = float(lin.intercept_)

        # Shifted fit line for visibility
        k_plot = np.linspace(k[fit_start], k[fit_end - 1], 200)
        C = np.exp(b)
        delta_x, delta_y = 0.3, 0.3
        C_shifted = C * np.exp(-m * delta_x + delta_y)
        loss_fit = C_shifted * (k_plot ** m)
        ax.loglog(k_plot, loss_fit, 'm--', linewidth=2.5, alpha=0.9, zorder=5)

        # Legend label formatting
        def _fmt(x, sig=3):
            if x == 0:
                return "0"
            sign = "-" if x < 0 else ""
            ax_val = abs(x)
            if ax_val < 1e-3 or ax_val >= 1e4:
                return sign + f"{{:.{sig - 1}e}}".format(ax_val)
            else:
                import math
                digits = sig - int(math.floor(math.log10(ax_val))) - 1
                digits = max(digits, 0)
                return sign + f"{{:.{digits}f}}".format(ax_val)

        m_str = _fmt(m, sig=3)
        b_str = _fmt(b, sig=3)
        ax.plot([], [], 'm--', linewidth=2.5, label=rf'Fit: $\log L = {m_str}\log k + {b_str}$')
        ax.legend(fontsize=12, loc='lower left')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=7, top=1e2)
        ax.set_yticks([1e1, 1e2])
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which='both', ls='--', lw=0.5)

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.90, wspace=0.40)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, "F_GH_loglog_combined.eps")
        plt.savefig(filename, format='eps', dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    data_path = os.path.join(DATA_DIR, 'gram_run')

    if os.path.exists(data_path + ".npz"):
        print("Loading cached data...")
        all_records, all_grad, meta = load_run_data(data_path)
        F_star = meta.get("F_star", 0.0)
    else:
        print("Finding reference initialization...")
        theta_ref, F_ref, X_ref, S_ref = find_theta_ref(seed=0)
        print(f"Reference F_ref = {F_ref:.6e}")

        # Adjust diagonal of S before training
        for i in range(S_ref.shape[0]):
            S_ref[i, i] *= 1.05
        F_star = compute_Fstar_via_eig_numpy(S_ref.cpu().numpy(), d)
        print(f"F_star = {F_star:.6e}")

        print("Running repeated experiments...")
        all_records, all_grad, minF = run_repeats_and_summarize(
            theta_ref, X_ref, S_ref, sigma=PERTURB_SIGMA, n_repeats=N_REPEATS)
        print(f"Min observed F = {minF:.6e}")

        # Convert to gaps
        all_gap = {}
        for key, lists in all_records.items():
            all_gap[key] = [[max(0.0, f - F_star) for f in seq] for seq in lists]

        meta = {
            "record_every": RECORD_EVERY,
            "n_repeats": N_REPEATS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": ETA,
            "F_star": F_star,
        }
        save_run_data(data_path, all_gap, all_grad, meta=meta)
        all_records = all_gap

    # Figure 2: objective gap and gradient norm
    summarize_and_plot(all_records, all_grad, RECORD_EVERY,
                       save_path=os.path.join(FIG_DIR, 'gram.eps'))

    # Figure 3: log-log diagnostics
    plot_GH_combined_loglog(all_records, RECORD_EVERY,
                            warmup_fraction=0.1, fit_fraction=0.6,
                            save_path=FIG_DIR)

    print("Figures saved to", FIG_DIR)
