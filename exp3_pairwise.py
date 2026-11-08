"""Pairwise coupling with balanced external forces: final MLP experiment."""

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import fit_power_law, rss_bytes, save_figure, save_json, schedule_epochs, theoretical_memory_bytes, threshold_epoch

N, INPUT_DIM, HIDDEN = 200, 32, 256
BETA, LAMBDA, A, T0 = 1.0, 1.0, 0.03, 1.0
DTYPE = torch.float64
BATCH_SIZE = 12
ETA = 0.025
EPOCHS = 6_000
N_REPEATS = 5
RATE_FRACTIONS = (0.3, 0.4, 0.6)
GAP_THRESHOLD = 1e-4
METHODS = ("G", "H", "naive_raw", "naive_q")


class SmoothLeakyReLU(nn.Module):
    def forward(self, x):
        return 0.01 * x + 0.99 * 0.02 * F.softplus(x / 0.02)


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN, dtype=DTYPE),
            SmoothLeakyReLU(),
            nn.Linear(HIDDEN, 1, dtype=DTYPE),
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)


def constants():
    pairs = N * (N - 1) // 2
    half = N // 2
    phi_2a = 0.5 * (2 * A) ** 2 + BETA / 4 * (2 * A) ** 4
    linear_coeff = half * (2 * A + BETA * (2 * A) ** 3) / pairs
    f_star = half * half * phi_2a / pairs
    return pairs, linear_coeff, f_star


PAIR_COUNT, LINEAR_COEFF, F_STAR = constants()


def setup(seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xi = torch.cat((torch.ones(N // 2, dtype=DTYPE), -torch.ones(N // 2, dtype=DTYPE)))
    X = torch.tensor(rng.standard_normal((N, INPUT_DIM)), dtype=DTYPE)
    X[:, 0] = xi
    model = SmallMLP()
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.fill_(T0)
    return X, xi, model


def full_value(y, xi):
    i, j = torch.triu_indices(N, N, offset=1)
    z = y - A * xi
    diff = z[i] - z[j]
    average = (y[i] + y[j]) / 2
    phi = 0.5 * diff.square() + BETA / 4 * diff.pow(4)
    return phi.sum() / PAIR_COUNT + LINEAR_COEFF * torch.dot(xi, y) + LAMBDA / (4 * PAIR_COUNT) * average.pow(4).sum()


def output_gradient_parts(y, xi, local=False):
    n = y.numel()
    z = y - A * xi
    z1, z2, z3 = z.sum(), z.square().sum(), z.pow(3).sum()
    y1, y2, y3 = y.sum(), y.square().sum(), y.pow(3).sum()
    pair = (n * z - z1 + BETA * (n * z.pow(3) - 3 * z.square() * z1 + 3 * z * z2 - z3)) / PAIR_COUNT
    quartic = LAMBDA / (16 * PAIR_COUNT) * (
        (n - 8) * y.pow(3) + 3 * y.square() * y1 + 3 * y * y2 + y3
    )
    external = LINEAR_COEFF * xi
    return external, pair + quartic


def output_gradient(y, xi, local=False):
    external, interaction = output_gradient_parts(y, xi, local=local)
    return external + interaction


def analytic_check():
    torch.manual_seed(227)
    xi = torch.cat((torch.ones(N // 2, dtype=DTYPE), -torch.ones(N // 2, dtype=DTYPE)))
    y = (0.1 * torch.randn(N, dtype=DTYPE)).requires_grad_()
    automatic = torch.autograd.grad(full_value(y, xi), y)[0]
    formula = output_gradient(y.detach(), xi)
    error = float((automatic - formula).abs().max())
    zero = torch.zeros(N, dtype=DTYPE)
    assert error < 1e-10
    assert float(output_gradient(zero, xi).abs().max()) < 1e-12
    assert abs(float(full_value(zero, xi)) - F_STAR) < 1e-12
    t = 0.02
    path_gap = float(full_value(torch.full((N,), t, dtype=DTYPE), xi)) - F_STAR
    assert abs(path_gap - LAMBDA * t ** 4 / 4) < 1e-12
    return {"max_autograd_error": error, "status": "passed"}


def flatten_grads(model):
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])


def parameter_gradient(model, X, output_grad):
    model.zero_grad(set_to_none=True)
    model(X).backward(output_grad)
    gradient = flatten_grads(model).clone()
    model.zero_grad(set_to_none=True)
    return gradient


def q_epoch():
    sizes = [min(BATCH_SIZE, N - start) for start in range(0, N, BATCH_SIZE)]
    return sum(size * (size - 1) for size in sizes) / (N * (N - 1))


def batches_for(seed):
    permutation = np.random.default_rng(seed).permutation(N)
    return [permutation[start:start + BATCH_SIZE] for start in range(0, N, BATCH_SIZE)]


def cosine(a, b):
    denominator = torch.linalg.norm(a) * torch.linalg.norm(b)
    return float(torch.dot(a, b) / denominator) if denominator > 0 else 0.0


def frozen_diagnostic(state, X, xi, seed):
    q = q_epoch()
    base = SmallMLP(); base.load_state_dict(state)
    with torch.no_grad():
        Y = base(X)
        full_output = output_gradient(Y, xi)
        external, interaction = output_gradient_parts(Y, xi)
    full_parameter = parameter_gradient(base, X, full_output)
    external_parameter = parameter_gradient(base, X, external)
    expected_naive_q_bias = (1 / q - 1) * external_parameter
    accum = {name: torch.zeros_like(full_parameter) for name in METHODS}
    batch_rows = []
    for batch in batches_for(seed):
        idx = torch.as_tensor(batch, dtype=torch.long)
        model = SmallMLP(); model.load_state_dict(state)
        with torch.no_grad():
            yb = model(X[idx])
            external_b, interaction_b = output_gradient_parts(yb, xi[idx], local=True)
            grads = {
                "G": full_output[idx],
                "H": full_output[idx],
                "naive_raw": external_b + interaction_b,
                "naive_q": (external_b + interaction_b) / q,
            }
        item = {"batch_size": int(len(batch))}
        for name, output_grad in grads.items():
            g = parameter_gradient(model, X[idx], output_grad)
            accum[name] += g
            item[f"{name}_norm"] = float(g.norm())
            item[f"{name}_cosine"] = cosine(full_parameter, g)
            item[f"{name}_inner_product"] = float(torch.dot(full_parameter, g))
        batch_rows.append(item)
    diagnostics = {}
    for name, estimate in accum.items():
        error = estimate - full_parameter
        diagnostics[name] = {
            "absolute_error": float(error.norm()),
            "relative_error": float(error.norm() / max(float(full_parameter.norm()), 1e-30)),
            "cosine_with_full": cosine(full_parameter, estimate),
            "gradient_norm": float(estimate.norm()),
        }
    return {
        "seed": seed, "q_epoch": q, "full_gradient_norm": float(full_parameter.norm()),
        "full_output_gradient_norm": float(full_output.norm()),
        "external_parameter_gradient_norm": float(external_parameter.norm()),
        "expected_naive_q_bias_norm": float(expected_naive_q_bias.norm()),
        "epoch_gradient_diagnostics": diagnostics,
        "batch_diagnostics": batch_rows,
    }


def mode_metrics(y, xi):
    common = y.mean()
    group = torch.dot(xi, y) / N
    residual = y - common - group * xi
    return {
        "common_mode": float(common),
        "group_mode": float(group),
        "orthogonal_mode_norm": float(residual.norm()),
    }


def train_method(name, state, X, xi, epochs, schedule, seed, q):
    model = SmallMLP(); model.load_state_dict(state)
    lr = ETA / q if name == "naive_q" else ETA
    optimizer = optim.SGD(model.parameters(), lr=lr)
    cache = None
    cache_gradient = None
    if name == "H":
        with torch.no_grad():
            cache = model(X).detach().clone()
    schedule_set = set(int(value) for value in schedule)
    records = []
    counters = {"forward_full": 0, "forward_batch": 0, "cache_refresh": 0}
    elapsed = 0.0
    rss_before = rss_bytes()
    batches_per_epoch = int(np.ceil(N / BATCH_SIZE))
    def measure(epoch):
        with torch.no_grad():
            y = model(X); counters["forward_full"] += 1
            value = full_value(y, xi)
            grad = output_gradient(y, xi)
            item = {
                "raw_epoch": epoch,
                "display_epoch": epoch,
                "update_step": epoch * batches_per_epoch,
                "gap": float(value) - F_STAR,
                "output_gradient_norm": float(grad.norm()),
                "output_norm": float(y.norm()),
                "parameter_norm": float(torch.sqrt(sum(p.square().sum() for p in model.parameters()))),
                "cache_error": float((cache - y).norm()) if name == "H" else 0.0,
                "cache_gradient_error": float((cache_gradient - grad).norm()) if name == "H" and cache_gradient is not None else 0.0,
                "cache_memory_bytes": int(cache.numel() * cache.element_size()) if name == "H" else 0,
            }
            item.update(mode_metrics(y, xi))
            records.append(item)
    measure(0)
    for epoch in range(1, epochs + 1):
        for batch in batches_for(seed + epoch):
            idx = torch.as_tensor(batch, dtype=torch.long)
            start = time.perf_counter()
            if name == "G":
                with torch.no_grad():
                    y_full = model(X); counters["forward_full"] += 1
                    grad_full = output_gradient(y_full, xi)
                    batch_grad = grad_full[idx]
                output = model(X[idx]); counters["forward_batch"] += 1
            elif name == "H":
                with torch.no_grad():
                    cache_external, cache_interaction = output_gradient_parts(cache, xi)
                    cache_gradient = cache_external + cache_interaction
                    batch_grad = cache_gradient[idx]
                output = model(X[idx]); counters["forward_batch"] += 1
            else:
                output = model(X[idx]); counters["forward_batch"] += 1
                with torch.no_grad():
                    external, interaction = output_gradient_parts(output.detach(), xi[idx], local=True)
                    batch_grad = external + interaction
            optimizer.zero_grad(set_to_none=True)
            output.backward(batch_grad)
            optimizer.step()
            if name == "H":
                with torch.no_grad():
                    cache[idx] = model(X[idx]).detach(); counters["forward_batch"] += 1
                    counters["cache_refresh"] += len(batch)
            elapsed += time.perf_counter() - start
        if epoch in schedule_set:
            measure(epoch)
    return {
        "name": name, "records": records, "wall_time_seconds": elapsed,
        "rss_before": rss_before, "rss_after": rss_bytes(), "counters": counters,
        "learning_rate": lr,
    }


def fit_rate(records):
    epochs = np.asarray([row["raw_epoch"] for row in records])
    gaps = np.asarray([row["gap"] for row in records])
    return fit_power_law(epochs, gaps, fractions=RATE_FRACTIONS, floor=1e-14)


def not_applicable_rate():
    return {"status": "not_applicable", "reason": "bias-related platform; no theoretical rate assigned"}


def load_saved_results(output_dir):
    output_dir = Path(output_dir)
    meta = json.loads((output_dir / "exp3_pairwise.meta.json").read_text(encoding="utf-8"))
    data = np.load(output_dir / "exp3_pairwise.npz")
    repeats = int(meta["repeats"])
    results = {name: [] for name in METHODS}
    keys = (
        "raw_epoch", "display_epoch", "update_step", "gap", "output_gradient_norm",
        "output_norm", "parameter_norm", "cache_error", "cache_gradient_error",
        "common_mode", "group_mode", "orthogonal_mode_norm",
    )
    for name in METHODS:
        for repeat in range(repeats):
            arrays = {key: data[f"{name}_repeat{repeat}_{key}"] for key in keys}
            results[name].append({
                "name": name,
                "records": [
                    {key: value[index].item() for key, value in arrays.items()}
                    for index in range(len(arrays["raw_epoch"]))
                ],
            })
    return results, meta["frozen_diagnostic"]


def plot_results(results, figure_dir, frozen=None):
    figure_dir = Path(figure_dir); figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"G": "#1f77b4", "H": "#ff7f0e", "naive_raw": "#d62728", "naive_q": "#2ca02c"}
    labels = {"G": "G", "H": "H", "naive_raw": "naive-raw", "naive_q": "naive-q"}
    linestyles = {"G": "-", "H": "--", "naive_raw": "-.", "naive_q": ":"}
    means = {}
    for name, runs in results.items():
        means[name] = {
            key: np.vstack([[row[key] for row in run["records"]] for run in runs]).mean(axis=0)
            for key in ("gap", "output_gradient_norm", "common_mode", "group_mode", "orthogonal_mode_norm", "cache_error")
        }
    x = np.asarray([row["display_epoch"] for row in results["G"][0]["records"]])
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for name, data in means.items():
        axis.semilogy(x, np.maximum(data["gap"], 1e-30), color=colors[name], linestyle=linestyles[name], label=labels[name])
    axis.set_xlabel("Data epoch"); axis.set_ylabel(r"$F-F^*$"); axis.set_title("Exp3 full-range gap")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5); axis.legend(); fig.tight_layout()
    save_figure(fig, figure_dir / "exp3_pairwise_total.pdf"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for name in ("G", "H"):
        axis.semilogy(x, np.maximum(means[name]["gap"], 1e-30), color=colors[name], linestyle=linestyles[name], label=labels[name])
    platform = min(float(data["gap"][-1]) for name, data in means.items() if name.startswith("naive"))
    axis.axhline(platform, color="#555555", linestyle="--", label="naive platform reference")
    axis.set_xlabel("Data epoch"); axis.set_ylabel(r"$F-F^*$"); axis.set_title("Exp3 G/H low-gap view")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5); axis.legend(); fig.tight_layout()
    save_figure(fig, figure_dir / "exp3_pairwise_GH_zoom.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, data in means.items():
        axes[0].plot(x, data["common_mode"], color=colors[name], linestyle=linestyles[name], label=labels[name])
        axes[1].plot(x, data["group_mode"], color=colors[name], linestyle=linestyles[name], label=labels[name])
        axes[2].semilogy(x, np.maximum(data["orthogonal_mode_norm"], 1e-30), color=colors[name], linestyle=linestyles[name], label=labels[name])
    for axis, title in zip(axes, ("Common mode", "Group mode", "Residual mode")):
        axis.set_title(title); axis.set_xlabel("Data epoch"); axis.grid(True, linestyle="--", linewidth=0.5)
    axes[0].legend(); fig.tight_layout()
    save_figure(fig, figure_dir / "exp3_pairwise_modes.pdf"); plt.close(fig)
    if frozen:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        markers = {"naive_raw": "o", "naive_q": "s"}
        for name in ("naive_raw", "naive_q"):
            diagnostics = [row["epoch_gradient_diagnostics"][name] for row in frozen]
            marker, linestyle = markers[name], linestyles[name]
            axes[0].plot([row["relative_error"] for row in diagnostics], marker=marker, linestyle=linestyle, color=colors[name], label=labels[name])
            axes[1].plot([row["cosine_with_full"] for row in diagnostics], marker=marker, linestyle=linestyle, color=colors[name], label=labels[name])
        axes[0].set_ylabel("Frozen epoch relative error")
        axes[1].set_ylabel("Cosine with full parameter gradient")
        for axis in axes:
            axis.set_xlabel("Repeat")
            axis.grid(True, linestyle="--", linewidth=0.5)
            axis.legend()
        fig.tight_layout()
        save_figure(fig, figure_dir / "exp3_pairwise_frozen.pdf"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for name in ("G", "H"):
        x = np.asarray([row["raw_epoch"] for row in results[name][0]["records"]])
        y = np.vstack([[row["gap"] for row in run["records"]] for run in results[name]]).mean(axis=0)
        fit = fit_power_law(x, y, fractions=RATE_FRACTIONS, floor=1e-14)
        selected = fit["selected"]
        start = selected["fit_start_index"]
        valid = np.isfinite(x[start:]) & np.isfinite(y[start:])
        valid &= (x[start:] > 0) & (y[start:] > 1e-14)
        x_tail, y_tail = x[start:][valid], y[start:][valid]
        if selected["status"] != "ok":
            continue
        axis.loglog(x_tail, y_tail, "o", color=colors[name], label=labels[name])
        fitted = np.exp(selected["intercept"] + selected["slope"] * np.log(x_tail))
        axis.loglog(x_tail, fitted, linestyle=linestyles[name], color=colors[name],
                    label=f"{labels[name]} fit: slope={selected['slope']:.3g}, $R^2$={selected['r_squared']:.3f}")
    axis.set_xlabel("Raw permutation epoch")
    axis.set_ylabel(r"$F-F^*$")
    axis.set_title("Exp3 G/H tail log-log rate diagnostic")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5); axis.legend()
    fig.tight_layout()
    save_figure(fig, figure_dir / "exp3_pairwise_GH_rate.pdf"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--figure-dir", default="fig")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        results, frozen = load_saved_results(args.output_dir)
        plot_results(results, args.figure_dir, frozen=frozen)
        return
    epochs = 3 if args.smoke_test else args.epochs
    repeats = 1 if args.smoke_test else args.repeats
    schedule = schedule_epochs(epochs, first=(1, 2, 3, 5, 10, 20), interval=100)
    q = q_epoch()
    check = analytic_check()
    base_X, base_xi, base_model = setup(0)
    results = {name: [] for name in METHODS}
    frozen = []
    for repeat in range(repeats):
        X, xi, initial = setup(repeat)
        state = copy.deepcopy(initial.state_dict())
        frozen.append(frozen_diagnostic(state, X, xi, 7919 + repeat))
        for name in METHODS:
            results[name].append(train_method(name, state, X, xi, epochs, schedule, 12000 + repeat, q))
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"X": base_X.numpy(), "xi": base_xi.numpy(), "F_star": np.asarray(F_STAR)}
    for repeat in range(repeats):
        repeat_X, repeat_xi, _ = setup(repeat)
        arrays[f"repeat{repeat}_X"] = repeat_X.numpy()
        arrays[f"repeat{repeat}_xi"] = repeat_xi.numpy()
    rate_fits = {}
    for name, runs in results.items():
        rate_fits[name] = [fit_rate(run["records"]) for run in runs] if name in ("G", "H") else [not_applicable_rate() for run in runs]
        for index, run in enumerate(runs):
            for key in ("raw_epoch", "display_epoch", "update_step", "gap", "output_gradient_norm", "output_norm", "parameter_norm", "cache_error", "cache_gradient_error", "common_mode", "group_mode", "orthogonal_mode_norm"):
                arrays[f"{name}_repeat{index}_{key}"] = np.asarray([row[key] for row in run["records"]])
    np.savez_compressed(output_dir / "exp3_pairwise.npz", **arrays)
    memory = {
        "X": base_X.numel() * base_X.element_size(),
        "xi": base_xi.numel() * base_xi.element_size(),
        "full_output": N * base_X.element_size(),
        "H_cache": N * base_X.element_size(),
    }
    meta = {
        "experiment": "exp3_pairwise_balanced_external_forces",
        "version": "final_v1", "N": N, "input_dim": INPUT_DIM, "hidden": HIDDEN,
        "batch_size": BATCH_SIZE, "epochs": epochs, "repeats": repeats,
        "eta": ETA, "q_epoch": q, "eta_naive_q": ETA / q,
        "beta": BETA, "lambda": LAMBDA, "A": A, "T0": T0,
        "pair_count": PAIR_COUNT, "linear_coefficient": LINEAR_COEFF, "F_star": F_STAR,
        "kl_exponent": 0.75, "expected_deterministic_reference_rate": "k^-2",
        "analytic_check": check, "frozen_diagnostic": frozen,
        "rate_fits": rate_fits,
        "rate_fit_interpretation": "Only G and H are primary rate references; naive methods reach a bias-related platform and are not assigned a theoretical rate.",
        "threshold_epochs": {
            name: [threshold_epoch(
                [row["raw_epoch"] for row in run["records"]],
                [row["gap"] for row in run["records"]],
                GAP_THRESHOLD,
            ) for run in runs]
            for name, runs in results.items()
        },
        "theoretical_memory_bytes": theoretical_memory_bytes(memory), "memory_breakdown_bytes": memory,
        "per_method_efficiency": {
            name: [{key: run[key] for key in ("wall_time_seconds", "rss_before", "rss_after", "counters", "learning_rate")} for run in runs]
            for name, runs in results.items()
        },
        "interpretation": "Pairwise non-Gram KL-3/4 benchmark; naive-q bias is a frozen-state epoch diagnostic, not an exact online-training recursion.",
    }
    save_json(output_dir / "exp3_pairwise.meta.json", meta)
    plot_results(results, args.figure_dir, frozen=frozen)


if __name__ == "__main__":
    main()
