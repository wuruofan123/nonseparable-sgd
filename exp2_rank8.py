"""Final Rank-8 Gram experiment with shared MLP and full diagnostics."""

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

from common import (
    METHODS, aggregate_schedule, fit_exponential, rss_bytes, save_json,
    theoretical_memory_bytes, threshold_epoch, save_figure
)

DTYPE = torch.float64
N, INPUT_DIM, HIDDEN, D = 200, 32, 256, 8
BATCH_SIZE = 16
OUTPUT_SCALE = 16.0
TARGET_SCALE = 0.25
MU_RATIO = 0.5
BASE_ETA = 1e-2
ETA = BASE_ETA / OUTPUT_SCALE ** 4
EPOCHS = 15_000
AGGREGATION_EPOCHS = 10
N_REPEATS = 5
DIRECTION_SEED = 7919
RATE_FRACTIONS = (0.3, 0.4, 0.6)
GAP_THRESHOLD = 1e-4


class SmoothLeakyReLU(nn.Module):
    def forward(self, x):
        return 0.01 * x + 0.99 * 0.02 * F.softplus(x / 0.02)


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN, dtype=DTYPE),
            SmoothLeakyReLU(),
            nn.Linear(HIDDEN, D, dtype=DTYPE),
        )

    def forward(self, X):
        return OUTPUT_SCALE * self.net(X)


def gram_value_grad(Y, S):
    residual = (Y @ Y.T) / D - S
    value = 0.5 * torch.sum(residual.square())
    gradient = ((residual + residual.T) @ Y) / D
    return value, gradient, residual


def orthogonal_direction(factor, seed):
    generator = torch.Generator(device=factor.device).manual_seed(seed)
    direction = torch.randn(N, dtype=DTYPE, generator=generator)
    basis, _ = torch.linalg.qr(factor, mode="reduced")
    direction = direction - basis @ (basis.T @ direction)
    direction = direction / torch.linalg.vector_norm(direction)
    return direction


def rank_factor(S):
    values, vectors = torch.linalg.eigh(S)
    order = torch.argsort(values, descending=True)
    values = torch.clamp(values[order[:D]], min=0.0)
    vectors = vectors[:, order[:D]]
    return vectors * torch.sqrt(D * values).reshape(1, -1)


def target_and_state(seed=0):
    torch.manual_seed(seed)
    X = torch.randn(N, INPUT_DIM, dtype=DTYPE)
    target = SmallMLP()
    with torch.no_grad():
        target.net[-1].weight.mul_(TARGET_SCALE)
        target.net[-1].bias.mul_(TARGET_SCALE)
        U = target(X).detach().clone()
    direction = orthogonal_direction(U, seed + 1009)
    S_reference = U @ U.T / D
    singular = torch.linalg.svdvals(U)
    lambda_min = singular[-1].square() / D
    mu = MU_RATIO * lambda_min
    S = S_reference + mu * torch.outer(direction, direction)
    Sstar = rank_factor(S)
    spectral_fstar = float(0.5 * torch.linalg.norm(S - Sstar @ Sstar.T / D).square())
    analytic_fstar = 0.5 * float(mu.square())
    if abs(spectral_fstar - analytic_fstar) > 1e-8 * max(1.0, float(S.square().sum())):
        raise RuntimeError("Rank-8 F* check failed")
    return {
        "X": X, "U": U, "S_reference": S_reference, "S": S,
        "direction": direction, "Sstar_factor": Sstar,
        "target_state": copy.deepcopy(target.state_dict()),
        "F_star": analytic_fstar,
        "target_diagnostics": {
            "target_rank": int(torch.linalg.matrix_rank(S)),
            "factor_condition": float(singular[0] / singular[-1]),
            "lambda_min_factor_gram": float(lambda_min),
            "mu": float(mu), "mu_ratio": MU_RATIO,
            "orthogonality_error": float(torch.linalg.norm(U.T @ direction)),
            "F_star_spectral": spectral_fstar,
            "F_star_analytic": analytic_fstar,
        },
    }


def fit_output_state(state_dict, X, desired):
    model = SmallMLP()
    model.load_state_dict(state_dict)
    with torch.no_grad():
        hidden = model.net[1](model.net[0](X))
        design = torch.cat((hidden, torch.ones(N, 1, dtype=DTYPE)), dim=1)
        singular = torch.linalg.svdvals(design)
        tolerance = max(design.shape) * torch.finfo(DTYPE).eps * singular[0]
        rank = int((singular > tolerance).sum())
        if rank != N:
            raise RuntimeError("Rank-8 feature design is not full row rank")
        coeff = torch.linalg.lstsq(design, desired / OUTPUT_SCALE, driver="gelsd").solution
        model.net[-1].weight.copy_(coeff[:-1].T)
        model.net[-1].bias.copy_(coeff[-1])
        actual = model(X).detach()
    relative_error = torch.linalg.norm(actual - desired) / max(1.0, float(desired.norm()))
    if float(relative_error) > 1e-8:
        raise RuntimeError("Rank-8 output initialization failed")
    return copy.deepcopy(model.state_dict()), {
        "feature_rank": rank,
        "feature_sigma_min": float(singular[-1]),
        "feature_sigma_max": float(singular[0]),
        "feature_condition": float(singular[0] / singular[-1]),
        "fit_relative_error": float(relative_error),
    }


def prepare_initial_state(data, seed):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((N, D), dtype=DTYPE, generator=generator)
    noise = noise / torch.linalg.vector_norm(noise)
    center = data["Sstar_factor"]
    desired = center + 0.01 * torch.linalg.vector_norm(center) * noise
    state, diagnostics = fit_output_state(data["target_state"], data["X"], desired)
    model = SmallMLP()
    model.load_state_dict(state)
    with torch.no_grad():
        Y = model(data["X"])
        value, gradient, residual = gram_value_grad(Y, data["S"])
    diagnostics.update({
        "seed": seed,
        "initial_gap": float(value) - data["F_star"],
        "initial_output_gradient_norm": float(gradient.norm()),
        "initial_residual_norm": float(residual.norm()),
        "initial_output_norm": float(Y.norm()),
    })
    return state, diagnostics


def flatten_grads(model):
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])


def parameter_gradient(model, X, output_gradient):
    model.zero_grad(set_to_none=True)
    model(X).backward(output_gradient)
    gradient = flatten_grads(model).clone()
    model.zero_grad(set_to_none=True)
    return gradient


def feature_diagnostic(model, X):
    with torch.no_grad():
        feature = model.net[1](model.net[0](X))
        design = torch.cat((feature, torch.ones(N, 1, dtype=DTYPE)), dim=1)
        singular = torch.linalg.svdvals(design)
    return {
        "rank": int(torch.linalg.matrix_rank(design)),
        "sigma_min": float(singular[-1]),
        "sigma_max": float(singular[0]),
        "condition": float(singular[0] / singular[-1]),
    }


def batches_for(seed):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(N)
    return [permutation[start:start + BATCH_SIZE] for start in range(0, N, BATCH_SIZE)]


def q_epoch():
    sizes = [min(BATCH_SIZE, N - start) for start in range(0, N, BATCH_SIZE)]
    return sum(size * (size - 1) for size in sizes) / (N * (N - 1))


def cosine(first, second):
    denominator = torch.linalg.norm(first) * torch.linalg.norm(second)
    return float(torch.dot(first, second) / denominator) if denominator > 0 else 0.0


def frozen_diagnostic(state, X, S, seed):
    batches = batches_for(seed)
    q = q_epoch()
    model_full = SmallMLP(); model_full.load_state_dict(state)
    model_g = SmallMLP(); model_g.load_state_dict(state)
    model_n = SmallMLP(); model_n.load_state_dict(state)
    with torch.no_grad():
        Y = model_full(X)
        full_value, full_output_grad, _ = gram_value_grad(Y, S)
    full_gradient = parameter_gradient(model_full, X, full_output_grad)
    g_sum = torch.zeros_like(full_gradient)
    n_sum = torch.zeros_like(full_gradient)
    batches_out = []
    for batch in batches:
        idx = torch.as_tensor(batch, dtype=torch.long)
        Yg = model_g(X[idx])
        Yg.backward(full_output_grad[idx])
        g_batch = flatten_grads(model_g).clone(); model_g.zero_grad(set_to_none=True)
        Yn = model_n(X[idx])
        with torch.no_grad():
            _, n_output, _ = gram_value_grad(Yn, S[idx][:, idx])
        Yn.backward(n_output)
        n_batch = flatten_grads(model_n).clone(); model_n.zero_grad(set_to_none=True)
        g_sum += g_batch; n_sum += n_batch
        g_norm = torch.linalg.norm(g_batch); n_norm = torch.linalg.norm(n_batch)
        batches_out.append({
            "batch_size": int(len(batch)),
            "inner_product_G": float(torch.dot(full_gradient, g_batch)),
            "inner_product_naive": float(torch.dot(full_gradient, n_batch)),
            "cosine_G": cosine(full_gradient, g_batch),
            "cosine_naive": cosine(full_gradient, n_batch),
            "G_norm": float(g_norm), "naive_norm": float(n_norm),
        })
    def diag(estimate):
        error = estimate - full_gradient
        return {"absolute_error": float(error.norm()),
                "relative_error": float(error.norm() / max(float(full_gradient.norm()), 1e-30)),
                "cosine_with_full": cosine(estimate, full_gradient),
                "norm": float(estimate.norm())}
    inner_g = np.asarray([item["inner_product_G"] for item in batches_out])
    inner_n = np.asarray([item["inner_product_naive"] for item in batches_out])
    return {
        "seed": seed, "q_epoch": q, "full_objective": float(full_value),
        "full_gradient_norm": float(full_gradient.norm()),
        "G": diag(g_sum), "naive_raw": diag(n_sum), "naive_q": diag(n_sum / q),
        "batch_diagnostics": batches_out,
        "batch_negative_inner_fraction_G": float(np.mean(inner_g < 0)),
        "batch_negative_inner_fraction_naive": float(np.mean(inner_n < 0)),
        "batch_mean_inner_product_G": float(inner_g.mean()),
        "batch_mean_inner_product_naive": float(inner_n.mean()),
    }


def train_method(name, state, X, S, F_star, epochs, schedule, aggregation_epochs, seed, q):
    model = SmallMLP(); model.load_state_dict(state)
    lr = ETA / q if name == "naive_q" else ETA
    optimizer = optim.SGD(model.parameters(), lr=lr)
    cache = None
    gram_cache = None
    if name == "H":
        with torch.no_grad():
            cache = model(X).detach().clone()
            gram_cache = cache @ cache.T / D
    batches_per_epoch = int(np.ceil(N / BATCH_SIZE))
    records = []
    counters = {"forward_full": 0, "forward_batch": 0, "cache_refresh": 0}
    timing = 0.0
    rss_before = rss_bytes()
    schedule_set = set(int(value) for value in schedule)
    def measure(raw_epoch):
        nonlocal cache, gram_cache
        with torch.no_grad():
            Y = model(X); counters["forward_full"] += 1
            value, output_grad, _ = gram_value_grad(Y, S)
            item = {
                "raw_epoch": raw_epoch,
                "display_epoch": raw_epoch / aggregation_epochs,
                "update_step": raw_epoch * batches_per_epoch,
                "gap": float(value) - F_star,
                "output_gradient_norm": float(output_grad.norm()),
                "output_norm": float(Y.norm()),
                "parameter_norm": float(torch.sqrt(sum(p.square().sum() for p in model.parameters()))),
            }
            if name == "H":
                item["cache_error_Y"] = float((cache - Y).norm())
                item["cache_error_Gram"] = float((gram_cache - Y @ Y.T / D).norm())
                item["cache_memory_bytes"] = int(cache.numel() * cache.element_size() + gram_cache.numel() * gram_cache.element_size())
            else:
                item["cache_error_Y"] = 0.0
                item["cache_error_Gram"] = 0.0
                item["cache_memory_bytes"] = 0
            feature = feature_diagnostic(model, X)
            item.update({
                "feature_rank": feature["rank"],
                "feature_sigma_min": feature["sigma_min"],
                "feature_sigma_max": feature["sigma_max"],
                "feature_condition": feature["condition"],
            })
            records.append(item)
    measure(0)
    for raw_epoch in range(1, epochs + 1):
        for batch in batches_for(seed + raw_epoch):
            idx = torch.as_tensor(batch, dtype=torch.long)
            step_start = time.perf_counter()
            if name == "G":
                with torch.no_grad():
                    Y = model(X); counters["forward_full"] += 1
                    _, output_grad, _ = gram_value_grad(Y, S)
                    batch_grad = output_grad[idx]
                output = model(X[idx]); counters["forward_batch"] += 1
            elif name == "H":
                with torch.no_grad():
                    residual_cache = gram_cache - S
                    batch_grad = ((residual_cache[idx] + residual_cache.T[idx]) @ cache) / D
                output = model(X[idx]); counters["forward_batch"] += 1
            else:
                output = model(X[idx]); counters["forward_batch"] += 1
                with torch.no_grad():
                    _, batch_grad, _ = gram_value_grad(output.detach(), S[idx][:, idx])
            optimizer.zero_grad(set_to_none=True)
            output.backward(batch_grad)
            optimizer.step()
            if name == "H":
                with torch.no_grad():
                    new_rows = model(X[idx]).detach().clone(); counters["forward_batch"] += 1
                    cache[idx] = new_rows
                    cross = new_rows @ cache.T / D
                    gram_cache[idx, :] = cross
                    gram_cache[:, idx] = cross.T
                    counters["cache_refresh"] += len(batch)
            timing += time.perf_counter() - step_start
        if raw_epoch in schedule_set:
            measure(raw_epoch)
    rss_after = rss_bytes()
    return {
        "name": name, "records": records, "wall_time_seconds": timing,
        "rss_before": rss_before, "rss_after": rss_after,
        "counters": counters, "learning_rate": lr,
    }


def fit_rate(records):
    epochs = np.asarray([row["display_epoch"] for row in records])
    gaps = np.asarray([row["gap"] for row in records])
    return fit_exponential(epochs, gaps, fractions=RATE_FRACTIONS, floor=1e-14)


def load_saved_results(output_dir):
    output_dir = Path(output_dir)
    meta = json.loads((output_dir / "exp2_rank8.meta.json").read_text(encoding="utf-8"))
    data = np.load(output_dir / "exp2_rank8.npz")
    repeats = int(meta["repeats"])
    results = {name: [] for name in METHODS}
    keys = (
        "raw_epoch", "display_epoch", "update_step", "gap", "output_gradient_norm",
        "output_norm", "parameter_norm", "cache_error_Y", "cache_error_Gram",
        "cache_memory_bytes", "feature_rank", "feature_sigma_min", "feature_sigma_max",
        "feature_condition",
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
    return results, meta["frozen_gradient_diagnostics"], int(meta["aggregation_epochs"])


def plot_results(results, figure_dir, prefix, aggregation_epochs, frozen=None):
    figure_dir = Path(figure_dir); figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"G": "#1f77b4", "H": "#ff7f0e", "naive_raw": "#d62728", "naive_q": "#2ca02c"}
    labels = {"G": "G", "H": "H", "naive_raw": "naive-raw", "naive_q": "naive-q"}
    linestyles = {"G": "-", "H": "--", "naive_raw": "-.", "naive_q": ":"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, runs in results.items():
        curves = [np.asarray([row["gap"] for row in run["records"]]) for run in runs]
        x = np.asarray([row["display_epoch"] for row in runs[0]["records"]])
        values = np.vstack(curves); mean = values.mean(axis=0)
        axes[0].semilogy(x, np.maximum(mean, 1e-30), color=colors[name], linestyle=linestyles[name], label=labels[name])
        axes[1].semilogy(x, np.maximum(np.vstack([[row["output_gradient_norm"] for row in run["records"]] for run in runs]).mean(axis=0), 1e-30), color=colors[name], linestyle=linestyles[name], label=labels[name])
    axes[0].set_ylabel(r"$F-F^*$"); axes[1].set_ylabel(r"$\|\nabla_YF\|_F$")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, which="both", linestyle="--", linewidth=0.5); axis.legend()
    save_figure(fig, figure_dir / f"{prefix}_summary.pdf"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for name, runs in results.items():
        if name != "H": continue
        x = np.asarray([row["display_epoch"] for row in runs[0]["records"]])
        values = np.vstack([[row["cache_error_Y"] for row in run["records"]] for run in runs])
        mean = values.mean(axis=0)
        positive = mean > 0
        axis.semilogy(x[positive], mean[positive], color=colors[name], linestyle=linestyles[name], label=r"$\|Y_{cache}-Y\|_F$")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cache error")
    axis.set_title("Exp2 H cache error (initial zero omitted on log scale)")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5); axis.legend(); fig.tight_layout()
    save_figure(fig, figure_dir / f"{prefix}_cache.pdf"); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    rate_fits = {}
    for name in ("G", "H"):
        runs = results[name]
        epochs = np.asarray([row["display_epoch"] for row in runs[0]["records"]], dtype=float)
        gaps = np.vstack([[row["gap"] for row in run["records"]] for run in runs])
        mean_gap = gaps.mean(axis=0)
        fit = fit_exponential(epochs, mean_gap, fractions=RATE_FRACTIONS, floor=1e-14)
        rate_fits[name] = fit
        selected = fit["selected"]
        if selected["status"] != "ok":
            continue
        start = selected["fit_start_index"]
        valid = np.isfinite(epochs[start:]) & np.isfinite(mean_gap[start:])
        valid &= (epochs[start:] > 0) & (mean_gap[start:] > 1e-14)
        fit_epochs = epochs[start:][valid]
        fit_gap = mean_gap[start:][valid]
        axis.semilogy(fit_epochs, fit_gap, "o", color=colors[name], markersize=3, label=labels[name])
        fitted = np.exp(selected["intercept"] + selected["slope"] * fit_epochs)
        axis.semilogy(fit_epochs, fitted, linestyle=linestyles[name], color=colors[name],
                      label=f"{labels[name]} fit: slope={selected['slope']:.3g} per epoch, $R^2$={selected['r_squared']:.3f}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(r"$F-F^*$")
    axis.set_title("Exp2 G/H empirical exponential rate diagnostic")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5); axis.legend()
    fig.tight_layout()
    save_figure(fig, figure_dir / f"{prefix}_rate.pdf"); plt.close(fig)
    if frozen:
        names = tuple(name for name in METHODS if name in frozen[0])
        error = {name: [row[name]["relative_error"] for row in frozen] for name in names}
        cosine = {name: [row[name]["cosine_with_full"] for row in frozen] for name in names}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        markers = {"naive_raw": "o", "naive_q": "s"}
        for name in names:
            marker, linestyle = markers.get(name, "o"), linestyles[name]
            axes[0].plot(error[name], marker=marker, linestyle=linestyle, color=colors[name], label=labels[name])
            axes[1].plot(cosine[name], marker=marker, linestyle=linestyle, color=colors[name], label=labels[name])
        axes[0].set_ylabel("Frozen relative gradient error")
        axes[1].set_ylabel("Cosine with full gradient")
        for axis in axes:
            axis.set_xlabel("Repeat")
            axis.grid(True, linestyle="--", linewidth=0.5)
            axis.legend()
        save_figure(fig, figure_dir / f"{prefix}_frozen.pdf"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--figure-dir", default="fig")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--aggregation-epochs", type=int, default=AGGREGATION_EPOCHS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        results, frozen, aggregation = load_saved_results(args.output_dir)
        meta_path = Path(args.output_dir) / "exp2_rank8.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["rate_fit_model"] = "log(gap) versus epoch; exponential/linear-rate diagnostic for alpha=1/2"
        meta["rate_fits"] = {
            name: [fit_rate(run["records"]) for run in runs]
            for name, runs in results.items()
        }
        save_json(meta_path, meta)
        plot_results(results, args.figure_dir, "exp2_rank8", aggregation, frozen=frozen)
        return
    epochs = 3 if args.smoke_test else args.epochs
    aggregation = 1 if args.smoke_test else args.aggregation_epochs
    data = target_and_state()
    schedule = aggregate_schedule(epochs, aggregation)
    q = q_epoch()
    results = {name: [] for name in METHODS}
    initial_diagnostics = []
    frozen = []
    for repeat in range(args.repeats if not args.smoke_test else 1):
        state, init_diag = prepare_initial_state(data, repeat)
        initial_diagnostics.append(init_diag)
        frozen.append(frozen_diagnostic(state, data["X"], data["S"], DIRECTION_SEED + repeat))
        for name in METHODS:
            results[name].append(train_method(name, state, data["X"], data["S"], data["F_star"], epochs, schedule, aggregation, 10000 + repeat, q))
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"X": data["X"].numpy(), "S": data["S"].numpy(), "S_reference": data["S_reference"].numpy(), "F_star": np.asarray(data["F_star"])}
    for name, runs in results.items():
        for index, run in enumerate(runs):
            for key in ("raw_epoch", "display_epoch", "update_step", "gap", "output_gradient_norm", "output_norm", "parameter_norm", "cache_error_Y", "cache_error_Gram", "cache_memory_bytes", "feature_rank", "feature_sigma_min", "feature_sigma_max", "feature_condition"):
                arrays[f"{name}_repeat{index}_{key}"] = np.asarray([row[key] for row in run["records"]])
    np.savez_compressed(output_dir / "exp2_rank8.npz", **arrays)
    memory = {
        "X": data["X"].numel() * data["X"].element_size(),
        "S": data["S"].numel() * data["S"].element_size(),
        "full_output": N * D * data["X"].element_size(),
        "H_cache_Y": N * D * data["X"].element_size(),
        "H_cache_Gram": N * N * data["X"].element_size(),
    }
    meta = {
        "experiment": "exp2_rank8_gram_shared_mlp", "version": "final_v1",
        "N": N, "input_dim": INPUT_DIM, "hidden": HIDDEN, "d": D,
        "batch_size": BATCH_SIZE, "epochs": epochs, "aggregation_epochs": aggregation,
        "display_epoch_definition": "one displayed epoch equals aggregation_epochs raw permutation epochs",
        "output_scale": OUTPUT_SCALE, "base_eta": BASE_ETA, "eta": ETA,
        "eta_G": ETA, "eta_H": ETA, "eta_naive_raw": ETA, "eta_naive_q": ETA / q,
        "p_sample_inclusion": BATCH_SIZE / N, "q_epoch": q,
        "batch_sizes": [min(BATCH_SIZE, N - start) for start in range(0, N, BATCH_SIZE)],
        "repeats": args.repeats if not args.smoke_test else 1,
        "F_star": data["F_star"], "target_diagnostics": data["target_diagnostics"],
        "initialization_diagnostics": initial_diagnostics,
        "frozen_gradient_diagnostics": frozen,
        "rate_fit_model": "log(gap) versus epoch; exponential/linear-rate diagnostic for alpha=1/2",
        "rate_fits": {
            name: [fit_rate(run["records"]) for run in runs]
            for name, runs in results.items()
        },
        "threshold_epochs": {
            name: [threshold_epoch(
                [row["raw_epoch"] for row in run["records"]],
                [row["gap"] for row in run["records"]],
                GAP_THRESHOLD,
            ) for run in runs]
            for name, runs in results.items()
        },
        "theoretical_memory_bytes": theoretical_memory_bytes(memory),
        "memory_breakdown_bytes": memory,
        "per_method_efficiency": {
            name: [{key: run[key] for key in ("wall_time_seconds", "rss_before", "rss_after", "counters", "learning_rate")} for run in runs]
            for name, runs in results.items()
        },
        "interpretation": "Rank-8 shared-MLP estimator/cache comparison; empirical diagnostics do not prove the local KL assumptions.",
    }
    save_json(output_dir / "exp2_rank8.meta.json", meta)
    plot_results(results, args.figure_dir, "exp2_rank8", aggregation, frozen=frozen)


if __name__ == "__main__":
    main()
