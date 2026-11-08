"""KL-rate calibration experiment for MSE and fourth-power losses."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import fit_exponential, fit_power_law, rss_bytes, save_figure, save_json

INPUT_DIM = 4
HIDDEN1 = 64
HIDDEN2 = 64
OUTPUT_DIM = 2
NUM_SAMPLES = 10
MSE_EPOCHS = 100_000
FOURTH_EPOCHS = 500_000
MSE_ETA = 0.01
FOURTH_ETA = 0.001
RECORD_EVERY_MSE = 100
RECORD_EVERY_FOURTH = 1000
N_REPEATS = 5


class SmoothLeaky:
    slope = 0.01
    temperature = 0.02

    @classmethod
    def value(cls, x):
        z = x / cls.temperature
        return cls.slope * x + (1.0 - cls.slope) * cls.temperature * np.logaddexp(0.0, z)

    @classmethod
    def derivative(cls, x):
        z = x / cls.temperature
        sigmoid = np.empty_like(z, dtype=float)
        positive = z >= 0.0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        negative = ~positive
        exp_z = np.exp(z[negative])
        sigmoid[negative] = exp_z / (1.0 + exp_z)
        return cls.slope + (1.0 - cls.slope) * sigmoid


class MLP:
    def __init__(self, rng):
        self.W1 = rng.standard_normal((INPUT_DIM, HIDDEN1)) * 0.01
        self.b1 = np.zeros(HIDDEN1)
        self.W2 = rng.standard_normal((HIDDEN1, HIDDEN2)) * 0.01
        self.b2 = np.zeros(HIDDEN2)
        self.W3 = rng.standard_normal((HIDDEN2, OUTPUT_DIM)) * 0.01
        self.b3 = np.zeros(OUTPUT_DIM)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.h1 = SmoothLeaky.value(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = SmoothLeaky.value(self.z2)
        return self.h2 @ self.W3 + self.b3

    def backward(self, X, output_grad):
        dz2 = (output_grad @ self.W3.T) * SmoothLeaky.derivative(self.z2)
        dz1 = (dz2 @ self.W2.T) * SmoothLeaky.derivative(self.z1)
        grads = {
            "W3": self.h2.T @ output_grad,
            "b3": output_grad.sum(axis=0),
            "W2": self.h1.T @ dz2,
            "b2": dz2.sum(axis=0),
            "W1": X.T @ dz1,
            "b1": dz1.sum(axis=0),
        }
        return grads

    def step(self, grads, eta):
        for name, grad in grads.items():
            setattr(self, name, getattr(self, name) - eta * grad)

    @staticmethod
    def grad_norm(grads):
        return float(np.sqrt(sum(np.sum(value * value) for value in grads.values())))


def loss_and_grad(output, target, kind):
    residual = output - target
    if kind == "mse":
        return 0.5 * np.sum(residual * residual), residual
    if kind == "fourth_power":
        return 0.25 * np.sum(residual ** 4), residual ** 3
    raise ValueError(kind)


def full_metrics(model, X, target, kind):
    output = model.forward(X)
    loss, output_grad = loss_and_grad(output, target, kind)
    grads = model.backward(X, output_grad)
    return {
        "loss": float(loss),
        "gap": float(loss),
        "output_gradient_norm": float(np.linalg.norm(output_grad)),
        "parameter_gradient_norm": model.grad_norm(grads),
        "parameter_norm": float(np.sqrt(sum(np.sum(value * value) for value in (
            model.W1, model.b1, model.W2, model.b2, model.W3, model.b3
        )))),
    }


def run_single(kind, epochs, eta, record_every, seed):
    rng = np.random.default_rng(seed)
    X = rng.random((NUM_SAMPLES, INPUT_DIM))
    target = rng.random((NUM_SAMPLES, OUTPUT_DIM))
    model = MLP(rng)
    records = []
    update_norm = 0.0
    next_record = 0
    start = time.perf_counter()
    initial_rss = rss_bytes()
    for epoch in range(epochs + 1):
        if epoch == next_record or epoch == epochs:
            metrics = full_metrics(model, X, target, kind)
            metrics["epoch"] = epoch
            metrics["update_norm_since_record"] = update_norm
            records.append(metrics)
            update_norm = 0.0
            next_record = epoch + record_every
            if epoch == epochs:
                break
        permutation = rng.permutation(NUM_SAMPLES)
        for index in permutation:
            output = model.forward(X[index:index + 1])
            _, output_grad = loss_and_grad(output, target[index:index + 1], kind)
            grads = model.backward(X[index:index + 1], output_grad)
            update_norm += eta * model.grad_norm(grads)
            model.step(grads, eta)
    final_rss = rss_bytes()
    return {
        "seed": seed,
        "records": records,
        "wall_time_seconds": time.perf_counter() - start,
        "rss_before": initial_rss,
        "rss_after": final_rss,
        "target": target,
        "X": X,
    }


def analytic_check():
    residual = np.asarray([[-0.3, 0.2]])
    mse_loss, mse_grad = loss_and_grad(residual, np.zeros_like(residual), "mse")
    fourth_loss, fourth_grad = loss_and_grad(residual, np.zeros_like(residual), "fourth_power")
    assert np.allclose(mse_grad, residual)
    assert np.allclose(fourth_grad, residual ** 3)
    assert mse_loss > 0 and fourth_loss > 0
    return {"status": "passed", "mse_gradient": mse_grad.tolist(), "fourth_gradient": fourth_grad.tolist()}


def save_experiment(kind, runs, output_dir, epochs, eta, record_every):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        str(run["seed"]): {
            key: np.asarray([row[key] for row in run["records"]])
            for key in run["records"][0]
        }
        for run in runs
    }
    arrays = {f"seed{seed}_{key}": value for seed, data in records.items() for key, value in data.items()}
    for run in runs:
        arrays[f"seed{run['seed']}_X"] = run["X"]
        arrays[f"seed{run['seed']}_target"] = run["target"]
    np.savez_compressed(output_dir / f"exp1_{kind}.npz", **arrays)
    fit = {}
    selected_fraction = 0.6 if kind == "mse" else 0.2
    for seed, data in records.items():
        fit[seed] = (
            fit_exponential(data["epoch"], data["gap"], fractions=(0.3, 0.4, 0.6), selected_fraction=selected_fraction)
            if kind == "mse" else
            fit_power_law(data["epoch"], data["gap"], fractions=(0.2, 0.3, 0.4, 0.6), floor=1e-14, selected_fraction=selected_fraction)
        )
    meta = {
        "experiment": "exp1_kl_rate_calibration",
        "kind": kind,
        "kl_exponent": 0.5 if kind == "mse" else 0.75,
        "expected_output_space_rate": "linear/exponential" if kind == "mse" else "polynomial k^-2 upper-bound regime",
        "N": NUM_SAMPLES,
        "input_dim": INPUT_DIM,
        "hidden": [HIDDEN1, HIDDEN2],
        "output_dim": OUTPUT_DIM,
        "epochs": epochs,
        "eta": eta,
        "record_every": record_every,
        "sample_order": "independent random permutation per data epoch",
        "repeats": [run["seed"] for run in runs],
        "f_star": 0.0,
        "activation": "SmoothLeakyReLU",
        "analytic_check": analytic_check(),
        "fit_results_per_seed": fit,
        "wall_time_seconds_per_seed": {str(run["seed"]): run["wall_time_seconds"] for run in runs},
        "rss_before_per_seed": {str(run["seed"]): run["rss_before"] for run in runs},
        "rss_after_per_seed": {str(run["seed"]): run["rss_after"] for run in runs},
        "interpretation": "Output-space KL calibration; not a proof of the full parameterized stochastic rate.",
    }
    save_json(output_dir / f"exp1_{kind}.meta.json", meta)


def plot_experiment(kind, output_dir, figure_dir):
    output_dir, figure_dir = Path(output_dir), Path(figure_dir)
    meta = json.loads((output_dir / f"exp1_{kind}.meta.json").read_text(encoding="utf-8"))
    seeds = [str(seed) for seed in meta["repeats"]]
    series = [
        np.load(output_dir / f"exp1_{kind}.npz")[f"seed{seed}_gap"]
        for seed in seeds
    ]
    data = np.load(output_dir / f"exp1_{kind}.npz")
    epochs = data[f"seed{seeds[0]}_epoch"]
    values = np.vstack(series)
    mean = values.mean(axis=0)
    selected_fraction = 0.6 if kind == "mse" else 0.2
    selected_fit = fit_power_law(epochs, mean, fractions=(0.2, 0.3, 0.4, 0.6), floor=1e-14, selected_fraction=selected_fraction) if kind != "mse" else fit_exponential(epochs, mean, fractions=(0.3, 0.4, 0.6), selected_fraction=selected_fraction)
    selected = selected_fit["selected"]
    fit_valid = np.isfinite(epochs) & np.isfinite(mean) & (epochs > 0) & (mean > 1e-14 if kind != "mse" else mean > 0)
    start = selected["fit_start_index"]
    fit_valid[:start] = False
    plot_valid = np.isfinite(epochs) & (epochs > 0)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for values_one in values:
        if kind == "mse":
            axis.semilogy(epochs, np.maximum(values_one, 1e-30), color="#a6c8e3", linewidth=0.8)
        else:
            axis.loglog(epochs[plot_valid], np.maximum(values_one[plot_valid], 1e-30), color="#a6c8e3", linewidth=0.8)
    if kind == "mse":
        axis.semilogy(epochs, np.maximum(mean, 1e-30), color="#1f77b4", linestyle="-", linewidth=2, label="mean gap")
    else:
        axis.loglog(epochs[plot_valid], np.maximum(mean[plot_valid], 1e-30), color="#1f77b4", linestyle="-", linewidth=2, label="mean gap")
    if selected["status"] == "ok":
        fit_epochs = epochs[fit_valid]
        if kind == "mse":
            fitted = np.exp(selected["intercept"] + selected["slope"] * fit_epochs)
            axis.semilogy(fit_epochs, fitted, color="#ff7f0e", linestyle="--", linewidth=1.5,
                          label=f"Mean empirical fit (last {selected_fraction:.0%}): slope={selected['slope']:.3g}, $R^2$={selected['r_squared']:.3f}")
        else:
            fitted = np.exp(selected["intercept"] + selected["slope"] * np.log(fit_epochs))
            axis.loglog(fit_epochs, fitted, color="#ff7f0e", linestyle="--", linewidth=1.5,
                        label=f"Mean empirical fit (last {selected_fraction:.0%}): slope={selected['slope']:.3g}, $R^2$={selected['r_squared']:.3f}")
    axis.set_xlabel("Data epoch")
    axis.set_ylabel(r"$F-F^*$")
    axis.set_title(f"Exp1 {kind}: KL rate calibration")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)
    axis.legend()
    fig.tight_layout()
    path = figure_dir / f"exp1_{kind}.pdf"
    save_figure(fig, path)
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--figure-dir", default="fig")
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        for kind in ("mse", "fourth_power"):
            meta_path = Path(args.output_dir) / f"exp1_{kind}.meta.json"
            data_path = Path(args.output_dir) / f"exp1_{kind}.npz"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            data = np.load(data_path)
            selected_fraction = 0.6 if kind == "mse" else 0.2
            fit = {}
            for seed in meta["repeats"]:
                fit[str(seed)] = (
                    fit_exponential(data[f"seed{seed}_epoch"], data[f"seed{seed}_gap"], fractions=(0.3, 0.4, 0.6), selected_fraction=selected_fraction)
                    if kind == "mse" else
                    fit_power_law(data[f"seed{seed}_epoch"], data[f"seed{seed}_gap"], fractions=(0.2, 0.3, 0.4, 0.6), floor=1e-14, selected_fraction=selected_fraction)
                )
            meta["fit_results_per_seed"] = fit
            save_json(meta_path, meta)
            plot_experiment(kind, args.output_dir, args.figure_dir)
        return
    if args.smoke_test:
        settings = (("mse", 3, MSE_ETA, 1), ("fourth_power", 3, FOURTH_ETA, 1))
    else:
        settings = (("mse", MSE_EPOCHS, MSE_ETA, RECORD_EVERY_MSE), ("fourth_power", FOURTH_EPOCHS, FOURTH_ETA, RECORD_EVERY_FOURTH))
    for kind, epochs, eta, record_every in settings:
        runs = [run_single(kind, epochs, eta, record_every, seed) for seed in range(args.repeats)]
        save_experiment(kind, runs, args.output_dir, epochs, eta, record_every)
        plot_experiment(kind, args.output_dir, args.figure_dir)


if __name__ == "__main__":
    main()
