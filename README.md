# Numerical Experiments

This directory contains the code for the three numerical experiments reported in the paper: separable losses with two KL exponents, rank-eight Gram approximation, and pairwise coupling with balanced external forces.

Run the commands below from this directory.

## Dependencies

```bash
pip install -r requirements.txt
```

The experiments use NumPy, Matplotlib, and PyTorch. Install `psutil` optionally to record process RSS; otherwise RSS fields are saved as `null`.

All networks use smooth LeakyReLU activation and double-precision arithmetic. Each experiment uses five repeats. Within each repeat of Experiments 2 and 3, all methods share the initialization and random-permutation sequence.

## Experiment 1: Separable Losses with Two KL Exponents

```bash
python exp1_kl.py
```

### (a) MSE loss

Output: `fig/exp1_mse.eps`

Settings: MLP (4-64-64-2), N=10, batch size 1, lr=0.01, 1e5 data epochs, 5 repeats. The objective is `F = 0.5 * sum((prediction - target)^2)`, with output-space KL exponent 1/2 and F*=0.

### (b) Fourth-power loss

Output: `fig/exp1_fourth_power.eps`

Settings: same MLP, N=10, batch size 1, lr=0.001, 5e5 data epochs, 5 repeats. The objective is `F = 0.25 * sum((prediction - target)^4)`, with output-space KL exponent 3/4 and F*=0.

Both plots show the full individual trajectories and their arithmetic mean. The mean-curve fits use the last 60% of stored records for MSE and the last 20% for fourth-power loss. The fourth-power metadata also retains the 30%, 40%, and 60% fits. These are empirical tail diagnostics; the selected window does not truncate the displayed trajectories.

## Experiment 2: Rank-Eight Gram Approximation

```bash
python exp2_rank8.py
```

Output:
- `fig/exp2_rank8_summary.eps` — full objective gap and output-gradient norm for G / H / naive-raw / naive-q.
- `fig/exp2_rank8_cache.eps` — output-cache discrepancy for H.
- `fig/exp2_rank8_frozen.eps` — frozen parameter-gradient errors and directional cosines.
- `fig/exp2_rank8_rate.eps` — empirical exponential fits for G and H.

Settings: Gram objective `F(Y) = 0.5 * ||YY^T/d - S||_F^2`, N=200, d=8, MLP (32-256-8), output scale 16, batch size 16, lr=0.01/16^4, 15000 data epochs, 5 repeats.

The target is `S = UU^T/8 + mu * zz^T`, where z is a unit vector orthogonal to U and mu is half the smallest positive eigenvalue of `UU^T/8`. The target has rank nine, and the best rank-eight approximation has `F* = mu^2/2`. Each repeat starts from a 1% perturbation of an optimal output factor, realized by fitting the last network layer.

One plotted epoch represents ten data epochs. The rate fits use this same plotted epoch unit and the last 60% of stored records. The rate figure fits the mean gap, while metadata records the individual-repeat fits. The exact initial cache error is zero and is omitted on the logarithmic cache plot.

## Experiment 3: Pairwise Coupling with Balanced External Forces

```bash
python exp3_pairwise.py
```

Output:
- `fig/exp3_pairwise_total.eps` — full-range objective gaps for all four methods.
- `fig/exp3_pairwise_GH_rate.eps` — empirical log-log tail fits for G and H.
- `fig/exp3_pairwise_frozen.eps` — frozen parameter-gradient errors and directional cosines.

Settings: MLP (32-256-1), N=200, batch size 12, lr=0.025, 6000 data epochs, 5 repeats. The objective combines pairwise quadratic/quartic interactions, balanced external forces, and a quartic pair-average term, with beta=lambda=1 and A=0.03. Initial outputs equal one. The analytic optimum is y=0; along a common translation, `F(t*1) - F* = t^4/4`.

The script also generates `exp3_pairwise_GH_zoom.eps` and `exp3_pairwise_modes.eps` as auxiliary plots. The modes plot is not included in the manuscript. The G/H rate figure fits the mean gap over the last 60% of stored records; all epochs in this experiment are complete data epochs.

## Methods and Saved Results

G uses the current full-data output gradient. H uses cached outputs, then refreshes the current batch after the parameter update. For the Gram objective, H also refreshes the corresponding cached Gram rows and columns.

Naive-raw differentiates the full objective restricted to the current batch, retaining its normalization constants. Naive-q uses the same local gradient with learning rate `lr/q`, where `q = sum(b_l * (b_l - 1)) / (N * (N - 1))` and b_l are the actual batch sizes in one data epoch. Frozen naive-raw and naive-q directional cosines coincide because positive scalar rescaling does not change direction.

Each experiment saves per-repeat trajectories in `.npz` files and configuration, fit results, and timing/memory measurements in `.meta.json` files directly under `data/`. All PDF and EPS figures are saved directly under `fig/`, without experiment or job-ID subdirectories. Experiment-specific filenames keep the three sets of outputs distinct.

```text
data/
  exp1_mse.npz
  exp1_mse.meta.json
  exp1_fourth_power.npz
  exp1_fourth_power.meta.json
  exp2_rank8.npz
  exp2_rank8.meta.json
  exp3_pairwise.npz
  exp3_pairwise.meta.json
fig/
  exp1_mse.eps
  exp1_fourth_power.eps
  exp2_rank8_summary.eps
  ...
```

Running without `--plot-only` starts training and replaces any same-named result files and figures. Existing results are not automatically reused. To preserve the included results, choose separate output locations, for example:

```bash
python exp2_rank8.py --output-dir new_data --figure-dir new_fig
```

## Regenerate Figures Without Retraining

The saved results are included in `data/`. Regenerate the figures using:

```bash
python exp1_kl.py --plot-only
python exp2_rank8.py --plot-only
python exp3_pairwise.py --plot-only
```

These commands read `data/` and replace figures in `fig/` without training. For Experiments 1 and 2, postprocessing also refreshes the fit fields in the existing metadata; the saved `.npz` trajectories remain unchanged. Use `--output-dir` and `--figure-dir` to select other directories when needed.
