# Numerical Experiments

This directory contains the code to reproduce the numerical experiments reported in Section 5 of the paper.

## Dependencies

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, NumPy, Matplotlib, SciPy, scikit-learn, PyTorch.

## Experiment 1: Convergence Rates under Different Convexity Assumptions

Reproduces Figure 1.

### (a) MSE loss — linear convergence

```bash
python exp1_mse.py
```

Output: `fig/mse.eps`

Settings: 3-layer MLP (4-64-64-2), N=10, lr=0.01, 1e5 epochs, 5 repeats.

### (b) Fourth-power loss — sublinear convergence

```bash
python exp1_fourth_power.py
```

Output: `fig/4power.eps`

Settings: same MLP, N=10, lr=0.001, 5e5 epochs, 5 repeats.

## Experiment 2: Gram Objective

Reproduces Figures 2 and 3.

```bash
python exp2_gram.py
```

Output:
- `fig/gram.eps` — full objective gap and gradient norm for G / H / naive.
- `fig/F_GH_loglog_combined.eps` — log-log diagnostics for G and H.

Settings: Gram objective F(Y)=0.5||YY^T/d - S||_F^2, d=8, Y in R^{200x8}, MLP (32-128-8), batch size 16, 5e4 epochs, 5 repeats.

The diagonal entries of the target Gram matrix S are scaled by 1.05 after computing the reference initialization, so that the optimization starts from a non-trivial neighborhood rather than extremely close to the optimum.

The script automatically trains all methods, caches results under `data/`, and generates the figures. To re-use cached data without retraining, the script loads existing `.npz` files when present.
