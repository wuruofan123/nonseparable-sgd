"""Small shared utilities for the final numerical experiments."""

import json
import math
import os
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


METHODS = ("G", "H", "naive_raw", "naive_q")


def rss_bytes():
    if psutil is None:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path, value):
    Path(path).write_text(
        json.dumps(json_ready(value), indent=2), encoding="utf-8"
    )


def save_figure(figure, path):
    path = Path(path)
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".eps"), format="eps")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def schedule_epochs(total, first=(1, 2, 5, 10, 20), interval=100):
    values = {0, total}
    values.update(value for value in first if value <= total)
    values.update(range(interval, total + 1, interval))
    return np.asarray(sorted(values), dtype=int)


def aggregate_schedule(total, aggregation_epochs=10):
    values = {0, total, 1}
    values.update(range(aggregation_epochs, total + 1, aggregation_epochs))
    return np.asarray(sorted(value for value in values if value <= total), dtype=int)


def fit_exponential(x, y, fractions=(0.3, 0.4, 0.6), floor=1e-14, selected_fraction=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fractions = tuple(float(value) for value in fractions)
    selected_fraction = fractions[-1] if selected_fraction is None else float(selected_fraction)
    if selected_fraction not in fractions:
        raise ValueError("selected_fraction must be one of fractions")
    result = {"fractions": list(fractions), "floor": floor}
    for fraction in fractions:
        start = max(0, int(math.ceil(len(x) * (1.0 - fraction))))
        valid = np.isfinite(x[start:]) & np.isfinite(y[start:])
        valid &= (x[start:] > 0) & (y[start:] > floor)
        xv, yv = x[start:][valid], y[start:][valid]
        item = {"fraction": float(fraction), "fit_start_index": int(start)}
        if len(xv) < 3:
            item.update({"status": "insufficient_valid_points", "valid_points": int(len(xv))})
        else:
            ly = np.log(yv)
            slope, intercept = np.polyfit(xv, ly, 1)
            residual = ly - (intercept + slope * xv)
            total = np.sum((ly - ly.mean()) ** 2)
            r2 = 1.0 - np.sum(residual ** 2) / total if total > 0 else None
            item.update({"status": "ok", "slope": float(slope), "intercept": float(intercept),
                         "r_squared": None if r2 is None else float(r2),
                         "valid_points": int(len(xv))})
        result[str(fraction)] = item
    result["selected_fraction"] = float(selected_fraction)
    result["selected"] = result[str(selected_fraction)]
    return result


def fit_power_law(x, y, fractions=(0.3, 0.4, 0.6), floor=1e-14, selected_fraction=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fractions = tuple(float(value) for value in fractions)
    selected_fraction = fractions[-1] if selected_fraction is None else float(selected_fraction)
    if selected_fraction not in fractions:
        raise ValueError("selected_fraction must be one of fractions")
    result = {"fractions": list(fractions), "floor": floor}
    for fraction in fractions:
        start = max(0, int(math.ceil(len(x) * (1.0 - fraction))))
        valid = np.isfinite(x[start:]) & np.isfinite(y[start:])
        valid &= (x[start:] > 0) & (y[start:] > floor)
        xv, yv = x[start:][valid], y[start:][valid]
        item = {"fraction": float(fraction), "fit_start_index": int(start)}
        if len(xv) < 3:
            item.update({"status": "insufficient_valid_points", "valid_points": int(len(xv))})
        else:
            lx, ly = np.log(xv), np.log(yv)
            slope, intercept = np.polyfit(lx, ly, 1)
            residual = ly - (intercept + slope * lx)
            total = np.sum((ly - ly.mean()) ** 2)
            r2 = 1.0 - np.sum(residual ** 2) / total if total > 0 else None
            item.update({
                "status": "ok", "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": None if r2 is None else float(r2),
                "valid_points": int(len(xv)),
            })
        result[str(fraction)] = item
    result["selected_fraction"] = float(selected_fraction)
    result["selected"] = result[str(selected_fraction)]
    return result


def threshold_epoch(epochs, gaps, threshold):
    epochs = np.asarray(epochs)
    gaps = np.asarray(gaps)
    valid = np.flatnonzero(np.isfinite(gaps) & (gaps <= threshold))
    return None if len(valid) == 0 else int(epochs[valid[0]])


def theoretical_memory_bytes(items):
    return int(sum(int(size) for size in items.values()))
