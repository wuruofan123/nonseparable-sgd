"""
Experiment 1 (Fourth-power): sublinear convergence under convexity (non-strongly-convex).
Reproduces Figure 1(b) in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
import os

# Hyperparameters matching the paper, Section 5.1
INPUT_DIM = 4
HIDDEN1 = 64
HIDDEN2 = 64
OUTPUT_DIM = 2
NUM_SAMPLES = 10
EPOCHS = 500_000
LEARNING_RATE = 0.001
RECORD_EVERY = 1000
N_REPEATS = 5

DATA_DIR = 'data'
FIG_DIR = 'fig'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


class SimpleDNN2:
    """Three-layer MLP with LeakyReLU activation."""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, alpha=0.01, rng=None):
        rng = rng or np.random
        self.W1 = rng.randn(input_dim, hidden_dim1) * 0.01
        self.b1 = np.zeros((1, hidden_dim1))
        self.W2 = rng.randn(hidden_dim1, hidden_dim2) * 0.01
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = rng.randn(hidden_dim2, output_dim) * 0.01
        self.b3 = np.zeros((1, output_dim))
        self.alpha = alpha

    def leaky_relu(self, x):
        return np.maximum(self.alpha * x, x)

    def leaky_relu_derivative(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        return dx

    def forward(self, X):
        self.hidden_layer1 = self.leaky_relu(np.dot(X, self.W1) + self.b1)
        self.hidden_layer2 = self.leaky_relu(np.dot(self.hidden_layer1, self.W2) + self.b2)
        self.output_layer = np.dot(self.hidden_layer2, self.W3) + self.b3
        return self.output_layer

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        d_output = (self.output_layer - y) ** 3
        d_hidden2 = np.dot(d_output, self.W3.T) * self.leaky_relu_derivative(self.hidden_layer2)
        d_hidden1 = np.dot(d_hidden2, self.W2.T) * self.leaky_relu_derivative(self.hidden_layer1)

        dW3 = np.dot(self.hidden_layer2.T, d_output) / m
        db3 = np.sum(d_output, axis=0, keepdims=True) / m
        dW2 = np.dot(self.hidden_layer1.T, d_hidden2) / m
        db2 = np.sum(d_hidden2, axis=0, keepdims=True) / m
        dW1 = np.dot(X.T, d_hidden1) / m
        db1 = np.sum(d_hidden1, axis=0, keepdims=True) / m

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


def run_single(seed=0):
    """Single training run with a fixed random seed."""
    rng = np.random.RandomState(seed)
    dnn = SimpleDNN2(INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM, rng=rng)
    X_train = rng.rand(NUM_SAMPLES, INPUT_DIM)
    y_train = rng.rand(NUM_SAMPLES, OUTPUT_DIM)

    steps, losses = [], []
    step = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for i in range(NUM_SAMPLES):
            X = X_train[i].reshape(1, -1)
            y = y_train[i].reshape(1, -1)
            out = dnn.forward(X)
            loss = np.sum((out - y) ** 4)
            epoch_loss += loss
            dnn.backward(X, y, LEARNING_RATE)
        avg_loss = epoch_loss / NUM_SAMPLES
        step += 1
        if step % RECORD_EVERY == 0 or step == 1:
            steps.append(step)
            losses.append(float(avg_loss))
    return np.array(steps), np.array(losses)


def repeat_runs(n_repeats=N_REPEATS):
    """Repeat training and return aggregated statistics."""
    runs = []
    steps = None
    for r in range(n_repeats):
        s, losses = run_single(seed=r)
        runs.append(losses)
        steps = s
    arr = np.vstack(runs)
    mean = arr.mean(axis=0)
    se_vals = sem(arr, axis=0)
    return steps, mean, se_vals, arr


def save_data(path, steps, mean, se, arr):
    np.savez(path, steps=steps, mean=mean, se=se, arr=arr)
    print(f"Data saved to {path}")


def load_data(path):
    data = np.load(path)
    return data['steps'], data['mean'], data['se'], data['arr']


def plot_figure(steps, errors, title="Fourth-power loss"):
    """Generate the log-log plot with a late-stage linear fit."""
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)

    valid_indices = (errors > 1e-10) & (steps > 0)
    steps_valid = steps[valid_indices]
    errors_valid = errors[valid_indices]
    log_k = np.log(steps_valid)
    log_errors = np.log(errors_valid)

    ax.loglog(steps_valid, errors_valid, 'g-', linewidth=2.0, label='Loss')
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True)

    if len(log_k) > 10:
        start_idx = int(len(log_k) * 2 / 3)
        x_fit = log_k[start_idx:].reshape(-1, 1)
        y_fit = log_errors[start_idx:]

        model = LinearRegression()
        model.fit(x_fit, y_fit)
        y_pred = model.predict(x_fit)

        ax.loglog(steps_valid[start_idx:], np.exp(y_pred), 'm--', linewidth=2.5, label='Fit')
        ax.legend(fontsize=16, loc='upper right')

        r_squared = model.score(x_fit, y_fit)
        print(f"{title} - Log-log fit R^2 = {r_squared:.6f}")
        print(f"{title} - Slope (convergence order) = {model.coef_[0]:.6f}")

        if abs(model.coef_[0] + 1) < 0.2 and r_squared > 0.9:
            print(f"{title} exhibits O(1/k) sublinear convergence (slope ~ -1)")
        elif model.coef_[0] < 0:
            print(f"{title} exhibits polynomial convergence with order ~ {-model.coef_[0]:.3f}")

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.90)
    plt.savefig(os.path.join(FIG_DIR, '4power.eps'), format='eps', dpi=300)
    plt.close()


if __name__ == "__main__":
    data_path = os.path.join(DATA_DIR, 'toy.npz')

    if os.path.exists(data_path):
        steps, mean, se, arr = load_data(data_path)
    else:
        steps, mean, se, arr = repeat_runs()
        print(f"Training completed. Records: {len(steps)} points.")
        save_data(data_path, steps, mean, se, arr)

    plot_figure(steps, mean, "Fourth-power loss")
