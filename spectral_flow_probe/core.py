"""Core spectral mathematics — slope, PR, PCA pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

__all__ = [
    "spectral_slope",
    "compute_pr",
    "run_pca_layer",
    "LayerSpectral",
]


def spectral_slope(eigenvalues: np.ndarray, n_fit: int = 20) -> tuple[float, float]:
    """Log-linear spectral decay slope.

    Returns (slope, r_squared).  More negative slope = more anisotropic.
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[ev > 1e-12]
    n = min(n_fit, len(ev))
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.log(ev[:n])
    s, b = np.polyfit(x, y, 1)
    yhat = s * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return float(s), r2


def compute_pr(eigenvalues: np.ndarray) -> float:
    """Participation Ratio — effective number of dimensions."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[ev > 1e-12]
    if len(ev) == 0:
        return 0.0
    return float(np.sum(ev) ** 2 / np.sum(ev ** 2))


class LayerSpectral:
    """PCA result for one layer."""
    __slots__ = ("S", "r2", "pr", "pc01", "eigenvalues")

    def __init__(self, S: float, r2: float, pr: float, pc01: float,
                 eigenvalues: np.ndarray):
        self.S = S
        self.r2 = r2
        self.pr = pr
        self.pc01 = pc01
        self.eigenvalues = eigenvalues


def run_pca_layer(
    hidden_states: np.ndarray,
    n_components: int = 30,
    n_fit: int = 20,
) -> LayerSpectral | None:
    """Run PCA on (N, d) hidden states and compute spectral metrics."""
    N = hidden_states.shape[0]
    if N < 5:
        return None
    nc = min(n_components, N - 1, hidden_states.shape[1])
    if nc < 3:
        return None
    pca = PCA(n_components=nc)
    pca.fit(hidden_states)
    ev = pca.explained_variance_
    vr = pca.explained_variance_ratio_
    S, r2 = spectral_slope(ev, n_fit)
    pr = compute_pr(ev)
    pc01 = float(vr[0] / vr[1]) if len(vr) > 1 and vr[1] > 1e-10 else float("inf")
    return LayerSpectral(S=S, r2=r2, pr=pr, pc01=pc01, eigenvalues=ev[:n_components])
