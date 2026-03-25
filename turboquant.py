"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Implementation based on the paper by Zandieh et al. (2025).
Core algorithms:
  - TurboQuant_mse: MSE-optimized vector quantizer (Algorithm 1)
  - TurboQuant_prod: Inner-product-optimized vector quantizer (Algorithm 2)
"""

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Beta distribution on the hypersphere (Lemma 1)
# ---------------------------------------------------------------------------

def beta_pdf(x, d):
    """PDF of a coordinate of a uniformly random point on S^{d-1}.
    Uses log-gamma for numerical stability at high dimensions."""
    x = np.asarray(x, dtype=np.float64)
    log_coeff = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    # For high dimensions, use Gaussian approximation to avoid (1-x^2)^large
    if d > 300:
        # In high d, fX -> N(0, 1/d)
        return np.sqrt(d / (2 * np.pi)) * np.exp(-0.5 * d * x**2)
    log_pdf = log_coeff + ((d - 3) / 2) * np.log(np.maximum(1 - x**2, 1e-300))
    return np.exp(log_pdf)


# ---------------------------------------------------------------------------
# Lloyd-Max optimal scalar quantizer for the Beta distribution
# ---------------------------------------------------------------------------

def lloyd_max_quantizer(d, b, max_iter=200, tol=1e-12):
    """
    Compute optimal Lloyd-Max centroids for b-bit scalar quantization
    of the Beta distribution induced by dimension d.

    Returns:
        centroids: sorted array of 2^b centroids in [-1, 1]
    """
    n_levels = 2 ** b

    # Adaptive grid range: in high dimensions, coordinates concentrate near 0
    # with std ~ 1/sqrt(d). Use ~5 sigma range.
    sigma = 1.0 / np.sqrt(d)
    grid_range = min(1.0 - 1e-10, 5.0 * sigma)

    # Initialize centroids within the effective range
    centroids = np.linspace(-grid_range, grid_range, n_levels + 2)[1:-1]

    # Fine grid for numerical integration
    grid = np.linspace(-grid_range, grid_range, 50000)
    pdf_vals = beta_pdf(grid, d)
    dx = grid[1] - grid[0]

    for _ in range(max_iter):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = np.concatenate([[-grid_range - 1e-10], (centroids[:-1] + centroids[1:]) / 2, [grid_range + 1e-10]])

        # Update centroids: weighted mean within each partition
        new_centroids = np.zeros_like(centroids)
        for i in range(n_levels):
            mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
            if i == n_levels - 1:
                mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            weighted_sum = np.sum(grid[mask] * pdf_vals[mask]) * dx
            weight_total = np.sum(pdf_vals[mask]) * dx
            if weight_total > 1e-15:
                new_centroids[i] = weighted_sum / weight_total
            else:
                new_centroids[i] = centroids[i]

        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    return np.sort(centroids)


# ---------------------------------------------------------------------------
# Codebook cache
# ---------------------------------------------------------------------------

_codebook_cache = {}

def get_codebook(d, b):
    """Get (or compute and cache) the optimal codebook for dimension d, bit-width b."""
    key = (d, b)
    if key not in _codebook_cache:
        _codebook_cache[key] = lloyd_max_quantizer(d, b)
    return _codebook_cache[key]


# ---------------------------------------------------------------------------
# TurboQuant_mse  (Algorithm 1)
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """MSE-optimized vector quantizer."""

    def __init__(self, d, b, seed=42):
        self.d = d
        self.b = b
        self.rng = np.random.RandomState(seed)
        # Random rotation via QR decomposition of a random Gaussian matrix
        G = self.rng.randn(d, d)
        self.Pi, _ = np.linalg.qr(G)
        # Precompute codebook
        self.centroids = get_codebook(d, b)

    def quantize(self, x):
        """
        Quantize a vector x (or batch of vectors, shape (n, d)).
        Returns index array of shape (n, d) with b-bit integers.
        """
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]

        # Rotate
        y = x @ self.Pi.T  # (n, d)

        # Find nearest centroid for each coordinate
        # centroids shape: (2^b,)
        diffs = np.abs(y[:, :, np.newaxis] - self.centroids[np.newaxis, np.newaxis, :])
        idx = np.argmin(diffs, axis=2)  # (n, d)

        if single:
            return idx[0]
        return idx

    def dequantize(self, idx):
        """
        Dequantize index array back to vectors.
        idx: shape (n, d) or (d,)
        """
        single = (idx.ndim == 1)
        if single:
            idx = idx[np.newaxis, :]

        y_hat = self.centroids[idx]  # (n, d)
        x_hat = y_hat @ self.Pi  # rotate back

        if single:
            return x_hat[0]
        return x_hat

    def quantize_dequantize(self, x):
        """Quantize and immediately dequantize."""
        return self.dequantize(self.quantize(x))


# ---------------------------------------------------------------------------
# QJL: 1-bit inner product quantizer (Definition 1)
# ---------------------------------------------------------------------------

class QJL:
    """Quantized Johnson-Lindenstrauss 1-bit quantizer."""

    def __init__(self, d, seed=123):
        self.d = d
        self.rng = np.random.RandomState(seed)
        self.S = self.rng.randn(d, d)

    def quantize(self, x):
        """Returns sign vector in {-1, +1}^d."""
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]
        z = np.sign(x @ self.S.T)  # (n, d)
        z[z == 0] = 1.0
        if single:
            return z[0]
        return z

    def dequantize(self, z, gamma=1.0):
        """
        Dequantize sign vector.
        gamma: L2 norm of the original (residual) vector.
        """
        single = (z.ndim == 1)
        if single:
            z = z[np.newaxis, :]
            gamma = np.array([gamma])

        scale = np.sqrt(np.pi / 2) / self.d
        x_hat = scale * (z @ self.S) * gamma[:, np.newaxis]  # (n, d)

        if single:
            return x_hat[0]
        return x_hat


# ---------------------------------------------------------------------------
# TurboQuant_prod  (Algorithm 2)
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """Inner-product-optimized vector quantizer (unbiased).

    For b >= 2: uses (b-1)-bit MSE quantizer + 1-bit QJL on residual.
    For b == 1: uses pure QJL (no MSE stage).
    """

    def __init__(self, d, b, seed=42):
        self.d = d
        self.b = b
        self.use_mse = (b >= 2)
        if self.use_mse:
            self.mse_quant = TurboQuantMSE(d, b - 1, seed=seed)
        self.qjl = QJL(d, seed=seed + 1000)

    def quantize(self, x):
        """
        Returns (idx, qjl_signs, residual_norms).
        idx is None when b=1 (pure QJL).
        """
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]

        if self.use_mse:
            idx = self.mse_quant.quantize(x)
            x_hat_mse = self.mse_quant.dequantize(idx)
            residual = x - x_hat_mse
        else:
            idx = None
            residual = x

        residual_norms = np.linalg.norm(residual, axis=1)
        safe_norms = np.maximum(residual_norms, 1e-15)
        residual_normalized = residual / safe_norms[:, np.newaxis]
        qjl_signs = self.qjl.quantize(residual_normalized)

        if single:
            return (idx[0] if idx is not None else None), qjl_signs[0], residual_norms[0]
        return idx, qjl_signs, residual_norms

    def dequantize(self, idx, qjl_signs, residual_norms):
        """Dequantize to get unbiased inner product estimator."""
        single = (qjl_signs.ndim == 1)
        if single:
            qjl_signs = qjl_signs[np.newaxis, :]
            residual_norms = np.array([residual_norms])
            if idx is not None:
                idx = idx[np.newaxis, :]

        x_hat_qjl = self.qjl.dequantize(qjl_signs, residual_norms)

        if self.use_mse and idx is not None:
            x_hat_mse = self.mse_quant.dequantize(idx)
            x_hat = x_hat_mse + x_hat_qjl
        else:
            x_hat = x_hat_qjl

        if single:
            return x_hat[0]
        return x_hat

    def quantize_dequantize(self, x):
        """Quantize and immediately dequantize."""
        idx, qjl_signs, norms = self.quantize(x)
        return self.dequantize(idx, qjl_signs, norms)


