"""
utils.py
--------
Helper utilities for the SVD steganography pipeline:
  • Quality metrics (PSNR, SSIM, BER)
  • Visualisation helpers (requires matplotlib)
  • File I/O for genomic matrices and stego matrices
  • Robustness / attack simulation
"""

import json
import os

import numpy as np

# Optional deps (graceful degradation)
try:
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


# --------------------------------------------------------------------------- #
#  Quality metrics
# --------------------------------------------------------------------------- #
def psnr(original: np.ndarray, modified: np.ndarray,
         max_val: float = 3.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    Higher is better.  Typical acceptable threshold ≥ 30 dB.
    *max_val* is the maximum possible pixel/base value (3 for A/T/G/C).
    """
    mse = np.mean((original.astype(float) - modified.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)


def ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Structural Similarity Index Measure (−1…1, higher is better).
    Returns NaN when skimage is not installed.
    """
    if not _SKIMAGE:
        return float('nan')
    data_range = float(original.max() - original.min()) or 1.0
    return float(skimage_ssim(original, modified, data_range=data_range))


def ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """Bit Error Rate between two bit arrays (0.0 = perfect recovery)."""
    n = min(len(original_bits), len(extracted_bits))
    if n == 0:
        return float('nan')
    return float(np.sum(original_bits[:n] != extracted_bits[:n]) / n)


def embedding_capacity_bits(matrix: np.ndarray,
                             start_sv: int = 2) -> int:
    """Maximum payload bits for a single-matrix SVD embedding."""
    min_dim = min(matrix.shape)
    return max(0, min_dim - start_sv - 32)   # subtract 32-bit header


def full_metrics(cover: np.ndarray,
                 stego: np.ndarray,
                 secret: str,
                 recovered: str) -> dict:
    """Compute and return all quality metrics in a dict."""
    import sys
    orig_bytes      = secret.encode('utf-8')
    rec_bytes       = recovered.encode('utf-8')
    n               = min(len(orig_bytes), len(rec_bytes))
    orig_bits       = np.unpackbits(np.frombuffer(orig_bytes, dtype=np.uint8))
    rec_bits        = np.unpackbits(np.frombuffer(rec_bytes[:n], dtype=np.uint8))

    return {
        'PSNR_dB'         : psnr(cover, stego),
        'SSIM'            : ssim(cover, stego),
        'BER'             : ber(orig_bits, rec_bits),
        'exact_match'     : secret == recovered,
        'payload_bytes'   : len(orig_bytes),
        'payload_chars'   : len(secret),
        'matrix_elements' : int(np.prod(cover.shape)),
        'embedding_rate'  : len(orig_bytes) * 8 / int(np.prod(cover.shape)),
    }


# --------------------------------------------------------------------------- #
#  Robustness tests (attack simulation)
# --------------------------------------------------------------------------- #
def attack_gaussian_noise(matrix: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to simulate transmission channel errors."""
    return matrix + np.random.normal(0, sigma, matrix.shape)


def attack_quantisation(matrix: np.ndarray, levels: int = 64) -> np.ndarray:
    """Simulate lossy quantisation (e.g. compression artefacts)."""
    mn, mx  = matrix.min(), matrix.max()
    span    = mx - mn if mx != mn else 1.0
    normed  = (matrix - mn) / span          # [0, 1]
    q       = np.round(normed * (levels - 1)) / (levels - 1)
    return q * span + mn


def attack_row_column_flip(matrix: np.ndarray) -> np.ndarray:
    """Flip a random row and column (mild geometric attack)."""
    m     = matrix.copy()
    r     = np.random.randint(0, m.shape[0])
    c     = np.random.randint(0, m.shape[1])
    m[r]  = m[r, ::-1]
    m[:, c] = m[::-1, c]
    return m


def attack_truncation(matrix: np.ndarray, keep: float = 0.95) -> np.ndarray:
    """SVD truncation (low-rank approximation — simulates heavy compression)."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    k        = max(1, int(len(S) * keep))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


# --------------------------------------------------------------------------- #
#  File I/O
# --------------------------------------------------------------------------- #
def save_matrix(matrix: np.ndarray, path: str) -> None:
    """Save a matrix to an .npy file."""
    np.save(path, matrix)
    print(f"Saved matrix → {path}  ({matrix.shape})")


def load_matrix(path: str) -> np.ndarray:
    """Load a matrix from an .npy file."""
    m = np.load(path)
    print(f"Loaded matrix ← {path}  ({m.shape})")
    return m


def save_metadata(meta: dict, path: str) -> None:
    """Save embedding metadata to JSON."""
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved metadata → {path}")


def load_metadata(path: str) -> dict:
    """Load embedding metadata from JSON."""
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Visualisation
# --------------------------------------------------------------------------- #
def plot_comparison(cover : np.ndarray,
                    stego : np.ndarray,
                    title : str = "Cover vs Stego Genomic Matrix",
                    save_path: str | None = None) -> None:
    """
    Side-by-side heatmap of cover and stego matrices plus difference map.
    Requires matplotlib.
    """
    if not _MPL:
        print("[visualisation skipped — pip install matplotlib]")
        return

    diff = np.abs(cover - stego)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for ax, data, lbl in zip(
        axes,
        [cover, stego, diff],
        ['Cover (original)', 'Stego (embedded)', 'Difference (×10)'],
    ):
        im = ax.imshow(data if lbl != 'Difference (×10)' else diff * 10,
                       cmap='viridis', aspect='auto', vmin=0, vmax=3)
        ax.set_title(lbl)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_singular_values(cover : np.ndarray,
                         stego : np.ndarray,
                         n_show: int = 40,
                         save_path: str | None = None) -> None:
    """
    Compare singular value spectra of cover and stego matrices.
    Useful for visualising the QIM modification extent.
    """
    if not _MPL:
        print("[visualisation skipped — pip install matplotlib]")
        return

    _, Sc, _ = np.linalg.svd(cover, full_matrices=False)
    _, Ss, _ = np.linalg.svd(stego, full_matrices=False)
    k        = min(n_show, len(Sc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(Sc[:k], 'b-o', ms=4, label='Cover σᵢ')
    axes[0].plot(Ss[:k], 'r--s', ms=4, label='Stego σᵢ')
    axes[0].set_title('Singular Value Spectra')
    axes[0].set_xlabel('Index i')
    axes[0].set_ylabel('σᵢ')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(k), np.abs(Sc[:k] - Ss[:k]), color='purple', alpha=0.7)
    axes[1].set_title('|σᵢ Cover − σᵢ Stego|')
    axes[1].set_xlabel('Index i')
    axes[1].set_ylabel('Absolute difference')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Singular value plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    from svd_steganography import embed, extract
    from genomic_encoder   import sequence_to_matrix

    seq   = 'ATGC' * 256
    cover = sequence_to_matrix(seq, rows=32, cols=32)
    stego, meta = embed(cover, "Demo patient data payload", delta=0.3)

    recovered = extract(stego, delta=meta['delta'], start_sv=meta['start_sv'])
    m         = full_metrics(cover, stego, "Demo patient data payload", recovered)

    print("\n=== Quality Metrics ===")
    for k, v in m.items():
        print(f"  {k:25s}: {v}")

    plot_comparison(cover, stego, save_path='/tmp/comparison.png')
    plot_singular_values(cover, stego, save_path='/tmp/sv_plot.png')