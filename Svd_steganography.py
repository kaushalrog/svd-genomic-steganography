"""
svd_steganography.py
--------------------
Core engine for SVD-based steganography over genomic matrices.

Algorithm — Quantisation Index Modulation (QIM) on singular values:
    embed   : For each bit b, modify σᵢ so that (σᵢ // Δ) % 2 == b.
    extract : Read back (σᵢ // Δ) % 2 for every embedded position.

This is robust to small floating-point perturbations and mild lossy
compression, unlike plain LSB substitution.
"""

import numpy as np
from typing import Tuple


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #
def _text_to_bits(text: str) -> np.ndarray:
    """UTF-8 text → flat array of bits (uint8, MSB-first per byte)."""
    byte_arr = text.encode('utf-8')
    bits = np.unpackbits(np.frombuffer(byte_arr, dtype=np.uint8))
    return bits


def _bits_to_text(bits: np.ndarray) -> str:
    """Flat bit array → UTF-8 text.  Drops partial trailing bytes."""
    n_bytes = len(bits) // 8
    bits    = bits[:n_bytes * 8]
    byte_arr = np.packbits(bits)
    return byte_arr.tobytes().decode('utf-8', errors='replace')


def _embed_bits_qim(singular_values: np.ndarray,
                    bits          : np.ndarray,
                    delta         : float,
                    start         : int = 0) -> np.ndarray:
    """
    Embed *bits* into a copy of *singular_values* using QIM.

    Parameters
    ----------
    singular_values : 1-D array σ from np.linalg.svd().
    bits            : Bit array to embed.
    delta           : Quantisation step (larger ⟹ more robust, less fidelity).
    start           : First singular value index to use.

    Returns
    -------
    Modified singular value array.
    """
    sv   = singular_values.copy()
    n    = len(bits)
    cap  = len(sv) - start

    if n > cap:
        raise ValueError(
            f"Payload too large: need {n} singular values but only {cap} "
            f"are available (start={start})."
        )

    for i, bit in enumerate(bits):
        idx  = start + i
        q    = sv[idx] // delta        # quantisation index
        parity = int(q) % 2
        if parity != int(bit):
            # Flip to nearest valid quantisation level
            if bit == 0:
                sv[idx] = (int(q) - (parity - 0)) * delta
            else:
                sv[idx] = (int(q) + 1) * delta
            # Safety: keep positive
            if sv[idx] < 0:
                sv[idx] = delta

    return sv


def _extract_bits_qim(singular_values: np.ndarray,
                      n_bits          : int,
                      delta           : float,
                      start           : int = 0) -> np.ndarray:
    """Extract *n_bits* bits from singular values using QIM."""
    bits = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        q       = singular_values[start + i] // delta
        bits[i] = int(q) % 2
    return bits


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def embed(cover_matrix: np.ndarray,
          secret_text  : str,
          delta        : float = 0.1,
          start_sv     : int   = 2) -> Tuple[np.ndarray, dict]:
    """
    Embed *secret_text* into *cover_matrix* via SVD–QIM.

    Parameters
    ----------
    cover_matrix : 2-D float array (genomic matrix).
    secret_text  : Patient data / any UTF-8 string.
    delta        : QIM step size.  Typical range: 0.05–0.5.
    start_sv     : Skip the first *start_sv* singular values (they carry the
                   most energy and are perceptually most important).

    Returns
    -------
    stego_matrix : Modified matrix with hidden data.
    metadata     : Dict with embedding parameters needed for extraction.
    """
    if cover_matrix.ndim != 2:
        raise ValueError("cover_matrix must be 2-D.")

    # Prepend a 32-bit length header so extraction knows how many bits to read
    bits_payload = _text_to_bits(secret_text)
    length_header = np.unpackbits(
        np.array([len(bits_payload)], dtype=np.uint32).view(np.uint8)
    )
    bits_total = np.concatenate([length_header, bits_payload])

    # SVD decomposition
    U, S, Vt = np.linalg.svd(cover_matrix, full_matrices=True)

    # Capacity check
    capacity = len(S) - start_sv
    if len(bits_total) > capacity:
        raise ValueError(
            f"Payload ({len(bits_total)} bits) exceeds matrix capacity "
            f"({capacity} bits). Use a larger matrix or increase block count."
        )

    # Modify singular values
    S_modified = _embed_bits_qim(S, bits_total, delta, start=start_sv)

    # Reconstruct stego matrix
    stego_matrix = U @ np.diag(S_modified) @ Vt

    # Compute quality metrics
    mse  = float(np.mean((cover_matrix - stego_matrix) ** 2))
    psnr = float(10 * np.log10((3.0 ** 2) / mse)) if mse > 0 else float('inf')

    metadata = {
        'delta'       : delta,
        'start_sv'    : start_sv,
        'n_bits_total': len(bits_total),
        'text_length' : len(secret_text),
        'matrix_shape': cover_matrix.shape,
        'mse'         : mse,
        'psnr_dB'     : psnr,
        'capacity_bits': capacity,
    }

    return stego_matrix, metadata


def extract(stego_matrix: np.ndarray,
            delta        : float,
            start_sv     : int = 2) -> str:
    """
    Extract hidden text from a stego genomic matrix.

    Parameters
    ----------
    stego_matrix : Matrix produced by embed().
    delta        : Must match the value used during embedding.
    start_sv     : Must match the value used during embedding.

    Returns
    -------
    Recovered secret text string.
    """
    if stego_matrix.ndim != 2:
        raise ValueError("stego_matrix must be 2-D.")

    _, S, _ = np.linalg.svd(stego_matrix, full_matrices=True)

    # Read the 32-bit length header first
    header_bits = _extract_bits_qim(S, n_bits=32, delta=delta, start=start_sv)
    payload_len = int(np.packbits(header_bits).view(np.uint32)[0])

    # Sanity-check
    available = len(S) - start_sv - 32
    if payload_len > available:
        raise ValueError(
            f"Extracted payload length {payload_len} exceeds available capacity "
            f"{available}. Matrix may be corrupted or parameters incorrect."
        )

    payload_bits = _extract_bits_qim(
        S, n_bits=payload_len, delta=delta, start=start_sv + 32
    )

    return _bits_to_text(payload_bits)


# --------------------------------------------------------------------------- #
#  Block-SVD (for long sequences that exceed single-matrix capacity)
# --------------------------------------------------------------------------- #
def embed_blocks(cover_matrix: np.ndarray,
                 secret_text  : str,
                 block_size   : int   = 16,
                 delta        : float = 0.1,
                 start_sv     : int   = 2) -> Tuple[np.ndarray, dict]:
    """
    Tile the cover matrix into (block_size × block_size) blocks and
    distribute the payload across all blocks.

    Useful when the payload is too large for a single SVD.
    """
    rows, cols = cover_matrix.shape
    br = rows // block_size
    bc = cols // block_size

    if br == 0 or bc == 0:
        raise ValueError(
            f"block_size ({block_size}) is larger than matrix dimensions "
            f"({rows}×{cols})."
        )

    bits_payload  = _text_to_bits(secret_text)
    length_header = np.unpackbits(
        np.array([len(bits_payload)], dtype=np.uint32).view(np.uint8)
    )
    bits_total    = np.concatenate([length_header, bits_payload])

    bits_per_block = block_size - start_sv
    n_blocks_needed = int(np.ceil(len(bits_total) / bits_per_block))
    n_blocks_avail  = br * bc

    if n_blocks_needed > n_blocks_avail:
        raise ValueError(
            f"Payload needs {n_blocks_needed} blocks but only "
            f"{n_blocks_avail} are available."
        )

    stego = cover_matrix.copy()
    bit_cursor = 0

    for r in range(br):
        for c in range(bc):
            if bit_cursor >= len(bits_total):
                break
            rs = r * block_size
            cs = c * block_size
            block   = stego[rs:rs+block_size, cs:cs+block_size]
            chunk   = bits_total[bit_cursor:bit_cursor + bits_per_block]
            U, S, Vt = np.linalg.svd(block, full_matrices=True)
            S_mod   = _embed_bits_qim(S, chunk, delta, start=start_sv)
            stego[rs:rs+block_size, cs:cs+block_size] = U @ np.diag(S_mod) @ Vt
            bit_cursor += len(chunk)

    mse  = float(np.mean((cover_matrix - stego) ** 2))
    psnr = float(10 * np.log10((3.0 ** 2) / mse)) if mse > 0 else float('inf')

    return stego, {
        'delta': delta, 'start_sv': start_sv, 'block_size': block_size,
        'n_bits_total': len(bits_total), 'mse': mse, 'psnr_dB': psnr,
        'blocks_used': n_blocks_needed, 'blocks_available': n_blocks_avail,
    }


def extract_blocks(stego_matrix: np.ndarray,
                   block_size   : int   = 16,
                   delta        : float = 0.1,
                   start_sv     : int   = 2) -> str:
    """Extract text hidden with embed_blocks()."""
    rows, cols     = stego_matrix.shape
    br             = rows // block_size
    bc             = cols // block_size
    bits_per_block = block_size - start_sv

    all_bits: list[np.ndarray] = []

    for r in range(br):
        for c in range(bc):
            rs    = r * block_size
            cs    = c * block_size
            block = stego_matrix[rs:rs+block_size, cs:cs+block_size]
            _, S, _ = np.linalg.svd(block, full_matrices=True)
            all_bits.append(
                _extract_bits_qim(S, n_bits=bits_per_block, delta=delta, start=start_sv)
            )

    flat  = np.concatenate(all_bits)
    # Decode 32-bit header
    hdr   = flat[:32]
    plen  = int(np.packbits(hdr).view(np.uint32)[0])
    pbits = flat[32:32 + plen]
    return _bits_to_text(pbits)


# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    np.random.seed(42)
    rng = np.random.default_rng(42)
    cover = rng.integers(0, 4, size=(64, 64)).astype(float)

    secret = "Patient ID: P-20240501 | Name: John Doe | Diagnosis: Hypertension"

    print("=" * 60)
    print("SVD-based Genomic Steganography — Demo")
    print("=" * 60)
    print(f"Secret text   : {secret}")
    print(f"Cover shape   : {cover.shape}")

    stego, meta = embed(cover, secret, delta=0.3, start_sv=3)

    print("\n--- Embedding Metadata ---")
    for k, v in meta.items():
        print(f"  {k:20s}: {v}")

    recovered = extract(stego, delta=meta['delta'], start_sv=meta['start_sv'])

    print(f"\n--- Extraction ---")
    print(f"Recovered text: {recovered}")
    print(f"Exact match   : {recovered == secret}")