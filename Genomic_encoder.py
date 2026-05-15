"""
genomic_encoder.py
------------------
Converts DNA/genomic sequences ↔ numeric matrices suitable for SVD processing.

Encoding scheme:
    A → 0,  T → 1,  G → 2,  C → 3
    Any ambiguous IUPAC bases are mapped to 0 by default.
"""

import numpy as np
from typing import Tuple


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
BASE_MAP  = {'A': 0, 'T': 1, 'G': 2, 'C': 3,
             'U': 1,                              # RNA uracil → T
             'N': 0, 'R': 0, 'Y': 1, 'S': 2,
             'W': 1, 'K': 1, 'M': 0, 'B': 2,
             'D': 0, 'H': 0, 'V': 2}

RBASE_MAP = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}


# --------------------------------------------------------------------------- #
#  Sequence → Matrix
# --------------------------------------------------------------------------- #
def sequence_to_matrix(sequence: str,
                       rows: int | None = None,
                       cols: int | None = None,
                       pad_value: int = 0) -> np.ndarray:
    """
    Convert a DNA/RNA sequence string into a 2-D numeric matrix.

    Parameters
    ----------
    sequence  : DNA/RNA string (upper or lower case).
    rows, cols: Desired matrix dimensions.  If only rows is given, cols is
                inferred.  If neither is given, the sequence is reshaped into
                the most square shape possible.
    pad_value : Integer (0-3) used to pad short sequences.

    Returns
    -------
    np.ndarray of dtype float64, shape (rows, cols).
    """
    sequence = sequence.upper().replace(' ', '').replace('\n', '')
    numeric  = np.array([BASE_MAP.get(b, 0) for b in sequence], dtype=np.float64)

    # Determine target dimensions
    n = len(numeric)
    if rows is None and cols is None:
        side = int(np.ceil(np.sqrt(n)))
        rows, cols = side, side
    elif rows is not None and cols is None:
        cols = int(np.ceil(n / rows))
    elif rows is None and cols is not None:
        rows = int(np.ceil(n / cols))

    total = rows * cols
    if n < total:
        numeric = np.pad(numeric, (0, total - n), constant_values=pad_value)
    else:
        numeric = numeric[:total]

    return numeric.reshape(rows, cols)


# --------------------------------------------------------------------------- #
#  Matrix → Sequence
# --------------------------------------------------------------------------- #
def matrix_to_sequence(matrix: np.ndarray,
                       original_length: int | None = None) -> str:
    """
    Convert a numeric matrix back to a DNA sequence string.

    Values are rounded and clipped to [0, 3] before mapping.

    Parameters
    ----------
    matrix          : 2-D numeric array produced by sequence_to_matrix().
    original_length : If given, the returned string is trimmed to this length
                      (removes padding).

    Returns
    -------
    DNA string (uppercase).
    """
    flat    = np.clip(np.round(matrix.flatten()), 0, 3).astype(int)
    seq_str = ''.join(RBASE_MAP.get(v, 'A') for v in flat)

    if original_length is not None:
        seq_str = seq_str[:original_length]

    return seq_str


# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #
def gc_content(sequence: str) -> float:
    """Return GC-content fraction of a sequence."""
    seq = sequence.upper()
    gc  = seq.count('G') + seq.count('C')
    return gc / len(seq) if seq else 0.0


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Check whether a sequence contains only recognised IUPAC bases.

    Returns (True, '') if valid, (False, reason) otherwise.
    """
    seq      = sequence.upper().replace(' ', '').replace('\n', '')
    invalid  = set(seq) - set(BASE_MAP.keys())
    if invalid:
        return False, f"Unknown bases found: {sorted(invalid)}"
    return True, ''


def matrix_stats(matrix: np.ndarray) -> dict:
    """Return basic statistics of a genomic matrix."""
    return {
        'shape': matrix.shape,
        'min'  : float(matrix.min()),
        'max'  : float(matrix.max()),
        'mean' : float(matrix.mean()),
        'std'  : float(matrix.std()),
        'rank' : int(np.linalg.matrix_rank(matrix)),
    }


# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    sample_dna = (
        "ATGCGATACGCTTACGATCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG"
    )

    valid, msg = validate_sequence(sample_dna)
    print(f"Sequence valid : {valid}  {msg}")
    print(f"GC content     : {gc_content(sample_dna):.2%}")

    mat = sequence_to_matrix(sample_dna, rows=8)
    print(f"\nMatrix shape   : {mat.shape}")
    print("Matrix:\n", mat)

    recovered = matrix_to_sequence(mat, original_length=len(sample_dna))
    print(f"\nRecovered seq  : {recovered}")
    print(f"Lossless match : {recovered == sample_dna.upper()}")
    print("\nMatrix stats:", matrix_stats(mat))