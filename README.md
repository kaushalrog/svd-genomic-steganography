# SVD-Based Genomic Steganography for Secure Patient Data

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2022b%2B-orange?logo=mathworks)](https://mathworks.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Bioinformatics%20%7C%20Security-purple)]()

> Safely conceal sensitive medical information **inside genomic sequences** using
> Singular Value Decomposition — without altering the biological integrity of the
> host data.

---

## Overview

This project implements a **steganographic system** that embeds confidential patient
records inside genomic (DNA/RNA) sequences by exploiting the mathematical properties
of **Singular Value Decomposition (SVD)**.

### Why SVD?

| Property | Benefit |
|---|---|
| Energy compaction | The dominant singular values carry biological meaning; modifications to lower-energy values are imperceptible |
| Robustness | QIM-encoded singular values survive mild noise, quantisation, and compression |
| No structural alteration | The base sequence topology is preserved within the quantisation tolerance |
| Dual-layer security | AES-256-GCM encryption is applied *before* embedding |

### Algorithm — Quantisation Index Modulation (QIM)

```
A = U · Σ · Vᵀ          ← SVD decompose the genomic matrix

For each bit b to embed:
    σᵢ′ ← nearest σᵢ s.t.  ⌊σᵢ′/Δ⌋ mod 2 = b

A′ = U · Σ′ · Vᵀ         ← reconstruct stego matrix

Extraction:
    b = ⌊σᵢ′/Δ⌋ mod 2    ← read parity of quantisation index
```

The **length header** (32 bits prepended to the payload) allows blind extraction
without side-channel knowledge of the payload size.

---

## Project Structure

```
svd-steganography/
├── python/
│   ├── genomic_encoder.py       # DNA ↔ numeric matrix conversion
│   ├── svd_steganography.py     # Core embed / extract (single + block SVD)
│   ├── patient_data_handler.py  # Patient record serialisation + AES-256-GCM
│   ├── utils.py                 # PSNR, SSIM, BER, attack simulation, plots
│   └── main.py                  # CLI — embed / extract / demo / robustness
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### Python

```bash
# 1. Clone and set up
git clone https://github.com/<your-username>/svd-genomic-steganography.git
cd svd-genomic-steganography

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the self-contained demo
cd python
python main.py demo
```

Expected output:
```
=================================================================
  SVD-Based Genomic Steganography — Full Pipeline Demo
=================================================================

  Genomic sequence : 4096 bp  (synthetic)
  Cover matrix     : (64, 64)
  Patient          : Arjun Reddy
  Diagnosis        : Chronic Kidney Disease Stage 3
  Payload (packed) : 312 chars

  Embedding PSNR   : 38.47 dB
  Embedding MSE    : 0.00000843
  Exact match      : True

  --- Quality Metrics ---
    PSNR_dB                     : 38.47
    SSIM                        : 0.9994
    BER                         : 0.0
    exact_match                 : True
    ...
```

---

## CLI Reference (Python)

### Embed

```bash
python main.py embed \
  --sequence  ../data/sample_sequence.fasta \
  --record    ../data/sample_patient_record.json \
  --out-dir   ../output \
  --rows      64 \
  --delta     0.3 \
  --start-sv  3
```

With AES-256-GCM encryption:
```bash
# Generate a key (save this!)
python -c "import os; print(os.urandom(32).hex())"

python main.py embed \
  --sequence  ../data/sample_sequence.fasta \
  --record    ../data/sample_patient_record.json \
  --encrypt-key <hex-key> \
  --out-dir   ../output
```

### Extract

```bash
python main.py extract \
  --stego ../output/stego_matrix.npy \
  --meta  ../output/meta.json

# With decryption
python main.py extract \
  --stego       ../output/stego_matrix.npy \
  --meta        ../output/meta.json \
  --decrypt-key <hex-key>
```

### Robustness test grid

```bash
python main.py robustness
```

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `delta` (Δ) | 0.3 | QIM step size. Larger → more robust, lower PSNR |
| `start_sv` | 3 | Skip top-N singular values. Higher → less distortion |
| `block_size` | 16 | Block side length for block-SVD mode |

### Choosing Δ

| Δ | PSNR (typical) | Robustness | Recommended for |
|---|---|---|---|
| 0.1 | > 45 dB | Low | Lossless channels only |
| 0.3 | ~38 dB | Medium | General use |
| 0.5 | ~32 dB | High | Noisy / compressed channels |

---

## Robustness Results

Tested against three attack types:

| Attack | Δ=0.1 | Δ=0.3 | Δ=0.5 |
|---|---|---|---|
| Gaussian noise (σ=0.05) | ✓ | ✓ | ✓ |
| Quantisation (64 levels) | ✓ | ✓ | ✓ |
| SVD truncation (95%) | ✓ | ✓ | ✓ |
| Quantisation (32 levels) | ~ | ✓ | ✓ |
| Gaussian noise (σ=0.10) | ✗ | ✓ | ✓ |
| SVD truncation (90%) | ✗ | ~ | ✓ |

✓ = exact recovery  ~  = partial  ✗ = fail

---

## Security Considerations

1. **Encryption first** — Always use AES-256-GCM (`--encrypt-key`) in production.
   The steganography layer provides *hiding*, not secrecy.
2. **Key management** — Store AES keys in a HSM or KMS; never commit them to git.
3. **PSNR threshold** — For genomic integrity, keep PSNR ≥ 35 dB (Δ ≤ 0.3).
4. **Audit trail** — Log embedding/extraction events separately from the genomic data.

---

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | SVD, matrix operations |
| `scipy` | Signal processing utilities |
| `matplotlib` | Visualisation |
| `scikit-image` | SSIM metric |
| `cryptography` | AES-256-GCM patient data encryption |
| `biopython` | FASTA file parsing (optional) |

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{svd_genomic_stego_2024,
  title  = {SVD-Based Steganography for Secure Patient Data in Genomic Sequences},
  author = {<Your Name>},
  year   = {2024},
  url    = {https://github.com/<your-username>/svd-genomic-steganography}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> **Disclaimer**: This software is a research prototype.  
> Do **not** use it as the sole security mechanism for real patient data  
> without a comprehensive HIPAA / GDPR compliance review.
