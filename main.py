"""
main.py
-------
End-to-end pipeline for SVD-based genomic steganography.

Modes:
  python main.py embed    — embed patient data into a genomic sequence
  python main.py extract  — extract patient data from a stego matrix
  python main.py demo     — run a self-contained demonstration
  python main.py robustness — run robustness attack tests
"""

import argparse
import os
import sys

import numpy as np

# Local modules
from genomic_encoder      import sequence_to_matrix, matrix_to_sequence, validate_sequence
from svd_steganography     import embed, extract, embed_blocks, extract_blocks
from patient_data_handler import (create_patient_record, pack_plain, unpack_plain,
                                  encrypt_record, decrypt_record, generate_key,
                                  record_to_json)
from utils                import (save_matrix, load_matrix, save_metadata, load_metadata,
                                  full_metrics, attack_gaussian_noise, attack_quantisation,
                                  attack_truncation, plot_comparison, plot_singular_values)


# ============================================================================ #
#  EMBED
# ============================================================================ #
def run_embed(args) -> None:
    """Read genomic sequence, embed patient record, save stego matrix."""
    print("\n[EMBED MODE]")

    # Load genomic cover sequence
    with open(args.sequence, 'r') as f:
        raw = f.read()
    # Strip FASTA header if present
    lines  = [l for l in raw.splitlines() if not l.startswith('>')]
    seq    = ''.join(lines).strip()

    valid, msg = validate_sequence(seq)
    if not valid:
        print(f"ERROR: {msg}")
        sys.exit(1)

    print(f"  Sequence length : {len(seq)} bp")
    cover = sequence_to_matrix(seq, rows=args.rows)
    print(f"  Cover matrix    : {cover.shape}")

    # Load patient record
    with open(args.record, 'r') as f:
        payload_text = f.read().strip()

    if args.encrypt_key:
        key_bytes = bytes.fromhex(args.encrypt_key)
        import json
        record    = json.loads(payload_text)
        payload_text = encrypt_record(record, key_bytes)
        print(f"  Encryption      : AES-256-GCM enabled")

    print(f"  Payload size    : {len(payload_text)} chars")

    # Embed
    if args.block:
        stego, meta = embed_blocks(cover, payload_text,
                                   block_size=args.block_size,
                                   delta=args.delta,
                                   start_sv=args.start_sv)
        print(f"  Mode            : Block-SVD ({args.block_size}×{args.block_size})")
    else:
        stego, meta = embed(cover, payload_text,
                            delta=args.delta,
                            start_sv=args.start_sv)
        print(f"  Mode            : Single-SVD")

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    stego_path = os.path.join(args.out_dir, 'stego_matrix.npy')
    meta_path  = os.path.join(args.out_dir, 'meta.json')
    save_matrix(stego, stego_path)
    save_metadata(meta, meta_path)

    print(f"\n  PSNR            : {meta['psnr_dB']:.2f} dB")
    print(f"  MSE             : {meta['mse']:.6f}")
    print(f"  Outputs in      : {args.out_dir}/")


# ============================================================================ #
#  EXTRACT
# ============================================================================ #
def run_extract(args) -> None:
    """Load stego matrix, extract hidden patient data, print/save."""
    print("\n[EXTRACT MODE]")

    stego = load_matrix(args.stego)
    meta  = load_metadata(args.meta)

    delta    = meta['delta']
    start_sv = meta['start_sv']

    if meta.get('block_size'):
        recovered = extract_blocks(stego,
                                   block_size=meta['block_size'],
                                   delta=delta,
                                   start_sv=start_sv)
    else:
        recovered = extract(stego, delta=delta, start_sv=start_sv)

    if args.decrypt_key:
        key_bytes = bytes.fromhex(args.decrypt_key)
        record    = decrypt_record(recovered, key_bytes)
        print("\n  Decrypted patient record:")
        import json
        print(json.dumps(record, indent=2))
    else:
        print(f"\n  Extracted payload:\n{recovered}")

    if args.out:
        with open(args.out, 'w') as f:
            f.write(recovered)
        print(f"\n  Saved to {args.out}")


# ============================================================================ #
#  DEMO
# ============================================================================ #
def run_demo(_) -> None:
    print("\n" + "=" * 65)
    print("  SVD-Based Genomic Steganography — Full Pipeline Demo")
    print("=" * 65)

    # --- Synthetic genomic sequence ---
    rng = np.random.default_rng(0)
    bases = list('ATGC')
    seq   = ''.join(rng.choice(bases, size=4096))
    print(f"\n  Genomic sequence : {len(seq)} bp  (synthetic)")
    print(f"  Preview          : {seq[:60]}...")

    cover = sequence_to_matrix(seq, rows=64)
    print(f"  Cover matrix     : {cover.shape}")

    # Use a larger 256×256 cover for meaningful capacity
    rng2      = np.random.default_rng(1)
    big_seq   = ''.join(rng2.choice(list('ATGC'), size=65536))
    cover256  = sequence_to_matrix(big_seq, rows=256)
    print(f"\n  256×256 cover    : {cover256.shape}  (65 536 bp)")

    # --- Demo A: Short token via single-SVD ---
    # 256×256 matrix → 256 singular values → capacity = 256-3 = 253 bits
    # A 21-char payload needs 168+32 = 200 bits → fits comfortably.
    short_payload = "P-20240501|CKD3|Reddy"
    print(f"\n  [Single-SVD Mode]  payload = '{short_payload}'  ({len(short_payload)} chars)")

    stego, meta = embed(cover256, short_payload, delta=0.3, start_sv=3)
    print(f"  Embedding PSNR   : {meta['psnr_dB']:.2f} dB")
    print(f"  Embedding MSE    : {meta['mse']:.8f}")
    print(f"  Bits used / cap  : {meta['n_bits_total']} / {meta['capacity_bits']}")

    recovered_payload = extract(stego, delta=meta['delta'], start_sv=meta['start_sv'])
    print(f"  Recovered text   : {recovered_payload!r}")
    print(f"  Exact match      : {recovered_payload == short_payload}")

    # --- Demo B: Full patient record via Block-SVD ---
    print(f"\n  [Block-SVD Mode]   full patient record")
    record  = create_patient_record(
        patient_id  = 'P-20240501',
        name        = 'Arjun Reddy',
        dob         = '1978-11-15',
        diagnosis   = 'Chronic Kidney Disease Stage 3',
        medications = ['Amlodipine 5 mg', 'Furosemide 20 mg', 'Calcitriol 0.25 μg'],
        notes       = 'eGFR 42 mL/min/1.73m². Next nephrology review in 3 months.',
        genome_ref  = 'GRCh38',
    )
    payload = pack_plain(record)
    print(f"  Patient          : {record['name']}")
    print(f"  Diagnosis        : {record['diagnosis']}")
    print(f"  Payload (packed) : {len(payload)} chars")

    b_stego, b_meta = embed_blocks(cover256, payload, block_size=8, delta=0.3, start_sv=1)
    print(f"  Blocks used/avail: {b_meta['blocks_used']} / {b_meta['blocks_available']}")
    print(f"  Block PSNR       : {b_meta['psnr_dB']:.2f} dB")

    recovered_payload = extract_blocks(b_stego, block_size=8, delta=0.3, start_sv=1)
    recovered_record  = unpack_plain(recovered_payload)

    print(f"  Recovered name   : {recovered_record['name']}")
    print(f"  Exact match      : {recovered_record == record}")

    # --- Quality metrics (on single-SVD stego) ---
    m = full_metrics(cover256, stego, short_payload, recovered_payload)
    print("\n  --- Quality Metrics (Single-SVD) ---")
    for k, v in m.items():
        print(f"    {k:28s}: {v}")

    # --- Robustness (on block-SVD stego) ---
    print("\n  --- Robustness Against Attacks (Block-SVD) ---")
    attacks = {
        'Gaussian Noise (σ=0.05)' : attack_gaussian_noise(b_stego, sigma=0.05),
        'Quantisation (64 levels)' : attack_quantisation(b_stego, 64),
        'SVD Truncation (95%)'     : attack_truncation(b_stego, 0.95),
    }
    for name, attacked in attacks.items():
        try:
            rec  = extract_blocks(attacked, block_size=8, delta=0.3, start_sv=1)
            ok   = (rec == payload)
        except Exception as e:
            ok   = False
        print(f"    {name:35s}: {'✓ PASS' if ok else '✗ FAIL'}")

    print("\n  Demo complete.\n")


# ============================================================================ #
#  ROBUSTNESS
# ============================================================================ #
def run_robustness(_) -> None:
    """Run a grid of attack tests across different delta values."""
    print("\n[ROBUSTNESS MODE]")

    rng    = np.random.default_rng(1)
    seq    = ''.join(rng.choice(list('ATGC'), size=2048))
    cover  = sequence_to_matrix(seq, rows=32)
    secret = "Confidential patient genomic steganography test string."

    deltas   = [0.1, 0.2, 0.3, 0.5]
    noise_s  = [0.02, 0.05, 0.10]
    q_levels = [256, 64, 32]
    trunc_k  = [0.99, 0.97, 0.95]

    header = f"{'Delta':>6} | {'Noise σ':>8} | {'Q-levels':>9} | {'Trunc%':>7} | {'Status':>6}"
    print(header)
    print("-" * len(header))

    for d in deltas:
        stego, meta = embed(cover, secret, delta=d, start_sv=2)
        for ns, ql, tr in zip(noise_s, q_levels, trunc_k):
            for attack_name, attacked in [
                (f"noise={ns}", attack_gaussian_noise(stego, ns)),
                (f"q={ql}",     attack_quantisation(stego, ql)),
                (f"trunc={tr}", attack_truncation(stego, tr)),
            ]:
                try:
                    rec = extract(attacked, delta=d, start_sv=2)
                    ok  = "✓ OK" if rec == secret else "~ PARTIAL"
                except Exception:
                    ok  = "✗ FAIL"

                print(f"  {d:>4} | {attack_name:>14} | {ok}")

    print()


# ============================================================================ #
#  CLI
# ============================================================================ #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='SVD-Based Genomic Steganography Tool'
    )
    sub = p.add_subparsers(dest='mode', required=True)

    # embed
    e = sub.add_parser('embed', help='Embed patient data into a genomic sequence.')
    e.add_argument('--sequence',   required=True,  help='Path to FASTA or plain DNA file.')
    e.add_argument('--record',     required=True,  help='Path to JSON patient record file.')
    e.add_argument('--out-dir',    default='output', dest='out_dir')
    e.add_argument('--rows',       type=int, default=64)
    e.add_argument('--delta',      type=float, default=0.3)
    e.add_argument('--start-sv',   type=int,   default=3, dest='start_sv')
    e.add_argument('--block',      action='store_true')
    e.add_argument('--block-size', type=int, default=16, dest='block_size')
    e.add_argument('--encrypt-key', default=None, dest='encrypt_key',
                   help='Hex-encoded 32-byte AES key.')
    e.set_defaults(func=run_embed)

    # extract
    x = sub.add_parser('extract', help='Extract hidden patient data.')
    x.add_argument('--stego',       required=True, help='Path to stego_matrix.npy.')
    x.add_argument('--meta',        required=True, help='Path to meta.json.')
    x.add_argument('--out',         default=None,  help='Save extracted text here.')
    x.add_argument('--decrypt-key', default=None,  dest='decrypt_key')
    x.set_defaults(func=run_extract)

    # demo
    d = sub.add_parser('demo', help='Run self-contained demonstration.')
    d.set_defaults(func=run_demo)

    # robustness
    r = sub.add_parser('robustness', help='Run attack robustness tests.')
    r.set_defaults(func=run_robustness)

    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    args.func(args)