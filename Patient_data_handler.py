"""
patient_data_handler.py
-----------------------
Serialise, encrypt, and package patient records before steganographic
embedding — and reverse the process after extraction.

Encryption : AES-256-GCM (authenticated encryption via cryptography lib).
Encoding   : JSON → UTF-8 bytes → encrypt → base64 string (safe for text I/O).

Install dependency:
    pip install cryptography
"""

import base64
import json
import os
from datetime import datetime
from typing import Any


# --------------------------------------------------------------------------- #
#  Optional: AES-256-GCM encryption (requires `cryptography` package)
# --------------------------------------------------------------------------- #
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False


# --------------------------------------------------------------------------- #
#  Patient record schema
# --------------------------------------------------------------------------- #
REQUIRED_FIELDS = ('patient_id', 'name', 'dob')


def create_patient_record(patient_id  : str,
                          name        : str,
                          dob         : str,
                          diagnosis   : str   = '',
                          medications : list  = None,
                          notes       : str   = '',
                          **extra            ) -> dict:
    """
    Build a validated patient record dictionary.

    Parameters
    ----------
    patient_id  : Unique patient identifier (e.g. 'P-20240501').
    name        : Full patient name.
    dob         : Date of birth, ISO-8601 (YYYY-MM-DD).
    diagnosis   : Primary diagnosis string.
    medications : List of medication strings.
    notes       : Free-text clinical notes.
    **extra     : Any additional key–value metadata.

    Returns
    -------
    dict with a timestamp injected at key 'record_timestamp'.
    """
    record = {
        'patient_id'       : patient_id,
        'name'             : name,
        'dob'              : dob,
        'diagnosis'        : diagnosis,
        'medications'      : medications or [],
        'notes'            : notes,
        'record_timestamp' : datetime.utcnow().isoformat() + 'Z',
        **extra,
    }
    _validate_record(record)
    return record


def _validate_record(record: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
    if missing:
        raise ValueError(f"Patient record missing required fields: {missing}")


# --------------------------------------------------------------------------- #
#  Serialisation helpers
# --------------------------------------------------------------------------- #
def record_to_json(record: dict, compact: bool = True) -> str:
    """Serialise a patient record to a JSON string."""
    sep = (',', ':') if compact else (', ', ': ')
    return json.dumps(record, separators=sep, ensure_ascii=True)


def json_to_record(json_str: str) -> dict:
    """Deserialise a JSON string back to a patient record dict."""
    record = json.loads(json_str)
    _validate_record(record)
    return record


# --------------------------------------------------------------------------- #
#  Encryption (AES-256-GCM)
# --------------------------------------------------------------------------- #
def generate_key() -> bytes:
    """Generate a random 256-bit AES key.  Store this securely."""
    return os.urandom(32)


def encrypt_record(record: dict, key: bytes) -> str:
    """
    Encrypt a patient record with AES-256-GCM.

    Parameters
    ----------
    record : Patient record dict.
    key    : 32-byte AES-256 key (from generate_key() or a KMS).

    Returns
    -------
    Base64-encoded string: nonce (12 B) || ciphertext+tag.
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError(
            "cryptography package not installed.  Run: pip install cryptography"
        )
    if len(key) != 32:
        raise ValueError("AES-256 requires a 32-byte key.")

    plaintext = record_to_json(record).encode('utf-8')
    aesgcm    = AESGCM(key)
    nonce     = os.urandom(12)                    # 96-bit nonce per GCM spec
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    blob = base64.b64encode(nonce + ciphertext).decode('ascii')
    return blob


def decrypt_record(blob: str, key: bytes) -> dict:
    """
    Decrypt a base64-encoded AES-256-GCM blob back to a patient record.

    Parameters
    ----------
    blob : Output of encrypt_record().
    key  : Same 32-byte key used during encryption.

    Returns
    -------
    Decrypted patient record dict.
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError(
            "cryptography package not installed.  Run: pip install cryptography"
        )

    raw        = base64.b64decode(blob.encode('ascii'))
    nonce      = raw[:12]
    ciphertext = raw[12:]

    aesgcm  = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return json_to_record(plaintext.decode('utf-8'))


# --------------------------------------------------------------------------- #
#  Plain-text fallback (no crypto dependency)
# --------------------------------------------------------------------------- #
def pack_plain(record: dict) -> str:
    """Pack record as base64-encoded JSON (no encryption — for testing only)."""
    raw = record_to_json(record).encode('utf-8')
    return base64.b64encode(raw).decode('ascii')


def unpack_plain(blob: str) -> dict:
    """Unpack a plain-packed record."""
    raw = base64.b64decode(blob.encode('ascii'))
    return json_to_record(raw.decode('utf-8'))


# --------------------------------------------------------------------------- #
#  Demo
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    print("=" * 60)
    print("Patient Data Handler — Demo")
    print("=" * 60)

    record = create_patient_record(
        patient_id  = 'P-20240501',
        name        = 'Jane Smith',
        dob         = '1985-03-22',
        diagnosis   = 'Type 2 Diabetes',
        medications = ['Metformin 500 mg', 'Lisinopril 10 mg'],
        notes       = 'Annual review scheduled for Dec 2024.',
        genome_ref  = 'GRCh38',
    )

    print("\n[Plain-text pack/unpack]")
    packed   = pack_plain(record)
    print(f"  Packed  : {packed[:80]}...")
    unpacked = unpack_plain(packed)
    print(f"  Patient : {unpacked['name']} | DOB: {unpacked['dob']}")

    if _CRYPTO_AVAILABLE:
        print("\n[AES-256-GCM encrypt/decrypt]")
        key       = generate_key()
        encrypted = encrypt_record(record, key)
        print(f"  Encrypted: {encrypted[:80]}...")
        decrypted = decrypt_record(encrypted, key)
        print(f"  Patient  : {decrypted['name']} | DOB: {decrypted['dob']}")
        print(f"  Match    : {decrypted == record}")
    else:
        print("\n[Skipping AES demo — run: pip install cryptography]")