"""CPU thread count for inference.

OpenMP, MKL, and OpenBLAS read their thread limits when a native library
first loads. This module only resolves ``ATTIRE_NUM_THREADS`` and sets those
variables. It must not import torch, cv2, or mediapipe.
"""

from __future__ import annotations

import os

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def resolve_num_threads() -> int:
    """Positive thread count from ``ATTIRE_NUM_THREADS``, or 1 when unset."""
    raw = os.environ.get("ATTIRE_NUM_THREADS")
    if raw is None or not raw.strip():
        return 1
    try:
        count = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"ATTIRE_NUM_THREADS must be a positive integer, got {raw!r}"
        ) from exc
    if count < 1:
        raise ValueError(
            f"ATTIRE_NUM_THREADS must be a positive integer, got {raw!r}"
        )
    return count


def configure_cpu_threads() -> int:
    """Set any missing BLAS thread variable to the resolved count.

    A variable that is already set is left as-is. Returns the count.
    """
    count = resolve_num_threads()
    value = str(count)
    for key in _BLAS_THREAD_VARS:
        os.environ.setdefault(key, value)
    return count
