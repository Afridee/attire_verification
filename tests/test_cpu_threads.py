"""CPU thread default. Does not load torch, MediaPipe, or FashionSigLIP."""

from __future__ import annotations

import os

import pytest

from attire_verification.cpu_threads import configure_cpu_threads

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def test_unset_resolves_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATTIRE_NUM_THREADS", raising=False)
    for key in _BLAS_THREAD_VARS:
        monkeypatch.delenv(key, raising=False)

    assert configure_cpu_threads() == 1
    for key in _BLAS_THREAD_VARS:
        assert os.environ[key] == "1"


def test_override_does_not_replace_existing_omp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATTIRE_NUM_THREADS", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    assert configure_cpu_threads() == 2
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
