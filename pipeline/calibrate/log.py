"""Elapsed-time logging for long calibration runs."""

from __future__ import annotations

import time

_t0 = time.time()


def reset_timer() -> None:
    global _t0
    _t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)