from __future__ import annotations

import hashlib
import io
import pickle
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


class _SafeUnpickler(pickle.Unpickler):
    """Restrict unpickling to safe built-in types only."""

    _SAFE = frozenset({"bool", "int", "float", "complex", "str", "bytes",
                        "bytearray", "list", "tuple", "set", "frozenset",
                        "dict", "NoneType"})

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in self._SAFE:
            return getattr(__builtins__ if isinstance(__builtins__, dict) else __builtins__, name)
        raise pickle.UnpicklingError(f"Disallowed: {module}.{name}")


def safe_pickle_load(path: str | Path) -> Any:
    """Load a pickle file with restricted unpickling (blocks arbitrary code execution)."""
    with open(path, "rb") as f:
        data = f.read()

    buf = io.BytesIO(data)
    return _SafeUnpickler(buf).load()


def _progressbar(iterable: Iterable[Any], total: int, prefix: str = "", width: int = 40) -> Iterator[Any]:
    if total == 0:
        yield from iterable
        return
    t0 = time.monotonic()
    for i, item in enumerate(iterable, 1):
        yield item
        frac = i / total
        filled = int(width * frac)
        elapsed = time.monotonic() - t0
        if i > 0 and elapsed > 0:
            rate = i / elapsed
            eta_sec = (total - i) / rate if rate > 0 else 0
            if eta_sec >= 86400:
                eta_str = f"{eta_sec / 86400:.0f}d"
            elif eta_sec >= 3600:
                eta_str = f"{eta_sec / 3600:.0f}h"
            elif eta_sec >= 60:
                eta_str = f"{eta_sec / 60:.0f}m"
            else:
                eta_str = f"{eta_sec:.0f}s"
        else:
            eta_str = "?"
        bar_str = "[" + "#" * filled + "." * (width - filled) + "]"
        sys.stdout.write(f"\r{prefix}{bar_str} {i}/{total}  ETA {eta_str}")
        sys.stdout.flush()
    sys.stdout.write("\n")


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, AttributeError):
        pass
    return 0.0


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
