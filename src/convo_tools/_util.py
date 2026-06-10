from __future__ import annotations

import hashlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _progressbar(iterable: Iterable[Any], total: int, prefix: str = "", width: int = 40) -> Iterator[Any]:
    if total == 0:
        yield from iterable
        return
    for i, item in enumerate(iterable, 1):
        yield item
        frac = i / total
        filled = int(width * frac)
        bar_str = "[" + "#" * filled + "." * (width - filled) + "]"
        sys.stdout.write(f"\r{prefix}{bar_str} {i}/{total}")
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
