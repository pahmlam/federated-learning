"""Lightweight process resource snapshots for smoke-run reports."""

from __future__ import annotations

import os
import platform
import time


def get_resource_snapshot() -> dict[str, int | float | str | bool | None]:
    """Return a JSON-ready resource snapshot using only the standard library."""

    snapshot: dict[str, int | float | str | bool | None] = {
        "pid": os.getpid(),
        "platform": platform.platform(),
        "timestamp_unix": time.time(),
        "max_rss_bytes": None,
        "max_rss_mb": None,
        "source": "resource.getrusage(RUSAGE_SELF)",
        "available": False,
    }
    try:
        import resource

        max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return snapshot

    # macOS reports bytes; Linux reports KiB.
    max_rss_bytes = max_rss if platform.system() == "Darwin" else max_rss * 1024
    snapshot["max_rss_bytes"] = max_rss_bytes
    snapshot["max_rss_mb"] = round(max_rss_bytes / (1024 * 1024), 2)
    snapshot["available"] = True
    return snapshot
