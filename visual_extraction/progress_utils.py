"""Helpers for periodic progress and warning logs in long-running stages."""

from __future__ import annotations


def progress_interval(total: int) -> int:
    """Return a reasonable log interval for a stage with *total* items."""
    if total <= 0:
        return 1
    return max(1, min(100, max(10, total // 20)))


def should_log_progress(index: int, total: int) -> bool:
    """Return True when a stage should emit a progress update."""
    interval = progress_interval(total)
    return index == 1 or index == total or index % interval == 0


def progress_percent(index: int, total: int) -> float:
    """Return completed percentage for progress logs."""
    if total <= 0:
        return 100.0
    return (index / total) * 100.0


def should_warn_count(count: int) -> bool:
    """Return True on a small set of warning milestones."""
    return count <= 3 or count in {5, 10, 25, 50, 100}
