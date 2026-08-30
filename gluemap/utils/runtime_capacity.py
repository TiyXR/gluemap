"""Resolve native CPU concurrency once from the current runtime limits."""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path


CPU_BUDGET_RATIO = 0.95


def _linux_cpu_quota_cores() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.max")
    if not path.is_file():
        return None
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] == "max":
        return None
    try:
        quota, period = (int(value) for value in fields)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


@functools.lru_cache(maxsize=1)
def probe_logical_processor_capacity() -> float:
    """Return the narrowest host, affinity and cgroup CPU capacity."""
    capacities: list[float] = []
    logical = os.cpu_count()
    if logical:
        capacities.append(float(logical))
    if hasattr(os, "sched_getaffinity"):
        try:
            capacities.append(float(len(os.sched_getaffinity(0))))
        except OSError:
            pass
    quota = _linux_cpu_quota_cores()
    if quota is not None:
        capacities.append(quota)
    if not capacities:
        raise RuntimeError("logical processor capacity is unavailable")
    return min(capacities)


def calculate_native_thread_count(
    logical_processor_capacity: float,
    *,
    budget_ratio: float = CPU_BUDGET_RATIO,
) -> int:
    """Apply the process CPU budget to one native threaded stage."""
    if logical_processor_capacity <= 0 or not 0 < budget_ratio <= 1:
        raise ValueError("native thread capacity input is invalid")
    return max(1, math.floor(logical_processor_capacity * budget_ratio))


@functools.lru_cache(maxsize=1)
def resolve_native_thread_count() -> int:
    """Resolve and cache the startup thread budget for this process."""
    return calculate_native_thread_count(probe_logical_processor_capacity())
