"""Resolve native CPU concurrency once from the current runtime limits."""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path

import psutil


CPU_BUDGET_RATIO = 0.95
MEMORY_BUDGET_RATIO = 0.90


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


def calculate_memory_cache_budget(
    *,
    physical_memory_bytes: int,
    startup_used_memory_bytes: int,
    cache_headroom_ratio: float,
    memory_budget_ratio: float = MEMORY_BUDGET_RATIO,
) -> dict[str, int | float | str]:
    """Allocate one cache from currently safe system-memory headroom."""
    if (
        physical_memory_bytes <= 0
        or startup_used_memory_bytes < 0
        or startup_used_memory_bytes > physical_memory_bytes
        or not 0.0 < memory_budget_ratio <= 1.0
        or not 0.0 < cache_headroom_ratio <= 1.0
    ):
        raise ValueError("memory cache budget input is invalid")
    system_memory_limit_bytes = math.floor(
        physical_memory_bytes * memory_budget_ratio
    )
    startup_headroom_bytes = max(
        0, system_memory_limit_bytes - startup_used_memory_bytes
    )
    cache_budget_bytes = math.floor(
        startup_headroom_bytes * cache_headroom_ratio
    )
    return {
        "contractId": "jarailsense.gluemap-memory-cache-budget/v1",
        "probeTiming": "runner-construction",
        "physicalMemoryBytes": physical_memory_bytes,
        "startupUsedMemoryBytes": startup_used_memory_bytes,
        "memoryBudgetRatio": memory_budget_ratio,
        "systemMemoryLimitBytes": system_memory_limit_bytes,
        "startupHeadroomBytes": startup_headroom_bytes,
        "cacheHeadroomRatio": cache_headroom_ratio,
        "cacheBudgetBytes": cache_budget_bytes,
    }


def probe_memory_cache_budget(
    *, cache_headroom_ratio: float
) -> dict[str, int | float | str]:
    """Resolve a cache budget from live host memory once at construction."""
    memory = psutil.virtual_memory()
    return calculate_memory_cache_budget(
        physical_memory_bytes=int(memory.total),
        startup_used_memory_bytes=int(memory.total - memory.available),
        cache_headroom_ratio=cache_headroom_ratio,
    )
