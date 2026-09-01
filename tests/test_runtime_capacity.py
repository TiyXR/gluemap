import pytest

from gluemap.utils.runtime_capacity import (
    calculate_memory_cache_budget,
    calculate_native_thread_count,
)


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [
        (32.0, 30),
        (16.0, 15),
        (1.0, 1),
        (3.5, 3),
    ],
)
def test_native_thread_count_uses_ninety_five_percent_budget(
    capacity: float, expected: int
) -> None:
    assert calculate_native_thread_count(capacity) == expected


def test_native_thread_count_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="invalid"):
        calculate_native_thread_count(0)


def test_memory_cache_budget_uses_live_headroom_below_system_cap():
    report = calculate_memory_cache_budget(
        physical_memory_bytes=64 * 1024**3,
        startup_used_memory_bytes=20 * 1024**3,
        cache_headroom_ratio=0.50,
    )

    assert report["memoryBudgetRatio"] == 0.90
    assert report["systemMemoryLimitBytes"] == int(64 * 1024**3 * 0.90)
    assert report["startupHeadroomBytes"] == (
        report["systemMemoryLimitBytes"] - 20 * 1024**3
    )
    assert report["cacheBudgetBytes"] == report["startupHeadroomBytes"] // 2


def test_memory_cache_budget_does_not_allocate_past_exhausted_headroom():
    report = calculate_memory_cache_budget(
        physical_memory_bytes=16 * 1024**3,
        startup_used_memory_bytes=15 * 1024**3,
        cache_headroom_ratio=0.75,
    )

    assert report["startupHeadroomBytes"] == 0
    assert report["cacheBudgetBytes"] == 0
