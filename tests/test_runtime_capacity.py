import pytest

from gluemap.utils.runtime_capacity import calculate_native_thread_count


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
