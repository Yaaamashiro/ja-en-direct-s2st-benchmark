from __future__ import annotations

from collections.abc import Iterable


def reduce_consecutive_units(units: Iterable[int]) -> list[int]:
    reduced: list[int] = []
    for unit in units:
        value = int(unit)
        if not reduced or reduced[-1] != value:
            reduced.append(value)
    return reduced


def validate_units(units: Iterable[int], *, clusters: int) -> list[int]:
    values = [int(unit) for unit in units]
    if not values:
        raise ValueError("unit sequence must not be empty")
    invalid = [value for value in values if not 0 <= value < clusters]
    if invalid:
        raise ValueError(
            f"unit IDs are out of range [0, {clusters}); received {invalid[0]}"
        )
    return values
