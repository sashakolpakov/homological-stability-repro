#!/usr/bin/env python3
"""Shared cold-start and steady-state timing summaries."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping


def aggregate_times(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return descriptive timing statistics without inventing missing repeats."""
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "n": 0,
        }
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }

def summarize_fit_times(
    records: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Keep the first configuration fit separate from subsequent steady fits.

    The archived benchmark executes repeated fits in one process. Its first
    successful fit can include library initialization, CUDA compilation, or
    Numba JIT work. Combining that configuration-start observation with later
    fits produces neither a cold-start statistic nor a steady-state statistic.
    """
    values = [
        float(record["fit_time_sec"])
        for record in records
        if "error" not in record
        and record.get("fit_time_sec") is not None
        and math.isfinite(float(record["fit_time_sec"]))
    ]
    cold_start = values[0] if values else None
    steady_values = values[1:]
    return {
        "protocol": (
            "first successful fit retained as configuration cold start; "
            "subsequent successful fits summarized as steady state"
        ),
        "cold_start_sec": cold_start,
        "steady_start_repeat": 2,
        "steady": aggregate_times(steady_values),
        "all_fits": aggregate_times(values),
    }
