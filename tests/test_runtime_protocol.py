from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean, stdev

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_figures import steady_runtime_statistics  # noqa: E402
from runtime_protocol import summarize_fit_times  # noqa: E402


def test_cold_start_is_not_combined_with_steady_repeats() -> None:
    records = [
        {"fit_time_sec": 12.0},
        {"fit_time_sec": 0.8},
        {"fit_time_sec": 1.0},
        {"fit_time_sec": 1.2},
    ]

    timing = summarize_fit_times(records)

    assert timing["cold_start_sec"] == 12.0
    assert timing["steady_start_repeat"] == 2
    assert timing["steady"]["n"] == 3
    assert timing["steady"]["mean"] == pytest.approx(1.0)
    assert timing["steady"]["std"] == pytest.approx(0.2)
    assert timing["all_fits"]["mean"] == pytest.approx(3.75)


def test_historical_logs_derive_steady_statistics_from_repeats() -> None:
    result = {
        "aggregate": {
            "time": {
                "mean": 3.75,
                "std": 5.5,
                "n": 4,
            }
        },
        "repeats": [
            {"fit_time_sec": 12.0},
            {"fit_time_sec": 0.8},
            {"fit_time_sec": 1.0},
            {"fit_time_sec": 1.2},
        ],
    }

    steady = steady_runtime_statistics(result)

    assert steady["n"] == 3
    assert steady["mean"] == pytest.approx(1.0)
    assert steady["std"] == pytest.approx(0.2)


def test_archived_dire_runtime_does_not_mix_compilation_with_steady_fits() -> None:
    root = Path(__file__).resolve().parents[1]
    archived = json.loads(
        (root / "data" / "archived" / "json_logs" / "blobs.json").read_text(
            encoding="utf-8"
        )
    )
    result = archived["methods"]["dire"]

    steady = steady_runtime_statistics(result)
    fit_times = [
        float(record["fit_time_sec"])
        for record in result["repeats"]
    ]
    steady_fit_times = fit_times[1:]

    assert len(fit_times) == 20
    assert result["timing"]["cold_start_sec"] == pytest.approx(fit_times[0])
    assert result["timing"]["steady_start_repeat"] == 2
    assert result["aggregate"]["time"]["mean"] == pytest.approx(
        fmean(fit_times)
    )
    assert result["aggregate"]["time"]["std"] == pytest.approx(
        stdev(fit_times)
    )
    assert steady["n"] == len(steady_fit_times)
    assert steady["mean"] == pytest.approx(fmean(steady_fit_times))
    assert steady["std"] == pytest.approx(stdev(steady_fit_times))
    assert steady["mean"] < result["aggregate"]["time"]["mean"]
