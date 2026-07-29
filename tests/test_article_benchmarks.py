from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from article_benchmarks import compute_atlas_topology_metrics  # noqa: E402


def test_small_suite_invokes_direct_rank_atlas_not_ripser(
    monkeypatch,
) -> None:
    import dire_rapids.betti_curve
    import fastdtw

    calls: list[dict] = []

    def fake_atlas(points, **kwargs):
        calls.append(
            {
                "shape": np.asarray(points).shape,
                "kwargs": kwargs,
            }
        )
        return {
            "filtration_values": np.asarray([1.0, 0.0]),
            "beta_0": np.asarray([4, 1]),
            "beta_1": np.asarray([0, 1]),
        }

    distances = iter((8.0, 4.0))
    monkeypatch.setattr(
        dire_rapids.betti_curve,
        "compute_betti_curve_gpu",
        fake_atlas,
    )
    monkeypatch.setattr(
        fastdtw,
        "fastdtw",
        lambda _left, _right, dist: (next(distances), [])
        if dist == 2
        else (-1.0, []),
    )
    args = SimpleNamespace(
        topology_steps=100,
        topology_atlas_neighbors=20,
        topology_atlas_density_threshold=0.8,
        topology_atlas_overlap_factor=1.5,
    )
    points = np.zeros((4, 3), dtype=np.float32)
    embedding = np.ones((4, 2), dtype=np.float32)

    result = compute_atlas_topology_metrics(
        points,
        embedding,
        args,
        seed=1042,
    )

    assert len(calls) == 2
    assert [call["shape"] for call in calls] == [(4, 3), (4, 2)]
    assert all(
        call["kwargs"]
        == {
            "k_neighbors": 20,
            "density_threshold": 0.8,
            "overlap_factor": 1.5,
            "n_steps": 100,
        }
        for call in calls
    )
    assert result["backend"] == "atlas"
    assert result["backend_detail"] == (
        "direct GPU rank-based local-kNN atlas"
    )
    assert result["prefer_ripser"] is False
    assert result["metrics"] == {
        "dtw_beta0": 2.0,
        "dtw_beta1": 1.0,
    }
