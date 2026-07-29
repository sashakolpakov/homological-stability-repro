from __future__ import annotations

import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import prepare_large_datasets  # noqa: E402


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int):
        super().__init__(payload)
        self.status = status

    def getcode(self) -> int:
        return self.status


def test_streaming_download_resumes_when_server_accepts_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "archive.tar.gz"
    partial = tmp_path / "archive.tar.gz.partial"
    partial.write_bytes(b"first-")
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse(b"second", 206)

    monkeypatch.setattr(
        prepare_large_datasets.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    prepare_large_datasets.download_streaming(
        "https://example.invalid/archive.tar.gz",
        destination,
    )

    assert destination.read_bytes() == b"first-second"
    assert not partial.exists()
    assert requests[0][0].get_header("Range") == "bytes=6-"
    assert requests[0][1] == 120


def test_streaming_download_restarts_when_server_ignores_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "archive.tar.gz"
    partial = tmp_path / "archive.tar.gz.partial"
    partial.write_bytes(b"stale")

    monkeypatch.setattr(
        prepare_large_datasets.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(b"complete", 200),
    )

    prepare_large_datasets.download_streaming(
        "https://example.invalid/archive.tar.gz",
        destination,
    )

    assert destination.read_bytes() == b"complete"
    assert not partial.exists()
