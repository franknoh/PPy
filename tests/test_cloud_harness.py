"""The RunPod harness's judgement, without RunPod: a Pod that is not gone fails the run,
and an AMD type the stock list lacks is still asked for where a datacenter names it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(name="harness")
def _harness(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "runpod_matrix", ROOT / "scripts" / "cloud" / "runpod_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    return module


class _FakeRunpod:
    """`runpodctl` as a dictionary: what each command answers, and what was asked."""

    def __init__(self, answers: dict, listed_after_delete: bool = False) -> None:
        self.answers = answers
        self.calls: list[tuple[str, ...]] = []
        self.listed_after_delete = listed_after_delete
        self.deleted: list[str] = []

    def __call__(self, *arguments: str, timeout: int = 120) -> object:
        self.calls.append(arguments)
        if arguments[:2] == ("pod", "delete"):
            if isinstance(self.answers.get("delete"), Exception):
                raise self.answers["delete"]
            self.deleted.append(arguments[2])
            return {"id": arguments[2]}
        if arguments[:2] == ("pod", "list"):
            return [{"id": "pod-1"}] if self.listed_after_delete else []
        return self.answers.get(arguments[0], None)


def _pod(harness, log: Path):
    pod = harness.Pod("ppy-test-cuda-abcd1234", harness.ENVIRONMENTS["cuda"], "NVIDIA A40", log)
    pod.id = "pod-1"
    return pod


def test_a_deleted_pod_is_a_clean_run(harness, tmp_path, monkeypatch):
    fake = _FakeRunpod({})
    monkeypatch.setattr(harness, "_runpod", fake)
    summary = {"verdict": "PASS"}
    harness._cleanup(summary, _pod(harness, tmp_path / "log"), keep=False)
    assert summary["cleanup_ok"] is True and summary["verdict"] == "PASS"
    assert fake.deleted == ["pod-1"]


def test_a_pod_still_listed_after_deletion_fails_the_run(harness, tmp_path, monkeypatch):
    fake = _FakeRunpod({}, listed_after_delete=True)
    monkeypatch.setattr(harness, "_runpod", fake)
    summary = {"verdict": "PASS"}
    harness._cleanup(summary, _pod(harness, tmp_path / "log"), keep=False)
    assert summary["cleanup_ok"] is False
    assert summary["verdict"] == "FAIL"
    assert "still listed after deletion" in summary["cleanup"]


def test_a_delete_that_fails_twice_fails_the_run(harness, tmp_path, monkeypatch):
    fake = _FakeRunpod({"delete": harness.Failed("runpodctl pod delete: 500")})
    monkeypatch.setattr(harness, "_runpod", fake)
    summary = {"verdict": "PASS"}
    harness._cleanup(summary, _pod(harness, tmp_path / "log"), keep=False)
    assert summary["cleanup_ok"] is False and summary["verdict"] == "FAIL"
    assert sum(call[:2] == ("pod", "delete") for call in fake.calls) == 2, "retried once"


def test_keep_leaves_the_pod_on_purpose_and_says_so(harness, tmp_path, monkeypatch):
    fake = _FakeRunpod({})
    monkeypatch.setattr(harness, "_runpod", fake)
    summary = {"verdict": "PASS"}
    harness._cleanup(summary, _pod(harness, tmp_path / "log"), keep=True)
    assert summary["cleanup_ok"] is None and summary["verdict"] == "PASS"
    assert "kept on request" in summary["cleanup"] and not fake.deleted


def test_the_exit_status_follows_every_verdict(harness):
    line = harness._line({"environment": "cuda", "verdict": "FAIL", "steps": {}, "cleanup": "x"})
    assert "FAIL" in line


def test_amd_is_asked_for_where_a_datacenter_names_it(harness, monkeypatch):
    """No AMD type in the stock list; an MI300X in one datacenter's inventory: that pair is
    a candidate, and the create request decides, not the listing."""
    fake = _FakeRunpod(
        {
            "gpu": [
                {"gpuId": "NVIDIA A40", "available": True},
                {"gpuId": "AMD Instinct MI250X", "available": False},
            ],
            "datacenter": [
                {"id": "EU-RO-1", "gpuAvailability": [{"gpuId": "AMD Instinct MI300X OAM"}]},
                {"id": "US-KS-2", "gpuAvailability": [{"gpuId": "AMD Instinct MI250X"}]},
                {"id": "US-TX-3", "gpuAvailability": [{"gpuId": "NVIDIA H100 80GB HBM3"}]},
            ],
        }
    )
    monkeypatch.setattr(harness, "_runpod", fake)
    assert harness._candidates(harness.ENVIRONMENTS["rocm"]) == [
        ("AMD Instinct MI300X OAM", "EU-RO-1"),
        ("AMD Instinct MI250X", "US-KS-2"),
    ]
    assert harness._candidates(harness.ENVIRONMENTS["cuda"]) == [("NVIDIA A40", "")], (
        "the stock list still decides for NVIDIA"
    )


def test_every_refusal_is_not_run_not_pass(harness, tmp_path, monkeypatch):
    fake = _FakeRunpod(
        {
            "gpu": [],
            "datacenter": [
                {"id": "EU-RO-1", "gpuAvailability": [{"gpuId": "AMD Instinct MI300X OAM"}]}
            ],
            "user": {"id": "u"},
        }
    )
    monkeypatch.setattr(harness, "_runpod", fake)

    def refused(self):
        raise harness.Failed(
            'runpodctl pod create: {"error":"There are no longer any instances available"}'
        )

    monkeypatch.setattr(harness.Pod, "create", refused)
    monkeypatch.setattr(harness, "_ssh_key", lambda: ("ssh-ed25519 AAAA test", tmp_path / "k"))
    summary = harness.validate("rocm", "deadbeef", tmp_path / "out", keep=False)
    assert summary["verdict"] == "NOT RUN"
    refusal = (
        "AMD Instinct MI300X OAM x1 in EU-RO-1: runpodctl pod create: "
        '{"error":"There are no longer any instances available"}'
    )
    assert summary["refused"] == [refusal]
    assert "cleanup_ok" not in summary, "no Pod was made, none is to delete"
