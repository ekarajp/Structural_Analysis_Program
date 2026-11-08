"""Cross-check against OpenSeesPy on random structures (skipped without openseespy).

The full campaign: ``python -m tests.xcheck.compare --n 10000``.
"""
import pytest

pytest.importorskip("openseespy.opensees")

from tests.xcheck.compare import run_many  # noqa: E402

N = 240  # ~10 s; every family appears


def test_random_structures_match_opensees():
    outs = run_many(N, start=50_000, jobs=1)
    bad = [o for o in outs if o.status in ("fail", "error")]
    msg = "\n".join(f"seed {o.seed} {o.family}: x{o.worst:.3g} {o.detail[:200]}" for o in bad[:10])
    assert not bad, msg
    assert sum(o.status == "ok" for o in outs) >= 0.95 * N
