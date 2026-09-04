"""Unit tests for the Modal promote-gate (scripts/pull_modal_artifacts.py +
scripts/modal_bundles.py + scripts/model_metrics_store.py).

The defect these cover: the gate compared the challenger's served Brier to the
incumbent's stored number even when the two were computed on DIFFERENT test sets
(train_all_models splits 70/15/15 by ROW RATIO, so the held-out set is just the
tail of whatever frame that run loaded). When the soccer frame grew ~7x the
comparison became meaningless and rejected the better model four weeks running;
for small bundles the same comparison ratcheted the other way, admitting a drift
of one tolerance-sized regression per run.

Pure unit: no Modal, no B2, no DB, no models. b2_io.download_prefix is stubbed,
the artifact tree is faked under tmp_path, and the paired re-scoring (the only
part that needs real artifacts + Postgres) is injected."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mb = _load("modal_bundles", SCRIPTS / "modal_bundles.py")
mstore = _load("model_metrics_store", SCRIPTS / "model_metrics_store.py")
pull = _load("pull_modal_artifacts", SCRIPTS / "pull_modal_artifacts.py")

BUNDLE = "nfl_total"
ENSEMBLE = mb.BUNDLE_TO_ENSEMBLE[BUNDLE]
SOCCER = "soccer_match_result"
SOCCER_ENSEMBLE = mb.BUNDLE_TO_ENSEMBLE[SOCCER]

RUN = "run-2026-09-06"


# ── fixtures / builders ──────────────────────────────────────────────


def fingerprint(
    rows: int, holdout_n, date_min="2019-08-01", date_max="2026-08-29", feature_count=42, target_classes=3
) -> dict:
    return {
        "rows": rows,
        "date_min": date_min,
        "date_max": date_max,
        "holdout_n": holdout_n,
        "feature_count": feature_count,
        "target_classes": target_classes,
    }


def make_bundle_dir(
    incoming: Path,
    bundle: str,
    *,
    brier,
    rows: int,
    holdout_n,
    date_min: str = "2019-08-01",
    date_max: str = "2026-08-29",
    status: str = "ok",
    kept: bool = False,
    with_report: bool = True,
) -> Path:
    """A fake modal-incoming/<run>/<bundle>/ tree: gate.json + the sibling
    training_report.json the fingerprint is derived from + a model dir."""
    d = incoming / bundle
    (d / f"ensemble_{bundle}" / "1.0.0").mkdir(parents=True)
    (d / f"ensemble_{bundle}" / "1.0.0" / "model.bin").write_text("{}")
    gate = {
        "kept": kept,
        "reason": "test",
        "n": holdout_n,
        "raw": {"brier": brier},
        "calibrated": {"brier": None if brier is None else brier - 0.01},
    }
    (d / "gate.json").write_text(json.dumps({"status": status, "bundle": bundle, "run_id": RUN, "gate": gate}))
    if with_report:
        (d / "training_report.json").write_text(
            json.dumps(
                {
                    "data_quality": {
                        "rows": rows,
                        "date_min": date_min,
                        "date_max": date_max,
                        "feature_count": 42,
                        "target_classes": 3,
                    },
                    "holdout_test": gate,
                }
            )
        )
    return d


def seed_incumbent(models_dir: Path, ensemble: str, entries: list[dict]) -> Path:
    """Write a production sidecar whose history is `entries` (oldest first),
    with the newest mirrored into the top-level legacy fields."""
    last = entries[-1]
    doc = {
        "ensemble_name": ensemble,
        "served_brier": last["served_brier"],
        "kept": last.get("kept", False),
        "n": last.get("n"),
        "run_id": last.get("run_id"),
        "trained_at": last.get("trained_at", "2026-08-30T00:00:00+00:00"),
        "fingerprint": last.get("fingerprint"),
        "history": entries,
    }
    p = models_dir / "production" / ensemble / mstore.SIDECAR_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return p


def entry(run_id: str, brier: float, n, fp: dict) -> dict:
    return {
        "run_id": run_id,
        "served_brier": brier,
        "kept": False,
        "n": n,
        "trained_at": "2026-08-01T00:00:00+00:00",
        "fingerprint": fp,
    }


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """A fake /app/models with a stubbed B2 pull and a muted Telegram."""
    monkeypatch.setattr(pull.b2_io, "download_prefix", lambda prefix, dest: 0)
    monkeypatch.setattr(pull, "_telegram", lambda text: pages.append(text))
    pages.clear()
    (tmp_path / "modal-incoming" / RUN).mkdir(parents=True)
    return tmp_path


pages: list[str] = []


def incoming_of(models_dir: Path) -> Path:
    return models_dir / "modal-incoming" / RUN


def boom(**kwargs):
    raise AssertionError("paired re-scoring must NOT run when the fingerprints match")


def row_for(summary: dict, bundle: str) -> dict:
    return next(d for d in summary["decisions"] if d["bundle"] == bundle)


# ── A. fingerprints ──────────────────────────────────────────────────


def test_fingerprint_read_from_training_report():
    report = {
        "data_quality": {"rows": 25342, "date_min": "2015-08-08T00:00:00+00:00", "date_max": "2026-08-29", "x": 1},
        "holdout_test": {"n": 3801, "raw": {"brier": 0.6}},
    }
    fp = mb.fingerprint_from_report(report)
    assert fp == {
        "rows": 25342,
        "date_min": "2015-08-08",
        "date_max": "2026-08-29",
        "holdout_n": 3801,
        # population keys: a different feature set or class count is a
        # different modelling problem, not a comparable Brier
        "feature_count": None,
        "target_classes": None,
    }


def test_fingerprint_none_without_rows():
    assert mb.fingerprint_from_report({"holdout_test": {"n": 10}}) is None
    assert mb.fingerprint_from_report(None) is None


def test_comparable_only_for_same_shaped_frames():
    small = fingerprint(3562 * 7, 3562)
    grown = fingerprint(3562 * 7 * 7, 25342)
    assert mb.fingerprints_comparable(small, small)
    # a normal week's growth stays comparable
    assert mb.fingerprints_comparable(small, fingerprint(int(3562 * 7 * 1.02), 3600))
    # the 2026-08-06 corpus jump does not
    assert not mb.fingerprints_comparable(small, grown)
    # a changed corpus start (loader scope moved) does not
    assert not mb.fingerprints_comparable(small, fingerprint(3562 * 7, 3562, date_min="2010-01-01"))
    # a changed feature set / class count is a different modelling problem
    assert not mb.fingerprints_comparable(small, fingerprint(3562 * 7, 3562, feature_count=43))
    assert not mb.fingerprints_comparable(small, fingerprint(3562 * 7, 3562, target_classes=2))
    # unknown fails CLOSED
    assert not mb.fingerprints_comparable(None, small)
    assert not mb.fingerprints_comparable(small, {})
    # so does an unknown held-out size: "averaged over how many rows" is half
    # of what makes two Briers comparable
    assert not mb.fingerprints_comparable(small, fingerprint(3562 * 7, None))
    assert not mb.fingerprints_comparable(fingerprint(3562 * 7, None), small)


def test_the_row_band_cannot_admit_disjoint_test_sets():
    """The held-out set is the LAST 15% of the frame, so a frame that grows by
    1/0.85 = 1.176x has a tail starting after the old frame ENDED — two Briers
    on completely disjoint populations. The band must sit below that."""
    assert mb.FINGERPRINT_ROW_BAND < 1 / 0.85
    base = fingerprint(1000, 150)
    assert not mb.fingerprints_comparable(base, fingerprint(1180, 177))


# ── B. scalar vs paired routing ──────────────────────────────────────


def test_matching_fingerprints_take_the_scalar_path(models_dir):
    fp = fingerprint(1000, 128)
    seed_incumbent(models_dir, ENSEMBLE, [entry("r1", 0.5052, 128, fp)])
    make_bundle_dir(incoming_of(models_dir), BUNDLE, brier=0.5000, rows=1010, holdout_n=129)

    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url=None, paired_fn=boom)
    row = row_for(summary, BUNDLE)

    assert row["comparison"] == "scalar"
    assert row["decision"] == "promote"
    assert row["paired"] is None
    assert "best-of-" in row["reason"]
    # the promoted sidecar carries the challenger's fingerprint + appended history
    doc = json.loads((models_dir / "staging" / ENSEMBLE / mstore.SIDECAR_NAME).read_text())
    assert doc["served_brier"] == 0.5000
    assert doc["fingerprint"]["rows"] == 1010
    assert [h["run_id"] for h in doc["history"]] == ["r1", RUN]


def test_differing_fingerprints_take_the_paired_path(models_dir):
    """The soccer case: incumbent scored on n=3,562 six-league rows, challenger on
    a 7x frame. The stored scalars must not decide; the paired delta does."""
    seed_incumbent(models_dir, SOCCER_ENSEMBLE, [entry("r1", 0.59345, 3562, fingerprint(23000, 3562))])
    make_bundle_dir(incoming_of(models_dir), SOCCER, brier=0.6015, rows=169000, holdout_n=25342)

    calls = []

    def fake_paired(**kwargs):
        calls.append(kwargs)
        return {
            "n": 4061,
            "delta": -0.0014,
            "se": 0.0010,
            "basis": "rows after the incumbent frame end 2026-08-05 (unseen by BOTH models)",
            "incumbent_brier": 0.6029,
            "challenger_brier": 0.6015,
        }

    summary = pull.gate_and_stage(
        RUN, str(models_dir), shadow=False, database_url="postgres://x", paired_fn=fake_paired
    )
    row = row_for(summary, SOCCER)

    assert row["comparison"] == "paired"
    # the challenger's scalar is WORSE than the incumbent's: the old gate rejected
    # it four weeks running. The paired delta promotes it.
    assert row["challenger_brier"] > row["incumbent_brier"]
    assert row["decision"] == "promote"
    assert row["paired"]["n"] == 4061
    assert "paired ΔBrier" in row["reason"]
    assert calls and calls[0]["bundle"] == SOCCER and calls[0]["incumbent_fp"]["rows"] == 23000


def test_paired_challenger_worse_is_rejected(models_dir):
    seed_incumbent(models_dir, SOCCER_ENSEMBLE, [entry("r1", 0.59345, 3562, fingerprint(23000, 3562))])
    make_bundle_dir(incoming_of(models_dir), SOCCER, brier=0.5800, rows=169000, holdout_n=25342)

    def worse(**kwargs):
        return {
            "n": 4061,
            "delta": 0.0120,
            "se": 0.0011,
            "basis": "shared tail",
            "incumbent_brier": 0.6,
            "challenger_brier": 0.612,
        }

    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url="postgres://x", paired_fn=worse)
    row = row_for(summary, SOCCER)

    # its scalar Brier (0.58) LOOKS better than the incumbent's 0.593 — different
    # test sets. Paired on shared rows it is worse, so it is kept out.
    assert row["decision"] == "reject"
    assert not (models_dir / "staging").exists() or not (models_dir / "staging" / SOCCER_ENSEMBLE).exists()
    assert summary["promoted"] == 0


def test_paired_too_small_rejects_loudly_and_never_falls_back(models_dir, caplog):
    seed_incumbent(models_dir, SOCCER_ENSEMBLE, [entry("r1", 0.59345, 3562, fingerprint(23000, 3562))])
    # a challenger whose scalar would sail through the broken comparison
    make_bundle_dir(incoming_of(models_dir), SOCCER, brier=0.4000, rows=169000, holdout_n=25342)

    def too_small(**kwargs):
        raise pull.PairedEvalError("shared evaluation set too small (n=12 < 100)")

    with caplog.at_level("ERROR"):
        summary = pull.gate_and_stage(
            RUN, str(models_dir), shadow=False, database_url="postgres://x", paired_fn=too_small
        )
    row = row_for(summary, SOCCER)

    assert row["decision"] == "reject"
    assert row["needs_manual"] is True
    assert summary["needs_manual"] == [SOCCER]
    assert "MANUAL DECISION NEEDED" in row["reason"] and "n=12 < 100" in row["reason"]
    assert any("MANUAL DECISION NEEDED" in r.getMessage() for r in caplog.records if r.levelname == "ERROR")
    assert not (models_dir / "staging" / SOCCER_ENSEMBLE).exists()
    assert pages and "MANUAL DECISION NEEDED" in pages[-1]


def test_unexpected_paired_error_also_rejects(models_dir):
    seed_incumbent(models_dir, SOCCER_ENSEMBLE, [entry("r1", 0.59345, 3562, fingerprint(23000, 3562))])
    make_bundle_dir(incoming_of(models_dir), SOCCER, brier=0.4000, rows=169000, holdout_n=25342)

    def kaboom(**kwargs):
        raise MemoryError("frame too big")

    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url="x", paired_fn=kaboom)
    row = row_for(summary, SOCCER)
    assert row["decision"] == "reject" and row["needs_manual"] is True
    assert "MemoryError" in row["reason"]


def test_paired_rescore_refuses_without_a_database(tmp_path):
    with pytest.raises(pull.PairedEvalError):
        pull.paired_rescore(
            bundle=SOCCER,
            production=str(tmp_path / "production"),
            challenger_dir=tmp_path / "challenger",
            database_url=None,
            incumbent_fp=None,
            incumbent_doc=None,
            challenger_fp=None,
        )


def test_paired_cutoff_falls_back_to_trained_at_for_a_legacy_sidecar():
    """EVERY incumbent sidecar in production today is a legacy one with no
    fingerprint. Without a cutoff the paired path scored both models on the
    challenger's whole tail — rows the incumbent trained and calibrated on."""
    import pandas as pd

    cutoff, why = pull._frame_cutoff(None, {"trained_at": "2026-08-06T09:00:00+00:00"})
    assert cutoff == pd.Timestamp("2026-08-06T09:00:00+00:00")
    assert "trained_at" in why

    # a fingerprint wins, and its date-precision end excludes that whole day
    cutoff, why = pull._frame_cutoff({"date_max": "2026-08-05"}, {"trained_at": "2026-08-06T09:00:00+00:00"})
    assert cutoff == pd.Timestamp("2026-08-06T00:00:00+00:00")
    assert "frame end" in why

    # nothing to bound with → the caller must refuse to decide
    cutoff, why = pull._frame_cutoff(None, {})
    assert cutoff is None and "neither" in why


def test_paired_decision_math():
    # not significantly worse and inside tolerance → promote
    ok, why = mb.paired_decision(-0.0014, 0.0010, 4061)
    assert ok and "ΔBrier -0.00140" in why
    # a real regression at large n is caught even though it is tiny
    assert not mb.paired_decision(0.0050, 0.0010, 4061)[0]
    # a tiny bundle's huge SE cannot wave a clearly-worse model through
    assert not mb.paired_decision(0.20, 0.10, 128)[0]


# ── C. the ratchet ───────────────────────────────────────────────────


def test_best_of_k_baseline_stops_the_tolerance_ratchet(models_dir):
    """nfl_tot drifted 0.5052 → 0.5430 over four promoted runs because each step
    was inside tolerance of LAST week's number. Against the best of the last K,
    the drift is caught."""
    fp = fingerprint(1000, 128)
    tol = mb.gate_tolerance(128)
    step = 0.6 * tol  # each step passes against its immediate predecessor…
    briers = [0.5052 + i * step for i in range(4)]
    assert all(briers[i + 1] <= briers[i] + tol for i in range(3))  # …under the OLD rule

    seed_incumbent(models_dir, ENSEMBLE, [entry("r0", briers[0], 128, fp)])
    decisions = []
    for i, brier in enumerate(briers[1:], start=1):
        d = incoming_of(models_dir) / BUNDLE
        if d.exists():
            import shutil

            shutil.rmtree(d)
        make_bundle_dir(incoming_of(models_dir), BUNDLE, brier=brier, rows=1000 + i, holdout_n=128)
        summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url=None, paired_fn=boom)
        row = row_for(summary, BUNDLE)
        decisions.append(row["decision"])
        if row["decision"] == "promote":
            # emulate swap_production: staging's sidecar becomes the incumbent
            staged = (models_dir / "staging" / ENSEMBLE / mstore.SIDECAR_NAME).read_text()
            live = models_dir / "production" / ENSEMBLE / mstore.SIDECAR_NAME
            live.parent.mkdir(parents=True, exist_ok=True)
            live.write_text(staged)

    # the first drift step is inside tolerance of the seed; the ones after it are
    # not, because they are measured against the BEST run, not the previous one.
    assert decisions == ["promote", "reject", "reject"]


def _promote_once(models_dir, brier, rows, i=0):
    """One gate_and_stage cycle on nfl_total, emulating swap_production."""
    import shutil

    d = incoming_of(models_dir) / BUNDLE
    if d.exists():
        shutil.rmtree(d)
    make_bundle_dir(incoming_of(models_dir), BUNDLE, brier=brier, rows=rows, holdout_n=128)
    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url=None, paired_fn=boom)
    row = row_for(summary, BUNDLE)
    if row["decision"] == "promote":
        staged = (models_dir / "staging" / ENSEMBLE / mstore.SIDECAR_NAME).read_text()
        live = models_dir / "production" / ENSEMBLE / mstore.SIDECAR_NAME
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(staged)
    return row


def test_the_champion_never_ages_out_of_the_window(models_dir):
    """`history` is a SLIDING window of K, so the best entry eventually falls
    off the end and the ratchet resumes — one tolerance per K promotions
    instead of one per promotion (simulated on the real modules: nfl_total
    reaches 0.741, worse than a coin flip, after 39 'PROMOTE' runs). The
    monotone champion is what makes the bar stop moving."""
    fp = fingerprint(1000, 128)
    tol = mb.gate_tolerance(128)
    seed = 0.5052
    seed_incumbent(models_dir, ENSEMBLE, [entry("r0", seed, 128, fp)])

    # Fill the window past K with runs that each sit just inside tolerance of
    # the seed: every one of them promotes, and every one is worse.
    drifted = seed + 0.9 * tol
    for i in range(mb.HISTORY_LIMIT + 2):
        assert _promote_once(models_dir, drifted, 1000 + i, i)["decision"] == "promote"

    doc = json.loads((models_dir / "production" / ENSEMBLE / mstore.SIDECAR_NAME).read_text())
    assert [h["served_brier"] for h in doc["history"]] == [drifted] * mb.HISTORY_LIMIT
    assert doc["champion"]["run_id"] == "r0", "the seed has aged out of the window but not out of the bar"
    assert doc["champion"]["served_brier"] == seed

    # Without the champion the window's best would now be `drifted`, and this
    # next step (a second tolerance above the seed) would promote.
    nxt = _promote_once(models_dir, drifted + 0.9 * tol, 1010, 99)
    assert nxt["decision"] == "reject"
    assert nxt["baseline_brier"] == seed and nxt["baseline_run_id"] == "r0"


def test_a_better_run_becomes_the_champion(models_dir):
    fp = fingerprint(1000, 128)
    seed_incumbent(models_dir, ENSEMBLE, [entry("r0", 0.5052, 128, fp)])
    assert _promote_once(models_dir, 0.4800, 1000, 1)["decision"] == "promote"
    doc = json.loads((models_dir / "production" / ENSEMBLE / mstore.SIDECAR_NAME).read_text())
    assert doc["champion"]["served_brier"] == 0.4800


def test_a_changed_frame_family_resets_the_champion(models_dir):
    """A Brier from another population is not a bar this one can be held to —
    it is exactly the comparison the fingerprint exists to refuse."""
    seed_incumbent(models_dir, ENSEMBLE, [entry("r0", 0.4000, 128, fingerprint(1000, 128))])
    payload = mstore.build_payload(
        ENSEMBLE,
        served_brier=0.6015,
        kept=False,
        n=25342,
        run_id="r1",
        fingerprint=fingerprint(169000, 25342),
        previous=json.loads((models_dir / "production" / ENSEMBLE / mstore.SIDECAR_NAME).read_text()),
    )
    assert payload["champion"]["run_id"] == "r1" and payload["champion"]["served_brier"] == 0.6015


def test_best_comparable_brier_ignores_incomparable_history():
    fp_old = fingerprint(23000, 3562)
    fp_new = fingerprint(169000, 25342)
    doc = {
        "served_brier": 0.6015,
        "fingerprint": fp_new,
        "history": [entry("r1", 0.4000, 3562, fp_old), entry("r2", 0.6015, 25342, fp_new)],
    }
    best, run_id = mstore.best_comparable_brier(doc, fp_new)
    assert (best, run_id) == (0.6015, "r2")  # the 0.40 was scored on another frame
    assert mstore.best_comparable_brier(doc, fingerprint(5, 5, date_min="1999-01-01")) == (None, None)


def test_history_is_capped_and_legacy_sidecars_still_read():
    legacy = {"ensemble_name": ENSEMBLE, "served_brier": 0.5, "kept": False, "n": 128, "run_id": "old"}
    assert mstore.incumbent_fingerprint(legacy) is None
    assert [e["run_id"] for e in mstore.history_entries(legacy)] == ["old"]

    doc = legacy
    for i in range(mb.HISTORY_LIMIT + 3):
        doc = mstore.build_payload(
            ENSEMBLE,
            served_brier=0.5,
            kept=False,
            n=128,
            run_id=f"r{i}",
            fingerprint=fingerprint(1000, 128),
            previous=doc,
        )
    assert len(doc["history"]) == mb.HISTORY_LIMIT
    assert doc["history"][-1]["run_id"] == f"r{mb.HISTORY_LIMIT + 2}"
    # original payload keys are unchanged
    assert {"ensemble_name", "served_brier", "kept", "n", "run_id", "trained_at"} <= set(doc)


# ── D. escape hatches + output shape ─────────────────────────────────


def test_seed_promote_when_there_is_no_incumbent(models_dir):
    make_bundle_dir(incoming_of(models_dir), BUNDLE, brier=0.5052, rows=1000, holdout_n=128)
    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url=None, paired_fn=boom)
    row = row_for(summary, BUNDLE)
    assert row["decision"] == "promote" and row["comparison"] == "seed"
    assert "seed" in row["reason"]
    assert (models_dir / "staging" / ENSEMBLE / mstore.SIDECAR_NAME).exists()


def test_no_holdout_test_still_promotes(models_dir):
    seed_incumbent(models_dir, ENSEMBLE, [entry("r1", 0.5052, 128, fingerprint(1000, 128))])
    make_bundle_dir(incoming_of(models_dir), BUNDLE, brier=None, rows=1000, holdout_n=None)
    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url=None, paired_fn=boom)
    row = row_for(summary, BUNDLE)
    assert row["decision"] == "promote" and row["comparison"] == "ungated"
    assert row["challenger_brier"] is None
    # no number → the incumbent sidecar is left alone
    assert not (models_dir / "staging" / ENSEMBLE / mstore.SIDECAR_NAME).exists()


def test_failed_training_is_rejected(models_dir):
    seed_incumbent(models_dir, ENSEMBLE, [entry("r1", 0.5052, 128, fingerprint(1000, 128))])
    make_bundle_dir(incoming_of(models_dir), BUNDLE, brier=0.1, rows=1000, holdout_n=128, status="error")
    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url=None, paired_fn=boom)
    row = row_for(summary, BUNDLE)
    assert row["decision"] == "reject" and "status='error'" in row["reason"]
    # staging holds only the decision log; no artifacts and no new sidecar
    assert list((models_dir / "staging").iterdir()) == [models_dir / "staging" / "promote_decisions.json"]


def test_missing_training_report_is_not_comparable(models_dir):
    """No report → no challenger fingerprint → the scalar comparison is unproven,
    so the gate takes the paired path instead of trusting it."""
    seed_incumbent(models_dir, ENSEMBLE, [entry("r1", 0.5052, 128, fingerprint(1000, 128))])
    make_bundle_dir(incoming_of(models_dir), BUNDLE, brier=0.5000, rows=1000, holdout_n=128, with_report=False)

    def unavailable(**kwargs):
        raise pull.PairedEvalError("no DATABASE_URL — cannot rebuild a shared evaluation set")

    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=False, database_url=None, paired_fn=unavailable)
    row = row_for(summary, BUNDLE)
    assert row["comparison"] == "paired" and row["decision"] == "reject" and row["needs_manual"] is True


def test_promote_decisions_json_keeps_its_shape(models_dir):
    seed_incumbent(models_dir, ENSEMBLE, [entry("r1", 0.5052, 128, fingerprint(1000, 128))])
    make_bundle_dir(incoming_of(models_dir), BUNDLE, brier=0.5000, rows=1005, holdout_n=128)
    summary = pull.gate_and_stage(RUN, str(models_dir), shadow=True, database_url=None, paired_fn=boom)

    written = json.loads((incoming_of(models_dir) / "promote_decisions.json").read_text())
    assert written == summary  # the file is exactly what the caller (airflow task) gets back
    assert {"run_id", "shadow", "promoted", "decisions"} <= set(written)
    assert written["run_id"] == RUN and written["shadow"] is True and written["promoted"] == 1
    legacy_keys = {
        "bundle",
        "ensemble_name",
        "challenger_brier",
        "incumbent_brier",
        "kept_calibration",
        "n",
        "decision",
        "reason",
    }
    assert legacy_keys <= set(written["decisions"][0])
    # shadow writes nothing into staging
    assert not (models_dir / "staging").exists()
    # the digest still renders
    assert pages and pages[-1].startswith("🤖 Modal retrain ") and "would promote 1/1" in pages[-1]


# ---------------------------------------------------------------------------
# Legacy-sidecar fingerprint recovery.
#
# Sidecars written before the fingerprint existed carry only a run_id, so the
# paired path fell back to the sidecar's `trained_at` — a wall-clock build
# timestamp, not a data cut-off. For an incumbent built the same morning as the
# challenger that leaves ~no rows after the cutoff, so every legacy bundle died
# with "shared evaluation set too small" and the gate DEADLOCKED: nothing could
# promote until sidecars had fingerprints, and sidecars only get one by being
# promoted. Each run's per-bundle training_report.json is retained under
# modal-incoming/<run_id>/<bundle>/, which is where the real frame end lives.
# ---------------------------------------------------------------------------


def _write_report(root: Path, run_id: str, bundle: str, rows: int, date_max: str, holdout_n: int = 500) -> None:
    d = root / run_id / bundle
    d.mkdir(parents=True, exist_ok=True)
    (d / "training_report.json").write_text(
        json.dumps(
            {
                "data_quality": {"rows": rows, "date_min": "2016-08-05T00:00:00+00:00", "date_max": date_max},
                "holdout_test": {"n": holdout_n},
            }
        )
    )


def test_recovers_frame_end_from_the_incumbents_own_run(tmp_path):
    _write_report(tmp_path, "manual__2026-08-06T0805", "soccer_match_result", 23746, "2026-08-05T00:00:00+00:00")
    fp, why = pull.recover_incumbent_fingerprint(tmp_path, "soccer_match_result", {"run_id": "manual__2026-08-06T0805"})
    assert fp is not None
    assert fp["rows"] == 23746
    assert fp["date_max"].startswith("2026-08-05")
    assert "manual__2026-08-06T0805" in why


def test_recovered_cutoff_is_the_data_boundary_not_the_build_time(tmp_path):
    """The whole point: trained_at would strand the evaluation set."""
    _write_report(tmp_path, "run-a", "soccer_match_result", 23746, "2026-08-05T00:00:00+00:00")
    doc = {"run_id": "run-a", "trained_at": "2026-08-30T04:05:31+00:00"}
    fp, _ = pull.recover_incumbent_fingerprint(tmp_path, "soccer_match_result", doc)
    cutoff, desc = pull._frame_cutoff(fp, doc)
    assert cutoff is not None
    # August 6th, not August 30th — three weeks of extra evaluation rows.
    assert str(cutoff).startswith("2026-08-06")
    assert "frame end" in desc


def test_no_run_id_recovers_nothing(tmp_path):
    fp, why = pull.recover_incumbent_fingerprint(tmp_path, "soccer_match_result", {"trained_at": "2026-08-30"})
    assert fp is None
    assert "run_id" in why


def test_missing_report_recovers_nothing(tmp_path):
    fp, why = pull.recover_incumbent_fingerprint(tmp_path, "soccer_match_result", {"run_id": "never-pulled"})
    assert fp is None
    assert "never-pulled" in why


def test_report_without_a_row_count_recovers_nothing(tmp_path):
    d = tmp_path / "run-b" / "soccer_match_result"
    d.mkdir(parents=True)
    (d / "training_report.json").write_text(json.dumps({"holdout_test": {"n": 10}}))
    fp, why = pull.recover_incumbent_fingerprint(tmp_path, "run-b", {"run_id": "run-b"})
    assert fp is None
    assert why


def test_recovery_is_skipped_when_no_incoming_root_is_known(tmp_path):
    fp, why = pull.recover_incumbent_fingerprint(None, "soccer_match_result", {"run_id": "run-a"})
    assert fp is None
    assert why
