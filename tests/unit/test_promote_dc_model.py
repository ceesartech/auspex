"""Unit tests for the Dixon-Coles side-artifact promoter
(scripts/promote_dc_model.py): the pure promote/noop/refuse decision, the
serve-path smoke check, the atomic rollback command, and the whole
check -> backup -> promote -> verify flow against real (tiny) artifacts in a
temp tree, loaded through precompute_predictions' own loader helper —
including a promote step that fails half-way, a failed verification with and
without --rollback-on-verify-failure, and the rollback command as the last
line of every failure."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "services" / "ml-models" / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


promoter = _load("promote_dc_model", "promote_dc_model.py")
trainer = _load("train_halftime_dixon_coles", "train_halftime_dixon_coles.py")
precompute = _load("precompute_predictions", "precompute_predictions.py")

HT = promoter.REGISTRY["ht"]
NEWEST = "2026-10-31T01:05:00+00:00"


def payload(teams=939, leagues=31, rho=-0.0018, config="dixon_coles_ht_soccer", baselines=True):
    return {
        "team_attack": {f"T{i}": 1.0 for i in range(teams)},
        "league_baselines": {f"L{i}": [0.68, 0.55] for i in range(leagues)} if baselines else {},
        "global_home_baseline": 0.677 if baselines else None,
        "rho": rho,
        "config": {"name": config},
    }


def manifest(
    sha="s-new",
    run_id="r1",
    holdout=0.0,
    model="dixon_coles_ht_soccer",
    newest=NEWEST,
    n_rows=108_226,
    observed=1.215,
):
    return {
        "model": model,
        "run_id": run_id,
        "model_sha256": sha,
        "holdout_fraction": holdout,
        "newest_match_date": newest,
        "n_rows": n_rows,
        "observed_mean_total_goals": observed,
    }


def decide(**kw):
    args = dict(
        staged_payload=payload(),
        staged_sha="s-new",
        staged_manifest=manifest(),
        production_payload=payload(),
        production_sha="s-old",
        production_manifest=None,
        expect_run_id=None,
    )
    args.update(kw)
    return promoter.evaluate_promotion(HT, **args)


def failed(decision):
    return [c.name for c in decision.refusals]


# ── the pure decision ────────────────────────────────────────────────


class TestEvaluatePromotion:
    def test_a_good_refit_is_promoted(self):
        d = decide()
        assert d.action == "promote" and d.refusals == []

    def test_no_manifest_refuses(self):
        d = decide(staged_manifest=None)
        assert d.action == "refuse" and failed(d) == ["staged manifest.json present"]

    def test_manifest_for_another_file_refuses(self):
        d = decide(staged_manifest=manifest(sha="something-else"))
        assert d.action == "refuse" and failed(d) == ["manifest model_sha256 matches the staged model.bin"]

    def test_run_id_is_enforced_only_when_expected(self):
        assert decide(expect_run_id="r1").action == "promote"
        d = decide(expect_run_id="scheduled__2026-12-01T053000")
        assert d.action == "refuse" and failed(d) == ["staged manifest is from the expected run"]

    def test_staged_equal_to_production_is_nothing_to_promote(self):
        d = decide(production_sha="s-new")
        assert d.action == "noop" and d.refusals == []

    def test_a_foreign_run_is_refused_even_when_it_is_already_production(self):
        # The run-id check comes first: the DAG must not go green on a file
        # its own train task did not write.
        d = decide(production_sha="s-new", expect_run_id="other-run")
        assert d.action == "refuse"

    @pytest.mark.parametrize(
        "kw",
        [
            {"staged_payload": payload(config="dixon_coles")},
            {"staged_manifest": manifest(model="dixon_coles_h2_soccer")},
        ],
    )
    def test_wrong_config_name_refuses(self, kw):
        d = decide(**kw)
        assert d.action == "refuse" and failed(d) == ["config name dixon_coles_ht_soccer"]

    @pytest.mark.parametrize("holdout", [0.2, None])
    def test_a_holdout_diagnostic_fit_is_never_promoted(self, holdout):
        d = decide(staged_manifest=manifest(holdout=holdout))
        assert d.action == "refuse" and failed(d) == ["fitted on every row (holdout_fraction 0)"]

    def test_legacy_shaped_staged_artifact_refuses(self):
        d = decide(staged_payload=payload(baselines=False))
        assert d.action == "refuse" and "per-league baselines fitted" in failed(d)

    @pytest.mark.parametrize("rho", [float("nan"), float("inf"), None, "0.1"])
    def test_non_finite_rho_refuses(self, rho):
        d = decide(staged_payload=payload(rho=rho))
        assert d.action == "refuse" and failed(d) == ["finite rho"]

    def test_team_coverage_regression(self):
        assert decide(staged_payload=payload(teams=921)).action == "promote"  # 921 >= 0.98 * 939 = 920.2
        d = decide(staged_payload=payload(teams=920))
        assert d.action == "refuse" and failed(d) == ["teams >= 98% of production"]

    def test_league_coverage_regression(self):
        assert decide(staged_payload=payload(leagues=30)).action == "promote"
        d = decide(staged_payload=payload(leagues=29))
        assert d.action == "refuse" and failed(d) == ["leagues >= production - 1"]

    def test_legacy_production_skips_the_coverage_comparisons(self):
        legacy = payload(teams=1500, leagues=0, baselines=False, config="dixon_coles")
        d = decide(staged_payload=payload(teams=939), production_payload=legacy)
        assert d.action == "promote"
        skipped = [c for c in d.checks if c.name == "teams / leagues vs production"]
        assert skipped and "legacy production" in skipped[0].detail

    def test_first_promotion_has_nothing_to_compare(self):
        d = decide(production_payload=None, production_sha=None)
        assert d.action == "promote"

    def test_older_data_than_production_refuses(self):
        newer_prod = manifest(newest="2026-11-30T20:00:00+00:00")
        d = decide(production_manifest=newer_prod)
        assert d.action == "refuse" and failed(d) == ["trained on data no older than production's"]
        assert decide(production_manifest=manifest(newest="2026-09-30T20:00:00+00:00")).action == "promote"
        assert decide(production_manifest=manifest(newest=NEWEST)).action == "promote"

    def test_unparseable_staged_date_refuses_when_production_has_one(self):
        d = decide(staged_manifest=manifest(newest="yesterday"), production_manifest=manifest())
        assert d.action == "refuse"


class TestCrossRunDataSanity:
    """Staged vs production manifest: the trainer's level canary only compares
    a fit with its own frame, so a wrong frame must be caught here."""

    LEVEL = "observed mean total goals within 5% of production's"
    ROWS = "rows >= 99% of production's"

    def test_the_same_data_plus_a_month_is_promoted(self):
        d = decide(staged_manifest=manifest(n_rows=109_100, observed=1.218), production_manifest=manifest())
        assert d.action == "promote"
        assert {self.LEVEL, self.ROWS} <= {c.name for c in d.checks}

    @pytest.mark.parametrize("observed", [1.27, 1.13, 2.7])  # +5.8%, -5.8%, full-time goals in the HT columns
    def test_a_level_shift_beyond_five_percent_refuses(self, observed):
        d = decide(staged_manifest=manifest(observed=observed), production_manifest=manifest(observed=1.20))
        assert d.action == "refuse" and failed(d) == [self.LEVEL]
        assert "vs production 1.2" in d.refusals[0].detail

    @pytest.mark.parametrize("observed", [1.25, 1.15, 1.20])  # +4.2%, -4.2%, flat
    def test_a_level_shift_inside_five_percent_is_promoted(self, observed):
        d = decide(staged_manifest=manifest(observed=observed), production_manifest=manifest(observed=1.20))
        assert d.action == "promote"

    def test_a_row_count_drop_beyond_one_percent_refuses(self):
        prod = manifest(n_rows=100_000)
        assert decide(staged_manifest=manifest(n_rows=99_000), production_manifest=prod).action == "promote"
        d = decide(staged_manifest=manifest(n_rows=98_999), production_manifest=prod)
        assert d.action == "refuse" and failed(d) == [self.ROWS]
        assert d.refusals[0].detail == "98999 vs production 100000"

    @pytest.mark.parametrize("bad", [None, float("nan"), 0.0, "1.2", True])
    def test_an_unusable_level_refuses_whichever_side_it_is_on(self, bad):
        d = decide(staged_manifest=manifest(observed=bad), production_manifest=manifest())
        assert d.action == "refuse" and failed(d) == [self.LEVEL]
        d = decide(staged_manifest=manifest(), production_manifest=manifest(observed=bad))
        assert d.action == "refuse" and failed(d) == [self.LEVEL]

    @pytest.mark.parametrize("bad", [None, 0, -5, 1.5e5, True])
    def test_unusable_row_counts_refuse_whichever_side_they_are_on(self, bad):
        d = decide(staged_manifest=manifest(n_rows=bad), production_manifest=manifest())
        assert d.action == "refuse" and failed(d) == [self.ROWS]
        d = decide(staged_manifest=manifest(), production_manifest=manifest(n_rows=bad))
        assert d.action == "refuse" and failed(d) == [self.ROWS]

    def test_legacy_production_without_a_manifest_skips_it(self):
        # A staged frame that would fail both checks against any manifest.
        d = decide(staged_manifest=manifest(n_rows=10, observed=2.7), production_manifest=None)
        assert d.action == "promote"
        skipped = [c for c in d.checks if c.name == "observed level / rows vs production"]
        assert skipped and skipped[0].ok and "skipped" in skipped[0].detail
        assert not {self.LEVEL, self.ROWS} & {c.name for c in d.checks}

    def test_a_noop_is_not_blocked_by_it(self):
        d = decide(production_sha="s-new", production_manifest=manifest(observed=3.0, n_rows=1))
        assert d.action == "noop"


class TestRollbackCommand:
    def test_restore_is_copy_beside_then_rename(self):
        cmd = promoter.rollback_command(HT, Path("/app/models/backup-x"), backup_has_manifest=True)
        prod = HT.production_dir
        assert cmd.startswith("cd /opt/auspex && docker compose exec -T api sh -c '")
        assert f"cp -a /app/models/backup-x/model.bin {prod}/model.bin.new" in cmd
        assert f"mv -f {prod}/model.bin.new {prod}/model.bin" in cmd
        assert f"mv -f {prod}/manifest.json.new {prod}/manifest.json" in cmd
        assert f"cp /app/models/backup-x/model.bin {prod}/model.bin" not in cmd  # never over the live file

    def test_legacy_backup_removes_the_promoted_manifest(self):
        cmd = promoter.rollback_command(HT, Path("/app/models/backup-x"), backup_has_manifest=False)
        assert cmd.endswith(f"rm -f {HT.production_dir}/manifest.json'")

    def test_first_promotion_rolls_back_to_nothing(self):
        cmd = promoter.rollback_command(HT, None, backup_has_manifest=False)
        assert f"rm -f {HT.production_dir}/model.bin {HT.production_dir}/manifest.json" in cmd


# ── registry ↔ trainer ↔ serve path ──────────────────────────────────


class TestRegistry:
    def test_ht_paths_match_the_trainer(self):
        assert HT.staging_dir == trainer.DEFAULT_OUTPUT_DIR
        assert HT.name == HT.expected_config_name == trainer.MODEL_NAME

    def test_the_trainer_writes_ht_staging_and_never_production(self):
        assert trainer.output_dir_refusal(HT.staging_dir) is None
        assert trainer.output_dir_refusal(HT.production_dir) is not None

    def test_ht_production_is_the_file_the_serve_loader_reads(self, monkeypatch):
        seen = []
        monkeypatch.setattr(precompute, "_load_dixon_coles_artifact", lambda path, label: seen.append(path))
        getattr(precompute, HT.serve_loader)()
        assert seen == [str(HT.production_dir / "model.bin")]


# ── the full flow on a temp tree ─────────────────────────────────────


def ht_frame(seed: int, rounds: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(rounds):
        for league, teams in {"L1": ["A", "B", "C", "D"], "L2": ["E", "F", "G", "H"]}.items():
            for home in teams:
                for away in teams:
                    if home != away:
                        hs, as_ = int(rng.poisson(0.7)), int(rng.poisson(0.55))
                        rows.append(
                            {
                                "match_date": pd.Timestamp("2025-01-04", tz="UTC") + pd.Timedelta(days=7 * r),
                                "league_id": league,
                                "home_team": home,
                                "away_team": away,
                                "home_score": hs,
                                "away_score": as_,
                                "match_outcome": 0 if hs > as_ else (1 if hs == as_ else 2),
                            }
                        )
    return pd.DataFrame(rows)


def write_fit(out_dir: Path, seed: int, run_id: str, with_manifest: bool = True, rounds: int = 6) -> str:
    from predictors.model_config import DIXON_COLES_CONFIG
    from predictors.poisson_models import DixonColesPredictor

    frame = ht_frame(seed, rounds)
    model = DixonColesPredictor(replace(DIXON_COLES_CONFIG, name="dixon_coles_ht_soccer"))
    result = model.train(frame)
    if with_manifest:
        fields = dict(result=result, train_df=frame, run_id=run_id, git_sha="abc", holdout_fraction=0.0)
        return trainer.write_artifact(model, out_dir, fields)["model_sha256"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "model.bin"))
    return promoter.sha256_file(out_dir / "model.bin")


def write_production(spec, run_id: str = "old", **kw) -> str:
    """Last month's fit."""
    return write_fit(spec.production_dir, seed=1, rounds=6, run_id=run_id, **kw)


def write_staged(spec, run_id: str = "new", **kw) -> str:
    """This month's refit: last month's frame plus one more round (the same
    rng stream, so a strict superset) — what the cross-run checks expect."""
    return write_fit(spec.staging_dir, seed=1, rounds=7, run_id=run_id, **kw)


@pytest.fixture
def tree(tmp_path):
    spec = replace(
        HT,
        staging_dir=tmp_path / "models" / "staging-ht" / "dixon_coles_ht_soccer" / "1.0.0",
        production_dir=tmp_path / "models" / "production" / "dixon_coles_ht_soccer" / "1.0.0",
    )
    return spec, tmp_path / "models"


def run(
    spec,
    models_root,
    *,
    check_only=False,
    expect_run_id="new",
    stamp="20261101T053000Z",
    rollback_on_verify_failure=False,
    **loaders,
):
    kw = dict(
        load_staged=lambda path: precompute._load_dixon_coles_artifact(str(path), "staged ht"),
        load_production=lambda: precompute._load_dixon_coles_artifact(str(spec.production_dir / "model.bin"), "HT"),
    )
    kw.update(loaders)
    return promoter.promote(
        spec,
        check_only=check_only,
        expect_run_id=expect_run_id,
        models_root=models_root,
        stamp=stamp,
        rollback_on_verify_failure=rollback_on_verify_failure,
        **kw,
    )


def backups(models_root: Path):
    return sorted(p.name for p in models_root.glob("backup-*"))


def leftovers(spec):
    return sorted(p.name for p in spec.production_dir.glob("*.new"))


BACKUP = "backup-dixon_coles_ht_soccer-20261101T053000Z"


def last_line(caplog) -> str:
    """The last record the promoter logged (its last stderr line)."""
    return [r for r in caplog.records if r.name == "promote_dc_model"][-1].getMessage()


def file_bytes(path: Path):
    return path.read_bytes() if path.exists() else None


class TestPromoteFlow:
    def test_promotes_backs_up_and_verifies(self, tree, caplog):
        spec, root = tree
        old_sha = write_production(spec)
        old_manifest = json.loads((spec.production_dir / "manifest.json").read_text())
        new_sha = write_staged(spec)
        assert old_sha != new_sha

        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root) == 0
        assert promoter.sha256_file(spec.production_dir / "model.bin") == new_sha
        assert json.loads((spec.production_dir / "manifest.json").read_text())["run_id"] == "new"
        backup = root / "backup-dixon_coles_ht_soccer-20261101T053000Z"
        assert promoter.sha256_file(backup / "model.bin") == old_sha
        assert json.loads((backup / "manifest.json").read_text()) == old_manifest
        assert leftovers(spec) == []
        assert "serve loader ok" in caplog.text and "match_result_ht home=" in caplog.text
        assert f"mv -f {spec.production_dir}/model.bin.new {spec.production_dir}/model.bin" in caplog.text

        # A second run finds staging == production: nothing to do, no new backup.
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root, stamp="20261101T060000Z") == 0
        assert backups(root) == ["backup-dixon_coles_ht_soccer-20261101T053000Z"]
        assert "Nothing to promote" in caplog.text

    def test_check_only_changes_nothing(self, tree):
        spec, root = tree
        old_sha = write_production(spec)
        write_staged(spec)
        assert run(spec, root, check_only=True) == 0
        assert promoter.sha256_file(spec.production_dir / "model.bin") == old_sha
        assert backups(root) == [] and leftovers(spec) == []

    def test_a_staged_file_from_another_run_is_refused(self, tree):
        spec, root = tree
        old_sha = write_production(spec)
        write_staged(spec, run_id="last-month")
        assert run(spec, root, expect_run_id="this-month") == 1
        assert promoter.sha256_file(spec.production_dir / "model.bin") == old_sha
        assert backups(root) == []

    def test_a_staged_file_without_a_manifest_is_refused(self, tree):
        spec, root = tree
        write_production(spec)
        write_staged(spec, with_manifest=False)
        assert run(spec, root, expect_run_id=None) == 1

    def test_missing_staged_artifact_is_refused(self, tree):
        spec, root = tree
        write_production(spec)
        assert run(spec, root) == 1

    def test_a_staged_file_the_serve_loader_cannot_load_is_refused(self, tree):
        spec, root = tree
        old_sha = write_production(spec)
        write_staged(spec)
        assert run(spec, root, load_staged=lambda path: None) == 1
        assert promoter.sha256_file(spec.production_dir / "model.bin") == old_sha
        assert backups(root) == []

    def test_staging_rewritten_mid_promotion_is_refused(self, tree):
        spec, root = tree
        old_sha = write_production(spec)
        write_staged(spec)

        def load_then_race(path):
            model = precompute._load_dixon_coles_artifact(str(path), "staged ht")
            write_fit(spec.staging_dir, seed=2, run_id="new")  # a concurrent trainer run lands
            return model

        assert run(spec, root, load_staged=load_then_race) == 1
        assert promoter.sha256_file(spec.production_dir / "model.bin") == old_sha
        assert leftovers(spec) == []

    def test_failed_verification_is_loud_and_prints_the_rollback(self, tree, caplog):
        spec, root = tree
        write_production(spec)
        new_sha = write_staged(spec)
        with caplog.at_level(logging.ERROR, logger="promote_dc_model"):
            assert run(spec, root, load_production=lambda: None) == 1
        assert "Verification FAILED: the serve loader cannot serve the promoted artifact" in caplog.text
        # Without the flag production keeps the unverified file...
        assert promoter.sha256_file(spec.production_dir / "model.bin") == new_sha
        # ...and the last line is the command that undoes it.
        assert last_line(caplog).startswith("Rollback: cd /opt/auspex && docker compose exec -T api sh -c '")
        assert f"{BACKUP}/model.bin" in last_line(caplog)

    def test_legacy_production_is_backed_up_and_promoted_over(self, tree, caplog):
        spec, root = tree
        legacy_sha = write_production(spec, with_manifest=False)
        new_sha = write_staged(spec)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root) == 0
        assert promoter.sha256_file(spec.production_dir / "model.bin") == new_sha
        backup = root / "backup-dixon_coles_ht_soccer-20261101T053000Z"
        assert promoter.sha256_file(backup / "model.bin") == legacy_sha
        assert not (backup / "manifest.json").exists()
        assert f"rm -f {spec.production_dir}/manifest.json" in caplog.text

    def test_a_truncated_staged_frame_is_refused(self, tree, caplog):
        spec, root = tree
        old_sha = write_production(spec)  # 144 rows
        write_fit(spec.staging_dir, seed=1, rounds=3, run_id="new")  # its first 3 of 6 weeks: 72 rows
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root) == 1
        assert promoter.sha256_file(spec.production_dir / "model.bin") == old_sha
        assert "FAIL  rows >= 99% of production's (72 vs production 144)" in caplog.text
        assert backups(root) == []

    def test_first_promotion(self, tree):
        spec, root = tree
        new_sha = write_staged(spec)
        assert run(spec, root) == 0
        assert promoter.sha256_file(spec.production_dir / "model.bin") == new_sha
        assert backups(root) == []


class TestPromoteStepFailures:
    """Step 3 (copy -> re-hash -> replace) failing part-way."""

    def test_a_failed_copy_leaves_production_untouched_and_no_new_files(self, tree, caplog, monkeypatch):
        spec, root = tree
        old_sha = write_production(spec)
        old_manifest = file_bytes(spec.production_dir / "manifest.json")
        write_staged(spec)
        real_copy = promoter._copy_durable

        def copy(src, dst):
            if Path(dst).name == "manifest.json.new":
                Path(dst).write_text("{half")  # a partial copy is left behind...
                raise OSError(28, "No space left on device")
            real_copy(src, dst)

        monkeypatch.setattr(promoter, "_copy_durable", copy)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root) == 1
        assert promoter.sha256_file(spec.production_dir / "model.bin") == old_sha
        assert file_bytes(spec.production_dir / "manifest.json") == old_manifest
        assert leftovers(spec) == []  # ...and removed, with model.bin.new
        assert "production untouched" in caplog.text and "No space left on device" in caplog.text
        # The rollback was built before copying and is still the last line.
        assert last_line(caplog).startswith("Rollback: cd /opt/auspex")
        assert f"{BACKUP}/model.bin" in last_line(caplog)

    def test_a_failed_manifest_replace_reports_a_half_promotion(self, tree, caplog, monkeypatch):
        spec, root = tree
        write_production(spec)
        old_manifest = file_bytes(spec.production_dir / "manifest.json")
        new_sha = write_staged(spec)
        real_replace = promoter.os.replace

        def replace_(src, dst):
            if Path(dst).name == "manifest.json":
                raise OSError(5, "Input/output error")
            real_replace(src, dst)

        monkeypatch.setattr(promoter.os, "replace", replace_)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root) == 1
        monkeypatch.undo()
        # model.bin went in, the manifest did not: that is what the log says.
        assert promoter.sha256_file(spec.production_dir / "model.bin") == new_sha
        assert file_bytes(spec.production_dir / "manifest.json") == old_manifest
        assert leftovers(spec) == []
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("HALF-PROMOTED" in m and "model.bin was replaced" in m for m in errors)
        assert last_line(caplog).startswith("Rollback: cd /opt/auspex")
        assert "4/4 verify" not in caplog.text

    def test_an_unexpected_exception_still_ends_with_the_rollback(self, tree, caplog, monkeypatch):
        spec, root = tree
        write_production(spec)
        write_staged(spec)

        def boom(*a, **k):
            raise RuntimeError("unforeseen")

        monkeypatch.setattr(promoter, "_verify_production", boom)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root) == 1
        assert "Unexpected failure while promoting" in caplog.text
        assert last_line(caplog).startswith("Rollback: cd /opt/auspex")


class TestRollbackOnVerifyFailure:
    def test_restores_the_backup_and_exits_1(self, tree, caplog):
        spec, root = tree
        old_sha = write_production(spec)
        old_manifest = file_bytes(spec.production_dir / "manifest.json")
        write_staged(spec)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root, rollback_on_verify_failure=True, load_production=lambda: None) == 1
        assert promoter.sha256_file(spec.production_dir / "model.bin") == old_sha
        assert file_bytes(spec.production_dir / "manifest.json") == old_manifest
        assert leftovers(spec) == []
        assert "restored to the backup" in caplog.text and "verified" in caplog.text
        assert last_line(caplog).startswith("Rollback: cd /opt/auspex")

    def test_a_sha_mismatch_after_promotion_is_also_restored(self, tree, caplog, monkeypatch):
        spec, root = tree
        old_sha = write_production(spec)
        write_staged(spec)
        real_sha = promoter.sha256_file
        calls = {"n": 0}

        def sha(path):
            # The first hash of production AFTER the swap (step 4) lies once.
            if Path(path) == spec.production_dir / "model.bin" and "4/4 verify" in caplog.text and not calls["n"]:
                calls["n"] += 1
                return "f" * 64
            return real_sha(path)

        monkeypatch.setattr(promoter, "sha256_file", sha)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root, rollback_on_verify_failure=True) == 1
        assert real_sha(spec.production_dir / "model.bin") == old_sha
        assert "production sha256 ffffffffffff != staged" in caplog.text

    def test_a_legacy_backup_restores_without_a_manifest(self, tree, caplog):
        spec, root = tree
        legacy_sha = write_production(spec, with_manifest=False)
        write_staged(spec)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root, rollback_on_verify_failure=True, load_production=lambda: None) == 1
        assert promoter.sha256_file(spec.production_dir / "model.bin") == legacy_sha
        assert not (spec.production_dir / "manifest.json").exists()
        assert leftovers(spec) == []

    def test_a_first_promotion_rolls_back_to_nothing(self, tree, caplog):
        spec, root = tree
        write_staged(spec)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root, rollback_on_verify_failure=True, load_production=lambda: None) == 1
        assert sorted(p.name for p in spec.production_dir.iterdir()) == []
        assert "Production restored to nothing (this was the first promotion) and verified" in caplog.text
        assert last_line(caplog).startswith("Rollback: cd /opt/auspex")

    def test_a_failed_restore_is_loud_and_leaves_no_new_files(self, tree, caplog, monkeypatch):
        spec, root = tree
        write_production(spec)
        new_sha = write_staged(spec)
        real_copy = promoter._copy_durable

        def copy(src, dst):
            if BACKUP in str(src):
                Path(dst).write_text("half")
                raise OSError(28, "No space left on device")
            real_copy(src, dst)

        monkeypatch.setattr(promoter, "_copy_durable", copy)
        with caplog.at_level(logging.INFO, logger="promote_dc_model"):
            assert run(spec, root, rollback_on_verify_failure=True, load_production=lambda: None) == 1
        assert promoter.sha256_file(spec.production_dir / "model.bin") == new_sha
        assert leftovers(spec) == []
        assert "Automatic restore FAILED" in caplog.text
        assert last_line(caplog).startswith("Rollback: cd /opt/auspex")
        assert f"{BACKUP}/model.bin" in last_line(caplog)

    def test_a_successful_promotion_ignores_the_flag(self, tree):
        spec, root = tree
        write_production(spec)
        new_sha = write_staged(spec)
        assert run(spec, root, rollback_on_verify_failure=True) == 0
        assert promoter.sha256_file(spec.production_dir / "model.bin") == new_sha


class TestRestoreBackup:
    def test_a_restore_that_does_not_match_the_backup_is_reported(self, tree):
        spec, root = tree
        write_production(spec)
        backup = root / "backup-x"
        backup.mkdir(parents=True)
        (backup / "model.bin").write_text("not what production held")
        err = promoter.restore_backup(spec, backup, False, production_sha="a" * 64)
        assert err and "!= pre-promotion aaaaaaaaaaaa" in err


class TestMain:
    @pytest.mark.parametrize("argv, flag", [([], False), (["--rollback-on-verify-failure"], True)])
    def test_the_flag_reaches_promote(self, monkeypatch, argv, flag):
        seen = {}
        monkeypatch.setattr(promoter, "promote", lambda spec, **kw: seen.update(kw, spec=spec) or 0)
        assert promoter.main(["--model", "ht", "--expect-run-id", "r1"] + argv) == 0
        assert seen["spec"] is HT and seen["rollback_on_verify_failure"] is flag
        assert seen["expect_run_id"] == "r1" and seen["check_only"] is False


class TestServeSmoke:
    def test_rejects_no_model(self):
        err, _ = promoter.serve_smoke(None, HT)
        assert err and "no model" in err

    def test_rejects_a_model_that_serves_nan(self):
        class NaNModel:
            team_attack = {"A": 1.0, "B": 1.0}
            league_baselines = {"L1": (0.7, 0.5)}
            rho = 0.0

            def lambdas_for_match(self, home, away, league=None):
                return float("nan"), 0.5

        err, _ = promoter.serve_smoke(NaNModel(), HT)
        assert err and "invalid scoreline matrix" in err
