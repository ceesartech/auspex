"""The 2026-09-21 Telegram drift page could not be acted on as written:

    ⚠️ MCE 0.253 >= 0.250 — worst-bucket gap too wide
    ⚠️ Accuracy 48.8% < 52.4% break-even — strategy unprofitable at -110 vig
    ...ten lines, no sport, no market, no n, re-sent every hour.

build_alert dropped f.sport / f.market on the floor (the markdown path
kept them), the monitor had no idea how many outcomes a market has, and
nothing stopped the identical alert set paging on every hourly run.
These tests pin the rendering, the class-count map and the dedup — and the
review findings on the first draft: dedup keys are marked only after a
successful send, class count is never inferred from a multi-line vector,
and the rec_gating lookup is keyed by (sport, prediction_type).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
ML_SRC = REPO_ROOT / "services" / "ml-models" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mm = _load("monitor_models", "monitor_models.py")
cm = _load("calibration_metrics", "calibration_metrics.py")


def _finding(sport, market, metric, severity="alert", n=84, current=0.253, message="MCE 0.253 >= 0.250", model=""):
    return cm.DriftFinding(
        sport=sport,
        market=market,
        metric=metric,
        severity=severity,
        current=current,
        threshold=0.25,
        message=message,
        n=n,
        model_name=model,
    )


class FakeRedis:
    def __init__(self):
        self.store = {}

    def mget(self, keys):
        return [self.store.get(k) for k in keys]

    def set(self, key, value, nx=False, ex=None):
        assert ex == 23 * 3600
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


class TestAlertBodyIsLabeled:
    def test_every_line_names_sport_market_and_n(self):
        alert = mm.build_alert(
            [
                _finding("mma", "moneyline", "mce", n=84),
                _finding("nfl", "total", "ece", n=30, message="ECE 0.218 >= 0.100"),
                _finding("horse_racing", "win", "mce", n=16948, message="MCE 0.923 >= 0.250"),
            ]
        )
        assert alert is not None
        lines = alert.probabilities["body"].splitlines()
        assert len(lines) == 3
        for line in lines:
            assert "(n=" in line and ": " in line, line
        assert any("(n=84)" in line and "MCE 0.253" in line for line in lines)
        assert any("(n=16948)" in line and "0.923" in line for line in lines)

    def test_two_models_on_one_market_render_as_two_distinct_lines(self):
        alert = mm.build_alert(
            [
                _finding("horse_racing", "win", "mce", n=16948, model="lightgbm_ranker_v1"),
                _finding("horse_racing", "win", "mce", n=16948, model="market_consensus_v1"),
            ]
        )
        lines = alert.probabilities["body"].splitlines()
        assert len(lines) == 2 and lines[0] != lines[1]
        assert "[lightgbm_ranker_v1]" in lines[0] and "[market_consensus_v1]" in lines[1]

    def test_markdown_findings_carry_n_too(self):
        md = mm.render_findings([_finding("nfl", "total", "ece", severity="warn", n=30)])
        assert "(n=30)" in md

    def test_unknown_n_renders_as_question_mark_not_zero(self):
        alert = mm.build_alert([_finding("soccer", "1x2", "canary", n=0)])
        assert "(n=?)" in alert.probabilities["body"]

    def test_warnings_never_page(self):
        assert mm.build_alert([_finding("nfl", "total", "ece", severity="warn")]) is None


class TestClassCount:
    def test_multi_line_two_way_markets_are_two_way_regardless_of_vector_shape(self):
        # asian_handicap stores 51 keys (17 lines x 3) but each pick is a 2-way bet.
        assert mm.n_classes_for("asian_handicap", 51) == 2
        assert mm.n_classes_for("over_under", 12) == 2

    def test_known_k_way_markets(self):
        assert mm.n_classes_for("match_result", 3) == 3
        assert mm.n_classes_for("double_chance", 3) == 3
        assert mm.n_classes_for("correct_score", 13) == 13

    def test_plain_two_way_markets(self):
        for pt in ("moneyline", "spread", "total", "btts", "draw_no_bet"):
            assert mm.n_classes_for(pt, 2) == 2

    def test_unknown_multi_key_type_errs_toward_not_paging(self):
        # Not in the map and the vector says 7 keys: treat as 7-way (floor off).
        assert mm.n_classes_for("some_new_market", 7) == 7
        assert mm.n_classes_for("some_new_market", None) == 2

    def test_slice_query_computes_the_modal_class_count_and_guards_scalars(self):
        import inspect

        src = inspect.getsource(mm.fetch_slices)
        assert "MODE() WITHIN GROUP (ORDER BY n_classes)" in src
        assert "jsonb_typeof(p.probabilities) = 'object'" in mm._DEDUPED_GRADED_PREDICTIONS_SQL


class TestGatedStreamLookup:
    def test_map_is_keyed_by_sport_and_type(self):
        assert mm._PREDICTION_TYPE_TO_BET_TYPE[("soccer", "match_result")] == "1x2"
        assert mm._PREDICTION_TYPE_TO_BET_TYPE[("nhl", "spread")] == "puck_line"

    def test_lookup_consults_the_bet_type_a_generator_actually_writes(self, monkeypatch):
        seen = []

        class Gate:
            enabled = False

        class RG:
            @staticmethod
            def gate_for(sport, bet_type):
                seen.append((sport, bet_type))
                return Gate()

        monkeypatch.setattr(mm, "rec_gating", RG)
        assert mm._stream_is_gated("soccer", "match_result") is True
        assert mm._stream_is_gated("nhl", "spread") is True
        assert mm._stream_is_gated("soccer", "btts") is True
        assert seen == [("soccer", "1x2"), ("nhl", "puck_line"), ("soccer", "btts")]

    def test_lookup_fails_open(self, monkeypatch):
        class Boom:
            @staticmethod
            def gate_for(sport, bet_type):
                raise RuntimeError("no")

        monkeypatch.setattr(mm, "rec_gating", Boom)
        assert mm._stream_is_gated("mma", "moneyline") is False

    def test_every_mapped_bet_type_is_one_a_generator_actually_writes(self):
        # gate_for() never fails (it falls through to a default), so asserting
        # it returns a gate proves nothing. Assert instead that each mapped
        # bet_type appears as a written bet_type in that sport's generator and
        # that rec_gating can key on it (exact gate or per-sport default).
        rg = _load("rec_gating", "rec_gating.py")
        generator_src = {
            "soccer": (SCRIPTS_DIR / "generate_recommendations.py").read_text(),
            "nhl": (SCRIPTS_DIR / "generate_recommendations_nhl.py").read_text(),
        }
        for (sport, _pt), bet_type in mm._PREDICTION_TYPE_TO_BET_TYPE.items():
            assert f'"{bet_type}"' in generator_src[sport], f"{sport} generator never writes bet_type {bet_type!r}"
            assert (sport, bet_type) in rg.GATES or sport in rg.SPORT_DEFAULTS, (sport, bet_type)

    def test_page_body_is_capped_so_it_can_never_chunk(self):
        alerts = [_finding("soccer", f"market_{i}", "ece", n=500) for i in range(20)]
        body = mm.build_alert(alerts).probabilities["body"]
        lines = body.splitlines()
        assert len(lines) == mm._MAX_PAGE_LINES + 1
        assert lines[-1].startswith("…") and "+8 more" in lines[-1]
        assert len(body) < 3900


class TestPageDedup:
    def test_new_alerts_are_selected_and_only_marked_after_success(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr(mm, "_dedup_client", lambda url=None: fake)
        a = _finding("mma", "moneyline", "mce")
        b = _finding("nfl", "total", "ece")
        # Nothing marked yet: both are new.
        assert mm.select_new_alerts([a, b]) == [a, b]
        # A send that FAILED must not mark — the next run still sees both.
        assert mm.select_new_alerts([a, b]) == [a, b]
        # A successful send marks; now nothing is new...
        assert mm.mark_paged([a, b]) == 2
        assert mm.select_new_alerts([a, b]) == []
        # ...but a genuinely new condition pages immediately, alongside the old ones.
        c = _finding("soccer", "btts", "mce")
        assert mm.select_new_alerts([a, b, c]) == [c]

    def test_keys_are_per_finding_so_a_shrinking_set_does_not_repage(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr(mm, "_dedup_client", lambda url=None: fake)
        a, b = _finding("mma", "moneyline", "mce"), _finding("nfl", "total", "ece")
        mm.mark_paged([a, b])
        assert mm.select_new_alerts([a]) == []

    def test_two_models_on_one_market_get_separate_keys(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr(mm, "_dedup_client", lambda url=None: fake)
        ranker = _finding("horse_racing", "win", "mce", model="lightgbm_ranker_v1")
        consensus = _finding("horse_racing", "win", "mce", model="market_consensus_v1")
        mm.mark_paged([ranker])
        assert mm.select_new_alerts([ranker, consensus]) == [consensus]

    def test_warnings_are_never_selected_or_marked(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr(mm, "_dedup_client", lambda url=None: fake)
        w = _finding("nfl", "total", "ece", severity="warn")
        assert mm.select_new_alerts([w]) == []
        assert mm.mark_paged([w]) == 0 and fake.store == {}

    def test_redis_unavailable_fails_open(self, monkeypatch):
        monkeypatch.setattr(mm, "_dedup_client", lambda url=None: None)
        a = _finding("mma", "moneyline", "mce")
        assert mm.select_new_alerts([a]) == [a]
        assert mm.mark_paged([a]) == 0

    def test_redis_error_fails_open(self, monkeypatch):
        class Broken:
            def mget(self, keys):
                raise ConnectionError("down")

            def set(self, *a, **k):
                raise ConnectionError("down")

        monkeypatch.setattr(mm, "_dedup_client", lambda url=None: Broken())
        a = _finding("mma", "moneyline", "mce")
        assert mm.select_new_alerts([a]) == [a]
        assert mm.mark_paged([a]) == 0

    def test_dedup_client_carries_socket_timeouts(self):
        import inspect

        src = inspect.getsource(mm._dedup_client)
        assert "socket_connect_timeout=2" in src and "socket_timeout=2" in src


class TestSendFailureIsLoud:
    def test_main_fails_the_task_when_a_page_could_not_be_delivered(self, monkeypatch):
        monkeypatch.setattr(
            mm,
            "parse_args",
            lambda argv=None: type(
                "A",
                (),
                {
                    "database_url": "postgresql://x",
                    "ece_warn": 0.05,
                    "ece_alert": 0.10,
                    "mce_warn": 0.15,
                    "mce_alert": 0.25,
                    "brier_drift_warn": 0.02,
                    "brier_drift_alert": 0.05,
                    "accuracy_floor": 0.524,
                    "days": 30,
                    "min_samples": 30,
                },
            )(),
        )
        monkeypatch.setattr(mm, "run", lambda *a, **k: {"telegram_send_failed": True})
        assert mm.main([]) == 1
        monkeypatch.setattr(mm, "run", lambda *a, **k: {"telegram_send_failed": False})
        assert mm.main([]) == 0
