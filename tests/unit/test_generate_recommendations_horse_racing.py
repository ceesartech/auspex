"""Unit tests for generate_recommendations_horse_racing — pure helpers
+ CLI shape. Focuses on the best-of-N pricing path and the risk-factor
heuristics since those are horse-racing-specific. EV / Kelly /
confidence_rating are shared with generate_recommendations.py and
tested there.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Loading the recs script also loads generate_recommendations.py
# (the shared helpers source) via sys.path insertion inside the
# module — fine for the test environment.
ghr = _load("generate_recommendations_horse_racing", "generate_recommendations_horse_racing.py")


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_kelly_fraction_is_quarter(self):
        # Same conservative quarter-Kelly the team-sport engines use.
        # Bumping this here without parallel changes elsewhere would
        # mean horse racing sizes more aggressively than soccer/NBA
        # for no documented reason.
        assert ghr.KELLY_FRACTION == 0.25

    def test_model_precedence_lock(self):
        # race_recommendations.race_prediction_id FKs into
        # race_predictions; that lookup is keyed on (model_name,
        # model_version). Renaming the consensus model without
        # updating this drops every rec to zero hits.
        #
        # Precedence rule (load-bearing for rec QUALITY): the
        # ranker is INTENTIONALLY ABSENT from this list even
        # though it has better top-1 accuracy. The recs engine
        # consumes probabilities directly (EV = prob × odds) and
        # the ranker's Brier on the 13k-race corpus (0.0898) is
        # WORSE than consensus's (0.0831). Listing it even as a
        # fallback silently leaks ranker probs into races where
        # consensus hasn't scored yet (verified empirically:
        # consensus-first-with-ranker-fallback fired 146 picks
        # vs ~30 consensus-only). Until isotonic re-calibration
        # closes the Brier gap, this list stays consensus-only.
        # The ranker keeps writing to race_predictions via the
        # precompute task in the DAG — analysis path stays intact.
        assert ghr.MODEL_PRECEDENCE == [
            ("market_consensus_v1", "1.0.0"),
        ]


# ── best_decimal: longest decimal across bookmakers ─────────────────


class TestBestDecimal:
    def test_returns_longest_decimal(self):
        # 8-book array; the bettor wants the highest decimal (=longest
        # price) since that maximises payout for a given probability.
        odds = [
            {"bookmaker": "A", "decimal": 3.0},
            {"bookmaker": "B", "decimal": 4.5},
            {"bookmaker": "C", "decimal": 3.75},
        ]
        best = ghr.best_decimal(odds)
        assert best == {"bookmaker": "B", "decimal": 4.5}

    def test_handles_string_decimals(self):
        # The Racing API returns decimal as a string; the upserter
        # normalises but the recs path should still tolerate either.
        odds = [{"bookmaker": "A", "decimal": "2.5"}, {"bookmaker": "B", "decimal": "3.0"}]
        best = ghr.best_decimal(odds)
        assert best["decimal"] == 3.0

    def test_returns_none_for_empty(self):
        assert ghr.best_decimal([]) is None
        assert ghr.best_decimal(None) is None

    def test_skips_entries_with_no_decimal(self):
        odds = [
            {"bookmaker": "A"},  # no decimal field
            {"bookmaker": "B", "decimal": None},
            {"bookmaker": "C", "decimal": 2.0},
        ]
        best = ghr.best_decimal(odds)
        assert best["bookmaker"] == "C"

    def test_skips_decimals_at_or_below_one(self):
        # decimal=1.0 implies 100% probability — usually a stub from
        # a bookmaker that hasn't priced yet. Treat as invalid.
        odds = [
            {"bookmaker": "A", "decimal": 1.0},
            {"bookmaker": "B", "decimal": 0.95},
            {"bookmaker": "C", "decimal": 1.01},
        ]
        best = ghr.best_decimal(odds)
        assert best["bookmaker"] == "C"
        assert best["decimal"] == 1.01

    def test_skips_non_dict_entries(self):
        # Defensive: a stray string / number in the array shouldn't crash.
        odds = [{"bookmaker": "A", "decimal": 2.0}, "garbage", 3.5]
        best = ghr.best_decimal(odds)
        assert best == {"bookmaker": "A", "decimal": 2.0}

    def test_tolerates_unparseable_decimal(self):
        odds = [
            {"bookmaker": "A", "decimal": "not-a-number"},
            {"bookmaker": "B", "decimal": "2.5"},
        ]
        best = ghr.best_decimal(odds)
        assert best == {"bookmaker": "B", "decimal": 2.5}


# ── _risk_factors: horse-racing-specific risk labels ────────────────


class TestRiskFactors:
    def test_no_risks_for_modal_pick(self):
        # 25% probability, 3.5 decimal, 8-runner field — bread-and-
        # butter mid-card pick. Shouldn't trip any flag.
        assert ghr._risk_factors(0.25, 3.5, 8) == []

    def test_longshot_flag_at_decimal_10(self):
        # decimal=10 → 9/1; the variance dominates at this end so we
        # flag for the user. Boundary is inclusive per the impl.
        assert "longshot" in ghr._risk_factors(0.15, 10.0, 8)
        assert "longshot" in ghr._risk_factors(0.05, 20.0, 8)

    def test_no_longshot_flag_below_decimal_10(self):
        assert "longshot" not in ghr._risk_factors(0.30, 8.0, 8)

    def test_low_consensus_probability_below_15_pct(self):
        # If the devigged consensus says <15%, betting it requires
        # disagreeing with the entire bookmaker board. Worth marking.
        assert "low_consensus_probability" in ghr._risk_factors(0.10, 12.0, 8)

    def test_no_low_consensus_flag_at_15_pct(self):
        # Strict < threshold — exactly 15% is borderline, not flagged.
        assert "low_consensus_probability" not in ghr._risk_factors(0.15, 8.0, 8)

    def test_large_field_flag_at_14_runners(self):
        # 14+ runner fields amplify variance + traffic / bad-luck
        # losses. Sprint fields in Britain regularly hit 20+ runners.
        assert "large_field" in ghr._risk_factors(0.20, 6.0, 14)
        assert "large_field" in ghr._risk_factors(0.20, 6.0, 20)

    def test_no_large_field_flag_below_14(self):
        assert "large_field" not in ghr._risk_factors(0.20, 6.0, 13)

    def test_multiple_flags_combine(self):
        # 16-runner sprint with a 14/1 outsider at 8% consensus —
        # all three risks trigger.
        risks = ghr._risk_factors(0.08, 15.0, 16)
        assert set(risks) == {"longshot", "low_consensus_probability", "large_field"}


# ── CLI ─────────────────────────────────────────────────────────────


class TestCli:
    def test_defaults(self):
        args = ghr.parse_args(["--database-url", "postgresql://x"])
        assert args.days == 2
        assert args.ev_threshold == 0.05
        assert args.prob_floor == 0.10

    def test_ev_threshold_parses_as_float(self):
        args = ghr.parse_args(["--ev-threshold", "0.08", "--database-url", "x"])
        assert args.ev_threshold == 0.08

    def test_prob_floor_parses_as_float(self):
        args = ghr.parse_args(["--prob-floor", "0.25", "--database-url", "x"])
        assert args.prob_floor == 0.25

    def test_days_parses_as_int(self):
        args = ghr.parse_args(["--days", "7", "--database-url", "x"])
        assert args.days == 7

    def test_no_notify_flag_defaults_off(self):
        # Default behaviour: enqueue alerts. The flag is opt-OUT so
        # the DAG path (which doesn't pass --no-notify) keeps the
        # Telegram digest fed.
        args = ghr.parse_args(["--database-url", "x"])
        assert args.no_notify is False

    def test_no_notify_flag_parses(self):
        args = ghr.parse_args(["--no-notify", "--database-url", "x"])
        assert args.no_notify is True


# ── Alert factory: shape + value-bet trigger ────────────────────────


class TestHorseRacingAlert:
    """horse_racing_alert is what bridges a value-bet rec row into
    the shared Alert dataclass that send_pipeline_digest renders.
    Because the dataclass was built for 2-team sports, the mapping
    isn't 1:1 — these tests lock the specific shape choices so a
    future refactor (e.g., adding a horse_name field to Alert) can
    update the digest renderer in lockstep."""

    def _alert(self, **overrides):
        from datetime import datetime

        kwargs = dict(
            track_name="Newton Abbot",
            race_date=datetime(2026, 6, 3, 15, 0),
            race_number=3,
            horse_name="Jena d'Oudairies",
            odds_decimal=3.75,
            bookmaker="Betfair Exchange",
            confidence=0.31,
            expected_value=0.16,
            recommended_stake=14.0,
        )
        kwargs.update(overrides)
        return ghr.horse_racing_alert(**kwargs)

    def test_sport_is_horse_racing(self):
        # Drives the 🐎 emoji in the digest. If this changes, also
        # update _SPORT_EMOJI in telegram_notify.py.
        assert self._alert().sport == "horse_racing"

    def test_home_team_carries_horse_name(self):
        # The {home_team} vs {away_team} render needs the horse name
        # to land in home so the digest reads "Jena d'Oudairies vs Field".
        assert self._alert().home_team == "Jena d'Oudairies"

    def test_away_team_is_field(self):
        # Consensus prob is derived across the field, so the contrast
        # is horse-vs-field, not horse-vs-horse. Locked here so a
        # future "away_team = next-best horse" tweak surfaces in CI.
        assert self._alert().away_team == "Field"

    def test_league_name_includes_track_and_race_number(self):
        # The digest shows league_name as the secondary label, so
        # the track + race number need to land there for the user to
        # know WHICH race the pick is on.
        a = self._alert()
        assert "Newton Abbot" in a.league_name
        assert "R3" in a.league_name

    def test_league_name_omits_race_number_when_none(self):
        a = self._alert(race_number=None)
        assert a.league_name == "Newton Abbot"
        assert "R" not in a.league_name

    def test_market_label_is_win(self):
        # Single market in v1 (the consensus baseline only predicts
        # the win market). Place / show would need separate models.
        assert self._alert().market_label == "Win"

    def test_value_bet_fields_populated(self):
        # The presence of expected_value flips
        # telegram_notify._format_alert_line into value-bet mode.
        # All four optional Alert fields land together — partial
        # populations break the formatter.
        a = self._alert()
        assert a.expected_value == 0.16
        assert a.odds_decimal == 3.75
        assert a.recommended_stake == 14.0
        assert a.bookmaker == "Betfair Exchange"

    def test_confidence_carries_consensus_prob(self):
        # `confidence` is the consensus win prob, used by the
        # formatter to render "model NN%" alongside the odds.
        assert self._alert(confidence=0.42).confidence == 0.42

    def test_probabilities_dict_has_win_entry(self):
        # Even though horse racing has no second outcome, the Alert
        # dataclass expects a dict; emit a single 'win' entry so any
        # downstream that iterates probabilities (e.g. logging) gets
        # something sensible.
        a = self._alert(confidence=0.31)
        assert a.probabilities == {"win": 0.31}

    def test_floats_coerced_from_ints(self):
        # Defensive: callers passing int (or numpy float) should
        # still produce a clean Alert with Python floats.
        a = self._alert(odds_decimal=4, recommended_stake=20, confidence=1)
        assert a.odds_decimal == 4.0
        assert a.recommended_stake == 20.0
        assert a.confidence == 1.0


# ── Hybrid filter: ranker rank + consensus prob ────────────────────


class _FakeCursor:
    """Captures execute() args + lets tests seed fetchall responses
    for recommend_for_race. The race-recommendations DELETE in
    delete_pending and the INSERT in insert_recommendation use
    execute() with no fetch, so seeding only fetchall is sufficient
    for the smoke shape we need here."""

    def __init__(self, fetchall_responses):
        self._fetchall = list(fetchall_responses)
        self.executions = []

    def execute(self, sql, params=None):
        self.executions.append((sql, params or ()))

    def fetchall(self):
        return self._fetchall.pop(0) if self._fetchall else []

    def fetchone(self):
        return None


def _cand(entrant_id, horse_name, confidence, ranker_confidence=None, bookmaker_odds=None):
    """Build one candidate row the way load_race_candidates returns it."""
    return {
        "entrant_id": entrant_id,
        "prediction_id": f"pred-{entrant_id}",
        "confidence": confidence,
        "bookmaker_odds": bookmaker_odds
        or [
            {"bookmaker": "Test Book", "decimal": 5.0},
        ],
        "horse_name": horse_name,
        "model_name": "market_consensus_v1",
        "ranker_confidence": ranker_confidence,
    }


class TestHybridFilter:
    """RANKER_TOP_N narrows each race to the top-N entrants by
    ranker confidence before EV evaluation, when the ranker has
    scored every entrant in the race. When the ranker hasn't scored
    (any candidate lacks ranker_confidence) the filter falls back
    to considering every entrant — the prior consensus-only
    behaviour."""

    def _patch_top_n(self, n, monkeypatch):
        monkeypatch.setattr(ghr, "RANKER_TOP_N", n)

    def test_top_n_filters_to_ranker_choices(self, monkeypatch):
        # 5-runner race with ranker rankings [A=0.4, B=0.3, C=0.2, D=0.05, E=0.05].
        # RANKER_TOP_N=3 should narrow to A, B, C — D and E never see EV math.
        self._patch_top_n(3, monkeypatch)
        candidates = [
            _cand("A", "Horse A", confidence=0.3, ranker_confidence=0.4),
            _cand("B", "Horse B", confidence=0.2, ranker_confidence=0.3),
            _cand("C", "Horse C", confidence=0.15, ranker_confidence=0.2),
            _cand("D", "Horse D", confidence=0.1, ranker_confidence=0.05),
            _cand("E", "Horse E", confidence=0.25, ranker_confidence=0.05),
        ]
        cur = _FakeCursor([candidates])
        race = {
            "race_id": "race-1",
            "track_name": "Newton Abbot",
            "race_date": __import__("datetime").datetime(2026, 6, 4, 15, 0),
            "race_number": 1,
        }
        alerts = ghr.recommend_for_race(cur, race, bankroll=1000.0, ev_threshold=0.0, prob_floor=0.0)
        # Up to 3 alerts; whichever passed EV/prob thresholds. The key
        # invariant: NONE of the alerts can be Horse D or E because
        # the ranker filter dropped them before EV evaluation.
        alert_horses = {a.home_team for a in alerts}
        assert "Horse D" not in alert_horses
        assert "Horse E" not in alert_horses

    def test_top_n_falls_back_when_ranker_missing(self, monkeypatch):
        # Ranker hasn't scored this race (None on every candidate).
        # Filter must fall back to considering every entrant rather
        # than silently dropping the race.
        self._patch_top_n(3, monkeypatch)
        candidates = [
            _cand("A", "Horse A", confidence=0.4, ranker_confidence=None),
            _cand("B", "Horse B", confidence=0.3, ranker_confidence=None),
            _cand("C", "Horse C", confidence=0.2, ranker_confidence=None),
            _cand("D", "Horse D", confidence=0.1, ranker_confidence=None),
        ]
        cur = _FakeCursor([candidates])
        race = {
            "race_id": "race-1",
            "track_name": "Curragh",
            "race_date": __import__("datetime").datetime(2026, 6, 4, 15, 0),
            "race_number": 2,
        }
        # Every candidate's EV vs 5.0 odds: prob × 5 − 1. Horse A
        # (0.4 × 5 - 1 = 1.0) and Horse B (0.5) clear any threshold.
        # If the filter mis-fired and dropped the race, alerts would
        # be empty.
        alerts = ghr.recommend_for_race(cur, race, bankroll=1000.0, ev_threshold=0.0, prob_floor=0.0)
        assert len(alerts) > 0  # Race was evaluated, not silently dropped.

    def test_top_n_falls_back_when_partial_ranker_coverage(self, monkeypatch):
        # Mixed coverage: 3 of 4 candidates have ranker_confidence,
        # one doesn't. The "every candidate has ranker_confidence"
        # guard fails, fallback fires — all 4 evaluated. Without
        # this guard, we'd silently dock the unscored entrant from
        # consideration even though we DON'T know whether the
        # ranker would have liked it.
        self._patch_top_n(2, monkeypatch)
        candidates = [
            _cand("A", "Horse A", confidence=0.4, ranker_confidence=0.5),
            _cand("B", "Horse B", confidence=0.3, ranker_confidence=0.3),
            _cand("C", "Horse C", confidence=0.2, ranker_confidence=0.1),
            _cand("D", "Horse D", confidence=0.4, ranker_confidence=None),  # unscored
        ]
        cur = _FakeCursor([candidates])
        race = {
            "race_id": "race-1",
            "track_name": "Ripon",
            "race_date": __import__("datetime").datetime(2026, 6, 4, 15, 0),
            "race_number": 3,
        }
        alerts = ghr.recommend_for_race(cur, race, bankroll=1000.0, ev_threshold=0.0, prob_floor=0.0)
        # Without the partial-coverage fallback the filter would have
        # taken top-2 (A, B) and dropped both C and D. With the
        # fallback all 4 are evaluated, so D's high-prob row CAN
        # produce an alert.
        alert_horses = {a.home_team for a in alerts}
        assert "Horse D" in alert_horses

    def test_filter_disabled_when_top_n_none(self, monkeypatch):
        # RANKER_TOP_N=None disables the hybrid filter entirely.
        # Every candidate evaluated regardless of ranker_confidence.
        self._patch_top_n(None, monkeypatch)
        candidates = [
            _cand("A", "Horse A", confidence=0.4, ranker_confidence=0.5),
            _cand("B", "Horse B", confidence=0.3, ranker_confidence=0.3),
            _cand("C", "Horse C", confidence=0.2, ranker_confidence=0.1),
            _cand("D", "Horse D", confidence=0.4, ranker_confidence=0.05),
        ]
        cur = _FakeCursor([candidates])
        race = {
            "race_id": "race-1",
            "track_name": "Warwick",
            "race_date": __import__("datetime").datetime(2026, 6, 4, 15, 0),
            "race_number": 4,
        }
        alerts = ghr.recommend_for_race(cur, race, bankroll=1000.0, ev_threshold=0.0, prob_floor=0.0)
        # Without the filter, Horse D (consensus 40%, ranker 5%) can
        # still fire an alert based on its consensus probability.
        alert_horses = {a.home_team for a in alerts}
        assert "Horse D" in alert_horses
