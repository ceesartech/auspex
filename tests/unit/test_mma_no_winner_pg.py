"""An MMA bout with no winner, end to end against a real Postgres (a throwaway
cluster built from services/data-ingestion/db/migrations, one database per
test from a template):

  * the results sweep (fetch_upcoming.record_results, ESPN's core API stubbed)
    records it finished with NULL scores and metadata.no_winner, and writes it
    again unchanged on the next sweep; a later winner for it is refused;
  * the match-outcome trigger grades nothing and settles nothing on it;
  * the grader records its predictions ('draw', is_correct NULL), voids its
    rec ('void: draw', profit 0) and then stops listing the match;
  * no model input counts it: compute_features_mma (form, head-to-head,
    rest), the rec eligibility gate, the MMA training query, the walk-forward
    test set; and monitor_models' graded set leaves it out.

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extras import RealDictCursor  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"
ML_SRC = REPO / "services" / "ml-models" / "src"
MIGRATIONS = REPO / "services" / "data-ingestion" / "db" / "migrations"

pytestmark = pytest.mark.skipif(
    not (shutil.which("initdb") and shutil.which("pg_ctl")), reason="no Postgres server binaries on PATH"
)

for _p in (str(SCRIPTS_DIR), str(ML_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming", "fetch_upcoming.py")
gcm = _load("grade_completed_matches", "grade_completed_matches.py")
fmma = _load("compute_features_mma", "compute_features_mma.py")
grm = _load("generate_recommendations_mma", "generate_recommendations_mma.py")
wf = _load("walk_forward_predictions", "walk_forward_predictions.py")
mon = _load("monitor_models", "monitor_models.py")
from utils.training_data import MMA_MONEYLINE_TRAINING_QUERY  # noqa: E402

MMA = fu.SPORT_CONFIGS["mma"]
NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)
CARD = NOW - timedelta(days=2)


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgn", dir="/tmp"))  # a Unix socket path is limited to ~100 bytes
    try:
        subprocess.run(["initdb", "-D", str(data), "-U", "postgres", "-A", "trust"], check=True, capture_output=True)
        subprocess.run(
            ["pg_ctl", "-D", str(data), "-w", "-l", str(base / "log"), "-o", f"-k {sock} -c listen_addresses=''"]
            + ["start"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:  # an environment that cannot host a cluster (root, no shm)
        shutil.rmtree(sock, ignore_errors=True)
        pytest.skip(f"cannot start a throwaway Postgres: {exc.stderr.decode(errors='replace')[-300:]}")
    try:
        admin = psycopg2.connect(host=str(sock), user="postgres", dbname="postgres")
        admin.autocommit = True
        admin.cursor().execute("CREATE DATABASE tpl")
        conn = psycopg2.connect(host=str(sock), user="postgres", dbname="tpl")
        conn.autocommit = True
        for f in sorted(MIGRATIONS.glob("*.sql")):
            conn.cursor().execute(f.read_text())
        conn.close()
        yield {"host": str(sock), "admin": admin}
        admin.close()
    finally:
        subprocess.run(["pg_ctl", "-D", str(data), "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(sock, ignore_errors=True)


@pytest.fixture
def dsn(cluster):
    name = f"t{uuid.uuid4().hex[:12]}"
    cluster["admin"].cursor().execute(f"CREATE DATABASE {name} TEMPLATE tpl")
    yield {"host": cluster["host"], "user": "postgres", "dbname": name}
    cluster["admin"].cursor().execute(f"DROP DATABASE {name} WITH (FORCE)")


@pytest.fixture
def conn(dsn):
    c = psycopg2.connect(**dsn)
    yield c
    c.rollback()
    c.close()


@pytest.fixture
def cur(conn):
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        yield c


@pytest.fixture(autouse=True)
def _core_says(monkeypatch):
    """ESPN's core API, stubbed: every no-winner bout is confirmed as answers[comp id] (default 'draw')."""
    answers: dict = {}
    monkeypatch.setattr(fu, "espn_core_result", lambda cfg, slug, ev, cid: (answers.get(str(cid), "draw"), True))
    for k in fu.IDENTITY_STATS:
        fu.IDENTITY_STATS[k] = 0
    yield answers
    for k in fu.IDENTITY_STATS:
        fu.IDENTITY_STATS[k] = 0


# ── fixtures in the database ──────────────────────────────────────────


def _league(cur) -> str:
    return str(fu.ensure_league(cur, MMA, "UFC", "UFC", "USA"))


def _fighter(cur, name: str) -> str:
    cur.execute(
        "INSERT INTO teams (name, normalized_name, sport) VALUES (%s, %s, 'mma') RETURNING id::text AS id",
        (name, fu.normalize_team(name)),
    )
    return cur.fetchone()["id"]


def _row(cur, league, home, away, kickoff, espn=None, status="scheduled", scores=(None, None), metadata=None):
    cur.execute(
        """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season, external_ids,
                             home_score, away_score, metadata)
        VALUES (%s, %s, %s, %s, %s, '2026', %s::jsonb, %s, %s, %s::jsonb) RETURNING id::text AS id
        """,
        (
            league,
            home,
            away,
            kickoff,
            status,
            json.dumps({"espn": espn} if espn else {}),
            scores[0],
            scores[1],
            json.dumps(metadata or {}),
        ),
    )
    return cur.fetchone()["id"]


def _prediction(cur, match_id, pick="away", created_at=None):
    cur.execute(
        """
        INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome,
                                 confidence, probabilities, created_at)
        VALUES (%s, 'ensemble_mma_ml', %s, 'moneyline', %s, 0.6, '{"home": 0.4, "away": 0.6}'::jsonb, %s)
        RETURNING id::text AS id
        """,
        (match_id, f"v{uuid.uuid4().hex[:6]}", pick, created_at or NOW - timedelta(days=5)),
    )
    return cur.fetchone()["id"]


def _rec(cur, match_id, prediction_id, created_at=None):
    cur.execute(
        """
        INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection, odds_at_recommendation,
                                             bookmaker, recommended_stake, status, created_at)
        VALUES (%s, %s, 'moneyline', 'away', 1.80, 'b', 25.00, 'pending', %s) RETURNING id::text AS id
        """,
        (prediction_id, match_id, created_at or NOW - timedelta(days=4)),
    )
    return cur.fetchone()["id"]


def _get(cur, table, row_id, *cols):
    cur.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE id = %s", (row_id,))
    return dict(cur.fetchone())


# ── ESPN payloads ─────────────────────────────────────────────────────


def _side(name, won, cards=None):
    side = {"athlete": {"displayName": name}, "winner": won}
    if cards:
        side["linescores"] = [{"value": float(sum(cards)), "linescores": [{"value": float(v)} for v in cards]}]
    return side


def _comp(comp_id, when, home, away, winner=None, home_cards=None, away_cards=None):
    return {
        "id": comp_id,
        "date": when.strftime("%Y-%m-%dT%H:%MZ"),
        "status": {
            "period": 3,
            "displayClock": "5:00",
            "type": {"state": "post", "completed": True, "name": "STATUS_FINAL", "detail": "Final"},
        },
        "competitors": [_side(home, winner == "home", home_cards), _side(away, winner == "away", away_cards)],
    }


def _draw(comp_id, when, home, away):
    return _comp(comp_id, when, home, away, home_cards=[28, 27, 28], away_cards=[28, 29, 28])


def _card(event_id, *comps):
    return {"id": event_id, "season": {"type": 2}, "competitions": list(comps)}


# ── the results sweep, the trigger and the grader ─────────────────────


class TestRecordAndGrade:
    def _setup(self, cur):
        league = _league(cur)
        f = {n: _fighter(cur, n) for n in ("Melissa Gatto", "Ernesta Kareckaite", "Al", "Bo")}
        draw_row = _row(cur, league, f["Melissa Gatto"], f["Ernesta Kareckaite"], CARD, espn="501")
        win_row = _row(cur, league, f["Al"], f["Bo"], CARD, espn="502")
        pid = _prediction(cur, draw_row)
        _prediction(cur, draw_row, pick="home")
        rid = _rec(cur, draw_row, pid)
        card = _card(
            "600", _draw("501", CARD, "Melissa Gatto", "Ernesta Kareckaite"), _comp("502", CARD, "Al", "Bo", "home")
        )
        return league, f, draw_row, win_row, rid, card

    def test_recorded_finished_with_null_scores_and_the_trigger_settles_nothing(self, cur):
        league, f, draw_row, win_row, rid, card = self._setup(cur)
        assert fu.record_results(cur, MMA, league, [card], "ufc") == 2
        row = _get(cur, "matches", draw_row, "status", "home_score", "away_score", "metadata", "external_ids")
        assert (row["status"], row["home_score"], row["away_score"]) == ("finished", None, None)
        assert row["metadata"]["no_winner"] == "draw"
        assert row["metadata"]["no_winner_detail"] == "majority draw; judges 28-28, 27-29, 28-28; ended R3 5:00"
        assert row["external_ids"] == {"espn": "501"}
        # migration 027's trigger fired on the flip: NULL scores grade nothing, so nothing settles.
        cur.execute("SELECT actual_outcome, is_correct FROM predictions WHERE match_id = %s", (draw_row,))
        assert all(r["actual_outcome"] is None and r["is_correct"] is None for r in cur.fetchall())
        assert _get(cur, "betting_recommendations", rid, "status")["status"] == "pending"
        assert _get(cur, "matches", win_row, "home_score", "away_score") == {"home_score": 1, "away_score": 0}

    def test_the_grader_records_the_predictions_voids_the_rec_and_stops_listing_it(self, cur):
        league, f, draw_row, win_row, rid, card = self._setup(cur)
        fu.record_results(cur, MMA, league, [card], "ufc")
        listed = [m for m in gcm.list_finished_matches(cur, 30, None) if m["match_id"] == draw_row]
        assert len(listed) == 1
        stats: dict = {}
        gcm.grade_match(cur, listed[0], stats)
        cur.execute("SELECT actual_outcome, is_correct FROM predictions WHERE match_id = %s", (draw_row,))
        assert [(r["actual_outcome"], r["is_correct"]) for r in cur.fetchall()] == [("draw", None), ("draw", None)]
        rec = _get(cur, "betting_recommendations", rid, "status", "actual_result", "profit_loss", "settled_at")
        assert (rec["status"], rec["actual_result"], rec["profit_loss"]) == ("void", "void: draw", Decimal("0.00"))
        assert rec["settled_at"] is not None
        assert stats == {"no_winner_predictions": 2, "no_winner_rec_voids": 1}
        assert draw_row not in {m["match_id"] for m in gcm.list_finished_matches(cur, 30, None)}
        assert gcm.list_ungraded_predictions(cur, draw_row) == []

    def test_the_next_sweep_writes_it_again_unchanged(self, cur):
        league, f, draw_row, win_row, rid, card = self._setup(cur)
        fu.record_results(cur, MMA, league, [card], "ufc")
        cur.execute("SELECT count(*) AS n FROM matches")
        before_n = cur.fetchone()["n"]
        before = _get(cur, "matches", draw_row, "status", "home_score", "away_score", "metadata")
        assert fu.record_results(cur, MMA, league, [card], "ufc") == 2
        assert _get(cur, "matches", draw_row, "status", "home_score", "away_score", "metadata") == before
        cur.execute("SELECT count(*) AS n FROM matches")
        assert cur.fetchone()["n"] == before_n
        assert sum(fu.IDENTITY_STATS.values()) == 0

    def test_a_later_winner_is_refused_and_counted(self, cur):
        league, f, draw_row, win_row, rid, card = self._setup(cur)
        fu.record_results(cur, MMA, league, [card], "ufc")
        corrected = _card("600", _comp("501", CARD, "Melissa Gatto", "Ernesta Kareckaite", "home"))
        assert fu.record_results(cur, MMA, league, [corrected], "ufc") == 0
        row = _get(cur, "matches", draw_row, "status", "home_score", "metadata")
        assert (row["status"], row["home_score"], row["metadata"]["no_winner"]) == ("finished", None, "draw")
        assert fu.IDENTITY_STATS["no_winner_refused"] == 1

    def test_a_later_no_winner_over_a_recorded_winner_is_refused(self, cur):
        league, f, draw_row, win_row, rid, card = self._setup(cur)
        fu.record_results(cur, MMA, league, [card], "ufc")
        overturned = _card("600", _draw("502", CARD, "Al", "Bo"))
        assert fu.record_results(cur, MMA, league, [overturned], "ufc") == 0
        row = _get(cur, "matches", win_row, "home_score", "away_score", "metadata")
        assert (row["home_score"], row["away_score"]) == (1, 0) and "no_winner" not in row["metadata"]
        assert fu.IDENTITY_STATS["no_winner_refused"] == 1

    def test_a_disagreeing_core_api_writes_nothing(self, cur, _core_says):
        league, f, draw_row, win_row, rid, card = self._setup(cur)
        _core_says["501"] = "no-contest"
        assert fu.record_results(cur, MMA, league, [card], "ufc") == 1
        assert _get(cur, "matches", draw_row, "status")["status"] == "scheduled"
        assert fu.IDENTITY_STATS["no_winner_refused"] == 1

    @pytest.mark.parametrize("answer", [None, "result-pending", "kotko"])
    def test_a_bout_still_unconfirmed_two_days_on_stays_scheduled_and_fails_the_run(self, cur, _core_says, answer):
        # The card is 2 days old (inside the 3-day window, past the 12-hour confirmation window): the bout is
        # not written, but the run exits 3 instead of letting it age out 'scheduled' with only log lines.
        league, f, draw_row, win_row, rid, card = self._setup(cur)
        _core_says["501"] = answer
        assert fu.record_results(cur, MMA, league, [card], "ufc") == 1
        assert _get(cur, "matches", draw_row, "status")["status"] == "scheduled"
        assert fu.IDENTITY_STATS["no_winner_unconfirmed"] == 1
        assert fu.identity_exit_code(MMA) == fu.IDENTITY_EXIT_CODE

    def test_an_unseen_bout_is_inserted_finished_with_no_winner(self, cur):
        league = _league(cur)
        card = _card("600", _draw("777", CARD, "Cy", "Di"))
        assert fu.record_results(cur, MMA, league, [card], "ufc") == 1
        cur.execute("SELECT status, home_score, away_score, metadata->>'no_winner' AS nw FROM matches")
        assert [dict(r) for r in cur.fetchall()] == [
            {"status": "finished", "home_score": None, "away_score": None, "nw": "draw"}
        ]


# ── no model input counts it ──────────────────────────────────────────


class TestNoModelInputCountsIt:
    def _history(self, cur, with_draw: bool):
        """Al: a win over Bo 100 days ago, (a draw with Cy 30 days ago), next: Al v Cy in 7 days."""
        league = _league(cur)
        f = {n: _fighter(cur, n) for n in ("Al", "Bo", "Cy")}
        _row(cur, league, f["Al"], f["Bo"], NOW - timedelta(days=100), status="finished", scores=(1, 0))
        draw = None
        if with_draw:
            meta = {"no_winner": "draw", "no_winner_detail": "split draw", "result_source": "espn_scoreboard"}
            draw = _row(cur, league, f["Al"], f["Cy"], NOW - timedelta(days=30), status="finished", metadata=meta)
            _prediction(cur, draw)
            gcm.grade_match(cur, gcm.list_finished_matches(cur, 0, draw)[0], {})
        nxt = _row(cur, league, f["Al"], f["Cy"], NOW + timedelta(days=7))
        return league, f, draw, nxt

    def test_features_are_what_they_were_without_the_draw(self, cur):
        _, _, draw, nxt = self._history(cur, with_draw=True)
        with_draw = fmma.compute_for_match(cur, nxt)
        cur.execute("DELETE FROM matches WHERE id = %s", (draw,))
        without = fmma.compute_for_match(cur, nxt)
        assert with_draw == without
        assert with_draw["home_roll_matches"] == 1.0 and with_draw["home_roll_wins"] == 1.0
        assert round(with_draw["home_days_rest"]) == 107  # the win, not the draw, is the last fight
        assert with_draw["h2h_matches"] == 0.0  # Al v Cy drew: no head-to-head fight
        assert with_draw["away_roll_matches"] == 5.0  # Cy has no counted fight: the neutral default

    def test_the_backfill_list_leaves_it_out(self, conn, cur):
        _, _, draw, _ = self._history(cur, with_draw=True)
        assert draw not in fmma.list_target_matches(conn, 7, False, True, None)

    def test_the_rec_eligibility_gate_does_not_count_it(self, cur):
        _, _, _, nxt = self._history(cur, with_draw=True)
        assert grm.list_upcoming_mma(cur, 14, min_fighter_history=1) == []  # Cy: 0 counted fights
        cur.execute(
            "UPDATE matches SET metadata = metadata - 'no_winner', home_score = 1, away_score = 0 "
            "WHERE metadata ? 'no_winner'"
        )
        assert grm.list_upcoming_mma(cur, 14, min_fighter_history=1) == [nxt]

    def test_the_training_query_leaves_it_out(self, cur):
        _, _, draw, _ = self._history(cur, with_draw=True)
        cur.execute(MMA_MONEYLINE_TRAINING_QUERY)
        ids = {r["match_id"] for r in cur.fetchall()}
        assert draw not in ids and len(ids) == 1

    def test_the_walk_forward_test_set_leaves_it_out(self, conn, cur):
        league, f, draw, _ = self._history(cur, with_draw=True)
        for mid in (draw,):
            cur.execute(
                "INSERT INTO features_cache (match_id, feature_set, feature_version, features, expires_at) "
                "VALUES (%s, 'mma_baseline', 'v1', '{}'::jsonb, NOW())",
                (mid,),
            )
        split = (NOW - timedelta(days=60)).date().isoformat()
        assert wf.list_test_matches(conn, "mma_moneyline", split) == []

    def test_the_monitor_never_sees_its_predictions(self, cur):
        _, _, draw, _ = self._history(cur, with_draw=True)
        cur.execute(f"SELECT match_id::text AS m FROM ({mon._DEDUPED_GRADED_PREDICTIONS_SQL}) d", {"days": "365"})
        assert draw not in {r["m"] for r in cur.fetchall()}
