"""scripts/settle_mma_held_rows.py end to end against a real Postgres (a
throwaway cluster built from services/data-ingestion/db/migrations), with the
prod state of the two held rows as of 2026-11-02 and ESPN's real payloads
(tests/unit/fixtures/espn_ufc_no_winner.json; the core API stubbed to its real
answer):

  * the dry run plans the reviewed plan, runs the whole transaction and
    commits nothing; the apply pinned to that plan records Gatto v Kareckaite
    as a draw through the results path, grades it, deletes Valesca Machado v
    Melissa Amaya with every row under it and the orphaned team row, archives
    all of it first, and leaves every other row alone; a re-run writes nothing
    and asks ESPN nothing; --check-after holds;
  * a plan other than --expect-plan, a rec on the replaced row, an extra
    prediction, a recorded winner, ESPN not listing the competition as
    reviewed, a core API that disagrees, or a write outside the listed rows:
    nothing is committed;
  * the other order (the sweep already recorded the draw, the replaced row
    already cancelled as re-drawn) settles to the same after-state;
  * the archive and the audit CSV cover the recorded bout's two fighter rows
    (an athlete-id resolver may stamp them through the results path);
  * run B's done check (scripts/ops/repair_stranded_recs_mma.sh: its SQL, read
    out of the wrapper) reads the held rows 'held' before and 'settled' after
    a real apply, and counts only the UFC recs created before its AS_OF, so a
    rec the pipeline writes for a later card never makes it read "not done".

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extras import RealDictCursor  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"
B_WRAPPER = SCRIPTS_DIR / "ops" / "repair_stranded_recs_mma.sh"
MIGRATIONS = REPO / "services" / "data-ingestion" / "db" / "migrations"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "espn_ufc_no_winner.json"

pytestmark = pytest.mark.skipif(
    not (shutil.which("initdb") and shutil.which("pg_ctl")), reason="no Postgres server binaries on PATH"
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming", "fetch_upcoming.py")
_load("grade_completed_matches", "grade_completed_matches.py")
st = _load("settle_mma_held_rows", "settle_mma_held_rows.py")
st.fu = fu  # the same module object the tool's fetch_upcoming calls go through

COMPS = {c["id"]: c for c in json.loads(FIXTURE.read_text())["competitions"]}
ENTRIES = st.load_list()
GATTO, MACHADO = ENTRIES
TINA_ROW = GATTO_PRED = None  # set by _seed
REVIEWED = {
    "record": 1,
    "record_done": 0,
    "record_predictions": 2,
    "record_recs_void": 0,
    "delete": 1,
    "delete_done": 0,
    "delete_predictions": 2,
    "delete_odds": 0,
    "delete_snapshots": 0,
    "delete_features": 2,
    "delete_other_rows": 0,
    "teams_delete": 1,
    "refuse": 0,
}
DONE = {**{k: 0 for k in REVIEWED}, "record_done": 1, "delete_done": 1}


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgs", dir="/tmp"))  # a Unix socket path is limited to ~100 bytes
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
def db(cluster):
    name = f"t{uuid.uuid4().hex[:12]}"
    cluster["admin"].cursor().execute(f"CREATE DATABASE {name} TEMPLATE tpl")
    dsn = {"host": cluster["host"], "user": "postgres", "dbname": name}
    _seed(dsn)
    yield dsn
    cluster["admin"].cursor().execute(f"DROP DATABASE {name} WITH (FORCE)")


def _connect(dsn):
    return psycopg2.connect(**dsn)


def _q(dsn, sql, params=None):
    with _connect(dsn) as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()] if cur.description else []
    c.close()
    return rows


def _x(dsn, sql, params=None):
    c = _connect(dsn)
    with c.cursor() as cur:
        cur.execute(sql, params)
    c.commit()
    c.close()


# ── the prod state of 2026-11-02 06:25Z (read only), in miniature ─────


def _seed(dsn):
    c = _connect(dsn)
    cur = c.cursor(cursor_factory=RealDictCursor)
    league = str(fu.ensure_league(cur, fu.SPORT_CONFIGS["mma"], "UFC", "UFC", "USA"))
    team = {}
    fixed = {"Valesca Machado": MACHADO.replaced_team_id}
    for name in ("Melissa Gatto", "Ernesta Kareckaite", "Valesca Machado", "Melissa Amaya", "Tina Black", "Other"):
        cur.execute(
            "INSERT INTO teams (id, name, normalized_name, sport) VALUES (%s, %s, %s, 'mma') RETURNING id::text AS id",
            (fixed.get(name, str(uuid.uuid4())), name, fu.normalize_team(name)),
        )
        team[name] = cur.fetchone()["id"]

    def row(mid, home, away, kickoff, status, espn=None, scores=(None, None), meta=None):
        cur.execute(
            """
            INSERT INTO matches (id, league_id, home_team_id, away_team_id, match_date, status, season, external_ids,
                                 home_score, away_score, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, '2026', %s::jsonb, %s, %s, %s::jsonb)
            """,
            (
                mid,
                league,
                team[home],
                team[away],
                kickoff,
                status,
                json.dumps({"espn": espn} if espn else {}),
                scores[0],
                scores[1],
                json.dumps(meta or {"season_type": "regular"}),
            ),
        )

    row(GATTO.match_id, "Melissa Gatto", "Ernesta Kareckaite", GATTO.kickoff, "scheduled", espn="401924511")
    row(MACHADO.match_id, "Valesca Machado", "Melissa Amaya", MACHADO.kickoff, "scheduled", espn="401914464")
    row(MACHADO.replacement_match_id, "Tina Black", "Melissa Amaya", MACHADO.kickoff, "finished", scores=(1, 0))
    row(
        str(uuid.uuid4()),
        "Other",
        "Melissa Gatto",
        datetime(2026, 6, 6, 22, tzinfo=timezone.utc),
        "finished",
        "x1",
        (1, 0),
    )
    for e in ENTRIES:
        for pid in e.prediction_ids:
            cur.execute(
                """
                INSERT INTO predictions (id, match_id, model_name, model_version, prediction_type, predicted_outcome,
                                         confidence, probabilities, created_at)
                VALUES (%s, %s, 'ensemble_mma_ml', %s, 'moneyline', 'away', 0.57,
                        '{"home": 0.43, "away": 0.57}'::jsonb, %s)
                """,
                (pid, e.match_id, f"v{pid[:6]}", datetime(2026, 9, 13, tzinfo=timezone.utc)),
            )
        for version in ("v1", "v0"):
            cur.execute(
                "INSERT INTO features_cache (match_id, feature_set, feature_version, features, expires_at) "
                "VALUES (%s, 'mma_baseline', %s, '{}'::jsonb, NOW())",
                (e.match_id, version),
            )
    c.commit()
    c.close()


def _event(cid):
    comp = copy.deepcopy(COMPS[cid])
    return {"id": comp["event_id"], "season": {"type": 2}, "competitions": [comp]}


class Espn:
    """fetch_day stand-in serving the real competitions, counting calls."""

    def __init__(self, days=None):
        self.calls: list = []
        self.days = days or {
            GATTO.scoreboard_date: [_event("401924511")],
            MACHADO.scoreboard_date: [_event("401914464")],
        }

    def __call__(self, cfg, slug, day):
        self.calls.append((slug, day))
        return copy.deepcopy(self.days.get(day, []))


@pytest.fixture(autouse=True)
def core(monkeypatch):
    answers = {"401924511": ("draw", True)}
    monkeypatch.setattr(fu, "espn_core_result", lambda cfg, slug, ev, cid: answers.get(str(cid), (None, True)))
    for k in fu.IDENTITY_STATS:
        fu.IDENTITY_STATS[k] = 0
    yield answers


def _run(dsn, tmp_path, apply=False, expect=None, espn=None, tag="r"):
    conn = _connect(dsn)
    out = tmp_path / f"{tag}_{uuid.uuid4().hex[:6]}.csv"
    arch = tmp_path / f"{tag}_{uuid.uuid4().hex[:6]}.jsonl.gz"
    try:
        rc = st.run(conn, ENTRIES, apply, out, arch, expect, fetch=espn or Espn())
    finally:
        conn.close()
    return rc, out, arch


def _check_after(dsn) -> int:
    conn = _connect(dsn)
    try:
        return st.run_check_after(conn, ENTRIES)
    finally:
        conn.close()


def _state(dsn):
    return {
        "matches": _q(dsn, "SELECT id::text, status, home_score, away_score, metadata FROM matches ORDER BY id"),
        "predictions": _q(dsn, "SELECT id::text, actual_outcome, is_correct FROM predictions ORDER BY id"),
        "features": _q(dsn, "SELECT match_id::text FROM features_cache ORDER BY 1"),
        "teams": _q(dsn, "SELECT id::text, name FROM teams ORDER BY id"),
    }


def _plan_logged(caplog) -> dict:
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("PLAN ")][-1]
    return {k: int(v) for k, v in (kv.split("=") for kv in line.split()[1:])}


class TestTheReviewedRun:
    def test_the_dry_run_plans_the_reviewed_plan_and_commits_nothing(self, db, tmp_path, caplog):
        before = _state(db)
        caplog.set_level("INFO")
        rc, out, arch = _run(db, tmp_path)
        assert rc == 0 and _plan_logged(caplog) == REVIEWED
        assert "DRY RUN: the whole transaction ran and was ROLLED BACK" in caplog.text
        assert _state(db) == before and not arch.exists()
        changes = Counter((r["table"], r["change"]) for r in __import__("csv").DictReader(out.open()))
        assert changes == {
            ("matches", "updated"): 1,
            ("predictions", "updated"): 2,
            ("matches", "deleted"): 1,
            ("predictions", "deleted"): 2,
            ("features_cache", "deleted"): 2,
            ("teams", "deleted"): 1,
        }

    def test_the_apply_settles_both_rows_and_nothing_else(self, db, tmp_path, caplog):
        before = _state(db)
        rc, out, arch = _run(db, tmp_path, apply=True, expect=REVIEWED)
        assert rc == 0
        gatto = _q(
            db,
            "SELECT status, home_score, away_score, metadata, external_ids FROM matches WHERE id = %s",
            (GATTO.match_id,),
        )
        assert gatto[0]["status"] == "finished" and (gatto[0]["home_score"], gatto[0]["away_score"]) == (None, None)
        assert gatto[0]["metadata"]["no_winner"] == "draw"
        assert gatto[0]["metadata"]["no_winner_detail"] == "majority draw; judges 28-28, 27-29, 28-28; ended R3 5:00"
        assert gatto[0]["external_ids"] == {"espn": "401924511"}
        preds = _q(db, "SELECT actual_outcome, is_correct FROM predictions WHERE match_id = %s", (GATTO.match_id,))
        assert preds == [{"actual_outcome": "draw", "is_correct": None}] * 2
        assert _q(db, "SELECT id FROM matches WHERE id = %s", (MACHADO.match_id,)) == []
        assert _q(db, "SELECT id FROM predictions WHERE match_id = %s", (MACHADO.match_id,)) == []
        assert _q(db, "SELECT id FROM features_cache WHERE match_id = %s", (MACHADO.match_id,)) == []
        assert _q(db, "SELECT id FROM teams WHERE id = %s", (MACHADO.replaced_team_id,)) == []
        after = _state(db)
        untouched = {GATTO.match_id, MACHADO.match_id}
        assert [m for m in after["matches"] if m["id"] not in untouched] == [
            m for m in before["matches"] if m["id"] not in untouched
        ]
        assert len(after["teams"]) == len(before["teams"]) - 1
        with gzip.open(arch, "rt") as fh:
            archived = Counter((json.loads(line)["table"], json.loads(line)["role"]) for line in fh)
        assert archived == {
            ("matches", "record_no_winner_before"): 1,
            ("matches", "delete_replaced_before"): 1,
            ("predictions", "child_before"): 4,
            ("features_cache", "child_before"): 4,
            ("teams", "team_before"): 1,
            ("teams", "fighter_before"): 2,
        }
        assert _check_after(db) == 0

    def test_a_re_run_writes_nothing_and_asks_espn_nothing(self, db, tmp_path, caplog):
        assert _run(db, tmp_path, apply=True, expect=REVIEWED)[0] == 0
        espn = Espn()
        caplog.set_level("INFO")
        rc, out, _ = _run(db, tmp_path, espn=espn)
        assert rc == 0 and _plan_logged(caplog) == DONE and espn.calls == []
        assert "NOTHING TO WRITE" in caplog.text and out.read_text().strip() == ",".join(st.CSV_FIELDS)
        rc, _, _ = _run(db, tmp_path, apply=True, expect=DONE, espn=espn)
        assert rc == 0 and espn.calls == []

    def test_check_after_fails_before_the_apply(self, db):
        assert _check_after(db) == 1

    @pytest.mark.parametrize("apply", [False, True])
    def test_the_log_carries_one_plan_the_plan(self, db, tmp_path, caplog, apply):
        # The wrapper reads the LAST 'PLAN k=v ...' of a run's log: the in-transaction re-plan (the done plan)
        # once printed as one, and a rehearsal's pre-check read 'done' before anything was written.
        caplog.set_level("INFO")
        assert _run(db, tmp_path, apply=apply, expect=REVIEWED if apply else None)[0] == 0
        found = re.findall(r"PLAN [a-z_]+=[0-9]+(?: [a-z_]+=[0-9]+)*", caplog.text)
        assert found and set(found) == {st.plan_line(REVIEWED)}, found
        caplog.clear()
        assert _check_after(db) == (0 if apply else 1)
        assert re.findall(r"PLAN [a-z_]+=[0-9]+", caplog.text) == []


class TestNothingIsCommitted:
    def test_a_plan_other_than_expect_plan(self, db, tmp_path, caplog):
        before = _state(db)
        rc, _, arch = _run(db, tmp_path, apply=True, expect={**REVIEWED, "teams_delete": 0})
        assert rc == 2 and _state(db) == before and not arch.exists()
        assert "the plan is not --expect-plan" in caplog.text

    def test_a_rec_on_the_replaced_row(self, db, tmp_path, caplog):
        pid = MACHADO.prediction_ids[0]
        _x(
            db,
            "INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection, odds_at_recommendation,"
            " bookmaker, recommended_stake, status) VALUES (%s, %s, 'moneyline', 'away', 1.9, 'b', 10, 'pending')",
            (pid, MACHADO.match_id),
        )
        before = _state(db)
        assert _run(db, tmp_path)[0] == 2 and _state(db) == before
        assert "rec(s) reference the row or its predictions" in caplog.text

    def test_an_extra_prediction(self, db, tmp_path, caplog):
        _x(
            db,
            "INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome, "
            "confidence, probabilities) VALUES (%s, 'ensemble_mma_ml', 'new', 'moneyline', 'home', 0.5, '{}'::jsonb)",
            (GATTO.match_id,),
        )
        assert _run(db, tmp_path)[0] == 2
        assert "are not the reviewed" in caplog.text

    def test_a_recorded_winner(self, db, tmp_path, caplog):
        _x(
            db,
            "UPDATE matches SET status = 'finished', home_score = 1, away_score = 0 WHERE id = %s",
            (GATTO.match_id,),
        )
        assert _run(db, tmp_path)[0] == 2
        assert "neither still to record nor recorded as a draw" in caplog.text

    def test_espn_not_listing_the_competition(self, db, tmp_path, caplog):
        espn = Espn(days={GATTO.scoreboard_date: [], MACHADO.scoreboard_date: [_event("401914464")]})
        assert _run(db, tmp_path, espn=espn)[0] == 2
        assert "does not list competition 401924511" in caplog.text

    def test_espn_still_listing_the_replaced_fighter(self, db, tmp_path, caplog):
        ev = _event("401914464")
        ev["competitions"][0]["competitors"][0]["athlete"]["displayName"] = "Valesca Machado"
        espn = Espn(days={GATTO.scoreboard_date: [_event("401924511")], MACHADO.scoreboard_date: [ev]})
        assert _run(db, tmp_path, espn=espn)[0] == 2
        assert "ESPN lists Valesca Machado v Melissa Amaya, not Tina Black v Melissa Amaya" in caplog.text

    @pytest.mark.parametrize("winner", [0, 1])
    def test_espn_now_showing_a_winner_refuses_before_any_write(self, db, tmp_path, caplog, winner):
        ev = _event("401924511")
        ev["competitions"][0]["competitors"][winner]["winner"] = True
        espn = Espn(days={GATTO.scoreboard_date: [ev], MACHADO.scoreboard_date: [_event("401914464")]})
        before = _state(db)
        assert _run(db, tmp_path, espn=espn)[0] == 2 and _state(db) == before
        assert "ESPN's scoreboard reads a winner or no result, not draw" in caplog.text

    def test_a_replacement_row_that_is_not_finished(self, db, tmp_path, caplog):
        _x(db, "UPDATE matches SET status = 'scheduled' WHERE id = %s", (MACHADO.replacement_match_id,))
        assert _run(db, tmp_path)[0] == 2
        assert "the replacement row" in caplog.text

    def test_a_core_api_that_disagrees_rolls_back(self, db, tmp_path, caplog, core):
        core["401924511"] = ("no-contest", True)
        before = _state(db)
        rc, _, _ = _run(db, tmp_path, apply=True, expect=REVIEWED)
        assert rc == 1 and _state(db) == before
        assert "the results path did not record the bout" in caplog.text

    def test_a_write_outside_the_listed_rows_rolls_back(self, db, tmp_path, caplog, monkeypatch):
        real = st.apply_grade

        def grade_and_stray(cur, p):
            real(cur, p)
            cur.execute("UPDATE matches SET venue = 'stray' WHERE id = %s", (MACHADO.replacement_match_id,))

        monkeypatch.setattr(st, "apply_grade", grade_and_stray)
        before = _state(db)
        rc, _, _ = _run(db, tmp_path, apply=True, expect=REVIEWED)
        assert rc == 1 and _state(db) == before
        assert "rows outside the listed ones changed: ['matches']" in caplog.text

    def test_a_results_path_that_counts_a_refusal_rolls_back(self, db, tmp_path, caplog, monkeypatch):
        real = fu._record_result

        def record_and_count(cur, cfg, league_id, comp, ctx):
            fu.IDENTITY_STATS["team_namesakes"] += 1  # e.g. a fighter inserted as a namesake on the way
            return real(cur, cfg, league_id, comp, ctx)

        monkeypatch.setattr(fu, "_record_result", record_and_count)
        before = _state(db)
        rc, _, _ = _run(db, tmp_path, apply=True, expect=REVIEWED)
        assert rc == 1 and _state(db) == before
        assert "the results path counted a refusal or a team problem" in caplog.text

    def test_a_results_path_that_writes_nothing_rolls_back(self, db, tmp_path, caplog, monkeypatch):
        monkeypatch.setattr(fu, "_record_result", lambda *a: True)
        before = _state(db)
        rc, _, _ = _run(db, tmp_path, apply=True, expect=REVIEWED)
        assert rc == 1 and _state(db) == before
        assert "after the results path the row reads" in caplog.text

    def test_a_grader_that_reports_other_counts_than_the_plan_rolls_back(self, db, tmp_path, caplog, monkeypatch):
        # The database ends right, but the grader's own count disagrees with the plan: the apply trusts neither.
        real = st.gcm.grade_match

        def grade_and_miscount(cur, match, stats=None):
            out = real(cur, match, stats)
            stats["no_winner_predictions"] -= 1
            return out

        monkeypatch.setattr(st.gcm, "grade_match", grade_and_miscount)
        before = _state(db)
        rc, _, _ = _run(db, tmp_path, apply=True, expect=REVIEWED)
        assert rc == 1 and _state(db) == before
        assert "the grader recorded 1 prediction(s) and voided 0 rec(s); the plan says 2 and 0" in caplog.text

    def test_a_re_plan_inside_the_transaction_that_is_not_done_rolls_back(self, db, tmp_path, caplog, monkeypatch):
        real = st.plan_entries

        def plan(cur, entries, schema, lock):
            plans = real(cur, entries, schema, lock)
            if not lock:  # only the in-transaction re-plan
                plans[0].record = True
            return plans

        monkeypatch.setattr(st, "plan_entries", plan)
        before = _state(db)
        rc, _, _ = _run(db, tmp_path, apply=True, expect=REVIEWED)
        assert rc == 1 and _state(db) == before
        assert "the re-plan is not done: record=1" in caplog.text

    def test_a_team_with_another_bout_is_kept_and_the_plan_says_so(self, db, tmp_path, caplog):
        team = {r["name"]: r["id"] for r in _q(db, "SELECT id::text, name FROM teams")}
        _x(
            db,
            "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season) "
            "SELECT league_id, %s, %s, '2026-12-01', 'scheduled', '2026' FROM matches WHERE id = %s",
            (team["Valesca Machado"], team["Other"], GATTO.match_id),
        )
        caplog.set_level("INFO")
        assert _run(db, tmp_path)[0] == 0
        assert _plan_logged(caplog)["teams_delete"] == 0


class TestTheOtherOrder:
    def test_already_recorded_by_the_sweep_and_already_cancelled_as_re_drawn(self, db, tmp_path, caplog):
        # What a deep results sweep with this commit's fetch_upcoming would leave: the draw recorded (its
        # predictions not yet graded), the replaced row cancelled as a re-drawn slot with its ESPN id released.
        _x(
            db,
            "UPDATE matches SET status = 'finished', metadata = metadata || %s::jsonb WHERE id = %s",
            (json.dumps({"no_winner": "draw", "no_winner_detail": "majority draw"}), GATTO.match_id),
        )
        _x(
            db,
            "UPDATE matches SET status = 'cancelled', external_ids = '{}'::jsonb, metadata = metadata || %s::jsonb "
            "WHERE id = %s",
            (json.dumps({"redrawn_espn_id": "401914464"}), MACHADO.match_id),
        )
        caplog.set_level("INFO")
        rc, _, _ = _run(db, tmp_path)
        plan = _plan_logged(caplog)
        assert rc == 0 and plan == {**REVIEWED, "record": 0, "record_done": 1}
        assert _run(db, tmp_path, apply=True, expect=plan)[0] == 0
        assert _check_after(db) == 0
        preds = _q(db, "SELECT actual_outcome, is_correct FROM predictions WHERE match_id = %s", (GATTO.match_id,))
        assert preds == [{"actual_outcome": "draw", "is_correct": None}] * 2


class TestTheFighterRows:
    def test_a_stamp_the_results_path_makes_is_archived_and_audited(self, db, tmp_path, monkeypatch):
        # With the mma_identity resolver deployed, the results path stamps ESPN athlete ids on the two fighters
        # (prod rehearsal: {} -> {"espn": "4420893"} / {"espn": "5149147"}). The fingerprint tolerates it; the
        # reversal record and the audit CSV must show it.
        real = fu._record_result
        gatto_team = _q(db, "SELECT id::text AS id FROM teams WHERE name = 'Melissa Gatto'")[0]["id"]

        def record_and_stamp(cur, cfg, league_id, comp, ctx):
            ok = real(cur, cfg, league_id, comp, ctx)
            cur.execute('UPDATE teams SET external_ids = \'{"espn": "4420893"}\'::jsonb WHERE id = %s', (gatto_team,))
            return ok

        monkeypatch.setattr(fu, "_record_result", record_and_stamp)
        rc, out, arch = _run(db, tmp_path, apply=True, expect=REVIEWED)
        assert rc == 0
        rows = [r for r in __import__("csv").DictReader(out.open()) if r["table"] == "teams"]
        stamped = [r for r in rows if r["row_id"] == gatto_team]
        assert len(stamped) == 1 and stamped[0]["change"] == "updated"
        assert stamped[0]["match_id"] == GATTO.match_id and stamped[0]["action"] == "record_no_winner"
        assert json.loads(stamped[0]["before"])["external_ids"] in ({}, None)
        assert json.loads(stamped[0]["after"])["external_ids"] == {"espn": "4420893"}
        with gzip.open(arch, "rt") as fh:
            fighters = [json.loads(line)["row"] for line in fh if json.loads(line)["role"] == "fighter_before"]
        assert sorted(r["name"] for r in fighters) == ["Ernesta Kareckaite", "Melissa Gatto"]

    def test_without_a_stamp_the_fighter_rows_are_not_in_the_audit_csv(self, db, tmp_path):
        rc, out, _ = _run(db, tmp_path)
        assert rc == 0
        teams = [r for r in __import__("csv").DictReader(out.open()) if r["table"] == "teams"]
        assert [r["row_id"] for r in teams] == [MACHADO.replaced_team_id]


# ── run B's done check, its own SQL, on the same database ─────────────


def _b_sql(name: str) -> str:
    """One of run B's SQL files as the wrapper writes it: the heredoc body, with the wrapper's own values for
    the variables it expands."""
    text = B_WRAPPER.read_text()
    m = re.search(r'cat > "\$W/' + name + r'\.sql" <<SQLEOF\n(.*?)\nSQLEOF', text, re.S)
    assert m, name
    values = dict(re.findall(r"^(AS_OF|SETTLED_DRAW_ID|SETTLED_DELETED_ID)=(\S+)", text, re.M))
    sql = m.group(1)
    for k, v in values.items():
        sql = sql.replace("$" + k, v)
    assert "$" not in sql, sql
    return sql


def _b_state(dsn) -> str:
    c = _connect(dsn)
    try:
        with c.cursor() as cur:
            cur.execute(_b_sql("state_b"))
            stale, recs = cur.fetchone()
            cur.execute(_b_sql("held"))
            held = cur.fetchone()[0]
        return f"{stale}|{recs} {held}"
    finally:
        c.rollback()
        c.close()


class TestRunBsDoneCheck:
    def _recs(self, dsn):
        """A UFC rec written before run B's AS_OF (2026-11-01) on a fought bout, and one the pipeline writes
        today for a card ahead."""
        league = _q(dsn, "SELECT league_id::text AS l FROM matches WHERE id = %s", (GATTO.match_id,))[0]["l"]
        team = {r["name"]: r["id"] for r in _q(dsn, "SELECT id::text, name FROM teams")}
        old_row = _q(dsn, "SELECT id::text AS id FROM matches WHERE external_ids->>'espn' = 'x1'")[0]["id"]  # _seed
        new_row = str(uuid.uuid4())
        _x(
            dsn,
            "INSERT INTO matches (id, league_id, home_team_id, away_team_id, match_date, status, season) "
            "VALUES (%s, %s, %s, %s, NOW() + interval '5 days', 'scheduled', '2026')",
            (new_row, league, team["Other"], team["Melissa Amaya"]),
        )
        for mid, created in ((old_row, "2026-08-29T00:00:00Z"), (new_row, None)):
            pid = str(uuid.uuid4())
            _x(
                dsn,
                "INSERT INTO predictions (id, match_id, model_name, model_version, prediction_type, predicted_outcome,"
                " confidence, probabilities) VALUES (%s, %s, 'ensemble_mma_ml', 'v', 'moneyline', 'home', 0.6,"
                " '{}'::jsonb)",
                (pid, mid),
            )
            _x(
                dsn,
                "INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection,"
                " odds_at_recommendation, bookmaker, recommended_stake, status, created_at)"
                " VALUES (%s, %s, 'moneyline', 'home', 1.9, 'b', 10, 'pending', COALESCE(%s::timestamptz, NOW()))",
                (pid, mid, created),
            )
        assert _q(dsn, "SELECT count(*) AS n FROM betting_recommendations")[0]["n"] == 2

    def test_held_before_and_settled_after_a_real_apply_and_new_recs_do_not_count(self, db, tmp_path):
        self._recs(db)
        # Before: both held rows stale and held; 1 UFC rec created before AS_OF (today's rec not counted).
        assert _b_state(db) == "2|1 held|held"
        assert _run(db, tmp_path, apply=True, expect=REVIEWED)[0] == 0
        # After: neither row stale any more, both settled as decided; the recs untouched.
        assert _b_state(db) == "0|1 settled|settled"

    def test_the_other_order_reads_one_settled_and_one_held(self, db):
        # A deep re-fetch recorded the draw first and cancelled the Machado row as a re-drawn slot.
        _x(
            db,
            "UPDATE matches SET status = 'finished', metadata = metadata || %s::jsonb WHERE id = %s",
            (json.dumps({"no_winner": "draw", "no_winner_detail": "majority draw"}), GATTO.match_id),
        )
        _x(db, "UPDATE matches SET status = 'cancelled' WHERE id = %s", (MACHADO.match_id,))
        assert _b_state(db) == "1|0 settled|held"

    def test_a_recorded_winner_on_the_draw_row_is_other(self, db):
        _x(
            db,
            "UPDATE matches SET status = 'finished', home_score = 1, away_score = 0 WHERE id = %s",
            (GATTO.match_id,),
        )
        assert _b_state(db).endswith(" other|held")
