"""fetch_upcoming.cancel_dropped_bouts and scripts/backfill_results.py against a
real Postgres (a throwaway cluster built from services/data-ingestion/db/migrations,
one database per test from a template):

  * a finished UFC card cancels the stale rows it no longer lists — an
    ESPN-identified bout whose competition is gone, an unidentified one whose
    fighters are both off the card — and leaves a listed id, a fought pair
    (either order; also under a stale ESPN id the sweep no longer lists, and
    its rec is then not voided), an unidentified row with one fighter on the
    card, an unfinished card, a row outside the card's window or in the
    future, a card with an unparseable competition, and every tennis sweep
    alone;
  * the cancel is correctable (a later result for the row takes) and the
    grader voids its open recs once the kickoff is 72 h old;
  * the backfill's dry run plans the results, the settlements and the voids
    and rolls everything back; an apply pinned to that plan commits it and a
    re-run plans no write; a different pin writes nothing; a rec written at or
    after kickoff is voided, never settled; only the recs on rows this run
    cancelled are voided (never another sport's, nor an older cancelled row's,
    nor a duplicate's); a write outside the sport rolls the run back; a commit
    by another session meanwhile does not reach the plan (REPEATABLE READ);
    --expect-recs commits only the exact rec outcomes;
  * a competition ESPN lists the other way round writes onto its row, and the
    logs name the fighters in the row's order.

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

import gzip
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
MIGRATIONS = REPO / "services" / "data-ingestion" / "db" / "migrations"

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
gcm = _load("grade_completed_matches", "grade_completed_matches.py")
bf = _load("backfill_results_pg", "backfill_results.py")

MMA = fu.SPORT_CONFIGS["mma"]
NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)  # ESPN's dates carry minutes, not seconds
CARD = NOW - timedelta(days=10)  # the card's main-card segment; prelims 3 h earlier


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgb", dir="/tmp"))  # a Unix socket path is limited to ~100 bytes
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
def cur(dsn):
    conn = psycopg2.connect(**dsn)
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        yield c
    conn.rollback()
    conn.close()


# ── fixtures in the database ──────────────────────────────────────────


def _league(cur) -> str:
    return str(fu.ensure_league(cur, MMA, "UFC", "UFC", "USA"))


def _fighter(cur, name: str, sport: str = "mma") -> str:
    cur.execute(
        "INSERT INTO teams (name, normalized_name, sport) VALUES (%s, %s, %s) RETURNING id::text AS id",
        (name, fu.normalize_team(name), sport),
    )
    return cur.fetchone()["id"]


def _row(cur, league, home, away, kickoff, espn=None, status="scheduled", metadata=None):
    cur.execute(
        """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season, external_ids, metadata)
        VALUES (%s, %s, %s, %s, %s, '2026', %s::jsonb, %s::jsonb) RETURNING id::text AS id
        """,
        (league, home, away, kickoff, status, json.dumps({"espn": espn} if espn else {}), json.dumps(metadata or {})),
    )
    return cur.fetchone()["id"]


def _rec(cur, match_id, selection="away", created_at=None, odds="3.0000", stake="10.00"):
    cur.execute(
        """
        INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome,
                                 confidence, probabilities, created_at)
        VALUES (%s, 'ensemble_mma_ml', 'v1', 'moneyline', 'home', 0.6, '{"home": 0.6, "away": 0.4}'::jsonb,
                NOW() - interval '30 days')
        RETURNING id::text AS id
        """,
        (match_id,),
    )
    pid = cur.fetchone()["id"]
    cur.execute(
        """
        INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection, odds_at_recommendation,
                                             bookmaker, recommended_stake, status, created_at)
        VALUES (%s, %s, 'moneyline', %s, %s, 'b', %s, 'pending', %s) RETURNING id::text AS id
        """,
        (pid, match_id, selection, odds, stake, created_at),
    )
    return cur.fetchone()["id"]


def _get(cur, table, row_id, *cols):
    cur.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE id = %s", (row_id,))
    return dict(cur.fetchone())


# ── ESPN payloads ─────────────────────────────────────────────────────


def _comp(comp_id, when, home, away, winner="home", state="post", name="STATUS_FINAL"):
    def side(n, won):
        return {"athlete": {"displayName": n}, "winner": won}

    return {
        "id": comp_id,
        "date": when.strftime("%Y-%m-%dT%H:%MZ"),
        "status": {"type": {"state": state, "completed": state == "post", "name": name, "detail": "Final"}},
        "competitors": [side(home, winner == "home"), side(away, winner == "away")],
    }


def _card(event_id, *comps):
    return {"id": event_id, "season": {"type": 2}, "competitions": list(comps)}


# ── cancel_dropped_bouts ──────────────────────────────────────────────


class TestCancelDroppedBouts:
    def _setup(self, cur):
        league = _league(cur)
        f = {n: _fighter(cur, n) for n in ("Al", "Bo", "Cy", "Di", "Ed", "Fy")}
        return league, f

    def test_a_finished_card_cancels_an_identified_bout_it_no_longer_lists(self, cur):
        league, f = self._setup(cur)
        dropped = _row(cur, league, f["Al"], f["Bo"], CARD, espn="111")
        card = _card("600", _comp("222", CARD - timedelta(hours=3), "Cy", "Di"), _comp("333", CARD, "Ed", "Fy"))
        assert fu.cancel_dropped_bouts(cur, MMA, league, [card]) == 1
        row = _get(cur, "matches", dropped, "status", "external_ids", "metadata")
        assert row["status"] == "cancelled" and "espn" not in row["external_ids"]
        assert row["metadata"] == {
            "espn_status": fu.DROPPED_FROM_CARD,
            "status_source": "espn_scoreboard",
            "dropped_from_event": "600",
            "cancelled_espn_id": "111",
        }

    def test_a_listed_id_and_a_fought_pair_in_either_order_are_left(self, cur):
        league, f = self._setup(cur)
        listed = _row(cur, league, f["Al"], f["Bo"], CARD, espn="111")  # listed by another event of the sweep
        dup = _row(cur, league, f["Di"], f["Cy"], CARD - timedelta(hours=2))  # Cy v Di fought, other order
        card = _card("600", _comp("222", CARD, "Cy", "Di"))
        elsewhere = _card("601", _comp("111", NOW + timedelta(days=20), "Al", "Bo", state="pre"))
        assert fu.cancel_dropped_bouts(cur, MMA, league, [card, elsewhere]) == 0
        assert _get(cur, "matches", listed, "status")["status"] == "scheduled"
        assert _get(cur, "matches", dup, "status")["status"] == "scheduled"

    @pytest.mark.parametrize("order", ["same", "reversed"])
    def test_an_identified_duplicate_of_a_fought_pair_is_left_and_its_rec_is_not_voided(self, cur, order):
        # ESPN reissued the competition (or an old duplicate holds a stale id): '999' is listed nowhere, but its
        # two fighters fought on the card under '222'. repair_stranded_recs settles its rec from the twin; a
        # cancel here would have the grader void it for good.
        league, f = self._setup(cur)
        home, away = (f["Cy"], f["Di"]) if order == "same" else (f["Di"], f["Cy"])
        dup = _row(cur, league, home, away, CARD - timedelta(hours=1), espn="999")
        rec = _rec(cur, dup, created_at=CARD - timedelta(days=1))
        card = _card("600", _comp("222", CARD, "Cy", "Di"), _comp("333", CARD, "Ed", "Fy"))
        assert fu.cancel_dropped_bouts(cur, MMA, league, [card]) == 0
        row = _get(cur, "matches", dup, "status", "external_ids")
        assert row == {"status": "scheduled", "external_ids": {"espn": "999"}}
        assert gcm.void_recs_on_cancelled_matches(cur) == (0, 0)
        assert _get(cur, "betting_recommendations", rec, "status")["status"] == "pending"

    def test_an_unidentified_row_is_cancelled_only_when_neither_fighter_is_on_the_card(self, cur):
        league, f = self._setup(cur)
        replaced = _row(cur, league, f["Al"], f["Bo"], CARD)  # Al fights Cy instead: or 'Bo' is Cy's other name
        scrapped = _row(cur, league, f["Ed"], f["Fy"], CARD)  # neither on the card
        card = _card("600", _comp("222", CARD, "Al", "Cy"))
        assert fu.cancel_dropped_bouts(cur, MMA, league, [card]) == 1
        assert _get(cur, "matches", replaced, "status")["status"] == "scheduled"
        row = _get(cur, "matches", scrapped, "status", "metadata")
        assert row["status"] == "cancelled" and "cancelled_espn_id" not in row["metadata"]

    @pytest.mark.parametrize(
        "card",
        [
            _card("600", _comp("222", CARD, "Cy", "Di"), _comp("333", CARD, "Ed", "Fy", state="in")),  # running
            _card("600", _comp("222", CARD, "Cy", "Di"), _comp("333", CARD, "TBA", "Fy")),  # unparseable
            _card("600"),  # no competition at all
        ],
    )
    def test_an_unfinished_or_unreadable_card_cancels_nothing(self, cur, card):
        league, f = self._setup(cur)
        row = _row(cur, league, f["Al"], f["Bo"], CARD, espn="111")
        assert fu.cancel_dropped_bouts(cur, MMA, league, [card]) == 0
        assert _get(cur, "matches", row, "status")["status"] == "scheduled"

    def test_rows_outside_the_window_or_in_the_future_are_left(self, cur):
        league, f = self._setup(cur)
        late = _row(cur, league, f["Al"], f["Bo"], CARD + timedelta(hours=fu.CARD_WINDOW_HOURS + 1), espn="111")
        recent = NOW - timedelta(hours=1)
        future = _row(cur, league, f["Ed"], f["Fy"], NOW + timedelta(hours=2), espn="444")
        cards = [_card("600", _comp("222", CARD, "Cy", "Di")), _card("601", _comp("555", recent, "Cy", "Di"))]
        assert fu.cancel_dropped_bouts(cur, MMA, league, cards) == 0
        assert _get(cur, "matches", late, "status")["status"] == "scheduled"
        assert _get(cur, "matches", future, "status")["status"] == "scheduled"

    def test_a_tennis_sweep_cancels_nothing(self, cur):
        league, f = self._setup(cur)
        row = _row(cur, league, f["Al"], f["Bo"], CARD, espn="111")
        tennis = fu.SPORT_CONFIGS["tennis"]
        assert fu.cancel_dropped_bouts(cur, tennis, league, [_card("600", _comp("222", CARD, "Cy", "Di"))]) == 0
        assert _get(cur, "matches", row, "status")["status"] == "scheduled"

    def test_the_cancel_stays_correctable_and_the_grader_voids_its_rec(self, cur):
        league, f = self._setup(cur)
        row = _row(cur, league, f["Al"], f["Bo"], CARD)
        rec = _rec(cur, row, created_at=CARD - timedelta(days=1))
        fu.cancel_dropped_bouts(cur, MMA, league, [_card("600", _comp("222", CARD, "Cy", "Di"))])
        assert gcm.void_recs_on_cancelled_matches(cur) == (1, 0)
        assert _get(cur, "betting_recommendations", rec, "status", "actual_result", "profit_loss") == {
            "status": "void",
            "actual_result": gcm.CANCELLED_VOID_RESULT,
            "profit_loss": Decimal("0.00"),
        }
        # ESPN lists the bout after all: its result takes, as for ESPN's own cancels.
        assert fu.insert_finished_match(cur, MMA, league, f["Al"], f["Bo"], CARD, None, 1, 0, {}) == 1
        assert _get(cur, "matches", row, "status")["status"] == "finished"


# ── backfill_results.run ──────────────────────────────────────────────


class TestBackfill:
    def _world(self, dsn):
        """A June-style card nobody resulted: Al v Bo fought (Bo won) with a pending rec on Bo; Cy v Di dropped
        (neither on the card) with a pending rec; Ed v Fy fought but we never had it. Returns ids and the payload."""
        conn = psycopg2.connect(**dsn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            league = _league(cur)
            f = {n: _fighter(cur, n) for n in ("Al", "Bo", "Cy", "Di")}
            fought = _row(cur, league, f["Al"], f["Bo"], CARD)
            dropped = _row(cur, league, f["Cy"], f["Di"], CARD)
            ids = {
                "fought": fought,
                "dropped": dropped,
                "rec_fought": _rec(cur, fought, "away", CARD - timedelta(minutes=15)),
                "rec_dropped": _rec(cur, dropped, "home", CARD - timedelta(days=2)),
            }
        conn.commit()
        conn.close()
        card = _card("600", _comp("222", CARD, "Al", "Bo", winner="away"), _comp("333", CARD, "Ed", "Fy"))
        return ids, {"ufc": [card]}

    def _run(self, dsn, events, tmp_path, name, **kw):
        conn = psycopg2.connect(**dsn)
        try:
            return bf.run(
                conn,
                "mma",
                [CARD.date()],
                events_by_slug=events,
                out_csv=tmp_path / f"{name}.csv",
                archive=tmp_path / f"{name}.jsonl.gz",
                **kw,
            )
        finally:
            conn.close()

    PLAN = {
        "results": 2,
        "flipped": 1,
        "inserted": 1,
        "cancelled": 1,
        "stamped": 0,
        "rescored": 0,
        "other": 0,
        "recs_settled": 1,
        "recs_voided": 1,
        "recs_other": 0,
        "recs_left": 0,
        "fetch_failed": 0,
        "identity_refused": 0,
    }

    def _state(self, dsn, ids):
        conn = psycopg2.connect(**dsn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            out = {
                "fought": _get(cur, "matches", ids["fought"], "status", "home_score", "away_score"),
                "dropped": _get(cur, "matches", ids["dropped"], "status"),
                "rec_fought": _get(cur, "betting_recommendations", ids["rec_fought"], "status", "profit_loss"),
                "rec_dropped": _get(cur, "betting_recommendations", ids["rec_dropped"], "status", "actual_result"),
            }
        conn.close()
        return out

    def test_the_dry_run_plans_everything_and_rolls_it_back(self, dsn, tmp_path):
        ids, events = self._world(dsn)
        before = self._state(dsn, ids)
        result = self._run(dsn, events, tmp_path, "dry")
        assert result["plan"] == self.PLAN and not result["plan_mismatch"]
        assert self._state(dsn, ids) == before
        assert not (tmp_path / "dry.jsonl.gz").exists()
        rows = (tmp_path / "dry.csv").read_text().splitlines()
        assert rows[0] == "kind,id,change,before,after" and len(rows) == 1 + 5

    def test_a_pinned_apply_commits_and_a_rerun_plans_no_write(self, dsn, tmp_path):
        ids, events = self._world(dsn)
        result = self._run(dsn, events, tmp_path, "apply", apply_changes=True, expect_plan=self.PLAN)
        assert not result["plan_mismatch"]
        assert self._state(dsn, ids) == {
            "fought": {"status": "finished", "home_score": 0, "away_score": 1},
            "dropped": {"status": "cancelled"},
            "rec_fought": {"status": "won", "profit_loss": Decimal("20.00")},
            "rec_dropped": {"status": "void", "actual_result": gcm.CANCELLED_VOID_RESULT},
        }
        archived = [json.loads(line) for line in gzip.open(tmp_path / "apply.jsonl.gz", "rt")]
        assert {a["change"] for a in archived} == {"flipped", "inserted", "cancelled", "recs_settled", "recs_voided"}
        assert next(a for a in archived if a["id"] == ids["rec_fought"])["before"]["status"] == "pending"
        again = self._run(dsn, events, tmp_path, "again")
        assert {k: again["plan"][k] for k in bf.WRITE_KEYS} == dict.fromkeys(bf.WRITE_KEYS, 0)
        assert again["plan"]["results"] == 2

    def test_a_different_pin_writes_nothing(self, dsn, tmp_path):
        ids, events = self._world(dsn)
        before = self._state(dsn, ids)
        result = self._run(dsn, events, tmp_path, "drift", apply_changes=True, expect_plan={**self.PLAN, "flipped": 2})
        assert result["plan_mismatch"]
        assert self._state(dsn, ids) == before and not (tmp_path / "drift.jsonl.gz").exists()

    def test_a_rec_written_at_or_after_kickoff_is_voided_not_settled(self, dsn, tmp_path):
        ids, events = self._world(dsn)
        conn = psycopg2.connect(**dsn)
        with conn.cursor() as c:
            c.execute("UPDATE betting_recommendations SET created_at = %s WHERE id = %s", (CARD, ids["rec_fought"]))
        conn.commit()
        conn.close()
        result = self._run(dsn, events, tmp_path, "pk", apply_changes=True, expect_plan=None)
        assert result["plan"]["recs_settled"] == 0 and result["plan"]["recs_voided"] == 2
        assert self._state(dsn, ids)["rec_fought"] == {"status": "void", "profit_loss": Decimal("0.00")}

    def _extra(self, dsn):
        """Beside the world: a soccer row and an MMA duplicate (orphan_of), both cancelled long ago, and an MMA
        duplicate still 'scheduled' that the card drops — each with a pending rec written before kickoff."""
        conn = psycopg2.connect(**dsn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            soccer = str(fu.ensure_league(cur, fu.SPORT_CONFIGS["soccer"], "E0", "Premier League", "England"))
            h, a = _fighter(cur, "Ham", "soccer"), _fighter(cur, "Ars", "soccer")
            soccer_row = _row(cur, soccer, h, a, CARD, status="cancelled")
            league = _league(cur)
            g, hy, ix, jo = (_fighter(cur, n) for n in ("Gu", "Hy", "Ix", "Jo"))
            old_dup = _row(
                cur, league, g, hy, CARD - timedelta(days=30), status="cancelled", metadata={"orphan_of": "x"}
            )
            new_dup = _row(cur, league, ix, jo, CARD, metadata={"orphan_of": "y"})
            out = {
                "soccer_row": soccer_row,
                "rec_soccer": _rec(cur, soccer_row, created_at=CARD - timedelta(days=1)),
                "rec_old_dup": _rec(cur, old_dup, created_at=CARD - timedelta(days=31)),
                "rec_new_dup": _rec(cur, new_dup, created_at=CARD - timedelta(days=1)),
                "new_dup": new_dup,
            }
        conn.commit()
        conn.close()
        return out

    def _rec_status(self, dsn, rec_id):
        conn = psycopg2.connect(**dsn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            out = _get(cur, "betting_recommendations", rec_id, "status")["status"]
        conn.close()
        return out

    def test_only_the_recs_on_rows_this_run_cancelled_are_voided(self, dsn, tmp_path, caplog):
        ids, events = self._world(dsn)
        extra = self._extra(dsn)
        # The run cancels the dropped row and the scheduled duplicate (neither pair is on the card); it voids the
        # dropped row's rec only. The two older MMA recs stay pending on rows that are not finished: recs_left.
        plan = {**self.PLAN, "cancelled": 2, "recs_left": 2}
        result = self._run(dsn, events, tmp_path, "scope", apply_changes=True, expect_plan=plan)
        assert not result["plan_mismatch"], result["plan"]
        assert self._state(dsn, ids)["rec_dropped"]["status"] == "void"
        for key in ("rec_soccer", "rec_old_dup", "rec_new_dup"):
            assert self._rec_status(dsn, extra[key]) == "pending", key
        assert any(
            "Not voiding rec" in r.getMessage() and extra["rec_new_dup"] in r.getMessage() for r in caplog.records
        )

    def test_a_write_outside_the_sport_rolls_the_run_back(self, dsn, tmp_path, monkeypatch):
        ids, events = self._world(dsn)
        extra = self._extra(dsn)
        before = self._state(dsn, ids)
        settle = bf.settle

        def settle_and_stray(cur, *a):
            out = settle(cur, *a)
            cur.execute("UPDATE betting_recommendations SET status = 'void' WHERE id = %s", (extra["rec_soccer"],))
            return out

        monkeypatch.setattr(bf, "settle", settle_and_stray)
        with pytest.raises(RuntimeError, match="outside --sport mma"):
            self._run(dsn, events, tmp_path, "stray", apply_changes=True, expect_plan=None)
        assert self._state(dsn, ids) == before and self._rec_status(dsn, extra["rec_soccer"]) == "pending"
        assert not (tmp_path / "stray.jsonl.gz").exists()

    def test_a_commit_by_another_session_meanwhile_does_not_reach_the_plan(self, dsn, tmp_path, monkeypatch):
        ids, events = self._world(dsn)
        conn = psycopg2.connect(**dsn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            league = _league(cur)
            other = _row(cur, league, _fighter(cur, "Ko"), _fighter(cur, "Lu"), NOW + timedelta(days=5))
        conn.commit()
        conn.close()
        settle = bf.settle

        def settle_while_the_sweep_commits(cur, *a):
            out = settle(cur, *a)
            sweep = psycopg2.connect(**dsn)
            with sweep.cursor() as c:  # the 15-min fixtures sweep moving an unrelated UFC row
                c.execute("UPDATE matches SET match_date = match_date + interval '1 hour' WHERE id = %s", (other,))
            sweep.commit()
            sweep.close()
            return out

        monkeypatch.setattr(bf, "settle", settle_while_the_sweep_commits)
        result = self._run(dsn, events, tmp_path, "rr", apply_changes=True, expect_plan=self.PLAN)
        assert result["plan"] == self.PLAN and not result["plan_mismatch"]

    def _expect(self, ids, **over):
        want = {
            ids["rec_fought"]: {"status": "won", "profit_loss": Decimal("20.00"), "actual_result": "won"},
            ids["rec_dropped"]: {
                "status": "void",
                "profit_loss": Decimal("0.00"),
                "actual_result": gcm.CANCELLED_VOID_RESULT,
            },
        }
        for rid, patch in over.items():
            want[ids[rid]] = {**want[ids[rid]], **patch}
        return want

    def test_expect_recs_commits_only_the_exact_outcomes(self, dsn, tmp_path):
        ids, events = self._world(dsn)
        before = self._state(dsn, ids)
        wrong = self._expect(
            ids, rec_fought={"status": "lost", "profit_loss": Decimal("-10.00"), "actual_result": "lost"}
        )
        result = self._run(dsn, events, tmp_path, "wrong", apply_changes=True, expect_plan=self.PLAN, expect_recs=wrong)
        assert result["plan_mismatch"] and self._state(dsn, ids) == before
        short = {k: v for k, v in self._expect(ids).items() if k == ids["rec_fought"]}
        result = self._run(dsn, events, tmp_path, "short", apply_changes=True, expect_plan=self.PLAN, expect_recs=short)
        assert result["plan_mismatch"] and self._state(dsn, ids) == before
        result = self._run(
            dsn, events, tmp_path, "exact", apply_changes=True, expect_plan=self.PLAN, expect_recs=self._expect(ids)
        )
        assert not result["plan_mismatch"]
        assert self._state(dsn, ids)["rec_fought"] == {"status": "won", "profit_loss": Decimal("20.00")}

    def test_another_sport_is_refused(self, dsn, tmp_path):
        conn = psycopg2.connect(**dsn)
        with pytest.raises(bf.UnsafeRequest):
            bf.run(conn, "soccer", [CARD.date()], events_by_slug={})
        conn.close()


# ── a competition ESPN lists the other way round ──────────────────────


class TestReversedListing:
    """UFC 2026-10-03: ESPN listed McGhee v Romero (401927906), then Romero v McGhee; the reversed parse forked a
    second row that took the result while the first stayed 'scheduled'. Both handlers now write in the row's order."""

    def _setup(self, cur, fork=False):
        league = _league(cur)
        al, bo = _fighter(cur, "Al"), _fighter(cur, "Bo")
        row = _row(cur, league, al, bo, CARD, espn="401927906")
        twin = _row(cur, league, bo, al, CARD) if fork else None
        return league, al, bo, row, twin

    def _pairs(self, cur, al, bo):
        cur.execute(
            "SELECT count(*) AS n FROM matches WHERE home_team_id IN (%s, %s) AND away_team_id IN (%s, %s)",
            (al, bo, al, bo),
        )
        return cur.fetchone()["n"]

    def test_the_fixtures_path_moves_the_row_instead_of_forking(self, cur):
        league, al, bo, row, _ = self._setup(cur)
        later = CARD + timedelta(hours=2)
        assert fu._process_competition(cur, MMA, league, _comp("401927906", later, "Bo", "Al", state="pre"), {})
        assert self._pairs(cur, al, bo) == 1
        got = _get(cur, "matches", row, "status", "match_date", "home_team_id")
        assert got == {"status": "scheduled", "match_date": later, "home_team_id": al}

    def test_the_results_path_writes_the_winner_onto_the_row_in_its_order(self, cur):
        league, al, bo, row, _ = self._setup(cur)
        comp = _comp("401927906", CARD, "Bo", "Al", winner="home")  # Bo, listed first, won
        assert fu._record_result(cur, MMA, league, comp, {})
        assert self._pairs(cur, al, bo) == 1
        assert _get(cur, "matches", row, "status", "home_score", "away_score") == {
            "status": "finished",
            "home_score": 0,
            "away_score": 1,
        }

    @pytest.mark.parametrize("handler", ["_record_result", "_process_competition"])
    def test_a_refused_fork_logs_the_names_in_the_rows_order(self, cur, monkeypatch, handler):
        league, al, bo, row, _ = self._setup(cur)
        seen = []

        def refuse(*a, **k):
            raise fu.ForkRefused(row, {"home_team_id": al, "away_team_id": bo})

        monkeypatch.setattr(fu, "_reconcile_match_row", refuse)
        monkeypatch.setattr(fu, "_log_refused_fork", lambda cur, cfg, espn_id, refusal, *rest: seen.append(rest))
        state = "post" if handler == "_record_result" else "pre"
        assert not getattr(fu, handler)(cur, MMA, league, _comp("401927906", CARD, "Bo", "Al", state=state), {})
        assert seen == [("Al", al, "Bo", bo, CARD)]

    def test_an_existing_fork_is_not_swapped_onto(self, cur, caplog):
        league, al, bo, row, twin = self._setup(cur, fork=True)
        assert fu._record_result(cur, MMA, league, _comp("401927906", CARD, "Bo", "Al", winner="home"), {})
        assert _get(cur, "matches", row, "status")["status"] == "scheduled"
        assert _get(cur, "matches", twin, "status", "home_score")["home_score"] == 1
        assert any("one bout on two rows" in r.message for r in caplog.records)
