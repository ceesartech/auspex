"""scripts/merge_mma_fighters.py against a real Postgres (a throwaway cluster
built from services/data-ingestion/db/migrations, one database per test):

  * a dry run writes nothing and a pinned apply commits the whole plan in one
    transaction: the stale copy folded into its finished row (its freshest
    pre-game prices, snapshots and pre-game predictions moved, the ESPN
    competition id adopted), the history re-pointed, the merged spelling an
    alias of the kept row with merged_from and the ESPN athlete id, the merged
    team deleted, a reversed listed copy folded with home/away swapped, a
    decomposed name composed; --check-after holds; a re-run plans 'done';
  * the deployed resolver then resolves every spelling to the kept row:
    fetch_upcoming.ensure_team mints no team for 'Zachary Reese' nor for the
    decomposed 'Rose Conceição' ESPN still sends;
  * an apply whose plan changed since the dry run refuses with nothing written;
    so does one whose (expected) plan refuses a pair; one whose
    in-transaction after-state fails rolls everything back — a resolver that
    misses every name, and one that misses only the merged spelling (the
    per-pair check alone: the soccer lesson); a copy a rec holds is left in
    place, its rec untouched;
  * --check-after counts a team re-minted under a merged spelling and a stale
    row of a folded bout left beside its finished row;
  * the SQL guards: a rec that arrives after the plan on a prediction the fold
    moves, or on the copy itself, stops the fold (GuardError, not a move or a
    foreign-key error);
  * the archive holds the stale copy and every child row before the first write.

Skipped where no Postgres server binaries are on PATH (initdb, pg_ctl).
"""

from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
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
mm = _load("merge_mma_fighters_pg", "merge_mma_fighters.py")

MMA = fu.SPORT_CONFIGS["mma"]
NOW = datetime.now(timezone.utc).replace(second=0, microsecond=0)
CARD = NOW - timedelta(days=20)
NFD_NAME = "Rose Conceição"
NFC_NAME = "Rose Conceição"


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg")
    data = base / "data"
    sock = Path(tempfile.mkdtemp(prefix="pgm", dir="/tmp"))  # a Unix socket path is limited to ~100 bytes
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


def _cur(conn):
    return conn.cursor(cursor_factory=RealDictCursor)


def _fighter(cur, name: str, normalized: str | None = None) -> str:
    cur.execute(
        "INSERT INTO teams (name, normalized_name, sport) VALUES (%s, %s, 'mma') RETURNING id::text AS id",
        (name, normalized or fu.normalize_team(name)),
    )
    return cur.fetchone()["id"]


def _row(cur, league, home, away, kickoff, status="scheduled", espn=None, score=None):
    cur.execute(
        """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, season, external_ids,
                             home_score, away_score)
        VALUES (%s, %s, %s, %s, %s, '2026', %s::jsonb, %s, %s) RETURNING id::text AS id
        """,
        (league, home, away, kickoff, status, json.dumps({"espn": espn} if espn else {}), *(score or (None, None))),
    )
    return cur.fetchone()["id"]


def _odds(cur, match_id, sel, ts, book="FanDuel", price="2.50"):
    cur.execute(
        """INSERT INTO odds (match_id, bookmaker, market_type, selection, odds_decimal, "timestamp", is_live)
           VALUES (%s, %s, 'moneyline', %s, %s, %s, false) RETURNING id::text AS id""",
        (match_id, book, sel, price, ts),
    )
    return cur.fetchone()["id"]


def _snap(cur, match_id, sel, ts, book="FanDuel"):
    cur.execute(
        """INSERT INTO odds_snapshots (match_id, bookmaker, market_type, selection, odds_decimal, captured_at)
           VALUES (%s, %s, 'moneyline', %s, 2.5, %s) RETURNING id::text AS id""",
        (match_id, book, sel, ts),
    )
    return cur.fetchone()["id"]


def _pred(cur, match_id, version, written):
    cur.execute(  # updated_at given on INSERT: trg_predictions_updated_at stamps NOW() on an UPDATE only
        """INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome,
                                    confidence, probabilities, created_at, updated_at)
           VALUES (%s, 'ensemble_mma_ml', %s, 'moneyline', 'home', 0.6, '{"home": 0.6, "away": 0.4}'::jsonb, %s, %s)
           RETURNING id::text AS id""",
        (match_id, version, written, written),
    )
    return cur.fetchone()["id"]


def _rec(cur, match_id, prediction_id=None):
    pid = prediction_id or _pred(cur, match_id, "rec-v", CARD - timedelta(days=2))
    cur.execute(
        """INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection, odds_at_recommendation,
                                                bookmaker, recommended_stake, status, created_at)
           VALUES (%s, %s, 'moneyline', 'home', 2.5, 'b', 10, 'won', %s) RETURNING id::text AS id""",
        (pid, match_id, CARD - timedelta(days=2)),
    )
    return cur.fetchone()["id"]


def _world(conn, nfd: bool = True) -> dict:
    """The merged fighter's copy beside the finished row, his history, a
    reversed duplicate, and (nfd) a decomposed name."""
    with _cur(conn) as cur:
        league = str(fu.ensure_league(cur, MMA, "UFC", "UFC", "USA"))
        w = {"league": league}
        for key, name in (("kept", "Zach Reese"), ("merged", "Zachary Reese"), ("opp", "Ryan Gandra")):
            w[key] = _fighter(cur, name)
        for key, name in (("old", "Eli Aronov"), ("mcghee", "Marcus McGhee"), ("romero", "Anthony Romero")):
            w[key] = _fighter(cur, name)
        if nfd:
            w["rose"] = _fighter(cur, NFD_NAME, fu.normalize_team(NFD_NAME).replace("çã", "çã"))
        w["fin"] = _row(cur, league, w["kept"], w["opp"], CARD, "finished", score=(0, 1))
        w["stale"] = _row(cur, league, w["merged"], w["opp"], CARD, espn="401874061")
        w["hist"] = _row(cur, league, w["merged"], w["old"], CARD - timedelta(days=400), "finished", score=(1, 0))
        t = CARD - timedelta(hours=6)
        w["odds_old"] = _odds(cur, w["stale"], "home", t, book="DraftKings")
        w["odds_new"] = _odds(cur, w["stale"], "home", t + timedelta(hours=1), price="2.60")
        w["odds_live"] = _odds(cur, w["stale"], "away", CARD + timedelta(minutes=10))  # after kickoff
        w["snap"] = _snap(cur, w["stale"], "home", t)
        w["pred"] = _pred(cur, w["stale"], "v1", t)
        w["pred_late"] = _pred(cur, w["stale"], "v2", CARD + timedelta(hours=1))
        when = CARD + timedelta(days=1)
        w["rev_fin"] = _row(cur, league, w["romero"], w["mcghee"], when, "finished", score=(0, 1))
        w["rev_stale"] = _row(cur, league, w["mcghee"], w["romero"], when, espn="401927906")
        w["rev_odds"] = _odds(cur, w["rev_stale"], "home", when - timedelta(hours=2), price="1.18")
        w["rev_odds2"] = _odds(cur, w["rev_stale"], "away", when - timedelta(hours=2), price="5.00")
        w["rev_snap"] = _snap(cur, w["rev_stale"], "away", when - timedelta(hours=3))
        w["rev_pred"] = _pred(cur, w["rev_stale"], "v1", when - timedelta(hours=4))
    conn.commit()
    w["pairs"] = [mm.MergePair(w["merged"], "Zachary Reese", w["kept"], "Zach Reese", "5143223", "m", "e")]
    w["folds"] = [
        mm.ListedFold(w["rev_stale"], w["rev_fin"], when, "Marcus McGhee", "Anthony Romero", "401927906", "e")
    ]
    return w


def _run(conn, w, tmp_path, apply=False, expect=None, resolvers=None, tag="x"):
    res, why = mm.load_resolvers()
    assert why is None, why
    plan_out = tmp_path / f"{tag}.plan.json"
    rc = mm.run(
        conn,
        apply,
        w["pairs"],
        w["folds"],
        tmp_path / f"{tag}.csv",
        tmp_path / f"{tag}.jsonl.gz",
        resolvers or res,
        plan_out=plan_out,
        expect_plan=expect,
    )
    return rc, json.loads(plan_out.read_text())


def _one(conn, sql, *args):
    with _cur(conn) as cur:
        cur.execute(sql, args)
        row = cur.fetchone()
    conn.rollback()
    return row


def test_a_pinned_apply_merges_folds_and_resolves(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0, dry
    s = dry["summary"]
    assert (s["pairs_merge"], s["folds"], s["folds_reversed"], s["repoints"], s["nfc_renamed"]) == (1, 2, 1, 1, 1)
    assert (s["odds_moved"], s["snapshots_moved"], s["predictions_moved"], s["espn_ids_adopted"]) == (4, 2, 1, 2)
    assert (
        _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["merged"])["n"] == 1
    )  # the dry run wrote nothing

    rc, applied = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0 and applied["digest"] == dry["digest"]
    assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["merged"])["n"] == 0
    kept = _one(conn, "SELECT external_ids, metadata FROM teams WHERE id = %s", w["kept"])
    assert kept["external_ids"] == {"espn": "5143223"} and kept["metadata"]["aliases"] == ["Zachary Reese"]
    (entry,) = kept["metadata"]["merged_from"]
    assert entry["team_id"] == w["merged"] and entry["espn_athlete_id"] == "5143223"
    assert _one(conn, "SELECT count(*) AS n FROM matches WHERE id IN (%s, %s)", w["stale"], w["rev_stale"])["n"] == 0
    fin = _one(conn, "SELECT external_ids FROM matches WHERE id = %s", w["fin"])
    assert fin["external_ids"] == {"espn": "401874061"}
    assert _one(conn, "SELECT home_team_id::text AS h FROM matches WHERE id = %s", w["hist"])["h"] == w["kept"]
    moved = _one(conn, "SELECT match_id::text AS m, odds_decimal FROM odds WHERE id = %s", w["odds_new"])
    assert moved["m"] == w["fin"] and str(moved["odds_decimal"]) == "2.6000"
    assert _one(conn, "SELECT match_id::text AS m FROM odds WHERE id = %s", w["odds_old"])["m"] == w["fin"]
    assert _one(conn, "SELECT count(*) AS n FROM odds WHERE id = %s", w["odds_live"])["n"] == 0  # in play: dropped
    assert _one(conn, "SELECT match_id::text AS m FROM predictions WHERE id = %s", w["pred"])["m"] == w["fin"]
    assert _one(conn, "SELECT count(*) AS n FROM predictions WHERE id = %s", w["pred_late"])["n"] == 0
    # the reversed copy: the 1.18 price of McGhee (home there) is the away price on Romero v McGhee
    rev = _one(conn, "SELECT match_id::text AS m, selection FROM odds WHERE id = %s", w["rev_odds"])
    assert rev == {"m": w["rev_fin"], "selection": "away"}
    assert _one(conn, "SELECT selection FROM odds WHERE id = %s", w["rev_odds2"])["selection"] == "home"
    assert _one(conn, "SELECT selection FROM odds_snapshots WHERE id = %s", w["rev_snap"])["selection"] == "home"
    assert _one(conn, "SELECT count(*) AS n FROM predictions WHERE id = %s", w["rev_pred"])["n"] == 0
    assert _one(conn, "SELECT external_ids FROM matches WHERE id = %s", w["rev_fin"])["external_ids"] == {
        "espn": "401927906"
    }
    rose = _one(conn, "SELECT name, normalized_name FROM teams WHERE id = %s", w["rose"])
    assert rose == {"name": NFC_NAME, "normalized_name": "rose conceição"}

    res, _ = mm.load_resolvers()
    with _cur(conn) as cur:
        out = mm.check_after(cur, dry, res)
    conn.rollback()
    assert mm.after_state_ok(out), out
    rc, again = _run(conn, w, tmp_path, tag="again")
    assert rc == 0 and again["summary"]["pairs_done"] == 1 and again["summary"]["folds_done"] == 1
    assert again["summary"]["folds"] == 0 and again["summary"]["nfc_renamed"] == 0

    # the next sweep: every spelling resolves, nothing is minted
    with _cur(conn) as cur:
        cur.execute("SELECT count(*) AS n FROM teams")
        before = cur.fetchone()["n"]
        assert fu.ensure_team(cur, MMA, "Zachary Reese", w["league"]) == w["kept"]
        assert fu.ensure_team(cur, MMA, "Zach Reese", w["league"]) == w["kept"]
        assert fu.ensure_team(cur, MMA, NFD_NAME, w["league"]) == w["rose"]
        assert fu.ensure_team(cur, MMA, NFC_NAME, w["league"]) == w["rose"]
        cur.execute("SELECT count(*) AS n FROM teams")
        assert cur.fetchone()["n"] == before
        # a new fighter ESPN sends decomposed is stored composed, name and normalized_name
        new = fu.ensure_team(cur, MMA, "Zoë Dupré", w["league"])
        cur.execute("SELECT name, normalized_name FROM teams WHERE id = %s", (new,))
        assert dict(cur.fetchone()) == {"name": "Zoë Dupré", "normalized_name": "zoë dupré"}
    conn.rollback()


def test_an_apply_whose_plan_changed_refuses_with_nothing_written(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0
    with _cur(conn) as cur:  # a fresher pre-game price lands on the copy after the dry run
        _odds(cur, w["stale"], "away", CARD - timedelta(hours=1), book="Bovada")
    conn.commit()
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 2
    assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["merged"])["n"] == 1
    assert _one(conn, "SELECT count(*) AS n FROM matches WHERE id = %s", w["stale"])["n"] == 1
    assert not (tmp_path / "apply.jsonl.gz").exists()  # refused before the archive


def test_an_apply_without_an_expected_plan_refuses(conn, tmp_path):
    w = _world(conn)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=None, tag="apply")
    assert rc == 2
    assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["merged"])["n"] == 1


def test_a_resolver_miss_inside_the_transaction_rolls_everything_back(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    res, _ = mm.load_resolvers()
    blind = dataclasses.replace(res, find_team=lambda cur, name, league: None)
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, resolvers=blind, tag="apply")
    assert rc == 1
    assert (tmp_path / "apply.jsonl.gz").exists()  # the archive landed, the writes did not
    assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["merged"])["n"] == 1
    assert _one(conn, "SELECT count(*) AS n FROM matches WHERE id = %s", w["stale"])["n"] == 1
    assert _one(conn, "SELECT name FROM teams WHERE id = %s", w["rose"])["name"] == NFD_NAME
    assert _one(conn, "SELECT match_id::text AS m FROM odds WHERE id = %s", w["odds_new"])["m"] == w["stale"]


def test_a_resolver_miss_on_the_merged_spelling_alone_rolls_everything_back(conn, tmp_path, caplog):
    # no NFC rename in this world: only the per-pair check (every merged and kept spelling -> the kept row) can trip
    w = _world(conn, nfd=False)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0 and dry["summary"]["nfc_renamed"] == 0
    res, _ = mm.load_resolvers()
    misses_old_name = dataclasses.replace(
        res,
        find_team=lambda cur, name, league: (
            None if fu.normalize_team(name) == "zachary reese" else res.find_team(cur, name, league)
        ),
    )
    caplog.set_level(logging.INFO, logger="merge_mma_fighters")
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, resolvers=misses_old_name, tag="apply")
    assert rc == 1
    (after,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("AFTER (in the transaction)")]
    assert " resolver_misses=1 " in after and " alias_misses=0 " in after, after
    assert _one(conn, "SELECT count(*) AS n FROM teams WHERE id = %s", w["merged"])["n"] == 1
    assert _one(conn, "SELECT count(*) AS n FROM matches WHERE id = %s", w["stale"])["n"] == 1
    assert _one(conn, "SELECT home_team_id::text AS h FROM matches WHERE id = %s", w["hist"])["h"] == w["merged"]


def test_an_expected_plan_that_refuses_a_pair_is_never_applied(conn, tmp_path):
    w = _world(conn)
    with _cur(conn) as cur:  # the kept row holds another athlete id than the map: the pair refuses
        cur.execute("""UPDATE teams SET external_ids = '{"espn": "999"}'::jsonb WHERE id = %s""", (w["kept"],))
    conn.commit()
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 1 and dry["summary"]["pairs_refuse"] == 1 and dry["summary"]["folds"] == 1  # the listed fold
    rc, applied = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 2 and applied["digest"] == dry["digest"]  # the same plan: refused for its refusal, not a drift
    assert not (tmp_path / "apply.jsonl.gz").exists()
    assert _one(conn, "SELECT count(*) AS n FROM matches WHERE id = %s", w["rev_stale"])["n"] == 1
    assert _one(conn, "SELECT name FROM teams WHERE id = %s", w["rose"])["name"] == NFD_NAME


def test_check_after_counts_a_reminted_name_and_a_stale_row_left_beside_a_folded_bout(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0
    with _cur(conn) as cur:
        _fighter(cur, "Zachary Reese")  # the next sweep minted the merged spelling again
        _row(cur, w["league"], w["opp"], w["kept"], CARD + timedelta(hours=2))  # a stale row of the folded bout
    conn.commit()
    res, _ = mm.load_resolvers()
    with _cur(conn) as cur:
        out = mm.check_after(cur, dry, res)
    conn.rollback()
    assert out["teams_reminted"] == 1 and out["stale_twins_left"] == 1, out
    assert not mm.after_state_ok(out) and not mm.after_state_ok(out, after_later_steps=True)


def _plan_now(conn, w):
    res, _ = mm.load_resolvers()
    with _cur(conn) as cur:
        schema, unknown = mm.load_schema(cur)
        assert schema is not None, unknown
        plan = mm.plan_run(w["pairs"], w["folds"], mm.load_state(cur, w["pairs"], w["folds"], schema), NOW, res)
    conn.rollback()
    (fold,) = [f for f in plan.folds() if f.stale_id == w["stale"]]
    assert fold.predictions == [w["pred"]]
    return plan, schema, fold


def test_the_sql_guards_stop_a_fold_when_a_rec_arrives_after_the_plan(conn, tmp_path):
    w = _world(conn)
    plan, schema, fold = _plan_now(conn, w)
    # a rec on the prediction the fold moves, filed on the finished row (as repair_stranded_recs files them)
    with _cur(conn) as cur:
        _rec(cur, w["fin"], prediction_id=w["pred"])
        _, held = mm.write_archive(cur, tmp_path / "a.jsonl.gz", plan, schema)
        with pytest.raises(mm.GuardError, match="move predictions"):
            mm.apply_fold(cur, fold, schema, held, Counter())
    conn.rollback()
    # a rec filed on the copy itself, its prediction on the finished row: the guarded delete changes nothing
    with _cur(conn) as cur:
        _rec(cur, w["stale"], prediction_id=_pred(cur, w["fin"], "rec-v", CARD - timedelta(days=2)))
        _, held = mm.write_archive(cur, tmp_path / "b.jsonl.gz", plan, schema)
        with pytest.raises(mm.GuardError, match="delete stale copy"):
            mm.apply_fold(cur, fold, schema, held, Counter())
    conn.rollback()
    assert _one(conn, "SELECT count(*) AS n FROM matches WHERE id = %s", w["stale"])["n"] == 1


def test_a_copy_a_rec_holds_is_left_in_place_with_its_rec(conn, tmp_path):
    w = _world(conn)
    with _cur(conn) as cur:
        cur.execute("UPDATE matches SET match_date = match_date - interval '1 hour' WHERE id = %s", (w["stale"],))
        rec = _rec(cur, w["stale"])
    conn.commit()
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    assert rc == 0 and dry["summary"]["left_rec_held"] == 1 and dry["summary"]["folds"] == 1  # only the listed one
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0
    row = _one(conn, "SELECT home_team_id::text AS h FROM matches WHERE id = %s", w["stale"])
    assert row["h"] == w["kept"]  # re-pointed, not folded
    r = _one(conn, "SELECT match_id::text AS m, status FROM betting_recommendations WHERE id = %s", rec)
    assert r == {"m": w["stale"], "status": "won"}


def test_the_archive_holds_the_copy_and_every_child_before_the_first_write(conn, tmp_path):
    w = _world(conn)
    rc, dry = _run(conn, w, tmp_path, tag="dry")
    rc, _ = _run(conn, w, tmp_path, apply=True, expect=dry, tag="apply")
    assert rc == 0
    recs = [json.loads(line) for line in gzip.open(tmp_path / "apply.jsonl.gz", "rt")]
    by_role = {}
    for r in recs:
        by_role.setdefault((r["table"], r["role"]), set()).add(r["row"]["id"])
    assert by_role[("matches", "stale_copy")] == {w["stale"], w["rev_stale"]}
    assert {w["odds_old"], w["odds_new"], w["odds_live"], w["rev_odds"]} <= by_role[("odds", "stale_child")]
    assert {w["pred"], w["pred_late"], w["rev_pred"]} <= by_role[("predictions", "stale_child")]
    assert {w["merged"], w["kept"], w["rose"]} <= by_role[("teams", "team_before")]
    assert w["hist"] in by_role[("matches", "repointed_before")]
    assert {w["fin"], w["rev_fin"]} == by_role[("matches", "finished_before")]
