"""Close out the pending horse-racing recs the drain will never settle:
settle each from its race's result when The Racing API has one, void it
when the API has none.

WHY THEY EXIST (prod 2026-11-01, read only)
-------------------------------------------
42 race_recommendations were still 'pending' on 30 races that ran 5 days
to 5 months earlier and were still 'scheduled' (no result ever landed).
Asked one by one (/results/{race_id}, the endpoint the drain uses):
  * 10 recs on 8 races HAVE a result. June 4 (3 races) and June 6 (4) fell
    past the first 50 of /results, which fetch_results read one page of
    (those days total 56 and 69 results); Deauville 23 Aug 15:35 (3 recs)
    was posted after the results pass's two-day window had closed.
  * 32 recs on 22 races have NONE: 404 "Result not found", and the day's
    full, paginated results list does not carry them either. They are
    abandoned meetings or races — Newbury 23-24 Oct, Aintree 25 Oct,
    Wexford 26 Oct, Galway 24 and 27 Oct (Galway's 24 Oct card was re-run
    on the 26th under new race ids, and that re-run is in the feed),
    Bellewstown 25 Aug after race 2 and 26 Aug after race 6, and single
    races on 26 Jun, 6 Jul, 20 Aug, 28 Aug, 4 Sep, 29 Sep. The feed drops
    an abandoned race without a status, so nothing ever told the pipeline.
    A bookmaker voids a bet on an abandoned race.
The first run (2026-11-02 03:50Z) settled those 10 and voided those 32.
Seven more recs were still inside the drain's window then: Southwell (AW)
31 Oct 17:40 (4 recs), 18:50 (2) and 19:25 (1), an evening card abandoned
for fog after its third race (the day's paginated results list has
Southwell 16:00, 16:35 and 17:05 only; /results/{race_id} answers 404
"Result not found" for all three). They become candidates one race at a
time once each is DRAIN_DAYS old: 2026-11-03 17:40Z, 18:50Z and 19:25Z.

WHAT THIS DOES
--------------
Candidates: pending/placed race recs whose race is not finished /
cancelled / abandoned, went off more than load_racing_api.DRAIN_DAYS before
the clock — outside the drain's window, so the drain and this tool never act
on the same race — and before --raced-before (the reviewed set; required).

The clock is the database's own NOW(). --as-of pins it to an earlier instant
(the wrapper pins every step of one run to the same T0, so the dry run, the
apply and the re-plan see the same set). It can only move the clock BACK: a
future --as-of is refused (exit 2) before anything is read, because moving
the clock forward would make candidates of races the drain still asks
about. Nothing in this file moves it forward; the tests and the rehearsal
harness do that by replacing db_now().
Per race, GET /results/{racing_api_race_id}:
  200   -> settle. The result is ingested exactly as the results pass does
           (load_racing_api.ingest_race: the race goes 'finished', runners
           get their positions and SPs, listed non-runners are scratched)
           and grade_completed_races.grade_race settles the race's recs and
           grades its predictions. The plan works each verdict out from the
           payload first (plan_race_settlement); the apply checks the grader
           wrote exactly that and rolls everything back otherwise.
  404 "Result not found" -> void: status 'void', profit_loss 0,
           actual_result 'void: no result in The Racing API feed
           (abandoned)', settled_at now. The race row is left as it is.
  other -> error.
'unmatched' (refuses the apply): a race whose result names two winners
(grade_race pays only one), or whose field does not reconcile with the
result — one of our entrants is in neither the runners nor the
non-runners, so a rec on it, or each-way place terms, could be settled
wrong. Any unmatched or error refuses the apply; nothing is written.

The record (all write-once): the CSV (every candidate, dry run included)
and, for an apply, <csv>.reversal.jsonl — the pre-image of every row the
apply can update or that sits where it can insert (ARCHIVE_FROM: the
candidate races' races, race_entrants, race_predictions and
race_recommendations rows; the horses, jockeys, trainers and leagues rows
the settled races' results name), written before the transaction — and
<csv>.inserted.jsonl, the ids of the rows the apply inserted (new entrants
for runners we did not hold, new horses / jockeys / trainers, a new track
row), written inside it just before the commit. Restoring the pre-images
and deleting the inserted ids reverses an apply. One transaction; every
void is guarded on the rec still being pending/placed.

The PLAN line (what scripts/ops/close_out_stranded_race_recs.sh pins):
    PLAN candidates=N races=N settle=N void=N unmatched=N errors=N
and, each on its own line (never touched, listed one per line below it):
    BEYOND_PIN recs=N            stranded recs --raced-before keeps out: they
                                 need a new reviewed pin;
    IN_WINDOW recs=N races=N until=<instant|none>
                                 pending recs on races before --raced-before
                                 that are still inside the drain's window:
                                 the drain asks about them every tick, and a
                                 run after <until> (the last of them leaves
                                 the window then) makes them candidates.

Usage (inside the api container):
    python /app/scripts/close_out_stranded_race_recs.py --raced-before 2026-11-01T00:00:00Z
    python /app/scripts/close_out_stranded_race_recs.py --raced-before 2026-11-01T00:00:00Z \\
        --as-of 2026-11-03T19:30:00Z --apply \\
        --expect-plan candidates=7,races=3,settle=0,void=7,unmatched=0,errors=0
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("close_out_stranded_race_recs")

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load(name: str):
    """A sibling script by path (scripts/ is not a package), reusing an
    already-imported copy so the tests' stubs and this module agree."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lra = _load("load_racing_api")
gcr = _load("grade_completed_races")

PLAN_KEYS = ("candidates", "races", "settle", "void", "unmatched", "errors")
VOID_LABEL = "void: no result in The Racing API feed (abandoned)"
CENT = Decimal("0.01")

CSV_COLUMNS = [
    "rec_id",
    "race_id",
    "racing_api_race_id",
    "race_date",
    "track_name",
    "bet_type",
    "selection",
    "horse",
    "odds",
    "stake",
    "prior_status",
    "action",  # settle | void | unmatched | error
    "new_status",
    "profit_loss",
    "actual_result",
    "winner",
    "position",
    "places_paid",
    "reason",
]

CANDIDATES_SQL = """
    SELECT rec.id::text AS rec_id,
           rec.race_id::text AS race_id,
           rec.entrant_id::text AS entrant_id,
           rec.bet_type,
           rec.selection,
           rec.status,
           rec.odds_at_recommendation AS odds,
           rec.recommended_stake AS stake,
           r.race_date,
           r.track_name,
           r.external_ids->>'racing_api_race_id' AS api_race_id,
           h.normalized_name AS horse
    FROM race_recommendations rec
    JOIN races r ON r.id = rec.race_id
    JOIN race_entrants e ON e.id = rec.entrant_id
    JOIN horses h ON h.id = e.horse_id
    WHERE rec.status IN ('pending', 'placed')
      AND r.status NOT IN ('finished', 'cancelled', 'abandoned')
      AND r.race_date < %(as_of)s - make_interval(days => %(drain_days)s)
      AND r.race_date < %(raced_before)s
    ORDER BY r.race_date, rec.id
"""

# Stranded recs the pin keeps out: what CANDIDATES_SQL would list without
# --raced-before. Reported (BEYOND_PIN), never touched: they need a new
# reviewed --raced-before.
BEYOND_PIN_SQL = """
    SELECT rec.id::text AS rec_id,
           r.race_date,
           r.track_name,
           rec.bet_type,
           h.normalized_name AS horse
    FROM race_recommendations rec
    JOIN races r ON r.id = rec.race_id
    JOIN race_entrants e ON e.id = rec.entrant_id
    JOIN horses h ON h.id = e.horse_id
    WHERE rec.status IN ('pending', 'placed')
      AND r.status NOT IN ('finished', 'cancelled', 'abandoned')
      AND r.race_date < %(as_of)s - make_interval(days => %(drain_days)s)
      AND r.race_date >= %(raced_before)s
    ORDER BY r.race_date, rec.id
"""

# Pending recs in the reviewed set (before --raced-before) that are not
# candidates YET: their race is still inside the drain's window (or not off).
# Reported (IN_WINDOW), never touched; each becomes a candidate once its race
# is DRAIN_DAYS old (candidate_after, worked out in SQL so it uses the same
# interval arithmetic as the window). A run that closes the candidates while
# any of these remain has not finished the reviewed set.
IN_WINDOW_SQL = """
    SELECT rec.id::text AS rec_id,
           rec.race_id::text AS race_id,
           r.race_date,
           r.track_name,
           rec.bet_type,
           h.normalized_name AS horse,
           r.race_date + make_interval(days => %(drain_days)s) AS candidate_after
    FROM race_recommendations rec
    JOIN races r ON r.id = rec.race_id
    JOIN race_entrants e ON e.id = rec.entrant_id
    JOIN horses h ON h.id = e.horse_id
    WHERE rec.status IN ('pending', 'placed')
      AND r.status NOT IN ('finished', 'cancelled', 'abandoned')
      AND r.race_date >= %(as_of)s - make_interval(days => %(drain_days)s)
      AND r.race_date < %(raced_before)s
    ORDER BY r.race_date, rec.id
"""

FIELD_SQL = """
    SELECT e.race_id::text AS race_id, h.normalized_name AS horse, e.scratched
    FROM race_entrants e
    JOIN horses h ON h.id = e.horse_id
    WHERE e.race_id = ANY(%s::uuid[])
"""

VOID_SQL = """
    UPDATE race_recommendations
    SET status = 'void',
        profit_loss = 0,
        actual_result = %s,
        settled_at = NOW(),
        updated_at = NOW()
    WHERE id = %s
      AND status IN ('pending', 'placed')
"""

# The reversal record selects, per table, every row an apply can UPDATE or
# INSERT. On the candidate races: races, race_entrants (the ingest and
# mark_non_runners), race_predictions and race_recommendations (the grader,
# the voids). From the payloads of the races it settles: horses, jockeys and
# trainers by the normalized names its runners carry (ensure_* insert a new
# name or rewrite an existing row's display name, on our card or not) and
# the track's leagues row (ensure_league inserts it — the Deauville result's
# course is 'Deauville (FR)' — or merges its external_ids). The archive
# writes each row's JSON text verbatim (numerics exact); read again inside
# the apply, the same selectors find the rows it inserted.
ARCHIVE_FROM = {
    "races": "FROM races t WHERE t.id = ANY(%(races)s::uuid[])",
    "race_entrants": "FROM race_entrants t WHERE t.race_id = ANY(%(races)s::uuid[])",
    "horses": (
        "FROM horses t WHERE t.normalized_name = ANY(%(horses)s::text[]) "
        "OR t.id IN (SELECT horse_id FROM race_entrants WHERE race_id = ANY(%(races)s::uuid[]))"
    ),
    "jockeys": "FROM jockeys t WHERE t.normalized_name = ANY(%(jockeys)s::text[])",
    "trainers": "FROM trainers t WHERE t.normalized_name = ANY(%(trainers)s::text[])",
    "leagues": (
        "FROM leagues t WHERE t.sport = 'horse_racing' AND (t.name, t.country) IN "
        "(SELECT u.name, u.country FROM unnest(%(league_names)s::text[], %(league_countries)s::text[]) "
        "AS u(name, country))"
    ),
    "race_predictions": "FROM race_predictions t WHERE t.race_id = ANY(%(races)s::uuid[])",
    "race_recommendations": "FROM race_recommendations t WHERE t.race_id = ANY(%(races)s::uuid[])",
}


class ApplyMismatch(RuntimeError):
    """The database did not end up where the plan said; the transaction is rolled back."""


class FutureAsOf(ValueError):
    """--as-of is later than the database's clock: refused before anything is read."""


def db_now(cur) -> datetime:
    """The clock every window is measured from: the database's own NOW() (the
    drain measures its window from the same clock). The tests and the
    rehearsal harness replace this function to move the clock forward;
    nothing else can (run() refuses a future --as-of)."""
    cur.execute("SELECT NOW() AS now")
    return cur.fetchone()["now"]


def plan_line(totals: dict[str, int]) -> str:
    return "PLAN " + " ".join(f"{k}={totals[k]}" for k in PLAN_KEYS)


def parse_expect_plan(text: str) -> dict[str, int]:
    """'candidates=42,races=30,...' -> {key: n}; ValueError on anything else."""
    out: dict[str, int] = {}
    for part in (p.strip() for p in text.split(",")):
        key, sep, value = part.partition("=")
        if not sep or key not in PLAN_KEYS or not value.isdigit() or key in out:
            raise ValueError(f"--expect-plan {text!r}: want KEY=N pairs with KEY one of {', '.join(PLAN_KEYS)}")
        out[key] = int(value)
    return out


def _money(value: Decimal) -> Decimal:
    """numeric(10,2) rounding (half away from zero), so plan and row compare exactly."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def plan_race_settlement(recs: list[dict], payload: dict, field: list[dict]) -> list[dict]:
    """Each rec's verdict once `payload` (one race's result) is ingested,
    reached the way grade_completed_races.settle_recommendations reaches
    it. Pure. `field` is the race's entrants as they are now
    ({horse: normalized name, scratched}). Returns one entry per rec with
    action 'settle' (new_status won/lost/void) or 'unmatched' (reason)."""
    runners, non_runners = lra.result_names(payload)
    winners = [n for n, ru in runners.items() if lra._safe_int(ru.get("position")) == 1]
    ours = {f["horse"]: bool(f["scratched"]) for f in field}
    # The grader's field after the ingest: our entrants plus a new row for any
    # runner we do not hold, minus the non-runners (marked scratched).
    after = dict(ours)
    for name in runners:
        after.setdefault(name, False)
    for name in non_runners:
        if name in after:
            after[name] = True
    places_paid = gcr.ew_places(sum(1 for scratched in after.values() if not scratched))
    unaccounted = lra.unaccounted_entrants(field, payload)
    race_problem = None
    if len(winners) > 1:
        race_problem = f"dead heat ({', '.join(winners)}): grade_race pays one winner"
    elif unaccounted:
        race_problem = f"our entrant(s) {', '.join(unaccounted)} in neither the runners nor the non-runners"
    winner_name = runners[winners[0]].get("horse") if winners else None

    out = []
    for rec in recs:
        entry = {
            "winner": winner_name or "",
            "places_paid": places_paid,
            "position": "",
            "reason": "",
        }
        ru = runners.get(rec["horse"])
        if ru is not None:
            entry["position"] = str(ru.get("position") or "")
        if race_problem:
            out.append({**entry, "action": "unmatched", "reason": race_problem})
            continue
        stake = Decimal(rec["stake"] or 0)
        odds = Decimal(rec["odds"] or 0)
        scratched = after.get(rec["horse"], False)
        if not winners:
            verdict = ("void", Decimal("0"), "void", "result has no winner")
        elif scratched:
            verdict = ("void", Decimal("0"), winner_name or "void", "non-runner")
        elif rec["bet_type"] == "place":
            fp = lra._safe_int(ru.get("position")) if ru is not None else None
            if places_paid >= 2 and fp is not None and 1 <= fp <= places_paid:
                verdict = ("won", stake * (odds - 1), f"finished {fp}", "")
            else:
                verdict = ("lost", -stake, f"finished {fp}" if fp is not None else "unplaced", "")
        elif rec["horse"] == winners[0]:
            verdict = ("won", stake * (odds - 1), winner_name or rec["selection"], "")
        else:
            verdict = ("lost", -stake, winner_name or "", "")
        status, pl, actual, why = verdict
        out.append(
            {
                **entry,
                "action": "settle",
                "new_status": status,
                "profit_loss": _money(pl),
                "actual_result": actual,
                "reason": why,
            }
        )
    return out


def window_params(raced_before: datetime, as_of: datetime) -> dict[str, Any]:
    return {"drain_days": lra.DRAIN_DAYS, "raced_before": raced_before, "as_of": as_of}


def build_plan(
    cur, raced_before: datetime, fetch, as_of: datetime
) -> tuple[list[dict], dict[str, dict], dict[str, int]]:
    """Read-only. Returns (entries, {race_id: payload} for the races to
    settle, PLAN totals). `fetch(api_race_id)` is
    load_racing_api.fetch_race_result with the credentials bound; it is
    called once per candidate race, so no candidate means no API call."""
    cur.execute(CANDIDATES_SQL, window_params(raced_before, as_of))
    candidates = [dict(r) for r in cur.fetchall()]
    by_race: dict[str, list[dict]] = {}
    for rec in candidates:
        by_race.setdefault(rec["race_id"], []).append(rec)
    fields: dict[str, list[dict]] = {}
    if by_race:
        cur.execute(FIELD_SQL, (list(by_race),))
        for row in cur.fetchall():
            fields.setdefault(row["race_id"], []).append(dict(row))

    entries: list[dict] = []
    payloads: dict[str, dict] = {}
    for n, (race_id, recs) in enumerate(by_race.items()):
        api_race_id = recs[0]["api_race_id"]
        verdicts: list[dict]
        if not api_race_id:
            verdicts = [{"action": "error", "reason": "race has no racing_api_race_id"} for _ in recs]
        else:
            if n:
                time.sleep(lra.REQUEST_DELAY_SEC)
            try:
                payload = fetch(api_race_id)
            except lra.RacingApiError as e:
                logger.error("%s: %s", api_race_id, e)
                verdicts = [{"action": "error", "reason": str(e)[:200]} for _ in recs]
            else:
                if payload is None:
                    verdicts = [
                        {
                            "action": "void",
                            "new_status": "void",
                            "profit_loss": Decimal("0.00"),
                            "actual_result": VOID_LABEL,
                            "reason": "404 Result not found",
                        }
                        for _ in recs
                    ]
                else:
                    verdicts = plan_race_settlement(recs, payload, fields.get(race_id, []))
                    if all(v["action"] == "settle" for v in verdicts):
                        payloads[race_id] = payload
        for rec, verdict in zip(recs, verdicts):
            entries.append(
                {
                    "rec_id": rec["rec_id"],
                    "race_id": race_id,
                    "racing_api_race_id": api_race_id or "",
                    "race_date": rec["race_date"].astimezone(timezone.utc).isoformat(),
                    "track_name": rec["track_name"],
                    "bet_type": rec["bet_type"],
                    "selection": rec["selection"],
                    "horse": rec["horse"],
                    "odds": rec["odds"],
                    "stake": rec["stake"],
                    "prior_status": rec["status"],
                    **verdict,
                }
            )
    totals = {
        "candidates": len(entries),
        "races": len(by_race),
        "settle": sum(1 for e in entries if e["action"] == "settle"),
        "void": sum(1 for e in entries if e["action"] == "void"),
        "unmatched": sum(1 for e in entries if e["action"] == "unmatched"),
        "errors": sum(1 for e in entries if e["action"] == "error"),
    }
    return entries, payloads, totals


def _exclusive_open(path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.open("x", newline="", encoding="utf-8")


def write_csv(path: str, entries: list[dict]) -> str:
    """Every candidate with its planned action, BEFORE any write (dry run
    too). Opened 'x': an existing file is an error, never an overwrite —
    with the archive it is the only record of what the apply changed."""
    with _exclusive_open(path) as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: ("" if e.get(k) is None else str(e.get(k))) for k in CSV_COLUMNS})
    logger.info("Wrote %d candidate row(s) to %s", len(entries), path)
    return path


def archive_params(race_ids: list[str], payloads: dict[str, dict]) -> dict[str, list[str]]:
    """ARCHIVE_FROM's parameters: the candidate races, and the names the
    payloads of the races to settle make the ingest upsert."""
    names: dict[str, set[str]] = {"horses": set(), "jockeys": set(), "trainers": set()}
    leagues: set[tuple[str, str]] = set()
    for payload in payloads.values():
        league = lra.league_key(payload)
        if league:
            leagues.add(league)
        for ru in payload.get("runners") or []:
            for table, key in (("horses", "horse"), ("jockeys", "jockey"), ("trainers", "trainer")):
                name = lra.normalize_name(ru.get(key) or "")
                if name:
                    names[table].add(name)
    ordered = sorted(leagues)
    return {
        "races": sorted(race_ids),
        **{table: sorted(v) for table, v in names.items()},
        "league_names": [name for name, _ in ordered],
        "league_countries": [country for _, country in ordered],
    }


def write_archive(cur, path: str, params: dict[str, list[str]]) -> dict[str, set[str]]:
    """The reversal archive: JSON lines {"table", "row"} for every row an
    apply can change (ARCHIVE_FROM), as it is now. Write-once. Returns the
    archived ids per table (what write_inserted leaves out)."""
    ids: dict[str, set[str]] = {}
    n = 0
    with _exclusive_open(path) as fh:
        for table, frm in ARCHIVE_FROM.items():
            cur.execute(f"SELECT t.id::text AS id, row_to_json(t)::text AS row {frm}", params)
            ids[table] = set()
            for r in cur.fetchall():
                fh.write(f'{{"table": {json.dumps(table)}, "row": {r["row"]}}}\n')
                ids[table].add(r["id"])
                n += 1
    logger.info("Wrote %d pre-image row(s) to %s", n, path)
    return ids


def write_inserted(cur, path: str, params: dict[str, list[str]], archived: dict[str, set[str]]) -> int:
    """Inside the apply, after its writes: JSON lines {"table", "id"} for
    every row ARCHIVE_FROM selects now that the archive did not hold and
    that this transaction created (created_at defaults to NOW(), the
    transaction's start, on every one of these tables) — the rows the apply
    inserted, which a reversal deletes (race_entrants first: they
    reference the horses). A row another writer committed since the
    archive (the drain ingesting the same jockey) is logged and left out.
    Write-once."""
    n = 0
    with _exclusive_open(path) as fh:
        for table, frm in ARCHIVE_FROM.items():
            cur.execute(f"SELECT t.id::text AS id, t.created_at = NOW() AS ours {frm}", params)
            for r in cur.fetchall():
                if r["id"] in archived[table]:
                    continue
                if not r["ours"]:
                    logger.warning(
                        "%s %s appeared after the archive but was not inserted by this apply", table, r["id"]
                    )
                    continue
                fh.write(json.dumps({"table": table, "id": r["id"]}) + "\n")
                n += 1
    logger.info("Wrote %d inserted row id(s) to %s", n, path)
    return n


def apply_plan(
    conn,
    entries: list[dict],
    payloads: dict[str, dict],
    record: Optional[tuple[str, dict[str, list[str]], dict[str, set[str]]]] = None,
) -> None:
    """One transaction: ingest + grade each race to settle, void the rest
    (guarded), then read every candidate back and compare with the plan.
    Any difference raises ApplyMismatch after a rollback. `record` is
    (inserted-rows path, archive params, archived ids): the rows the
    apply inserted are written there just before the commit."""
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for race_id, payload in payloads.items():
                got, _ = lra.ingest_race(cur, payload, finished=True)
                if got != race_id:
                    raise ApplyMismatch(f"the result for race {race_id} upserted race {got}")
                gcr.grade_race(cur, race_id)
            for e in entries:
                if e["action"] != "void":
                    continue
                cur.execute(VOID_SQL, (VOID_LABEL, e["rec_id"]))
                if cur.rowcount != 1:
                    raise ApplyMismatch(
                        f"rec {e['rec_id']} was no longer pending at the void (rowcount {cur.rowcount})"
                    )
            cur.execute(
                "SELECT id::text AS rec_id, status, profit_loss, actual_result FROM race_recommendations "
                "WHERE id = ANY(%s::uuid[])",
                ([e["rec_id"] for e in entries],),
            )
            rows = {r["rec_id"]: r for r in cur.fetchall()}
            wrong = []
            for e in entries:
                r = rows.get(e["rec_id"])
                got_pl = None if r is None or r["profit_loss"] is None else _money(Decimal(r["profit_loss"]))
                if r is None or (r["status"], got_pl, r["actual_result"]) != (
                    e["new_status"],
                    e["profit_loss"],
                    e["actual_result"],
                ):
                    wrong.append(f"{e['rec_id']}: planned {e['new_status']} {e['profit_loss']}, got {r and dict(r)}")
            if wrong:
                raise ApplyMismatch(f"{len(wrong)} rec(s) not as planned: " + "; ".join(wrong[:5]))
            if record:
                write_inserted(cur, *record)
        conn.commit()
    except Exception:
        conn.rollback()
        logger.error("Apply failed; transaction rolled back — nothing changed", exc_info=True)
        raise


def default_out_csv() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/closed_out_race_recs_{stamp}.csv"


def summarize(entries: list[dict]) -> None:
    settled = [e for e in entries if e["action"] == "settle"]
    for status in ("won", "lost", "void"):
        rows = [e for e in settled if e["new_status"] == status]
        if rows:
            pl = sum((e["profit_loss"] for e in rows), Decimal("0"))
            logger.info("  settle %-4s %3d rec(s)  P/L %+.2f", status, len(rows), pl)
    for e in entries:
        logger.info(
            "  %s rec %s  %s %s %s/%s -> %s %s %s",
            e["action"],
            e["rec_id"],
            e["race_date"][:16],
            e["track_name"],
            e["bet_type"],
            e["horse"],
            e.get("new_status") or "-",
            e.get("profit_loss") if e.get("profit_loss") is not None else "",
            e.get("reason") or "",
        )


def _utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def run(
    conn,
    raced_before: datetime,
    fetch,
    apply_changes: bool = False,
    out_csv: Optional[str] = None,
    expect_plan: Optional[dict[str, int]] = None,
    as_of: Optional[datetime] = None,
) -> dict[str, Any]:
    out_csv = out_csv or default_out_csv()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        now = db_now(cur)
        if as_of is None:
            as_of = now
        elif as_of > now:
            conn.rollback()
            raise FutureAsOf(
                f"--as-of {_utc(as_of)} is after the database's clock ({_utc(now)}): measured from the future, "
                "the window would make candidates of races the drain still asks about"
            )
        entries, payloads, totals = build_plan(cur, raced_before, fetch, as_of)
        cur.execute(BEYOND_PIN_SQL, window_params(raced_before, as_of))
        beyond = [dict(r) for r in cur.fetchall()]
        cur.execute(IN_WINDOW_SQL, window_params(raced_before, as_of))
        waiting = [dict(r) for r in cur.fetchall()]
    conn.rollback()  # close the read snapshot before any write
    write_csv(out_csv, entries)
    logger.info(
        "AS_OF %s — the clock the drain window is measured from: candidates went off before %s",
        _utc(as_of),
        _utc(as_of - timedelta(days=lra.DRAIN_DAYS)),
    )
    logger.info("%s", plan_line(totals))
    summarize(entries)
    # Its own line, not a PLAN key: the wrapper pins every PLAN key, and these are not part of the reviewed set.
    logger.info(
        "BEYOND_PIN recs=%d — stranded rec(s) on races at or after --raced-before %s: not candidates, never "
        "touched; a new reviewed --raced-before covers them",
        len(beyond),
        raced_before.isoformat(),
    )
    for b in beyond:
        logger.info(
            "  beyond pin: rec %s  %s %s %s/%s",
            b["rec_id"],
            b["race_date"].astimezone(timezone.utc).isoformat()[:16],
            b["track_name"],
            b["bet_type"],
            b["horse"],
        )
    until = _utc(max(w["candidate_after"] for w in waiting)) if waiting else "none"
    # Its own line too: these are in the reviewed set but not candidates yet, so a run that leaves any has not
    # finished it (the wrapper never reports ALREADY DONE while this is above 0).
    logger.info(
        "IN_WINDOW recs=%d races=%d until=%s — pending rec(s) on races before --raced-before still inside the "
        "drain's %d-day window as of %s: the drain asks about them every tick; each becomes a candidate once its "
        "race is %d days old. Not touched",
        len(waiting),
        len({w["race_id"] for w in waiting}),
        until,
        lra.DRAIN_DAYS,
        _utc(as_of),
        lra.DRAIN_DAYS,
    )
    for w in waiting:
        logger.info(
            "  in window: rec %s  %s %s %s/%s — a candidate after %s",
            w["rec_id"],
            _utc(w["race_date"])[:16],
            w["track_name"],
            w["bet_type"],
            w["horse"],
            _utc(w["candidate_after"]),
        )
    mismatch = "; ".join(
        f"{totals[k]} {k}, expected {n}" for k, n in sorted((expect_plan or {}).items()) if totals[k] != n
    )
    refused = ""
    if totals["unmatched"] or totals["errors"]:
        refused = f"{totals['unmatched']} unmatched, {totals['errors']} error(s)"
        logger.error("REFUSED: %s — nothing written; review %s", refused, out_csv)
    if mismatch:
        logger.error("The plan is not the one pinned (%s): nothing written; review %s", mismatch, out_csv)
    applied = False
    if apply_changes and not mismatch and not refused and entries:
        params = archive_params(sorted({e["race_id"] for e in entries}), payloads)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            archived = write_archive(cur, str(Path(out_csv).with_suffix(".reversal.jsonl")), params)
        conn.rollback()
        apply_plan(conn, entries, payloads, (str(Path(out_csv).with_suffix(".inserted.jsonl")), params, archived))
        applied = True
        logger.info("Applied: %d settled, %d voided (one transaction)", totals["settle"], totals["void"])
    elif not apply_changes:
        logger.info("DRY-RUN: %d rec(s) would be written. Re-run with --apply.", totals["settle"] + totals["void"])
    return {
        "plan": totals,
        "beyond_pin": len(beyond),
        "in_window": len(waiting),
        "in_window_until": max(w["candidate_after"] for w in waiting) if waiting else None,
        "as_of": as_of,
        "plan_mismatch": bool(mismatch),
        "refused": bool(refused),
        "applied": applied,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--raced-before",
        required=True,
        help="Only races off before this UTC instant (ISO 8601) are candidates: the reviewed set.",
    )
    p.add_argument(
        "--as-of",
        default=None,
        help="Measure the drain window from this UTC instant (ISO 8601 with a zone) instead of the database's "
        "NOW(); refused when it is later than the database's clock. The wrapper pins one run's steps to one T0.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument(
        "--out-csv",
        default=None,
        help="Candidate CSV (default /app/backups/closed_out_race_recs_<utc>.csv). Write-once.",
    )
    p.add_argument(
        "--expect-plan",
        default=None,
        help=f"Write nothing unless the plan has exactly these counts (KEY=N,... with KEY one of "
        f"{', '.join(PLAN_KEYS)}); --apply needs it, with every key.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only (DEFAULT).")
    mode.add_argument("--apply", action="store_true", help="Settle / void in one transaction.")
    args = p.parse_args(argv)
    try:
        args.raced_before = datetime.fromisoformat(args.raced_before.replace("Z", "+00:00"))
    except ValueError:
        p.error(f"--raced-before {args.raced_before!r} is not an ISO 8601 instant")
    if args.raced_before.tzinfo is None:
        p.error("--raced-before needs a timezone (e.g. 2026-11-01T00:00:00Z)")
    if args.as_of is not None:
        try:
            args.as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
        except ValueError:
            p.error(f"--as-of {args.as_of!r} is not an ISO 8601 instant")
        if args.as_of.tzinfo is None:
            p.error("--as-of needs a timezone (e.g. 2026-11-03T19:30:00Z)")
    if args.expect_plan is not None:
        try:
            args.expect_plan = parse_expect_plan(args.expect_plan)
        except ValueError as e:
            p.error(str(e))
    if args.apply:
        missing = [k for k in PLAN_KEYS if k not in (args.expect_plan or {})]
        if missing:
            p.error(f"--apply needs --expect-plan with every key; missing: {', '.join(missing)}")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set (pass --database-url or export DATABASE_URL)")
        return 2
    auth = lra._auth()

    def fetch(api_race_id: str):
        if auth is None:
            raise lra.RacingApiError("THE_RACING_API_USER / THE_RACING_API_PASS not set")
        return lra.fetch_race_result(api_race_id, auth)

    conn = psycopg2.connect(args.database_url)
    try:
        result = run(
            conn,
            raced_before=args.raced_before,
            fetch=fetch,
            apply_changes=args.apply,
            out_csv=args.out_csv,
            expect_plan=args.expect_plan,
            as_of=args.as_of,
        )
    except FutureAsOf as exc:
        logger.error("REFUSED: %s — nothing read or written", exc)
        return 2
    except ApplyMismatch as exc:
        logger.error("%s", exc)
        return 1
    finally:
        conn.close()
    return 1 if (result["plan_mismatch"] or result["refused"]) else 0


if __name__ == "__main__":
    sys.exit(main())
