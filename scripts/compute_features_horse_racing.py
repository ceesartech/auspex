"""Compute baseline horse racing features and write them to race_features_cache.

Horse racing features split into two layers:

  * Race-level features (one set per race):
      - field_size, distance_meters, surface_<one_hot>,
        track_condition_<one_hot>, race_class_tier
      - venue weather (when available via match_weather equivalent
        on race_id — v2 follow-up)

  * Per-entrant features (one set per horse-in-race):
      - morning line implied probability
      - rolling form: last 5 races' avg finish position, win rate,
        place rate
      - course form: this horse's record at this specific track
      - surface form: this horse's record on this surface
      - distance form: avg finish at similar distances (±10%)
      - days since last race (rest)
      - jockey 30-day win rate (across all rides)
      - trainer 30-day win rate (across all runners)
      - weight carried, post position (draw)

The training pipeline (Phase 14 H4) consumes the per-entrant
features with race-level features broadcast. Each entrant becomes
one training row; the label is 1 if the horse won, 0 otherwise.
A softmax normalisation across the race converts the per-entrant
scores back into a proper distribution at predict time.

Storage shape: one race_features_cache row per (race, feature_set,
version) with JSONB:
    {
      "race_level": {...},
      "entrants": [{"entrant_id": ..., "features": {...}}, ...]
    }

Avoids N rows per race when N=14 entrants while keeping the
training query simple (flatten on read).

Usage:
    python /app/scripts/compute_features_horse_racing.py
    python /app/scripts/compute_features_horse_racing.py --days 14
    python /app/scripts/compute_features_horse_racing.py --all-finished
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("compute_features_horse_racing")

FEATURE_SET = "horse_racing_baseline"
FEATURE_VERSION = "v1"

# Rolling windows for entity-level form features.
HORSE_WINDOW = 5  # Last 5 starts for a horse
JOCKEY_DAYS = 30  # 30-day jockey win rate
TRAINER_DAYS = 30  # 30-day trainer win rate

CACHE_TTL_SECONDS = 3600

# Surface buckets — collapse vendor-specific labels into 4 canonical
# values so the one-hot stays tractable.
SURFACE_TURF = "turf"
SURFACE_DIRT = "dirt"
SURFACE_SYNTHETIC = "synthetic"
SURFACE_UNKNOWN = "unknown"


# Modal racecard defaults. Used when a feature can't be computed
# (typically because the horse is making its racing debut so there's
# no rolling form yet). Mid-pack values keep the model from over-
# weighting on missing data.
NEUTRAL_DEFAULTS: dict[str, float] = {
    # Race-level
    "field_size": 10.0,
    "distance_meters": 1600.0,  # ~8 furlongs, common mile race
    "surface_turf": 0.0,
    "surface_dirt": 0.0,
    "surface_synthetic": 0.0,
    "surface_unknown": 1.0,
    # Going / track condition collapsed to a numeric severity:
    # 0 = firm/fast, 1 = good, 2 = soft/sloppy, 3 = heavy/muddy.
    # NULL → 1.0 (good) as the modal racing condition.
    "going_severity": 1.0,
    # Race class tier: 1 = Group/Grade 1, 2 = Group/Grade 2, 3 = G3,
    # 4 = Listed, 5 = Handicap, 6 = Maiden / Claiming. Higher class =
    # more elite field, more predictable favorites.
    "race_class_tier": 5.0,
    # Per-entrant — morning-line implied probability. 1/field_size
    # = uniform prior; 0.10 is the modal at field_size=10.
    "morning_line_implied_prob": 0.10,
    # Rolling form (last 5 starts). Defaults to a mid-pack horse.
    "horse_starts_last_5": 0.0,  # 0 = no prior starts (debut)
    "horse_avg_finish_last_5": 5.0,  # mid-pack
    "horse_win_rate_last_5": 0.0,
    "horse_place_rate_last_5": 0.0,  # top-3 finishes
    # Course-specific form at this track.
    "horse_course_starts": 0.0,
    "horse_course_wins": 0.0,
    # Surface-specific form on the current surface.
    "horse_surface_starts": 0.0,
    "horse_surface_wins": 0.0,
    # Distance-specific form (within ±10% of today's distance).
    "horse_distance_starts": 0.0,
    "horse_distance_wins": 0.0,
    # Days since last race. NULL → 30 (modal mid-tier campaign).
    "days_since_last_race": 30.0,
    # Jockey + trainer 30-day strike rates. NULL → 0.10 (modal).
    "jockey_30d_win_rate": 0.10,
    "trainer_30d_win_rate": 0.10,
    # Schedule context.
    "weight_carried_lbs": 126.0,  # standard scale weight
    "post_position": 6.0,  # middle of a 10-horse field
}


def _with_defaults(features: dict) -> dict:
    """Replace any None / non-numeric value with the neutral default."""
    out: dict = {}
    for k, default in NEUTRAL_DEFAULTS.items():
        v = features.get(k)
        out[k] = v if isinstance(v, (int, float)) and v is not None else default
    for k, v in features.items():
        if k not in out:
            out[k] = v
    return out


# ── Race selectors ────────────────────────────────────────────────


def list_target_races(conn, days: int, force: bool, all_finished: bool, race_ids: Optional[list[str]]) -> list[str]:
    """Races needing feature computation. Same selection logic as the
    other compute_features scripts."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if race_ids:
            cur.execute("SELECT id::text AS id FROM races WHERE id::text = ANY(%s)", (race_ids,))
            return [r["id"] for r in cur.fetchall()]

        if all_finished:
            cur.execute(
                """
                SELECT id::text AS id FROM races
                WHERE status = 'finished'
                ORDER BY race_date ASC
                """
            )
            return [r["id"] for r in cur.fetchall()]

        if force:
            cur.execute(
                """
                SELECT id::text AS id FROM races
                WHERE status = 'scheduled'
                  AND race_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                ORDER BY race_date ASC
                """,
                (str(days),),
            )
            return [r["id"] for r in cur.fetchall()]

        cur.execute(
            """
            SELECT r.id::text AS id FROM races r
            LEFT JOIN race_features_cache fc
              ON fc.race_id = r.id
             AND fc.feature_set = %s
             AND fc.feature_version = %s
             AND fc.expires_at > NOW()
            WHERE r.status = 'scheduled'
              AND r.race_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
              AND fc.id IS NULL
            ORDER BY r.race_date ASC
            """,
            (FEATURE_SET, FEATURE_VERSION, str(days)),
        )
        return [r["id"] for r in cur.fetchall()]


def fetch_race_meta(cur, race_id: str) -> Optional[dict]:
    cur.execute(
        """
        SELECT id::text AS id, race_date, track_name, distance_meters,
               surface, track_condition, race_class, field_size
        FROM races WHERE id = %s
        """,
        (race_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def fetch_entrants(cur, race_id: str) -> list[dict]:
    cur.execute(
        """
        SELECT e.id::text AS id, e.horse_id::text AS horse_id,
               e.jockey_id::text AS jockey_id, e.trainer_id::text AS trainer_id,
               e.program_number, e.post_position, e.weight_carried_lbs,
               e.morning_line_odds, e.scratched
        FROM race_entrants e
        WHERE e.race_id = %s AND NOT e.scratched
        ORDER BY e.program_number ASC
        """,
        (race_id,),
    )
    return [dict(r) for r in cur.fetchall()]


# ── Race-level features ───────────────────────────────────────────


def race_level_features(meta: dict) -> dict:
    """One-hot surface + going-severity + class-tier + field size."""
    out: dict[str, float] = {}

    out["field_size"] = float(meta.get("field_size") or 0) or NEUTRAL_DEFAULTS["field_size"]
    if meta.get("distance_meters"):
        out["distance_meters"] = float(meta["distance_meters"])

    surface_bucket = _bucket_surface(meta.get("surface"))
    out["surface_turf"] = 1.0 if surface_bucket == SURFACE_TURF else 0.0
    out["surface_dirt"] = 1.0 if surface_bucket == SURFACE_DIRT else 0.0
    out["surface_synthetic"] = 1.0 if surface_bucket == SURFACE_SYNTHETIC else 0.0
    out["surface_unknown"] = 1.0 if surface_bucket == SURFACE_UNKNOWN else 0.0

    severity = _going_severity(meta.get("track_condition"))
    if severity is not None:
        out["going_severity"] = severity

    tier = _race_class_tier(meta.get("race_class"))
    if tier is not None:
        out["race_class_tier"] = tier

    return out


def _bucket_surface(raw: Optional[str]) -> str:
    if not raw:
        return SURFACE_UNKNOWN
    s = raw.strip().lower()
    if "turf" in s or "grass" in s:
        return SURFACE_TURF
    if "dirt" in s:
        return SURFACE_DIRT
    if "synth" in s or "polytrack" in s or "tapeta" in s or "all_weather" in s or "all weather" in s:
        return SURFACE_SYNTHETIC
    return SURFACE_UNKNOWN


_GOING_SEVERITY_MAP = {
    # Dirt / synthetic
    "fast": 0,
    "wet_fast": 0,
    "wet fast": 0,
    "standard": 1,
    "sloppy": 2,
    "muddy": 3,
    "slow": 2,
    "sealed": 1,
    # Turf
    "firm": 0,
    "good to firm": 0,
    "good": 1,
    "good to soft": 2,
    "soft": 2,
    "heavy": 3,
    "yielding": 2,
    "good to yielding": 1,
    "yielding to soft": 2,
}


def _going_severity(raw: Optional[str]) -> Optional[float]:
    """Collapse vendor going labels into a 0-3 severity scale."""
    if not raw:
        return None
    key = " ".join(raw.strip().lower().split())
    if key in _GOING_SEVERITY_MAP:
        return float(_GOING_SEVERITY_MAP[key])
    return None


_RACE_CLASS_TIER_MAP = {
    # North America
    "g1": 1,
    "grade 1": 1,
    "grade i": 1,
    "g2": 2,
    "grade 2": 2,
    "grade ii": 2,
    "g3": 3,
    "grade 3": 3,
    "grade iii": 3,
    "listed": 4,
    "allowance": 5,
    "claiming": 6,
    "maiden": 6,
    "msw": 6,
    # UK/Ireland (Group races)
    "group 1": 1,
    "group 2": 2,
    "group 3": 3,
    "class 1": 2,
    "class 2": 3,
    "class 3": 4,
    "class 4": 5,
    "class 5": 6,
    "class 6": 6,
    "class 7": 6,
}


def _race_class_tier(raw: Optional[str]) -> Optional[float]:
    """1 = top-grade stakes, 6 = maiden/claiming entry-level."""
    if not raw:
        return None
    key = " ".join(raw.strip().lower().split())
    for k, v in _RACE_CLASS_TIER_MAP.items():
        if k in key:
            return float(v)
    return None


# ── Per-entrant features ──────────────────────────────────────────


def horse_rolling_form(cur, horse_id: str, before_date) -> dict:
    """Horse's last HORSE_WINDOW finished starts. Returns avg finish,
    win rate, place rate, total starts."""
    cur.execute(
        """
        WITH recent AS (
            SELECT e.finish_position
            FROM race_entrants e
            JOIN races r ON r.id = e.race_id
            WHERE e.horse_id = %s
              AND r.status = 'finished'
              AND r.race_date < %s
              AND e.finish_position IS NOT NULL
              AND NOT e.scratched
            ORDER BY r.race_date DESC
            LIMIT %s
        )
        SELECT
            COUNT(*) AS starts,
            AVG(finish_position) AS avg_finish,
            SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) AS places
        FROM recent
        """,
        (horse_id, before_date, HORSE_WINDOW),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row and row.get("starts", 0):
        starts = float(row["starts"])
        out["horse_starts_last_5"] = starts
        if row.get("avg_finish") is not None:
            out["horse_avg_finish_last_5"] = float(row["avg_finish"])
        out["horse_win_rate_last_5"] = float(row["wins"] or 0) / starts if starts else 0.0
        out["horse_place_rate_last_5"] = float(row["places"] or 0) / starts if starts else 0.0
    return out


def horse_course_form(cur, horse_id: str, track_name: str, before_date) -> dict:
    """Horse's career record at this specific course."""
    cur.execute(
        """
        SELECT
            COUNT(*) AS starts,
            SUM(CASE WHEN e.finish_position = 1 THEN 1 ELSE 0 END) AS wins
        FROM race_entrants e
        JOIN races r ON r.id = e.race_id
        WHERE e.horse_id = %s
          AND r.track_name = %s
          AND r.status = 'finished'
          AND r.race_date < %s
          AND NOT e.scratched
        """,
        (horse_id, track_name, before_date),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row and row.get("starts", 0):
        out["horse_course_starts"] = float(row["starts"])
        out["horse_course_wins"] = float(row["wins"] or 0)
    return out


def horse_surface_form(cur, horse_id: str, surface_bucket: str, before_date) -> dict:
    """Horse's record on this surface (turf / dirt / synthetic)."""
    if surface_bucket == SURFACE_UNKNOWN:
        return {}
    cur.execute(
        """
        SELECT
            COUNT(*) AS starts,
            SUM(CASE WHEN e.finish_position = 1 THEN 1 ELSE 0 END) AS wins
        FROM race_entrants e
        JOIN races r ON r.id = e.race_id
        WHERE e.horse_id = %s
          AND r.status = 'finished'
          AND r.race_date < %s
          AND NOT e.scratched
          AND LOWER(COALESCE(r.surface, '')) LIKE %s
        """,
        (horse_id, before_date, f"%{surface_bucket}%"),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row and row.get("starts", 0):
        out["horse_surface_starts"] = float(row["starts"])
        out["horse_surface_wins"] = float(row["wins"] or 0)
    return out


def horse_distance_form(cur, horse_id: str, distance_m: Optional[int], before_date) -> dict:
    """Horse's record at distances within ±10% of today's race."""
    if not distance_m:
        return {}
    low, high = int(distance_m * 0.9), int(distance_m * 1.1)
    cur.execute(
        """
        SELECT
            COUNT(*) AS starts,
            SUM(CASE WHEN e.finish_position = 1 THEN 1 ELSE 0 END) AS wins
        FROM race_entrants e
        JOIN races r ON r.id = e.race_id
        WHERE e.horse_id = %s
          AND r.status = 'finished'
          AND r.race_date < %s
          AND NOT e.scratched
          AND r.distance_meters BETWEEN %s AND %s
        """,
        (horse_id, before_date, low, high),
    )
    row = cur.fetchone()
    out: dict[str, float] = {}
    if row and row.get("starts", 0):
        out["horse_distance_starts"] = float(row["starts"])
        out["horse_distance_wins"] = float(row["wins"] or 0)
    return out


def horse_days_since_last_race(cur, horse_id: str, before_date) -> dict:
    cur.execute(
        """
        SELECT r.race_date
        FROM race_entrants e
        JOIN races r ON r.id = e.race_id
        WHERE e.horse_id = %s
          AND r.status = 'finished'
          AND r.race_date < %s
        ORDER BY r.race_date DESC
        LIMIT 1
        """,
        (horse_id, before_date),
    )
    row = cur.fetchone()
    if not row:
        return {}
    delta = (before_date - row["race_date"]).total_seconds() / 86400.0
    return {"days_since_last_race": float(delta)}


def entity_30d_win_rate(cur, entity_id: str, entity_col: str, before_date) -> Optional[float]:
    """Win rate for a jockey or trainer over the last 30 days."""
    if not entity_id:
        return None
    cur.execute(
        f"""
        SELECT
            COUNT(*) AS starts,
            SUM(CASE WHEN e.finish_position = 1 THEN 1 ELSE 0 END) AS wins
        FROM race_entrants e
        JOIN races r ON r.id = e.race_id
        WHERE e.{entity_col} = %s
          AND r.status = 'finished'
          AND r.race_date < %s
          AND r.race_date >= %s - INTERVAL '{JOCKEY_DAYS} days'
          AND NOT e.scratched
        """,
        (entity_id, before_date, before_date),
    )
    row = cur.fetchone()
    if not row or not row.get("starts"):
        return None
    return float(row["wins"] or 0) / float(row["starts"])


def per_entrant_features(cur, entrant: dict, meta: dict, race_field_size: int) -> dict:
    """Assemble per-entrant features. The race-level features are
    duplicated onto every entrant at training time by the caller."""
    out: dict[str, float] = {}
    when = meta["race_date"]
    surface_bucket = _bucket_surface(meta.get("surface"))

    if entrant.get("morning_line_odds"):
        try:
            decimal = float(entrant["morning_line_odds"])
            if decimal > 1.0:
                out["morning_line_implied_prob"] = 1.0 / decimal
        except (TypeError, ValueError):
            pass

    if entrant.get("weight_carried_lbs") is not None:
        out["weight_carried_lbs"] = float(entrant["weight_carried_lbs"])
    if entrant.get("post_position") is not None:
        out["post_position"] = float(entrant["post_position"])

    out.update(horse_rolling_form(cur, entrant["horse_id"], when))
    out.update(horse_course_form(cur, entrant["horse_id"], meta["track_name"], when))
    out.update(horse_surface_form(cur, entrant["horse_id"], surface_bucket, when))
    out.update(horse_distance_form(cur, entrant["horse_id"], meta.get("distance_meters"), when))
    out.update(horse_days_since_last_race(cur, entrant["horse_id"], when))

    jockey_rate = entity_30d_win_rate(cur, entrant.get("jockey_id"), "jockey_id", when)
    if jockey_rate is not None:
        out["jockey_30d_win_rate"] = jockey_rate
    trainer_rate = entity_30d_win_rate(cur, entrant.get("trainer_id"), "trainer_id", when)
    if trainer_rate is not None:
        out["trainer_30d_win_rate"] = trainer_rate

    return _with_defaults(out)


# ── Orchestration ──────────────────────────────────────────────────


def compute_for_race(cur, race_id: str) -> Optional[dict]:
    """End-to-end feature computation for one race. Returns the
    JSONB payload to store, or None if the race can't be found."""
    meta = fetch_race_meta(cur, race_id)
    if not meta:
        return None
    entrants = fetch_entrants(cur, race_id)
    if not entrants:
        return None

    race_level = race_level_features(meta)
    race_level["field_size"] = float(len(entrants))  # actual non-scratched count

    entrant_rows = []
    for ent in entrants:
        feats = per_entrant_features(cur, ent, meta, len(entrants))
        entrant_rows.append({"entrant_id": ent["id"], "features": feats})

    return {"race_level": race_level, "entrants": entrant_rows}


def write_features(cur, race_id: str, payload: dict) -> None:
    cur.execute(
        """
        INSERT INTO race_features_cache
            (race_id, feature_set, feature_version, features, expires_at)
        VALUES (%s, %s, %s, %s::jsonb, NOW() + (%s || ' seconds')::interval)
        ON CONFLICT (race_id, feature_set, feature_version) DO UPDATE
            SET features = EXCLUDED.features,
                expires_at = EXCLUDED.expires_at,
                computed_at = NOW()
        """,
        (race_id, FEATURE_SET, FEATURE_VERSION, json.dumps(payload), str(CACHE_TTL_SECONDS)),
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--force", action="store_true")
    p.add_argument("--all-finished", action="store_true")
    p.add_argument("--race-ids", help="Comma-separated UUID list to recompute specific races.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def run(database_url: str, days: int, force: bool, all_finished: bool, race_ids: Optional[list[str]]) -> dict:
    written = 0
    skipped = 0
    with psycopg2.connect(database_url) as conn:
        targets = list_target_races(conn, days, force, all_finished, race_ids)
        if not targets:
            logger.info("No horse races need feature computation")
            return {"written": 0, "skipped": 0}
        logger.info("Computing features for %d races", len(targets))
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for rid in targets:
                payload = compute_for_race(cur, rid)
                if payload is None:
                    skipped += 1
                    continue
                write_features(cur, rid, payload)
                written += 1
            conn.commit()
    logger.info("Wrote %d race feature rows (%d skipped)", written, skipped)
    return {"written": written, "skipped": skipped}


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    race_ids = [s.strip() for s in args.race_ids.split(",") if s.strip()] if args.race_ids else None
    run(args.database_url, args.days, args.force, args.all_finished, race_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
