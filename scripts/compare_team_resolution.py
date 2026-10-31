"""Compare soccer team resolution between two versions of the resolver (read-only).

Why: a change to fetch_upcoming.find_team can stop resolving a name the
previous version resolved. That club then gets a history-less twin stamped
with its ESPN id, and each of its ESPN-identified fixtures raises ForkRefused
(c5dfeb2 dropped the in-league ratio step and lost 'Espanyol' -> 'Espanol'
and 'Levadiakos' -> 'Levadeiakos' that way). A review of the new rules cannot
see it, because a name the old step resolved never produced a twin to look at.

For every team ESPN lists in each swept soccer league (the public /teams
endpoint of every SOCCER_LEAGUES slug, international competitions included,
or a directory of saved responses), this resolves its display name and ESPN
team id with fetch_upcoming.find_team as two versions of scripts/ implement
it, against ONE database. Each version runs in its own subprocess
(`python -I`), connected read-only: find_team never writes, and the read-only
session makes sure. The default old version is the git ref --old-ref,
extracted with `git archive`; the new one is this working tree.

Report:
  REGRESSION  the old version resolved the name and the new one resolves it
              to another row or to nothing. Exit 1, unless the ESPN id is
              pinned in the new version's seed (scripts/data/
              soccer_team_espn_seed.csv): then it is SEEDED, a reviewed
              correction, printed with the seed's evidence.
  GAIN        only the new version resolves the name.
  COLLISION   one row that two distinct ESPN team ids resolve to under the new
              version (whichever is swept first would stamp it) — with
              --merge-map, also once the merge folds each twin into its kept
              row — and ESPN display names that two ids share.
  FOREIGN     a domestic-league team resolved (new version) onto a row that
              also holds matches in another country's domestic league: two
              clubs on one row, or a mis-assigned league.

Usage:
    # against a local copy of prod (throwaway Postgres):
    python scripts/compare_team_resolution.py --database-url postgresql://... --old-ref a071d6a
    # saved ESPN responses instead of fetching, and a CSV of every row:
    python scripts/compare_team_resolution.py --old-ref a071d6a --espn-dir /tmp/espn_teams --out-csv /tmp/cmp.csv
    # without git (e.g. inside a container): point at an extracted old scripts/ tree
    python scripts/compare_team_resolution.py --old-scripts-dir /tmp/old/scripts

Exit status: 0 no unexplained regression; 1 at least one; 2 bad arguments, a
missing league row, or a worker failure.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("compare_team_resolution")

HERE = Path(__file__).resolve().parent
ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams"
CROSS_COUNTRY = ("International", "World")
CSV_FIELDS = (
    "slug",
    "league",
    "espn_id",
    "espn_name",
    "old_team_id",
    "old_team",
    "old_tier",
    "new_team_id",
    "new_team",
    "new_tier",
    "verdict",
    "note",
)


# ── worker: one version of scripts/, one read-only session ────────────


def run_worker(scripts_dir: str, input_path: str, database_url: str) -> int:
    """Resolve every entry of `input_path` with the find_team of
    `scripts_dir`; print one JSON object per entry on stdout."""
    sys.path.insert(0, scripts_dir)
    import fetch_upcoming as fu  # the version under test
    import psycopg2
    from psycopg2.extras import RealDictCursor

    logging.getLogger().setLevel(logging.ERROR)  # its tier WARNINGs are not the report
    cfg = fu.SPORT_CONFIGS["soccer"]
    entries = json.loads(Path(input_path).read_text(encoding="utf-8"))
    takes_espn = "espn_id" in inspect.signature(fu.find_team).parameters
    conn = psycopg2.connect(database_url)
    conn.set_session(readonly=True)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for e in entries:
                kwargs = {"espn_id": e["espn_id"]} if takes_espn else {}
                tier = ""
                private = getattr(fu, "_find_team", None)
                if takes_espn and private is not None:
                    found = private(cur, cfg, e["name"], e["league_id"], e["espn_id"])
                    team_id, tier = found[0], found[1]
                else:
                    team_id = fu.find_team(cur, cfg, e["name"], e["league_id"], **kwargs)
                print(json.dumps({"key": e["key"], "team_id": str(team_id) if team_id else None, "tier": tier}))
    finally:
        conn.rollback()
        conn.close()
    return 0


# ── main ──────────────────────────────────────────────────────────────


def espn_teams(slug: str, espn_dir: str | None, save_dir: str | None) -> list[dict]:
    """ESPN's team list for a league slug: [{id, displayName, slug}]."""
    if espn_dir:
        payload = json.loads((Path(espn_dir) / f"{slug}.json").read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(ESPN_TEAMS_URL.format(slug=slug), timeout=30) as resp:  # noqa: S310
            body = resp.read()
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_dir) / f"{slug}.json").write_bytes(body)
        payload = json.loads(body)
    return [
        x["team"]
        for sport in payload.get("sports", [])
        for league in sport.get("leagues", [])
        for x in league.get("teams", [])
        if (x.get("team") or {}).get("id") and (x.get("team") or {}).get("displayName")
    ]


def scripts_tree(ref: str | None, scripts_dir: str | None, workdir: str) -> str:
    """The scripts/ directory of a version: given, or extracted from git."""
    if scripts_dir:
        return str(Path(scripts_dir).resolve())
    if not ref:
        return str(HERE)
    repo = HERE.parent
    dest = Path(workdir) / ref.replace("/", "_")
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(repo), "archive", ref, "scripts"], check=True, capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    return str(dest / "scripts")


def resolve_all(scripts_dir: str, entries: list[dict], database_url: str, workdir: str, label: str) -> dict:
    """{key: (team_id, tier)} from a worker subprocess running `scripts_dir`."""
    inp = Path(workdir) / f"entries_{label}.json"
    inp.write_text(json.dumps(entries), encoding="utf-8")
    err = Path(workdir) / f"worker_{label}.log"
    with err.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--worker",
                "--scripts-dir",
                scripts_dir,
                "--input",
                str(inp),
                "--database-url",
                database_url,
            ],
            stdout=subprocess.PIPE,
            stderr=fh,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} worker ({scripts_dir}) exited {proc.returncode}; see {err}")
    out = {}
    for line in proc.stdout.splitlines():
        row = json.loads(line)
        out[row["key"]] = (row["team_id"], row["tier"])
    if len(out) != len(entries):
        raise RuntimeError(f"{label} worker resolved {len(out)} of {len(entries)} entries; see {err}")
    return out


def load_seed(scripts_dir: str) -> dict[str, dict]:
    path = Path(scripts_dir) / "data" / "soccer_team_espn_seed.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["espn_id"].strip(): row for row in csv.DictReader(fh)}


def compare(old: dict, new: dict, entries: list[dict], seed: dict) -> list[dict]:
    """One verdict per entry: same | REGRESSION | SEEDED | GAIN."""
    rows = []
    for e in entries:
        (old_id, old_tier), (new_id, new_tier) = old[e["key"]], new[e["key"]]
        if old_id == new_id:
            verdict = "same"
        elif old_id is None:
            verdict = "GAIN"
        elif e["espn_id"] in seed:
            verdict = "SEEDED"
        else:
            verdict = "REGRESSION"
        rows.append(
            {
                **e,
                "old_team_id": old_id,
                "old_tier": old_tier,
                "new_team_id": new_id,
                "new_tier": new_tier,
                "verdict": verdict,
                "note": (seed.get(e["espn_id"]) or {}).get("evidence", "") if verdict == "SEEDED" else "",
            }
        )
    return rows


def collisions(rows: list[dict], entries: list[dict], fold: dict[str, str] | None = None) -> tuple[list, list]:
    """Rows two ESPN ids resolve to (new version), and names two ids share.
    `fold` maps a team id onto the row it will be merged into (the team
    merge map's new_team_id -> old_team_id): two ids landing on one row only
    after the merge collide too ('Nacional' / 'C.D. Nacional')."""
    fold = fold or {}
    by_row: dict[str, set] = defaultdict(set)
    by_name: dict[str, set] = defaultdict(set)
    for r in rows:
        if r["new_team_id"]:
            by_row[fold.get(r["new_team_id"], r["new_team_id"])].add((r["espn_id"], r["name"], r["slug"]))
    for e in entries:
        by_name[" ".join(e["name"].lower().split())].add((e["espn_id"], e["name"], e["slug"]))
    shared_rows = [(tid, sorted(v)) for tid, v in by_row.items() if len({x[0] for x in v}) > 1]
    shared_names = [(n, sorted(v)) for n, v in by_name.items() if len({x[0] for x in v}) > 1]
    return shared_rows, shared_names


def load_merge_map(path: str | None) -> dict[str, str]:
    """{new_team_id: old_team_id} from a team merge map CSV ('#' lines are
    comments), or {}."""
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        return {row["new_team_id"]: row["old_team_id"] for row in reader}


def foreign_rows(cur, rows: list[dict]) -> list[tuple]:
    """(entry, other countries) for domestic-league resolutions onto a row
    that also holds another country's domestic-league matches."""
    team_ids = sorted({r["new_team_id"] for r in rows if r["new_team_id"] and r["country"] not in CROSS_COUNTRY})
    if not team_ids:
        return []
    cur.execute(
        """
        SELECT t.team_id, l.country, count(*) AS n
          FROM (SELECT home_team_id AS team_id, league_id FROM matches WHERE home_team_id = ANY(%(ids)s::uuid[])
                UNION ALL
                SELECT away_team_id, league_id FROM matches WHERE away_team_id = ANY(%(ids)s::uuid[])) t
          JOIN leagues l ON l.id = t.league_id
         WHERE l.country NOT IN %(cross)s
         GROUP BY 1, 2
        """,
        {"ids": team_ids, "cross": CROSS_COUNTRY},
    )
    countries: dict[str, dict[str, int]] = defaultdict(dict)
    for row in cur.fetchall():
        countries[str(row["team_id"])][row["country"]] = row["n"]
    out = []
    for r in rows:
        if r["new_team_id"] and r["country"] not in CROSS_COUNTRY:
            other = {c: n for c, n in countries.get(r["new_team_id"], {}).items() if c != r["country"]}
            if other:
                out.append((r, other))
    return out


def team_names(cur, ids) -> dict[str, str]:
    ids = sorted({i for i in ids if i})
    if not ids:
        return {}
    cur.execute("SELECT id::text AS id, name FROM teams WHERE id = ANY(%s::uuid[])", (ids,))
    return {r["id"]: r["name"] for r in cur.fetchall()}


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — a linear report
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--old-ref", help="git ref of the old version (extracted with git archive)")
    p.add_argument("--old-scripts-dir", help="an extracted old scripts/ tree instead of --old-ref")
    p.add_argument("--new-ref", help="git ref of the new version (default: this working tree)")
    p.add_argument("--new-scripts-dir", help="an extracted new scripts/ tree")
    p.add_argument("--espn-dir", help="saved ESPN /teams responses, <slug>.json (default: fetch them)")
    p.add_argument("--save-espn-dir", help="when fetching, also save each response here")
    p.add_argument("--leagues", help="comma-separated slugs (default: every SOCCER_LEAGUES slug)")
    p.add_argument("--out-csv", help="write every entry with both verdicts here")
    p.add_argument("--merge-map", help="team merge map CSV (new_team_id -> old_team_id): collisions after the merge")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--scripts-dir", help=argparse.SUPPRESS)
    p.add_argument("--input", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2
    if args.worker:
        return run_worker(args.scripts_dir, args.input, args.database_url)
    if not (args.old_ref or args.old_scripts_dir):
        logger.error("give the old version: --old-ref or --old-scripts-dir")
        return 2

    import psycopg2
    from psycopg2.extras import RealDictCursor

    with tempfile.TemporaryDirectory(prefix="cmp_team_res_") as work:
        old_dir = scripts_tree(args.old_ref, args.old_scripts_dir, work)
        new_dir = scripts_tree(args.new_ref, args.new_scripts_dir, work)
        sys.path.insert(0, new_dir)
        import fetch_upcoming as fu_new  # SOCCER_LEAGUES of the new version

        leagues = fu_new.SOCCER_LEAGUES
        slugs = [s.strip() for s in args.leagues.split(",")] if args.leagues else list(leagues)
        conn = psycopg2.connect(args.database_url)
        conn.set_session(readonly=True)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                entries = []
                for slug in slugs:
                    _code, name, country = leagues[slug]
                    cur.execute(
                        "SELECT id::text AS id FROM leagues WHERE name = %s AND country = %s AND sport = 'soccer'",
                        (name, country),
                    )
                    league = cur.fetchone()
                    if not league:
                        logger.error("No leagues row for %s (%s, %s) — cannot compare it", slug, name, country)
                        return 2
                    for t in espn_teams(slug, args.espn_dir, args.save_espn_dir):
                        entries.append(
                            {
                                "key": f"{slug}|{t['id']}",
                                "slug": slug,
                                "league": name,
                                "country": country,
                                "league_id": league["id"],
                                "espn_id": str(t["id"]),
                                "name": t["displayName"],
                                "espn_name": t["displayName"],
                            }
                        )
                logger.info("%d ESPN teams across %d leagues", len(entries), len(slugs))
                old = resolve_all(old_dir, entries, args.database_url, work, "old")
                new = resolve_all(new_dir, entries, args.database_url, work, "new")
                rows = compare(old, new, entries, load_seed(new_dir))
                names = team_names(cur, [r["old_team_id"] for r in rows] + [r["new_team_id"] for r in rows])
                for r in rows:
                    r["old_team"], r["new_team"] = names.get(r["old_team_id"] or ""), names.get(r["new_team_id"] or "")
                shared_rows, shared_names = collisions(rows, entries, load_merge_map(args.merge_map))
                foreign = foreign_rows(cur, rows)
        finally:
            conn.rollback()
            conn.close()

    counts = defaultdict(int)
    for r in rows:
        counts[r["verdict"]] += 1
    for verdict in ("REGRESSION", "SEEDED", "GAIN"):
        for r in (x for x in rows if x["verdict"] == verdict):
            logger.log(
                logging.ERROR if verdict == "REGRESSION" else logging.WARNING,
                "%-10s %-22s %-6s %-30r old %-8s %-28r new %-9s %r%s",
                verdict,
                r["slug"],
                r["espn_id"],
                r["name"],
                r["old_tier"] or "-",
                r["old_team"],
                r["new_tier"] or "-",
                r["new_team"],
                f"  [{r['note']}]" if r["note"] else "",
            )
    for tid, hits in shared_rows:
        logger.warning("COLLISION  row %s (%r) is resolved by several ESPN ids: %s", tid, names.get(tid), hits)
    for name, hits in shared_names:
        logger.warning("COLLISION  ESPN display name %r is shared by several ESPN ids: %s", name, hits)
    for r, other in foreign:
        logger.warning(
            "FOREIGN    %s %s %r -> %r also holds domestic matches in %s",
            r["slug"],
            r["espn_id"],
            r["name"],
            r["new_team"],
            other,
        )
    if args.out_csv:
        with open(args.out_csv, "x", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    logger.info(
        "%d entries: %s; %d shared rows, %d shared names, %d foreign",
        len(rows),
        dict(sorted(counts.items())),
        len(shared_rows),
        len(shared_names),
        len(foreign),
    )
    if counts["REGRESSION"]:
        logger.error("%d name(s) the old version resolved now resolve differently or not at all", counts["REGRESSION"])
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(2)
