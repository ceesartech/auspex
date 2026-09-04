-- 023: Index the ESPN competition id now stored on matches.external_ids
--
-- WHY: `matches` permits exactly one row per (home_team_id, away_team_id,
-- match_date) and scripts/fetch_upcoming.py upserted on that triple, while no
-- ESPN id was stored anywhere (external_ids = '{}' on all 2,416 recent
-- soccer rows and all 46,308 tennis rows). ESPN slides scheduled kickoff
-- times, so ONE real fixture became SEVERAL matches rows: predictions,
-- odds and recommendations attached to the early row while the result
-- landed on a later one. Measured on prod 2026-09-02/04 — tennis: 19,008
-- stale 'scheduled' rows and 1,727 recommendations that can never settle
-- (121,668 of stake); soccer: 372 stale rows, 119 with a finished twin,
-- 42 stranded recs; MMA: 110 stale rows, 33 with twins. Of the 119 soccer
-- twin pairs the orphan was created first in 119/119 cases (99 before
-- kickoff via the fixtures path, 20 after via the results path).
--
-- fetch_upcoming.py now stores the ESPN COMPETITION id as
-- external_ids->>'espn' (a competition, not the event, is the unit of a
-- match — tennis nests competitions two layers deep under
-- event.groupings[].competitions[]) and resolves an incoming competition
-- to an existing row by that id BEFORE inserting. This index is what
-- keeps that lookup off a sequential scan.
--
-- DESIGN NOTE — this index deliberately does NOT enforce one-row-per-key
-- (contrast 020's idx_races_racing_api_id, which does): identity here is
-- enforced by the resolution ORDER in code (espn id -> exact (home, away,
-- match_date) -> a single unidentified same-pair 'scheduled' row within
-- 72 h), not by the database. An ESPN id distinguishes a match only
-- WITHIN a sport — the same numeric id can legitimately appear for a
-- soccer competition and an NFL one — so a table-wide exclusivity
-- constraint would raise on a cross-sport collision and abort the entire
-- ingest run (every league in that pass loses its fixtures), which is a
-- far worse failure than the duplicate row it would prevent.
-- resolve_match_id() scopes its lookup by leagues.sport for the same
-- reason.
--
-- Additive and reversible: index only, no data changes, no table
-- rewrite. To roll back: DROP INDEX CONCURRENTLY idx_matches_espn_id;

BEGIN;

-- Partial so it only covers rows that actually carry the key (rows
-- written before this work have external_ids = '{}' and are excluded).
--
-- The predicate is deliberately written on the SAME EXPRESSION the
-- resolver filters on. Postgres' predicate prover is intentionally
-- limited: `external_ids->>'espn' = $1` does NOT imply
-- `external_ids ? 'espn'` (unrelated operators), so a `?` predicate here
-- leaves the index unusable and step 1 of resolve_match_id sequential
-- scans `matches` on every competition — ~8k full scans per tennis
-- pipeline run alone. `IS NOT NULL` IS implied, because `=` is strict.
CREATE INDEX IF NOT EXISTS idx_matches_espn_id
    ON matches ((external_ids->>'espn'))
    WHERE (external_ids->>'espn') IS NOT NULL;

COMMENT ON INDEX idx_matches_espn_id IS
    'ESPN competition id lookup for fetch_upcoming.resolve_match_id. Non-exclusive on '
    'purpose: an ESPN id distinguishes a match only within a sport, so identity is '
    'enforced by the resolution order in code rather than by a constraint that would '
    'abort a whole ingest run on a cross-sport collision.';

COMMIT;
