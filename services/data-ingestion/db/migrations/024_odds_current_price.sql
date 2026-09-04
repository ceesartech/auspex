-- 024: `odds` becomes a CURRENT-price table (additive; no data change)
--
-- WHY (2026-09 prod audit, defect 1): scripts/fetch_live_odds.py inserted a
-- row for a (match, bookmaker, market_type, selection, line) key the FIRST
-- time it saw it and then skipped that key forever behind a NOT EXISTS
-- guard, so `odds` held first-seen quotes, not live ones. Measured on prod
-- the price a recommendation was written at averaged 178h old (1x2), 89h
-- (asian_handicap), 149h (over_under), 197h (mma). 48% of settled soccer
-- 1x2 recs failed their own 5% EV gate at the price actually available when
-- they were written, average claimed EV fell 0.132 -> 0.079, realized ROI
-- was +4.4% not the +14.1% claimed, and because recs are re-written every
-- 15 minutes while pending (median 0.23h before kickoff) vw_rec_clv was
-- measuring nothing but that staleness. The ingest now UPDATEs the
-- non-opening row in place when the price moves; odds_snapshots (022)
-- remains the append-only price history, and the is_opening=true row
-- remains the immutable opening line.
--
-- APPLY THIS BEFORE (or with) the code that ships alongside it: the
-- recommendation generators age their prices off odds.last_seen_at, which
-- this migration adds. Until it runs they will fail loudly (UndefinedColumn)
-- on every cycle and write NO recommendations -- the safe failure, but a
-- visible one. See OPERATIONS.md for the by-hand apply on the VM.
--
-- It rewrites NO row: the new column is NULLable with no default (a
-- non-volatile-default ADD COLUMN is metadata-only in PG11+, and DEFAULT NOW()
-- would rewrite the whole table), and readers COALESCE it back to `timestamp`
-- so pre-migration rows simply read as older than they are -- the safe
-- direction. Every row the ingest touches gets a real last_seen_at on the
-- next run. The rest of the file tries to add the uniqueness the new upsert
-- assumes -- one current price per key -- and refuses to do so if the
-- existing data would violate it, because a failed CREATE UNIQUE INDEX aborts
-- the whole deploy.
--
-- REVERSIBLE:  DROP INDEX IF EXISTS uq_odds_current_price_key;
--              ALTER TABLE odds DROP COLUMN IF EXISTS last_seen_at;

BEGIN;

-- WHY a second timestamp (2026-09 prod audit, defect 1 + review). `timestamp`
-- advances only when the PRICE MOVES -- that is what the cross-book feature
-- builders and the API price views have always read it as, and the ingest
-- keeps it that way. But "when did the price last change" is NOT "when did we
-- last see the book quote it", and only the second one can tell a live market
-- that has sat still from a book that pulled the market and left its last
-- number behind to win the best-price pick forever. That is the residual half
-- of defect 1: the in-place update fixes stale prices for books that are still
-- quoting; last_seen_at is what lets a recommendation refuse a book that is
-- not. scripts/fetch_live_odds.py stamps it on EVERY observation;
-- scripts/generate_recommendations.py ages MAX_ODDS_AGE_HOURS off it.
ALTER TABLE odds ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ;

COMMENT ON COLUMN odds.last_seen_at IS
    'When the ingest last SAW this bookmaker quote this price (stamped every '
    'observation, unchanged prices included). Contrast odds."timestamp", which '
    'advances only when the price MOVES. NULL for rows written before '
    'migration 024; readers COALESCE to "timestamp".';

-- The key is byte-for-byte the predicate scripts/fetch_live_odds.py
-- insert_odds_row matches a row on, so the writer can never permit a row the
-- index would reject:
--   * COALESCE(line, -1) because `line` is NULLable and NULLs are DISTINCT in
--     a plain unique index -- an unlined market (1x2, moneyline, btts, ...)
--     would otherwise never dedupe at all.
--   * is_opening because the opening line is a separate, immutable row for the
--     same market (written by scripts/promote_raw.py, never touched by the
--     ingest).
--   * is_live because nothing writes is_live=true today, but if a live-odds
--     feed is ever added its prices are a different series that must neither
--     block nor be clobbered by the pre-match row.
-- COALESCE on both flags because both are NULLable (DEFAULT false, no NOT
-- NULL) and a legacy NULL would otherwise index as its own key.
-- (scripts/promote_raw.py's own guard is narrower -- it does not test is_live
-- -- but it only ever writes is_live=false, so it cannot violate this index.)
--
-- The guard: if ANY duplicate key already exists we skip the index and say
-- so loudly rather than failing the migration. Duplicates can predate the
-- writers' guard, or come from a hand-run backfill. Re-run this file after
-- deduping to pick the index up -- it is idempotent either way.
DO $$
DECLARE
    dup_keys BIGINT;
BEGIN
    SELECT COUNT(*) INTO dup_keys FROM (
        SELECT 1
        FROM odds
        GROUP BY match_id, bookmaker, market_type, selection,
                 COALESCE(line, -1),
                 COALESCE(is_opening, false),
                 COALESCE(is_live, false)
        HAVING COUNT(*) > 1
    ) d;

    IF dup_keys > 0 THEN
        RAISE WARNING
            '024: % duplicate odds key(s) present - uq_odds_current_price_key NOT created. '
            'fetch_live_odds.insert_odds_row does not depend on it (it does '
            'UPDATE-then-INSERT-if-no-rows), but until it exists nothing stops a second '
            '"current" price appearing for a key. Dedupe with the query in this file, then re-run it.',
            dup_keys;
    ELSE
        CREATE UNIQUE INDEX IF NOT EXISTS uq_odds_current_price_key
            ON odds (match_id, bookmaker, market_type, selection,
                     (COALESCE(line, -1)),
                     (COALESCE(is_opening, false)),
                     (COALESCE(is_live, false)));
        RAISE NOTICE '024: uq_odds_current_price_key created.';
    END IF;
END
$$;

COMMIT;

-- Inspect the duplicates the guard found (read-only; run by hand):
--
--   SELECT match_id, bookmaker, market_type, selection,
--          COALESCE(line, -1) AS line_key,
--          COALESCE(is_opening, false) AS opening,
--          COALESCE(is_live, false)    AS live,
--          COUNT(*) AS n, MIN(timestamp) AS first_seen, MAX(timestamp) AS last_seen
--   FROM odds
--   GROUP BY 1, 2, 3, 4, 5, 6, 7
--   HAVING COUNT(*) > 1
--   ORDER BY n DESC
--   LIMIT 50;
--
-- Deduping keeps the FRESHEST row per key (the current price) and is a
-- destructive operation on a live table: take a verified backup first
-- (scripts/backup_postgres.py, OPERATIONS.md) and get operator sign-off.
