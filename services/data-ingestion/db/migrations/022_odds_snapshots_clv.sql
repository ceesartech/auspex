-- 022: Odds price-change snapshots + closing-line-value (CLV) views
--
-- WHY (audit doc §2.2): CLV — did a rec beat the closing price? — is the
-- industry-standard edge test and converges in weeks, vs the months/years
-- settled-ROI needs. It was uncomputable: scripts/fetch_live_odds.py's
-- dedup keeps only the FIRST price ever seen per (match, book, market,
-- selection, line); every later price, including the closing one, was
-- silently dropped despite being fetched every 90 minutes.
--
-- DESIGN NOTE — separate table, NOT extra rows in `odds`: appending
-- price-changes to `odds` would silently change the semantics of every
-- AVG(odds_decimal) reader (compute_features _odds_avg, the training-frame
-- odds subqueries in utils/training_data.py, the best-odds pickers) from
-- "average across BOOKS" to "average across books AND TIME" — a
-- train/serve-drift bug of the same class as the §1.1 constant-prior
-- incident. odds_snapshots is append-only, written only when a price
-- CHANGES (bounded growth: a line that never moves = 1 row), and read
-- only by the CLV views.

BEGIN;

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    bookmaker VARCHAR(50) NOT NULL,
    market_type VARCHAR(50) NOT NULL,
    selection VARCHAR(100) NOT NULL,
    line DECIMAL(10, 2),
    odds_decimal DECIMAL(10, 4) NOT NULL,
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Serves both the writer's latest-price lookup and the views' last-before-
-- commence lookup.
CREATE INDEX IF NOT EXISTS idx_odds_snapshots_key_time
    ON odds_snapshots (match_id, market_type, bookmaker, selection, line, captured_at DESC);

-- Per-recommendation CLV. closing_odds = the last snapshot strictly before
-- the match's commence time, at the SAME bookmaker the rec was priced at
-- (same-book CLV is the honest form — cross-book best-price CLV flatters).
-- Lined markets store the rec selection with the line embedded
-- ('over_2.5', 'home_-1.5'); snapshots store (selection, line) split, so
-- the join tries the plain form first, else parses the '_<line>' suffix.
-- clv > 0 means we beat the close: the market moved toward our pick.
CREATE OR REPLACE VIEW vw_rec_clv AS
SELECT
    br.id                     AS recommendation_id,
    l.sport,
    br.bet_type,
    br.selection,
    br.bookmaker,
    br.status,
    br.odds_at_recommendation,
    cs.odds_decimal           AS closing_odds,
    cs.captured_at            AS closing_captured_at,
    ROUND(br.odds_at_recommendation / cs.odds_decimal - 1.0, 5) AS clv,
    br.created_at             AS rec_created_at,
    m.match_date
FROM betting_recommendations br
JOIN matches m  ON m.id = br.match_id
JOIN leagues l  ON l.id = m.league_id
LEFT JOIN LATERAL (
    SELECT s.odds_decimal, s.captured_at
    FROM odds_snapshots s
    WHERE s.match_id = br.match_id
      AND s.market_type = br.bet_type
      AND s.bookmaker = br.bookmaker
      AND s.captured_at < m.match_date
      AND (
            (s.selection = br.selection AND s.line IS NULL)
         OR (br.selection ~ '_-?[0-9.]+$'
             AND s.selection = regexp_replace(br.selection, '_-?[0-9.]+$', '')
             AND s.line = (regexp_match(br.selection, '_(-?[0-9.]+)$'))[1]::numeric)
      )
    ORDER BY s.captured_at DESC
    LIMIT 1
) cs ON true;

-- The market-gating rollup (audit §2.3): per sport+market, how often do
-- our recs beat the close and by how much? Positive avg_clv sustained
-- over 60-90 days is the go/no-go signal per category. Only recs whose
-- closing snapshot was captured count toward the CLV columns.
CREATE OR REPLACE VIEW vw_clv_summary AS
SELECT
    sport,
    bet_type,
    COUNT(*)                                    AS recs_total,
    COUNT(clv)                                  AS recs_with_clv,
    ROUND(AVG(clv), 5)                          AS avg_clv,
    ROUND(100.0 * COUNT(*) FILTER (WHERE clv > 0) / NULLIF(COUNT(clv), 0), 1) AS pct_beat_close
FROM vw_rec_clv
GROUP BY sport, bet_type;

COMMIT;
