"""Statistical screen: can ANY pre-match feature family move soccer 1X2 (or
over-2.5) beyond the de-vigged closing price? (audit §4 row 2026-10-25)

Three stages, all offline, all against the FULL prod frame (170k finished
soccer matches, 2007-2026), run on a laptop in ~5 minutes:

  engineer  build strictly pre-match engineered features from the raw export:
            Elo (K=20, HA=60, MOV multiplier), rest days + congestion
            (games in last 7/14 d), venue-specific form (home-at-home,
            away-at-away, 5/10), 10/20-match + EWMA form, season-to-date
            PPG/GD/games played + days into season, league-season draw rate,
            rolling draw rate, H2H (last 5), rolling cards/fouls, rolling
            shots/SoT/corners/xG (where match_stats exist), rolling totals,
            cross-book micro-structure (implied-prob std, max/min spread,
            book count, overround), calendar.
  screen    for each family: paired dBrier (multiclass sum-of-squares, the
            repo convention) vs (a) a market-only multinomial LR on the three
            implied logits and (b) a prod-like LightGBM on the baseline feature
            set, fixed test 2024-26 (n~23.8k, SE~0.0001-0.0002), plus a
            per-season walk-forward sign check 2019-26. Also prints the raw
            de-vigged market Brier and the LR / GBM deltas against it.
  market    where is the MARKET miscalibrated? realized-minus-implied by
            league, favourite band, year, season stage, rest-diff, book era,
            overround, plus flat-stake returns by implied band at the average
            and at the best (Max) price.

Result 2026-10-25 (see docs/SYSTEM_AUDIT_AND_ROADMAP.md §4): market Brier
0.59496 on n=23,784; every family |dBrier| <= 0.00033 — the market is the
ceiling for 1x2. The market scan found the favourite-longshot bias of the
proportional de-vig (0.7-0.8 implied realized +3 pts, <0.10 implied -1.5 pts,
stable across eras) — a calibration finding, not a standalone edge.

Export (run inside the postgres container; COPY needs a superuser):

    docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/export_soccer.sql'

with /tmp/export_soccer.sql = the SQL in EXPORT_SQL below, then

    python scripts/ab_soccer_feature_screen.py --csv soccer_export.csv --stage all --workdir /tmp/screen
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

EXPORT_SQL = r"""
COPY (
WITH o AS (
  SELECT match_id,
    AVG(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='home' AND NOT is_live) AS oh,
    AVG(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='draw' AND NOT is_live) AS od,
    AVG(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='away' AND NOT is_live) AS oa,
    MAX(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='home' AND NOT is_live) AS oh_max,
    MAX(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='draw' AND NOT is_live) AS od_max,
    MAX(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='away' AND NOT is_live) AS oa_max,
    MIN(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='home' AND NOT is_live) AS oh_min,
    MIN(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='draw' AND NOT is_live) AS od_min,
    MIN(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='away' AND NOT is_live) AS oa_min,
    STDDEV_SAMP(1.0/NULLIF(odds_decimal,0))
        FILTER (WHERE market_type='1x2' AND selection='home' AND NOT is_live) AS ih_sd,
    STDDEV_SAMP(1.0/NULLIF(odds_decimal,0))
        FILTER (WHERE market_type='1x2' AND selection='away' AND NOT is_live) AS ia_sd,
    COUNT(DISTINCT bookmaker) FILTER (WHERE market_type='1x2' AND NOT is_live) AS n_books,
    AVG(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='home' AND is_opening) AS oh_open,
    AVG(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='draw' AND is_opening) AS od_open,
    AVG(odds_decimal) FILTER (WHERE market_type='1x2' AND selection='away' AND is_opening) AS oa_open,
    AVG(odds_decimal)
        FILTER (WHERE market_type='over_under' AND selection='over' AND line=2.5 AND NOT is_live) AS o_over25,
    AVG(odds_decimal)
        FILTER (WHERE market_type='over_under' AND selection='under' AND line=2.5 AND NOT is_live) AS o_under25,
    AVG(odds_decimal) FILTER (WHERE market_type='asian_handicap' AND selection='home' AND NOT is_live) AS ah_home,
    AVG(line) FILTER (WHERE market_type='asian_handicap' AND selection='home' AND NOT is_live) AS ah_line,
    AVG(odds_decimal) FILTER (WHERE market_type='btts' AND selection='yes' AND NOT is_live) AS btts_yes,
    MIN(timestamp) FILTER (WHERE market_type='1x2') AS odds_first_ts,
    MAX(timestamp) FILTER (WHERE market_type='1x2') AS odds_last_ts
  FROM odds GROUP BY match_id
), pr AS (
  SELECT DISTINCT ON (match_id) match_id, probabilities, model_version, created_at AS pred_at
  FROM predictions WHERE prediction_type='match_result' AND model_name='ensemble'
  ORDER BY match_id, created_at DESC
)
SELECT m.id AS match_id, m.match_date, l.name AS league, l.country, l.tier, m.season, m.round, m.venue,
       m.attendance, m.referee_id,
       m.home_team_id, m.away_team_id, ht.name AS home_team, at.name AS away_team,
       m.home_score, m.away_score, m.half_time_home, m.half_time_away,
       m.external_ids->>'espn' AS espn_id, m.metadata->>'season_type' AS season_type,
              m.metadata->>'source' AS src, m.created_at AS m_created,
       o.oh,o.od,o.oa,o.oh_max,o.od_max,o.oa_max,o.oh_min,o.od_min,o.oa_min,o.ih_sd,o.ia_sd,o.n_books,o.oh_open,o.od_open,o.oa_open,o.o_over25,o.o_under25,o.ah_home,o.ah_line,o.btts_yes,o.odds_first_ts,o.odds_last_ts,
       hs.shots AS h_shots, hs.shots_on_target AS h_sot, hs.expected_goals AS h_xg, hs.corners AS h_corners,
              hs.possession AS h_poss, hs.fouls AS h_fouls, hs.yellow_cards AS h_yc, hs.red_cards AS h_rc,
       aw.shots AS a_shots, aw.shots_on_target AS a_sot, aw.expected_goals AS a_xg, aw.corners AS a_corners,
              aw.possession AS a_poss, aw.fouls AS a_fouls, aw.yellow_cards AS a_yc, aw.red_cards AS a_rc,
       w.temperature_c, w.wind_kmh, w.precipitation_mm,
       pr.probabilities::text AS pred_probs, pr.model_version AS pred_version, pr.pred_at,
       fc.features::text AS fc_features
FROM matches m
JOIN leagues l ON l.id=m.league_id
JOIN teams ht ON ht.id=m.home_team_id JOIN teams at ON at.id=m.away_team_id
LEFT JOIN o ON o.match_id=m.id
LEFT JOIN match_stats hs ON hs.match_id=m.id AND hs.team_id=m.home_team_id
LEFT JOIN match_stats aw ON aw.match_id=m.id AND aw.team_id=m.away_team_id
LEFT JOIN LATERAL (
    SELECT temperature_c, wind_kmh, precipitation_mm FROM match_weather mw
    WHERE mw.match_id=m.id ORDER BY fetched_at DESC LIMIT 1) w ON true
LEFT JOIN pr ON pr.match_id=m.id
LEFT JOIN LATERAL (
    SELECT features FROM features_cache f WHERE f.match_id=m.id AND f.feature_set='baseline'
    ORDER BY computed_at DESC LIMIT 1) fc ON true
WHERE l.sport='soccer' AND m.status='finished' AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
ORDER BY m.match_date
) TO '/tmp/soccer_export.csv' WITH CSV HEADER
"""


def _pm(d) -> str:
    """mean ± standard error of a paired per-row delta, formatted."""
    return f"{d.mean():+.5f}±{d.std() / np.sqrt(len(d)):.5f}"


def load_export(csv_path: str) -> pd.DataFrame:
    """Load the prod CSV export, coerce numerics, derive target + implied probs."""
    df = pd.read_csv(
        csv_path, low_memory=False, parse_dates=["match_date", "odds_first_ts", "odds_last_ts", "pred_at", "m_created"]
    )
    num = [
        c
        for c in df.columns
        if c
        not in (
            "match_id",
            "match_date",
            "league",
            "country",
            "season",
            "round",
            "venue",
            "referee_id",
            "home_team_id",
            "away_team_id",
            "home_team",
            "away_team",
            "espn_id",
            "season_type",
            "src",
            "m_created",
            "odds_first_ts",
            "odds_last_ts",
            "pred_probs",
            "pred_version",
            "pred_at",
            "fc_features",
        )
    ]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)
    df["y"] = np.select([df.home_score > df.away_score, df.home_score == df.away_score], [0, 1], 2)
    df["year"] = df.match_date.dt.year
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add strictly pre-match engineered features (see module docstring)."""
    # clean odds
    for c in ["oh", "od", "oa", "oh_max", "od_max", "oa_max", "oh_min", "od_min", "oa_min", "o_over25", "o_under25"]:
        df.loc[(df[c] < 1.01) | (df[c] > 200), c] = np.nan
    inv = 1 / df[["oh", "od", "oa"]]
    S = inv.sum(axis=1)
    df["overround"] = S - 1
    df["ph"] = (1 / df.oh) / S
    df["pd"] = (1 / df.od) / S
    df["pa"] = (1 / df.oa) / S
    df["has_odds"] = df[["oh", "od", "oa"]].notna().all(axis=1) & (df.overround.between(0, 0.35))
    df["pou"] = (1 / df.o_over25) / ((1 / df.o_over25) + (1 / df.o_under25))
    df["tot"] = df.home_score + df.away_score
    df["over25"] = (df.tot > 2.5).astype(int)
    df["dow"] = df.match_date.dt.dayofweek
    df["month"] = df.match_date.dt.month
    df["hour"] = df.match_date.dt.hour
    # season start per league-season for stage features
    df["season_key"] = df.league + "|" + df.season.astype(str)
    sstart = df.groupby("season_key").match_date.transform("min")
    df["days_into_season"] = (df.match_date - sstart).dt.days

    # --- sequential team-level features ---
    n = len(df)
    K = 20.0
    HA = 60.0
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(list)  # team -> list of dicts (date, gf, ga, pts, venue, shots..., yc, rc, over25, draw)
    h2h = defaultdict(list)  # frozenset pair -> list of (date, home_team, hs, as)
    season_pts = defaultdict(lambda: [0, 0, 0])  # (team, season_key)->[pts, gd, gp]
    league_draws = defaultdict(lambda: [0, 0])  # season_key -> [draws, games]

    cols = defaultdict(lambda: np.full(n, np.nan))

    def roll(lst, k, key, venue=None):
        sub = [r for r in lst if venue is None or r["venue"] == venue][-k:]
        if not sub:
            return np.nan
        return (
            float(np.mean([r[key] for r in sub if r[key] is not None]))
            if any(r[key] is not None for r in sub)
            else np.nan
        )

    dates = df.match_date.values
    H = df.home_team_id.values
    A = df.away_team_id.values
    hs = df.home_score.values
    as_ = df.away_score.values
    sk = df.season_key.values
    h_shots = df.h_shots.values
    a_shots = df.a_shots.values
    h_sot = df.h_sot.values
    a_sot = df.a_sot.values
    h_yc = df.h_yc.values
    a_yc = df.a_yc.values
    h_rc = df.h_rc.values
    a_rc = df.a_rc.values
    h_xg = df.h_xg.values
    a_xg = df.a_xg.values
    h_corn = df.h_corners.values
    a_corn = df.a_corners.values
    for i in range(n):
        d = dates[i]
        h = H[i]
        a = A[i]
        # Elo
        eh, ea = elo[h], elo[a]
        cols["elo_h"][i] = eh
        cols["elo_a"][i] = ea
        cols["elo_diff"][i] = eh + HA - ea
        cols["elo_p_home"][i] = 1 / (1 + 10 ** (-(eh + HA - ea) / 400))
        # history-based
        lh, la = hist[h], hist[a]
        for side, l in (("h", lh), ("a", la)):
            if l:
                last = l[-1]
                cols[f"{side}_rest"][i] = min((d - last["date"]) / np.timedelta64(1, "D"), 30)
                cols[f"{side}_games7"][i] = sum(1 for r in l[-8:] if (d - r["date"]) / np.timedelta64(1, "D") <= 7)
                cols[f"{side}_games14"][i] = sum(1 for r in l[-10:] if (d - r["date"]) / np.timedelta64(1, "D") <= 14)
                cols[f"{side}_last_rc"][i] = last["rc"] if last["rc"] is not None else np.nan
                cols[f"{side}_n_hist"][i] = len(l)
            for k in (5, 10, 20):
                cols[f"{side}_pts{k}"][i] = roll(l, k, "pts")
                cols[f"{side}_gd{k}"][i] = roll(l, k, "gd")
            cols[f"{side}_draw20"][i] = roll(l, 20, "draw")
            cols[f"{side}_over25_10"][i] = roll(l, 10, "over25")
            cols[f"{side}_tot10"][i] = roll(l, 10, "tot")
            cols[f"{side}_yc10"][i] = roll(l, 10, "yc")
            cols[f"{side}_rc10"][i] = roll(l, 10, "rc")
            cols[f"{side}_shots10"][i] = roll(l, 10, "shots")
            cols[f"{side}_shots_ag10"][i] = roll(l, 10, "shots_ag")
            cols[f"{side}_sot10"][i] = roll(l, 10, "sot")
            cols[f"{side}_sot_ag10"][i] = roll(l, 10, "sot_ag")
            cols[f"{side}_xg10"][i] = roll(l, 10, "xg")
            cols[f"{side}_xga10"][i] = roll(l, 10, "xga")
            cols[f"{side}_corn10"][i] = roll(l, 10, "corn")
            # EWMA goal diff
            if l:
                w = np.array([0.5 ** ((len(l) - 1 - j) / 8.0) for j in range(len(l))][-40:])
                g = np.array([r["gd"] for r in l][-40:])
                cols[f"{side}_ewgd"][i] = float((w * g).sum() / w.sum())
        # venue-specific
        for k in (5, 10):
            cols[f"h_home_pts{k}"][i] = roll(lh, k, "pts", "H")
            cols[f"h_home_gd{k}"][i] = roll(lh, k, "gd", "H")
            cols[f"a_away_pts{k}"][i] = roll(la, k, "pts", "A")
            cols[f"a_away_gd{k}"][i] = roll(la, k, "gd", "A")
        # season-to-date
        sp_h = season_pts[(h, sk[i])]
        sp_a = season_pts[(a, sk[i])]
        cols["h_ppg_sea"][i] = sp_h[0] / sp_h[2] if sp_h[2] else np.nan
        cols["a_ppg_sea"][i] = sp_a[0] / sp_a[2] if sp_a[2] else np.nan
        cols["h_gd_sea"][i] = sp_h[1] / sp_h[2] if sp_h[2] else np.nan
        cols["a_gd_sea"][i] = sp_a[1] / sp_a[2] if sp_a[2] else np.nan
        cols["h_gp_sea"][i] = sp_h[2]
        cols["a_gp_sea"][i] = sp_a[2]
        ld = league_draws[sk[i]]
        cols["league_draw_rate_sea"][i] = ld[0] / ld[1] if ld[1] >= 20 else np.nan
        # h2h
        key = (h, a) if h < a else (a, h)
        hh = h2h[key][-5:]
        if hh:
            pts = []
            tots = []
            for r in hh:
                hw = r["hs"] > r["as"]
                if r["home"] == h:
                    p = 3 if hw else (1 if r["hs"] == r["as"] else 0)
                else:
                    p = 3 if r["hs"] < r["as"] else (1 if r["hs"] == r["as"] else 0)
                pts.append(p)
                tots.append(r["hs"] + r["as"])
            cols["h2h_pts"][i] = np.mean(pts)
            cols["h2h_n"][i] = len(hh)
            cols["h2h_tot"][i] = np.mean(tots)
        # --- update state with this match ---
        gh, ga = hs[i], as_[i]
        res = 0.5 if gh == ga else (1.0 if gh > ga else 0.0)
        exp = 1 / (1 + 10 ** (-(eh + HA - ea) / 400))
        mult = (
            np.log(abs(gh - ga) + 1) * (2.2 / ((eh - ea) * 0.001 * (1 if gh > ga else -1) + 2.2)) if gh != ga else 1.0
        )
        delta = K * mult * (res - exp)
        elo[h] = eh + delta
        elo[a] = ea - delta
        rec_h = dict(
            date=d,
            gf=gh,
            ga=ga,
            gd=gh - ga,
            pts=3 if gh > ga else (1 if gh == ga else 0),
            venue="H",
            draw=int(gh == ga),
            over25=int(gh + ga > 2.5),
            tot=gh + ga,
            shots=None if np.isnan(h_shots[i]) else h_shots[i],
            shots_ag=None if np.isnan(a_shots[i]) else a_shots[i],
            sot=None if np.isnan(h_sot[i]) else h_sot[i],
            sot_ag=None if np.isnan(a_sot[i]) else a_sot[i],
            yc=None if np.isnan(h_yc[i]) else h_yc[i],
            rc=None if np.isnan(h_rc[i]) else h_rc[i],
            xg=None if np.isnan(h_xg[i]) else h_xg[i],
            xga=None if np.isnan(a_xg[i]) else a_xg[i],
            corn=None if np.isnan(h_corn[i]) else h_corn[i],
        )
        rec_a = dict(
            date=d,
            gf=ga,
            ga=gh,
            gd=ga - gh,
            pts=3 if ga > gh else (1 if gh == ga else 0),
            venue="A",
            draw=int(gh == ga),
            over25=int(gh + ga > 2.5),
            tot=gh + ga,
            shots=None if np.isnan(a_shots[i]) else a_shots[i],
            shots_ag=None if np.isnan(h_shots[i]) else h_shots[i],
            sot=None if np.isnan(a_sot[i]) else a_sot[i],
            sot_ag=None if np.isnan(h_sot[i]) else h_sot[i],
            yc=None if np.isnan(a_yc[i]) else a_yc[i],
            rc=None if np.isnan(a_rc[i]) else a_rc[i],
            xg=None if np.isnan(a_xg[i]) else a_xg[i],
            xga=None if np.isnan(h_xg[i]) else h_xg[i],
            corn=None if np.isnan(a_corn[i]) else a_corn[i],
        )
        lh.append(rec_h)
        la.append(rec_a)
        if len(lh) > 60:
            del lh[0]
        if len(la) > 60:
            del la[0]
        h2h[key].append(dict(home=h, hs=gh, **{"as": ga}))
        sp_h[0] += rec_h["pts"]
        sp_h[1] += gh - ga
        sp_h[2] += 1
        sp_a[0] += rec_a["pts"]
        sp_a[1] += ga - gh
        sp_a[2] += 1
        ld[0] += int(gh == ga)
        ld[1] += 1
        if i % 20000 == 0:
            print(i, flush=True)

    F = pd.DataFrame(dict(cols), index=df.index)
    out = pd.concat([df, F], axis=1)
    # diffs
    out["rest_diff"] = out.h_rest - out.a_rest
    out["games7_diff"] = out.h_games7 - out.a_games7
    out["games14_diff"] = out.h_games14 - out.a_games14
    for k in (5, 10, 20):
        out[f"pts{k}_diff"] = out[f"h_pts{k}"] - out[f"a_pts{k}"]
        out[f"gd{k}_diff"] = out[f"h_gd{k}"] - out[f"a_gd{k}"]
    out["venue_pts5_diff"] = out.h_home_pts5 - out.a_away_pts5
    out["venue_pts10_diff"] = out.h_home_pts10 - out.a_away_pts10
    out["venue_gd10_diff"] = out.h_home_gd10 - out.a_away_gd10
    out["ppg_sea_diff"] = out.h_ppg_sea - out.a_ppg_sea
    out["gd_sea_diff"] = out.h_gd_sea - out.a_gd_sea
    out["ewgd_diff"] = out.h_ewgd - out.a_ewgd
    out["draw20_mean"] = (out.h_draw20 + out.a_draw20) / 2
    out["over25_10_mean"] = (out.h_over25_10 + out.a_over25_10) / 2
    out["tot10_mean"] = (out.h_tot10 + out.a_tot10) / 2
    out["shots10_diff"] = out.h_shots10 - out.a_shots10
    out["sot10_diff"] = out.h_sot10 - out.a_sot10
    out["xg10_diff"] = out.h_xg10 - out.a_xg10
    out["xgd10_diff"] = (out.h_xg10 - out.h_xga10) - (out.a_xg10 - out.a_xga10)
    out["yc10_diff"] = out.h_yc10 - out.a_yc10
    out["corn10_diff"] = out.h_corn10 - out.a_corn10
    out["book_spread_h"] = (out.oh_max - out.oh_min) / out.oh
    out["book_spread_a"] = (out.oa_max - out.oa_min) / out.oa
    out["logit_ph"] = np.log(out.ph / (1 - out.ph))
    out["logit_pd"] = np.log(out.pd / (1 - out.pd))
    out["logit_pa"] = np.log(out.pa / (1 - out.pa))
    out["elo_minus_mkt"] = out.elo_p_home - out.ph / (
        out.ph + out.pa
    )  # elo home-win prob (draw-less) vs market draw-no-bet prob
    return out


def screen(df: pd.DataFrame, workdir: str) -> None:
    """Paired dBrier of each feature family vs market-only LR and prod-like GBM."""
    U = df[df.has_odds & (df.year >= 2012) & (df.h_n_hist >= 5) & (df.a_n_hist >= 5)].copy()
    print("universe", len(U), "test(2024+)", (U.year >= 2024).sum())
    Y = U.y.values

    def onehot(y):
        o = np.zeros((len(y), 3))
        o[np.arange(len(y)), y] = 1
        return o

    def brier_rows(P, y):
        return ((P - onehot(y)) ** 2).sum(axis=1)

    def ll_rows(P, y):
        return -np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))

    MKT = ["logit_ph", "logit_pd", "logit_pa"]
    PROD = [
        "oh",
        "od",
        "oa",
        "ph",
        "pd",
        "pa",
        "overround",
        "pou",
        "h_pts5",
        "h_gd5",
        "a_pts5",
        "a_gd5",
        "pts5_diff",
        "gd5_diff",
        "shots10_diff",
        "sot10_diff",
        "corn10_diff",
    ]
    GROUPS = {
        "rest_congestion": [
            "h_rest",
            "a_rest",
            "rest_diff",
            "h_games7",
            "a_games7",
            "games7_diff",
            "h_games14",
            "a_games14",
            "games14_diff",
        ],
        "elo": ["elo_diff", "elo_p_home", "elo_minus_mkt"],
        "venue_form": [
            "venue_pts5_diff",
            "venue_pts10_diff",
            "venue_gd10_diff",
            "h_home_pts10",
            "a_away_pts10",
            "h_home_gd10",
            "a_away_gd10",
        ],
        "long_form": ["pts10_diff", "pts20_diff", "gd10_diff", "gd20_diff", "ewgd_diff", "h_ewgd", "a_ewgd"],
        "season_stage": [
            "h_ppg_sea",
            "a_ppg_sea",
            "ppg_sea_diff",
            "gd_sea_diff",
            "h_gp_sea",
            "a_gp_sea",
            "days_into_season",
        ],
        "draw_propensity": ["draw20_mean", "h_draw20", "a_draw20", "league_draw_rate_sea"],
        "h2h": ["h2h_pts", "h2h_n", "h2h_tot"],
        "discipline": ["yc10_diff", "h_yc10", "a_yc10", "h_rc10", "a_rc10", "h_last_rc", "a_last_rc"],
        "market_micro": ["ih_sd", "ia_sd", "book_spread_h", "book_spread_a", "n_books", "overround"],
        "calendar": ["dow", "month", "hour"],
        "stats_form": [
            "shots10_diff",
            "sot10_diff",
            "corn10_diff",
            "h_shots10",
            "a_shots10",
            "h_sot_ag10",
            "a_sot_ag10",
        ],
        "xg_form": ["xg10_diff", "xgd10_diff", "h_xg10", "a_xg10"],
        "totals_form": ["over25_10_mean", "tot10_mean", "h_tot10", "a_tot10"],
    }
    GROUPS["all_new"] = sorted(set(sum([v for k, v in GROUPS.items() if k not in ("stats_form", "xg_form")], [])))

    def with_missing(X):
        X = X.copy()
        for c in list(X.columns):
            if X[c].isna().any():
                X[c + "_na"] = X[c].isna().astype(float)
        return X

    def fit_lr(Xtr, ytr, Xte):
        pipe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)
        )
        pipe.fit(Xtr, ytr)
        return pipe.predict_proba(Xte)

    def fit_lgb(Xtr, ytr, Xva, yva, Xte):
        m = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            verbose=-1,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.predict_proba(Xte), m

    tr = U.year <= 2022
    va = U.year == 2023
    te = U.year >= 2024
    ytr, yva, yte = Y[tr.values], Y[va.values], Y[te.values]
    res = []
    # market itself
    Pm = U.loc[te, ["ph", "pd", "pa"]].values
    b_m = brier_rows(Pm, yte)
    acc_m = (Pm.argmax(1) == yte).mean()
    print(
        f"MARKET (devig avg odds)  Brier={b_m.mean():.5f}  logloss={ll_rows(Pm, yte).mean():.5f}",
        f"acc={acc_m:.4f} n={len(yte)}",
    )
    # LR market-only
    P0 = fit_lr(U.loc[tr | va, MKT], Y[(tr | va).values], U.loc[te, MKT])
    b0 = brier_rows(P0, yte)
    print(f"LR market-only          Brier={b0.mean():.5f} d_vs_mkt={_pm(b0 - b_m)}")
    # prod-like LGBM
    X = with_missing(U[PROD])
    P1, m1 = fit_lgb(X[tr.values], ytr, X[va.values], yva, X[te.values])
    b1 = brier_rows(P1, yte)
    print(
        f"LGBM prod-like features Brier={b1.mean():.5f} d_vs_mkt={_pm(b1 - b_m)} d_vs_LR0={_pm(b1 - b0)}",
        f"iters={m1.best_iteration_}",
    )
    imp = pd.Series(m1.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("prod-like gain importance:", (imp / imp.sum()).round(3).head(12).to_dict())
    for g, cols in GROUPS.items():
        t0 = time.time()
        cov = U.loc[te, cols].notna().all(axis=1).mean()
        Xg = with_missing(U[list(dict.fromkeys(MKT + cols))])
        Pg = fit_lr(Xg[(tr | va).values], Y[(tr | va).values], Xg[te.values])
        bg = brier_rows(Pg, yte)
        d_lr = bg - b0
        Xg2 = with_missing(U[list(dict.fromkeys(PROD + cols))])
        Pg2, mg = fit_lgb(Xg2[tr.values], ytr, Xg2[va.values], yva, Xg2[te.values])
        bg2 = brier_rows(Pg2, yte)
        d_gb = bg2 - b1
        # per-season walk-forward sign check (LR, market+group vs market)
        signs = []
        for s in range(2019, 2027):
            trm = (U.year < s).values
            tem = (U.year == s).values
            if tem.sum() < 500:
                continue
            Pa = fit_lr(U.loc[trm, MKT], Y[trm], U.loc[tem, MKT])
            Pb = fit_lr(Xg[trm], Y[trm], Xg[tem])
            signs.append(round(float((brier_rows(Pb, Y[tem]) - brier_rows(Pa, Y[tem])).mean()), 5))
        impg = pd.Series(mg.booster_.feature_importance("gain"), index=Xg2.columns)
        share = float(impg[[c for c in impg.index if c in cols or c.replace("_na", "") in cols]].sum() / impg.sum())
        res.append(
            dict(
                group=g,
                n_feat=len(cols),
                coverage=round(cov, 3),
                dBrier_LR=round(d_lr.mean(), 5),
                se_LR=round(d_lr.std() / np.sqrt(len(d_lr)), 5),
                dBrier_GBM=round(d_gb.mean(), 5),
                se_GBM=round(d_gb.std() / np.sqrt(len(d_gb)), 5),
                gbm_gain_share=round(share, 3),
                wf_by_season=signs,
                neg_seasons=f"{sum(1 for x in signs if x < 0)}/{len(signs)}",
            )
        )
        print(
            f"{g:16s} cov={cov:.2f} LR d={_pm(d_lr)}  GBM d={_pm(d_gb)} gainshare={share:.3f}",
            f"wf={signs} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(workdir, "screen_results.csv"), index=False)
    print(R.to_string())


def market_scan(df: pd.DataFrame) -> None:
    """Realized-minus-implied by segment and flat-stake returns by implied band."""
    U = df[df.has_odds & (df.year >= 2012)].copy()
    U["fav"] = U[["ph", "pd", "pa"]].max(axis=1)
    U["home_fav"] = U.ph > U.pa
    U["hw"] = (U.y == 0).astype(float)
    U["dr"] = (U.y == 1).astype(float)
    U["aw"] = (U.y == 2).astype(float)

    def tab(g, by):
        t = g.groupby(by, observed=True).agg(
            n=("y", "size"),
            home_act=("hw", "mean"),
            home_imp=("ph", "mean"),
            draw_act=("dr", "mean"),
            draw_imp=("pd", "mean"),
            away_act=("aw", "mean"),
            away_imp=("pa", "mean"),
        )
        for s in ("home", "draw", "away"):
            t[f"{s}_gap"] = t[f"{s}_act"] - t[f"{s}_imp"]
            t[f"{s}_z"] = t[f"{s}_gap"] / np.sqrt(t[f"{s}_imp"] * (1 - t[f"{s}_imp"]) / t.n)
        return t.round(3)

    print("=== by league (realized minus implied; z>2.5 flagged) ===")
    t = tab(U, "league")
    t = t[t.n >= 1000]
    print(t[["n", "home_gap", "home_z", "draw_gap", "draw_z", "away_gap", "away_z"]].sort_values("home_z").to_string())
    print("\n=== by favourite-probability band ===")
    U["band"] = pd.cut(U.fav, [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0])
    print(tab(U, "band")[["n", "home_gap", "home_z", "draw_gap", "draw_z", "away_gap", "away_z"]].to_string())
    print("\n=== by year ===")
    print(tab(U, "year")[["n", "home_gap", "home_z", "draw_gap", "draw_z", "away_gap", "away_z"]].to_string())
    print("\n=== by season stage (days into season) ===")
    U["stage"] = pd.cut(U.days_into_season, [-1, 30, 90, 180, 270, 400])
    print(tab(U, "stage")[["n", "home_gap", "home_z", "draw_gap", "draw_z", "away_gap", "away_z"]].to_string())
    print("\n=== by rest-diff (home rest - away rest, days) ===")
    U["rd"] = pd.cut(U.rest_diff, [-31, -4, -2, -0.5, 0.5, 2, 4, 31])
    print(tab(U, "rd")[["n", "home_gap", "home_z", "draw_gap", "draw_z", "away_gap", "away_z"]].to_string())
    print("\n=== by n_books (odds source era) ===")
    print(tab(U, "n_books")[["n", "home_gap", "home_z", "draw_gap", "draw_z", "away_gap", "away_z"]].to_string())
    print("\n=== by overround band ===")
    U["ovr"] = pd.cut(U.overround, [0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.35])
    print(tab(U, "ovr")[["n", "home_gap", "home_z", "draw_gap", "draw_z", "away_gap", "away_z"]].to_string())
    # longshot: realized return of betting each outcome at avg odds by implied-prob bucket (flat stake)
    print("\n=== flat-stake return by implied prob of the selection (all three outcomes pooled) ===")
    rows = []
    for sel, p, o, act in (("home", "ph", "oh", "hw"), ("draw", "pd", "od", "dr"), ("away", "pa", "oa", "aw")):
        rows.append(pd.DataFrame({"p": U[p], "ret": U[act] * U[o] - 1, "sel": sel}))
    R = pd.concat(rows)
    R["b"] = pd.cut(R.p, [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    print(
        R.groupby("b", observed=True)
        .agg(n=("ret", "size"), roi=("ret", "mean"), se=("ret", lambda s: s.std() / np.sqrt(len(s))))
        .round(4)
        .to_string()
    )
    print(
        R.groupby(["sel", "b"], observed=True)
        .agg(n=("ret", "size"), roi=("ret", "mean"), se=("ret", lambda s: s.std() / np.sqrt(len(s))))
        .round(4)
        .to_string()
    )
    # Max (best) odds: does betting at best price make favourites +EV?
    print("\n=== best-price (Max) flat-stake return by implied band, 2019+ (n_books>=4 era) ===")
    V = U[(U.n_books >= 4) & (U.oh_max.notna())]
    rows = []
    for sel, p, o, act in (
        ("home", "ph", "oh_max", "hw"),
        ("draw", "pd", "od_max", "dr"),
        ("away", "pa", "oa_max", "aw"),
    ):
        rows.append(pd.DataFrame({"p": V[p], "ret": V[act] * V[o] - 1, "sel": sel}))
    R = pd.concat(rows)
    R["b"] = pd.cut(R.p, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    print(len(V))
    print(
        R.groupby("b", observed=True)
        .agg(n=("ret", "size"), roi=("ret", "mean"), se=("ret", lambda s: s.std() / np.sqrt(len(s))))
        .round(4)
        .to_string()
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="prod export (EXPORT_SQL)")
    ap.add_argument("--stage", choices=["engineer", "screen", "market", "all"], default="all")
    ap.add_argument("--workdir", default=".", help="where soccer_feat.parquet / screen_results.csv land")
    ap.add_argument("--print-sql", action="store_true", help="print EXPORT_SQL and exit")
    args = ap.parse_args(argv)
    if args.print_sql:
        print(EXPORT_SQL)
        return 0
    os.makedirs(args.workdir, exist_ok=True)
    feat_path = os.path.join(args.workdir, "soccer_feat.parquet")
    if args.stage in ("engineer", "all"):
        if not args.csv:
            ap.error("--csv is required for the engineer stage")
        df = engineer(load_export(args.csv))
        df.to_parquet(feat_path)
        print("engineered", df.shape, "->", feat_path)
    if args.stage in ("screen", "market", "all"):
        df = pd.read_parquet(feat_path)
        if args.stage in ("screen", "all"):
            screen(df, args.workdir)
        if args.stage in ("market", "all"):
            market_scan(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
