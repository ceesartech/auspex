"""Re-validate the SHIPPED cross-book features on v2 (strictly pre-game) odds.

The original ab_*_cross_book.py harnesses compute their own bare-named
cross-book columns and drop only those from the control arm; since the
features shipped, the production frames already carry them as feature__*,
so the old harness control would contain them too. Here: production frame
(features recomputed from v2 odds 2026-10-30) with vs without the shipped
feature__ cross-book keys, identical XGB params to the harnesses,
season walk-forward, PAIRED per-row Brier deltas with SE.
Brier = sklearn binary (p - y)^2, the scale the original claims used.

Result 2026-10-30 (audit §4 "Cross-book odds-dispersion features"): none of
the five shipped wins survives on v2 odds — every pooled delta is inside noise.

    docker compose exec -T api python /app/scripts/ab_cross_book_v2_recheck.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/app/services/ml-models/src")
from utils import training_data as td  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

XGB = dict(
    objective="binary:logistic",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=400,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.01,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    eval_metric="logloss",
)


def keys(*names):
    return tuple(f"feature__{n}" for n in names)


NFL_TOTAL = keys(
    "total_book_count", "total_consensus_mean", "total_max_minus_min", "total_std", "total_consensus_implied_prob"
)
NFL_ML = keys("ml_book_count", "ml_consensus_home_prob", "ml_max_minus_min_home_prob", "ml_std_home_prob")
SPREAD = keys(
    "spread_book_count", "spread_consensus_mean", "spread_max_minus_min", "spread_std", "spread_consensus_implied_prob"
)
TOTAL = NFL_TOTAL
SPECS = [
    # label, loader, non-feature set, target, own family, every shipped cross-book key of the sport, claim
    (
        "NFL total",
        "load_nfl_total_frame",
        "NFL_TOTAL_NON_FEATURE_COLUMNS",
        "NFL_TOTAL_TARGET",
        NFL_TOTAL,
        NFL_TOTAL + NFL_ML,
        -0.0110,
    ),
    (
        "NFL moneyline",
        "load_nfl_moneyline_frame",
        "NFL_MONEYLINE_NON_FEATURE_COLUMNS",
        "NFL_MONEYLINE_TARGET",
        NFL_ML,
        NFL_TOTAL + NFL_ML,
        -0.0154,
    ),
    (
        "NBA spread",
        "load_nba_spread_frame",
        "NBA_SPREAD_NON_FEATURE_COLUMNS",
        "NBA_SPREAD_TARGET",
        SPREAD,
        SPREAD,
        -0.0061,
    ),
    (
        "NHL puck line",
        "load_nhl_puck_line_frame",
        "NHL_PUCK_LINE_NON_FEATURE_COLUMNS",
        "NHL_PUCK_LINE_TARGET",
        SPREAD,
        SPREAD + TOTAL,
        -0.0186,
    ),
    (
        "NHL total",
        "load_nhl_total_frame",
        "NHL_TOTAL_NON_FEATURE_COLUMNS",
        "NHL_TOTAL_TARGET",
        TOTAL,
        SPREAD + TOTAL,
        -0.0231,
    ),
]
LEAKY = {"home_score", "away_score", "total_points", "margin"}


def fit_proba(train, test, cols, target):
    clf = XGBClassifier(**XGB)
    clf.fit(train[cols].values, train[target].astype(int).values)
    return clf.predict_proba(test[cols].values)[:, 1]


def paired(y, p_a, p_b):
    d = (p_b - y) ** 2 - (p_a - y) ** 2
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d


url = os.environ["DATABASE_URL"]
only = sys.argv[1:]  # optional label filter
for label, loader, nonfeat, tgt, own, allcb, claim in SPECS:
    if only and not any(o.lower() in label.lower() for o in only):
        continue
    frame = getattr(td, loader)(database_url=url)
    target = getattr(td, tgt)
    frame = frame[frame[target].notna()].copy()
    excluded = set(getattr(td, nonfeat)) | {target}
    numeric = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    prod = [c for c in numeric if c not in excluded]
    leak = sorted(set(prod) & LEAKY)
    missing = [k for k in allcb if k not in frame.columns]
    no_own = [c for c in prod if c not in own]
    no_all = [c for c in prod if c not in allcb]
    print(
        f"\n=== {label}: {len(frame)} rows, prod features {len(prod)}, -own {len(no_own)}, -all {len(no_all)}"
        f"{'  LEAK ' + str(leak) if leak else ''}{'  MISSING ' + str(missing) if missing else ''}",
        flush=True,
    )
    bc, sd = own[0], [k for k in own if k.endswith("_std") or k.endswith("std_home_prob")][0]
    seasons = sorted(frame["season"].dropna().unique())
    disp = frame.groupby("season").agg(
        n=(target, "size"), books=(bc, "mean"), sdv=(sd, "mean"), neutral=(bc, lambda s: float((s <= 1).mean()))
    )
    print("dispersion by season (mean books, mean " + sd.replace("feature__", "") + ", share at 1 book):")
    print(
        "  "
        + "; ".join(
            f"{s}: n={r.n} books={r.books:.1f} std={r.sdv:.3f} 1book={r.neutral:.0%}" for s, r in disp.iterrows()
        ),
        flush=True,
    )
    pooled_own, pooled_all = [], []
    for s in seasons:
        train, test = frame[frame["season"] < s], frame[frame["season"] == s]
        if len(test) < 150 or train["season"].nunique() < 2:
            continue
        y = test[target].astype(int).values
        p_prod = fit_proba(train, test, prod, target)
        p_own = fit_proba(train, test, no_own, target)
        d_own, se_own, dd_own = paired(y, p_own, p_prod)
        line = (
            f"  test {s}: n={len(test)} brier no-own={np.mean((p_own - y) ** 2):.5f} "
            f"prod={np.mean((p_prod - y) ** 2):.5f}  dOWN={d_own:+.5f}±{se_own:.5f} (z {d_own / se_own:+.2f})"
        )
        ref = np.mean((train[target].astype(int).mean() - y) ** 2)
        line += f"  [const={ref:.5f}"
        if "moneyline" in label and "feature__implied_prob_home_ml" in test.columns:
            m = 1.0 - test["feature__implied_prob_home_ml"].values
            line += f" market={np.mean((m - y) ** 2):.5f}"
        line += "]"
        pooled_own.append(dd_own)
        if set(allcb) != set(own):
            p_all = fit_proba(train, test, no_all, target)
            d_all, se_all, dd_all = paired(y, p_all, p_prod)
            line += f"  dALL={d_all:+.5f}±{se_all:.5f}"
            pooled_all.append(dd_all)
        print(line, flush=True)
    for name, pool in (("own family", pooled_own), ("all cross-book", pooled_all)):
        if pool:
            d = np.concatenate(pool)
            print(
                f"  POOLED {name}: n={len(d)} dBrier={d.mean():+.5f}±{d.std(ddof=1) / np.sqrt(len(d)):.5f} "
                f"(claim {claim:+.4f})",
                flush=True,
            )
