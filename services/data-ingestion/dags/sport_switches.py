"""Per-sport on/off switches for the Airflow DAGs.

HORSE_RACING_ENABLED — the ONE switch for horse racing. False since
2026-11-01 (operator: "there is no current use for the horse racing").

False (dormant), in auspex_pipeline:
  * none of the seven horse-racing tasks exist — no racecards
    (load_racing_api --upcoming), no every-15-min results sweep, no
    features, no consensus / ranker predictions, no recommendations, no
    grade_completed_races task of its own;
  * ONE task, drain_horse_racing, settles what is still pending:
    `load_racing_api.py --drain` asks The Racing API for the result of each
    race that carries a pending rec, has been off 10 minutes and is at most
    load_racing_api.DRAIN_DAYS (3) old — one call per race, none at all
    when nothing is pending — then grade_completed_races.py settles them.
    Recs on races older than that are stranded and only logged; they are
    closed by scripts/ops/close_out_stranded_race_recs.sh (operator-run);
  * every other sport is untouched; send_pipeline_digest fans in the six
    other recommendation tasks.
Nothing else schedules horse racing: retrain_models has no horse bundle
(train_horse_racing_win.py runs by hand), monitor_models stops paging on
the horse streams once no race prediction has been created for 72 h
(monitor_models.HORSE_DORMANT_AFTER_HOURS; its findings are then logged
as warnings), and the API only lists races and recs, so it goes quiet on its
own. The frontend mirrors this constant in
services/frontend/src/lib/sport-switches.ts (HORSE_RACING_ENABLED): while
False it hides the Race cards nav entry, the /races pages show a "paused"
notice instead of empty lists, and the predictions sport filter and the
recommendations market filter stop offering horse racing.
tests/unit/test_frontend_sport_switches.py fails if the two differ.

True: the original branch — racecards >> results >> features >>
consensus >> ranker >> recommendations, and grade_completed_races.

Re-enable: set HORSE_RACING_ENABLED = True here AND in
services/frontend/src/lib/sport-switches.ts, and push to main. That is the
whole code change: no test pins either value (the mirror test only keeps the
two equal; the dormant and enabled states are each tested with the switch
forced, and the Playwright paused flows skip themselves). The DAG files
are bind-mounted, so the dag-processor re-parses within ~5 min; the next
tick fetches today's racecards (the Standard tier serves today only) and
recs follow within that tick. The frontend change ships with the next CI
image build. Before that, check:
  * the Racing API subscription still exists (THE_RACING_API_USER/PASS in
    the prod .env) — without it every horse task fails loudly;
  * the ranker in /app/models/horse_racing_ranker_v1 is as old as the
    dormancy; retrain with scripts/train_horse_racing_win.py if that is
    more than a few weeks;
  * rec_gating's horse_racing gates (scripts/rec_gating.py) still say what
    you want the streams to do.

Why a constant in git rather than an Airflow Variable: the dag-processor
re-reads a Variable on every parse (a DB round-trip in the parse loop), and
its value lives in no commit — this file is reviewed, versioned, and the
same on every machine that parses the DAGs.
"""

from __future__ import annotations

HORSE_RACING_ENABLED: bool = False
