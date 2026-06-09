"""Weather-data automation DAG.

Two cadences:

  * DAILY at 03:00 UTC — `fetch_weather_visual_crossing.py` for each
    outdoor sport. Pulls forecasts for upcoming matches (next 14 days)
    and backfills any recently-finished matches that didn't get
    covered live. The script's WHERE EXISTS clause skips matches that
    already have a `vc_*` row, so daily volume is small after the
    initial backfill — typically 100-300 records/day across all
    sports, well under VC Pro's 100k/mo quota.

  * WEEKLY Sundays at 02:00 UTC — regenerates the soccer team →
    home-stadium fallback map from Wikidata, clears the `vc_unknown`
    sentinels, and re-runs the soccer backfill. This catches:
    (a) new teams that joined a league mid-season, (b) Wikidata
    label additions / aliases we've added since the last run,
    (c) any matches that hit `no_venue` on the daily pass because
    their home team wasn't yet in the map.

Both schedules are off-peak so they don't compete with the every-
15-min auspex_pipeline DAG for the api container's CPU.

Soccer is the most quota-heavy sport (~50 matches/day) so it's first
in the daily loop — runs early in case we ever hit a quota wall, we
get soccer (the busiest sport) first.

`fetch_weather.py` (the Open-Meteo fetcher) is intentionally NOT in
this DAG. Per memory `weather-features-attempted`, Open-Meteo's 11km
grid showed no signal on NFL or tennis. We're running VC as the
replacement; Open-Meteo data still in the DB stays for retrospective
A/B but doesn't get refreshed.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from alerting import notify_failure  # shared Telegram failure alerting

DOCKER_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T api"
POSTGRES_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T postgres"

default_args = {
    "owner": "auspex",
    "on_failure_callback": notify_failure,
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ── DAILY: fetch VC weather for upcoming + recent matches ────────


with DAG(
    dag_id="weather_daily_vc_fetch",
    description="Pull VC forecast + recent backfill for outdoor sports (daily 03:00 UTC)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["weather", "vc", "daily"],
) as daily_dag:

    # Soccer first because it's the highest-volume sport — if anything
    # is going to consume the VC quota it's this. NFL + tennis chain
    # after so a soccer failure doesn't block the rest.
    fetch_vc_soccer = BashOperator(
        task_id="fetch_vc_soccer",
        # 14 days lookahead for forecasts + 7-day backfill for any
        # recently-finished matches that didn't get covered live.
        bash_command=(
            f"{DOCKER_EXEC} python /app/scripts/fetch_weather_visual_crossing.py "
            "--sport soccer --days 14 --backfill-days 7"
        ),
    )

    fetch_vc_nfl = BashOperator(
        task_id="fetch_vc_nfl",
        bash_command=(
            f"{DOCKER_EXEC} python /app/scripts/fetch_weather_visual_crossing.py "
            "--sport nfl --days 14 --backfill-days 7"
        ),
        # ALL_DONE so an NFL failure doesn't cascade to tennis.
        trigger_rule=TriggerRule.ALL_DONE,
    )

    fetch_vc_tennis = BashOperator(
        task_id="fetch_vc_tennis",
        bash_command=(
            f"{DOCKER_EXEC} python /app/scripts/fetch_weather_visual_crossing.py "
            "--sport tennis --days 14 --backfill-days 7"
        ),
        trigger_rule=TriggerRule.ALL_DONE,
    )

    fetch_vc_soccer >> fetch_vc_nfl >> fetch_vc_tennis


# ── WEEKLY: regenerate the soccer team → stadium fallback map ─────


with DAG(
    dag_id="weather_weekly_stadium_refresh",
    description="Refresh soccer team→stadium map from Wikidata + re-run soccer VC backfill (weekly Sun 02:00 UTC)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 2 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["weather", "vc", "weekly"],
) as weekly_dag:

    # Step 1: regenerate the JSON from Wikidata. Writes DIRECTLY to
    # /app/data/ inside the api container — that path is bind-mounted
    # to /opt/auspex/data/ on the host, so the file appears on the host
    # automatically. (Prior versions of this task tried `docker compose
    # cp api:/tmp/... /opt/auspex/data/...` from inside the Airflow
    # worker, which doesn't work because the airflow worker doesn't
    # have /opt/auspex/data/ in its filesystem — the bind mount goes
    # to the api container, not the airflow container.) The `-w /app`
    # flag MUST come before the `api` service name; placing it after
    # treats it as part of the command to run inside the container,
    # which is what broke the previous version.
    regenerate_stadium_map = BashOperator(
        task_id="regenerate_stadium_map",
        bash_command=(
            "docker compose -f /opt/auspex/docker-compose.yml exec -T -w /app api "
            "python /app/scripts/build_soccer_stadium_map.py "
            "--output /app/data/soccer_team_stadiums.json"
        ),
    )

    # Step 2: clear the vc_unknown sentinels so the next backfill pass
    # retries matches whose home_team_id wasn't in the prior JSON
    # (new teams, newly-aliased teams). DELETE on data_kind='vc_unknown'
    # is safe — these rows have no real weather data, just a "tried"
    # marker. Forecast/actual rows are untouched.
    clear_vc_unknown_sentinels = BashOperator(
        task_id="clear_vc_unknown_sentinels",
        bash_command=(
            f"{POSTGRES_EXEC} sh -c "
            '\'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" '
            "-c \"DELETE FROM match_weather WHERE data_kind = '\\''vc_unknown'\\'';\"'"
        ),
    )

    # Step 3: full soccer backfill (365 days). Catches both the
    # cleared sentinels AND any historical matches whose home team
    # just got added to the JSON via new aliases. Quota-bounded by
    # the WHERE EXISTS skip on already-covered matches.
    soccer_backfill = BashOperator(
        task_id="soccer_backfill",
        bash_command=(
            f"{DOCKER_EXEC} python /app/scripts/fetch_weather_visual_crossing.py "
            "--sport soccer --days 14 --backfill-days 365"
        ),
    )

    regenerate_stadium_map >> clear_vc_unknown_sentinels >> soccer_backfill
