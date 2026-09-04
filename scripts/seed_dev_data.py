#!/usr/bin/env python3
"""Seed the local database with realistic development data."""

import os
import random
import sys
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://betting_user:change_me_in_production_MUST_BE_STRONG@localhost:5432/betting_system",
)

TEAMS = [
    ("Man City", "ENG", "Premier League"),
    ("Arsenal", "ENG", "Premier League"),
    ("Liverpool", "ENG", "Premier League"),
    ("Chelsea", "ENG", "Premier League"),
    ("Man United", "ENG", "Premier League"),
    ("Real Madrid", "ESP", "La Liga"),
    ("Barcelona", "ESP", "La Liga"),
    ("Atletico Madrid", "ESP", "La Liga"),
    ("Bayern Munich", "GER", "Bundesliga"),
    ("Borussia Dortmund", "GER", "Bundesliga"),
]

OUTCOMES = ["home_win", "draw", "away_win"]


def connect():
    return psycopg2.connect(DATABASE_URL)


def seed_leagues(cur):
    leagues = list({(t[2], t[1]) for t in TEAMS})
    execute_values(
        cur,
        """
        INSERT INTO leagues (league_name, country, sport_type)
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        [(name, country, "soccer") for name, country in leagues],
    )
    print(f"  Seeded {len(leagues)} leagues")


def seed_teams(cur):
    execute_values(
        cur,
        """
        INSERT INTO teams (team_name, country, league_name)
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        [(name, country, league) for name, country, league in TEAMS],
    )
    print(f"  Seeded {len(TEAMS)} teams")


def seed_matches(cur, count: int = 100):
    cur.execute("SELECT team_id, team_name FROM teams LIMIT 20")
    teams = cur.fetchall()
    if not teams:
        print("  No teams found — skipping matches")
        return

    rows = []
    now = datetime.now(timezone.utc)
    for i in range(count):
        home, away = random.sample(teams, 2)
        match_date = now + timedelta(days=random.randint(-30, 14))
        status = "finished" if match_date < now else "scheduled"
        outcome = random.choice(OUTCOMES) if status == "finished" else None
        home_goals = random.randint(0, 5) if status == "finished" else None
        away_goals = random.randint(0, 5) if status == "finished" else None

        if outcome == "home_win":
            home_goals = max(home_goals, away_goals + 1)
        elif outcome == "away_win":
            away_goals = max(away_goals, home_goals + 1)
        elif outcome == "draw":
            away_goals = home_goals

        rows.append(
            (
                home[0],
                away[0],
                match_date.date(),
                status,
                outcome,
                home_goals,
                away_goals,
            )
        )

    execute_values(
        cur,
        """
        INSERT INTO matches
          (home_team_id, away_team_id, match_date, status,
           actual_outcome, home_goals, away_goals)
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        rows,
    )
    print(f"  Seeded {count} matches")


def seed_predictions(cur):
    cur.execute(
        """
        SELECT match_id FROM matches
        WHERE status IN ('finished','scheduled')
        LIMIT 80
    """
    )
    match_ids = [r[0] for r in cur.fetchall()]

    rows = []
    for match_id in match_ids:
        probs = [random.random() for _ in range(3)]
        total = sum(probs)
        ph, pd, pa = [p / total for p in probs]
        outcome = OUTCOMES[probs.index(max(probs))]
        confidence = max(probs) / total
        rows.append(
            (
                match_id,
                outcome,
                round(ph, 4),
                round(pd, 4),
                round(pa, 4),
                round(confidence, 4),
                "ensemble_v1.0",
            )
        )

    execute_values(
        cur,
        """
        INSERT INTO predictions
          (match_id, predicted_outcome, probability_home, probability_draw,
           probability_away, confidence, model_version)
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        rows,
    )
    print(f"  Seeded {len(rows)} predictions")


def seed_model_performance(cur):
    rows = []
    for days_ago in range(30, 0, -1):
        eval_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
        rows.append(
            (
                eval_date,
                "ensemble_v1.0",
                "ensemble",
                7,
                random.randint(50, 200),  # total
                random.randint(25, 130),  # correct
                round(random.uniform(0.52, 0.68), 4),  # accuracy
                round(random.uniform(0.9, 1.1), 4),  # log_loss
                round(random.uniform(0.18, 0.26), 4),  # brier
                round(random.uniform(-0.05, 0.12), 4),  # roi
                round(random.uniform(0.60, 0.80), 4),  # avg_confidence
                round(random.uniform(0.02, 0.08), 4),  # calibration_error
            )
        )

    execute_values(
        cur,
        """
        INSERT INTO model_performance_logs
          (evaluation_date, model_version, model_type, evaluation_period_days,
           total_predictions, correct_predictions, accuracy, log_loss,
           brier_score_avg, roi, avg_confidence, calibration_error)
        VALUES %s
        ON CONFLICT DO NOTHING
    """,
        rows,
    )
    print(f"  Seeded {len(rows)} model performance records")


def seed_user(cur):
    import hashlib

    # Simple demo user (password: "demo1234")
    pw_hash = hashlib.sha256(b"demo1234").hexdigest()
    cur.execute(
        """
        INSERT INTO users (username, email, password_hash, bankroll, timezone)
        VALUES ('demo', 'demo@betting-system.local', %s, 1000.00, 'America/Denver')
        ON CONFLICT (email) DO NOTHING
    """,
        (pw_hash,),
    )
    print("  Seeded demo user  (demo / demo1234)")


def main():
    print("Seeding development data...")
    try:
        conn = connect()
        conn.autocommit = False
        cur = conn.cursor()

        seed_leagues(cur)
        seed_teams(cur)
        seed_matches(cur)
        seed_predictions(cur)
        seed_model_performance(cur)
        seed_user(cur)

        conn.commit()
        print("Done.")
    except psycopg2.errors.UndefinedTable as exc:
        print(f"Schema not ready yet ({exc}). Run migrations first.")
        sys.exit(1)
    except Exception as exc:
        print(f"Seed failed: {exc}")
        sys.exit(1)
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
