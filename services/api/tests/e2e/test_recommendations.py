"""E2E: betting recommendations endpoint with seeded prediction + recommendation."""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import text


def _seed_recommendation(db, *, confidence: str = "high", odds: float = 1.85) -> str:
    """Create a league/teams/match/prediction/betting_recommendation chain.

    Names get a per-call uuid suffix so multiple calls inside one test don't
    collide on UNIQUE(name, country, sport) for leagues or
    UNIQUE(normalized_name, sport) for teams.
    Returns the recommendation_id.
    """
    suffix = uuid.uuid4().hex[:8]

    league_id = db.execute(
        text("""
            INSERT INTO leagues (name, country, sport)
            VALUES (:name, 'Spain', 'soccer')
            RETURNING id
            """),
        {"name": f"Rec League {suffix}"},
    ).scalar_one()

    home_id = db.execute(
        text("""
            INSERT INTO teams (name, normalized_name, league_id, country, sport)
            VALUES (:n, :nn, :lid, 'Spain', 'soccer')
            RETURNING id
            """),
        {"n": f"Rec Home {suffix}", "nn": f"rec home {suffix}", "lid": league_id},
    ).scalar_one()

    away_id = db.execute(
        text("""
            INSERT INTO teams (name, normalized_name, league_id, country, sport)
            VALUES (:n, :nn, :lid, 'Spain', 'soccer')
            RETURNING id
            """),
        {"n": f"Rec Away {suffix}", "nn": f"rec away {suffix}", "lid": league_id},
    ).scalar_one()

    match_id = db.execute(
        text("""
            INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, season, status)
            VALUES (:lid, :h, :a, :d, '2025-2026', 'scheduled')
            RETURNING id
            """),
        {
            "lid": league_id,
            "h": home_id,
            "a": away_id,
            "d": datetime.now(timezone.utc) + timedelta(days=3),
        },
    ).scalar_one()

    prediction_id = db.execute(
        text("""
            INSERT INTO predictions (match_id, model_name, model_version, prediction_type,
                                     predicted_outcome, confidence, probabilities)
            VALUES (:mid, 'ensemble', 'v1.0', 'match_result', 'home', 0.72,
                    '{"home": 0.72, "draw": 0.18, "away": 0.10}'::jsonb)
            RETURNING id
            """),
        {"mid": match_id},
    ).scalar_one()

    recommendation_id = db.execute(
        text("""
            INSERT INTO betting_recommendations
                (prediction_id, match_id, bet_type, selection,
                 odds_at_recommendation, bookmaker, confidence_rating,
                 expected_value, recommended_stake, reasoning, status)
            VALUES (:pid, :mid, '1x2', 'home', :odds, 'bet365', :conf,
                    0.12, 25.0, 'Strong home form', 'pending')
            RETURNING id
            """),
        {"pid": prediction_id, "mid": match_id, "odds": odds, "conf": confidence},
    ).scalar_one()

    db.commit()
    return str(recommendation_id)


def test_recommendations_returns_seeded_recommendation(client, auth_headers, db):
    rec_id = _seed_recommendation(db)

    response = client.get("/api/v1/recommendations/", headers=auth_headers)
    assert response.status_code == 200, response.text

    recs = response.json()
    assert isinstance(recs, list)
    ids = {r.get("recommendation_id") or r.get("id") for r in recs}
    assert rec_id in ids


def test_recommendations_filters_by_confidence(client, auth_headers, db):
    high_id = _seed_recommendation(db, confidence="high")
    medium_id = _seed_recommendation(db, confidence="medium")

    high_only = client.get(
        "/api/v1/recommendations/",
        params={"confidence_level": "HIGH"},
        headers=auth_headers,
    )
    assert high_only.status_code == 200
    ids = {r.get("recommendation_id") or r.get("id") for r in high_only.json()}
    assert high_id in ids
    assert medium_id not in ids


def test_recommendations_filters_by_min_odds(client, auth_headers, db):
    short_id = _seed_recommendation(db, odds=1.50)
    long_id = _seed_recommendation(db, odds=2.50)

    response = client.get(
        "/api/v1/recommendations/",
        params={"min_odds": 2.0},
        headers=auth_headers,
    )
    assert response.status_code == 200
    ids = {r.get("recommendation_id") or r.get("id") for r in response.json()}
    assert long_id in ids
    assert short_id not in ids
