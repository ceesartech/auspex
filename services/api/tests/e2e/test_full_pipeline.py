"""End-to-end pipeline test: full user journey through the API.

Walks through the same sequence a real user (and the upstream services
that feed the database) would produce, exercising every endpoint that
matters for day-to-day operation:

    1. login (POST /api/v1/user/login)
    2. simulate scraper output: insert league + teams + match + odds
    3. browse matches (GET /api/v1/matches/upcoming)
    4. simulate ML output: insert prediction
    5. simulate recommender output: insert betting_recommendation rows
    6. browse recommendations (GET /api/v1/recommendations/)
    7. set bankroll (PUT /api/v1/user/preferences)
    8. record placed bets (POST /api/v1/user/betting-history)
    9. summary (GET /api/v1/user/dashboard)

Direct DB seeding stands in for the data-ingestion / feature-engineering
/ ml-models services. They have their own integration tests; here we
prove the API ties everything together end-to-end.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import text


def _seed_match_with_odds(db, *, days_ahead: int = 2) -> dict:
    """Insert league + 2 teams + 1 scheduled match + 1x2 odds. Returns ids."""
    ids = {}

    ids["league_id"] = db.execute(
        text(
            """
            INSERT INTO leagues (name, country, sport)
            VALUES ('Pipeline League', 'England', 'soccer')
            RETURNING id
            """
        )
    ).scalar_one()

    ids["home_id"] = db.execute(
        text(
            """
            INSERT INTO teams (name, normalized_name, league_id, country, sport)
            VALUES ('Pipeline Home', 'pipeline home', :lid, 'England', 'soccer')
            RETURNING id
            """
        ),
        {"lid": ids["league_id"]},
    ).scalar_one()

    ids["away_id"] = db.execute(
        text(
            """
            INSERT INTO teams (name, normalized_name, league_id, country, sport)
            VALUES ('Pipeline Away', 'pipeline away', :lid, 'England', 'soccer')
            RETURNING id
            """
        ),
        {"lid": ids["league_id"]},
    ).scalar_one()

    match_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
    ids["match_id"] = db.execute(
        text(
            """
            INSERT INTO matches (league_id, home_team_id, away_team_id,
                                 match_date, season, status)
            VALUES (:lid, :h, :a, :d, '2025-2026', 'scheduled')
            RETURNING id
            """
        ),
        {"lid": ids["league_id"], "h": ids["home_id"], "a": ids["away_id"], "d": match_date},
    ).scalar_one()

    for selection, price in [("home", 1.85), ("draw", 3.40), ("away", 4.50)]:
        db.execute(
            text(
                """
                INSERT INTO odds (match_id, bookmaker, market_type, selection,
                                  odds_decimal, implied_probability, timestamp, is_live)
                VALUES (:mid, 'bet365', '1x2', :sel, :p, 1.0/:p, NOW(), false)
                """
            ),
            {"mid": ids["match_id"], "sel": selection, "p": price},
        )

    db.commit()
    return ids


def _seed_prediction_and_recommendation(db, ids: dict) -> dict:
    """Insert a prediction + matching betting_recommendation. Returns row ids."""
    ids["prediction_id"] = db.execute(
        text(
            """
            INSERT INTO predictions (match_id, model_name, model_version,
                                     prediction_type, predicted_outcome,
                                     confidence, probabilities)
            VALUES (:mid, 'ensemble', 'v1.0', 'match_result', 'home', 0.71,
                    '{"home": 0.71, "draw": 0.19, "away": 0.10}'::jsonb)
            RETURNING id
            """
        ),
        {"mid": ids["match_id"]},
    ).scalar_one()

    ids["recommendation_id"] = db.execute(
        text(
            """
            INSERT INTO betting_recommendations
                (prediction_id, match_id, bet_type, selection,
                 odds_at_recommendation, bookmaker, confidence_rating,
                 expected_value, recommended_stake, reasoning, status)
            VALUES (:pid, :mid, '1x2', 'home', 1.85, 'bet365', 'high',
                    0.15, 30.0, 'Pipeline E2E recommendation', 'pending')
            RETURNING id
            """
        ),
        {"pid": ids["prediction_id"], "mid": ids["match_id"]},
    ).scalar_one()

    db.commit()
    return ids


def test_full_user_journey(client, auth_headers, db):
    # ── 1. Auth ───────────────────────────────────────────────────────────
    # auth_headers fixture already proved login works.
    me = client.get("/api/v1/user/preferences", headers=auth_headers)
    assert me.status_code == 200

    # ── 2. Scraper-equivalent seeding ──────────────────────────────────────
    ids = _seed_match_with_odds(db)

    # ── 3. Browse matches ─────────────────────────────────────────────────
    upcoming = client.get("/api/v1/matches/upcoming", headers=auth_headers)
    assert upcoming.status_code == 200
    matches = upcoming.json()
    assert any(m["match_id"] == str(ids["match_id"]) for m in matches)

    seeded = next(m for m in matches if m["match_id"] == str(ids["match_id"]))
    assert seeded["odds"]["home"] == 1.85
    assert seeded["odds"]["bookmaker"] == "bet365"

    # ── 4 & 5. ML + recommendation pipeline ────────────────────────────────
    ids = _seed_prediction_and_recommendation(db, ids)

    # ── 6. Browse recommendations ─────────────────────────────────────────
    recs_resp = client.get("/api/v1/recommendations/", headers=auth_headers)
    assert recs_resp.status_code == 200
    recs = recs_resp.json()
    rec_ids = {r.get("recommendation_id") or r.get("id") for r in recs}
    assert str(ids["recommendation_id"]) in rec_ids

    # ── 7. Set bankroll ───────────────────────────────────────────────────
    pref_resp = client.put(
        "/api/v1/user/preferences",
        json={"bankroll": 1000.0, "risk_tolerance": "MEDIUM"},
        headers=auth_headers,
    )
    assert pref_resp.status_code in (200, 204)

    prefs = client.get("/api/v1/user/preferences", headers=auth_headers).json()
    assert prefs["bankroll"]["value"] == 1000.0

    # ── 8. Record a bet against the recommendation ────────────────────────
    bet_resp = client.post(
        "/api/v1/user/betting-history",
        json={
            "recommendation_id": str(ids["recommendation_id"]),
            "match_id": str(ids["match_id"]),
            "bookmaker": "bet365",
            "bet_type": "1x2",
            "selection": "home",
            "odds": 1.85,
            "stake": 50.0,
        },
        headers=auth_headers,
    )
    assert bet_resp.status_code in (200, 201), bet_resp.text
    assert bet_resp.json().get("status") == "recorded"

    # ── 9. Dashboard reflects the activity ────────────────────────────────
    dash = client.get("/api/v1/user/dashboard", headers=auth_headers)
    assert dash.status_code == 200
    body = dash.json()
    assert body["total_bets"] == 1
    assert body["total_staked"] == 50.0
    # The recommendation we seeded is still pending → counts as active.
    assert body["active_recommendations"] >= 1
    assert body["upcoming_matches"] >= 1


def test_recommendation_filters_compose(client, auth_headers, db):
    """Multiple recommendations with different attributes; query filters compose."""
    ids = _seed_match_with_odds(db, days_ahead=3)

    # Seed two predictions with two recommendations of different confidence.
    pred_a = db.execute(
        text(
            """
            INSERT INTO predictions (match_id, model_name, model_version,
                                     prediction_type, predicted_outcome,
                                     confidence, probabilities)
            VALUES (:mid, 'ensemble', 'v1.0', 'match_result', 'home', 0.80,
                    '{"home": 0.80, "draw": 0.12, "away": 0.08}'::jsonb)
            RETURNING id
            """
        ),
        {"mid": ids["match_id"]},
    ).scalar_one()

    high_rec = db.execute(
        text(
            """
            INSERT INTO betting_recommendations
                (prediction_id, match_id, bet_type, selection,
                 odds_at_recommendation, bookmaker, confidence_rating,
                 expected_value, recommended_stake, reasoning, status)
            VALUES (:pid, :mid, '1x2', 'home', 2.20, 'bet365', 'high',
                    0.20, 40.0, 'high-confidence', 'pending')
            RETURNING id
            """
        ),
        {"pid": pred_a, "mid": ids["match_id"]},
    ).scalar_one()

    low_rec = db.execute(
        text(
            """
            INSERT INTO betting_recommendations
                (prediction_id, match_id, bet_type, selection,
                 odds_at_recommendation, bookmaker, confidence_rating,
                 expected_value, recommended_stake, reasoning, status)
            VALUES (:pid, :mid, 'over_under', 'over_2.5', 1.55, 'betmgm', 'low',
                    0.05, 10.0, 'low-confidence', 'pending')
            RETURNING id
            """
        ),
        {"pid": pred_a, "mid": ids["match_id"]},
    ).scalar_one()

    db.commit()

    # confidence + min_odds filters together: only the high_rec qualifies.
    filtered = client.get(
        "/api/v1/recommendations/",
        params={"confidence_level": "HIGH", "min_odds": 2.0},
        headers=auth_headers,
    )
    assert filtered.status_code == 200
    ids_seen = {r.get("recommendation_id") or r.get("id") for r in filtered.json()}
    assert str(high_rec) in ids_seen
    assert str(low_rec) not in ids_seen
