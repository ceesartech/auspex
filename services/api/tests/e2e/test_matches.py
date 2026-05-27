"""E2E: matches endpoints with seeded match + odds rows."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import text


def _seed_match(db, *, match_date=None) -> tuple[str, str, str, str]:
    """Insert a league, two teams, a scheduled match, and one row of 1x2 odds.

    Returns (league_id, home_team_id, away_team_id, match_id).
    """
    if match_date is None:
        match_date = datetime.now(timezone.utc) + timedelta(days=2)

    league_id = db.execute(
        text(
            """
            INSERT INTO leagues (name, country, sport)
            VALUES ('E2E League', 'England', 'soccer')
            RETURNING id
            """
        )
    ).scalar_one()

    home_id = db.execute(
        text(
            """
            INSERT INTO teams (name, normalized_name, league_id, country, sport)
            VALUES ('E2E Home FC', 'e2e home fc', :lid, 'England', 'soccer')
            RETURNING id
            """
        ),
        {"lid": league_id},
    ).scalar_one()

    away_id = db.execute(
        text(
            """
            INSERT INTO teams (name, normalized_name, league_id, country, sport)
            VALUES ('E2E Away United', 'e2e away united', :lid, 'England', 'soccer')
            RETURNING id
            """
        ),
        {"lid": league_id},
    ).scalar_one()

    match_id = db.execute(
        text(
            """
            INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, season, status)
            VALUES (:lid, :h, :a, :d, '2025-2026', 'scheduled')
            RETURNING id
            """
        ),
        {"lid": league_id, "h": home_id, "a": away_id, "d": match_date},
    ).scalar_one()

    for selection, price in [("home", 1.85), ("draw", 3.50), ("away", 4.20)]:
        db.execute(
            text(
                """
                INSERT INTO odds (match_id, bookmaker, market_type, selection,
                                  odds_decimal, implied_probability, timestamp, is_live)
                VALUES (:mid, 'bet365', '1x2', :sel, :p, 1.0/:p, NOW(), false)
                """
            ),
            {"mid": match_id, "sel": selection, "p": price},
        )

    db.commit()
    return league_id, home_id, away_id, match_id


def test_upcoming_matches_returns_seeded_match(client, auth_headers, db):
    _, _, _, match_id = _seed_match(db)

    response = client.get("/api/v1/matches/upcoming", headers=auth_headers)
    assert response.status_code == 200, response.text

    matches = response.json()
    assert isinstance(matches, list)
    ids = {m["match_id"] for m in matches}
    assert str(match_id) in ids

    # Pick our match out of the response and check the shape.
    seeded = next(m for m in matches if m["match_id"] == str(match_id))
    assert seeded["league_name"] == "E2E League"
    assert seeded["home_team"] == "E2E Home FC"
    assert seeded["away_team"] == "E2E Away United"
    assert seeded["odds"]["home"] == 1.85
    assert seeded["odds"]["bookmaker"] == "bet365"


def test_upcoming_matches_filters_by_league(client, auth_headers, db):
    _seed_match(db)

    # An unrelated league should produce no matches.
    response = client.get(
        "/api/v1/matches/upcoming",
        params={"league": "Nonexistent Premier"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json() == []


def test_match_detail_returns_404_for_unknown_id(client, auth_headers, db):
    unknown = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/api/v1/matches/{unknown}", headers=auth_headers)
    assert response.status_code == 404


def test_past_match_is_excluded_from_upcoming(client, auth_headers, db):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    _, _, _, match_id = _seed_match(db, match_date=yesterday)

    response = client.get("/api/v1/matches/upcoming", headers=auth_headers)
    assert response.status_code == 200
    ids = {m["match_id"] for m in response.json()}
    assert str(match_id) not in ids
