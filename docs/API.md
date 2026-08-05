# API Reference

## Base URLs

| Environment | URL |
|---|---|
| Local | `http://localhost:8000` |
| Production | `https://api.YOUR_DOMAIN.com` |

## Interactive docs

- Swagger UI: `{BASE_URL}/docs`
- ReDoc: `{BASE_URL}/redoc`
- OpenAPI JSON: `{BASE_URL}/openapi.json`

---

## Authentication

All protected endpoints require a **Bearer token** in the `Authorization` header.

### Login

```
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "yourpassword"
}
```

**Response 200:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

**Errors:** `401` invalid credentials, `422` validation error

### Using the token

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/predictions
```

### Refresh token

```
POST /api/v1/auth/refresh
Authorization: Bearer <current_token>
```

---

## Health & Observability

### Health check

```
GET /health
```

**Response 200:**
```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "models_loaded": true,
  "version": "1.0.0"
}
```

### Prometheus metrics

```
GET /metrics
```

Returns Prometheus text format. Metrics include:

| Metric | Type | Description |
|---|---|---|
| `http_requests_total` | Counter | Requests by endpoint, method, status |
| `http_request_duration_seconds` | Histogram | Latency by endpoint |
| `prediction_confidence` | Histogram | Distribution of prediction confidence |
| `predictions_total` | Counter | Predictions by outcome and model version |
| `model_accuracy_7d` | Gauge | Rolling 7-day accuracy |
| `model_roi_7d` | Gauge | Rolling 7-day ROI |
| `model_calibration_error` | Gauge | Expected calibration error |
| `drift_score` | Gauge | Feature drift score (0–1) |
| `active_connections` | Gauge | Active HTTP connections |
| `active_websocket_connections` | Gauge | Active WebSocket connections |
| `cache_hits_total` | Counter | Redis cache hits |
| `cache_requests_total` | Counter | Total cache lookups |

---

## Predictions

### List predictions

```
GET /api/v1/predictions
Authorization: Bearer <token>
```

**Query parameters:**

| Parameter | Type | Description |
|---|---|---|
| `sport` | string | Filter: `soccer`, `nfl`, `nhl`, `tennis`, `horse_racing`, `mma` |
| `league_id` | integer | Filter by league |
| `date` | string | Match date (YYYY-MM-DD) |
| `min_confidence` | float | Minimum confidence threshold (0.0–1.0) |
| `limit` | integer | Page size (default: 50, max: 200) |
| `offset` | integer | Pagination offset |

**Response 200:**
```json
[
  {
    "prediction_id": 1234,
    "match_id": 5678,
    "match_date": "2026-03-10",
    "home_team": "Man City",
    "away_team": "Arsenal",
    "predicted_outcome": "home_win",
    "probability_home": 0.52,
    "probability_draw": 0.25,
    "probability_away": 0.23,
    "confidence": 0.72,
    "model_version": "ensemble_v1.0",
    "shap_values": {
      "home_form_5": 0.12,
      "h2h_home_wins": 0.08,
      "odds_movement": -0.05
    },
    "prediction_timestamp": "2026-03-09T14:00:00Z"
  }
]
```

### Get prediction by ID

```
GET /api/v1/predictions/{prediction_id}
Authorization: Bearer <token>
```

**Response 200:** Single prediction object (same schema as above).
**Response 404:** `{ "detail": "Prediction not found" }`

---

## Recommendations

### List recommendations

```
GET /api/v1/recommendations
Authorization: Bearer <token>
```

**Query parameters:**

| Parameter | Type | Description |
|---|---|---|
| `confidence_level` | string | `LOW`, `MEDIUM`, `HIGH`, `VERY_HIGH` |
| `market_type` | string | `1X2`, `over_under_2.5`, `btts`, `asian_handicap` |
| `active_only` | boolean | Default `true` — exclude resolved bets |
| `limit` | integer | Default: 20, max: 100 |

**Response 200:**
```json
[
  {
    "recommendation_id": 99,
    "match_id": 5678,
    "predicted_outcome": "home_win",
    "recommended_outcome": "home_win",
    "recommended_odds": 1.85,
    "recommended_stake": 25.00,
    "kelly_fraction": 0.025,
    "expected_value": 0.043,
    "confidence": 0.72,
    "confidence_level": "HIGH",
    "market_type": "1X2",
    "bookmaker": "Bet365",
    "is_active": true,
    "expires_at": "2026-03-10T15:00:00Z"
  }
]
```

---

## Matches

### Upcoming matches

```
GET /api/v1/matches/upcoming
Authorization: Bearer <token>
```

**Query parameters:** `sport`, `league_id`, `date`, `limit`, `offset`

**Response 200:**
```json
[
  {
    "match_id": 5678,
    "home_team_id": 10,
    "away_team_id": 11,
    "home_team": "Man City",
    "away_team": "Arsenal",
    "league": "Premier League",
    "match_date": "2026-03-10T15:00:00Z",
    "status": "scheduled",
    "venue": "Etihad Stadium",
    "odds": {
      "home": 1.85,
      "draw": 3.50,
      "away": 4.20
    },
    "prediction": {
      "predicted_outcome": "home_win",
      "confidence": 0.72
    }
  }
]
```

### Match detail

```
GET /api/v1/matches/{match_id}
Authorization: Bearer <token>
```

Returns full match object including odds history, prediction, and recommendation.

---

## Betting History

### Get history

```
GET /api/v1/betting/history
Authorization: Bearer <token>
```

**Query parameters:** `limit`, `offset`, `from_date`, `to_date`

**Response 200:**
```json
{
  "total_bets": 42,
  "total_staked": 420.00,
  "total_profit": 38.50,
  "roi": 0.0917,
  "win_rate": 0.619,
  "bets": [
    {
      "bet_id": 1,
      "match_id": 5678,
      "outcome": "home_win",
      "stake": 25.00,
      "odds": 1.85,
      "profit": 21.25,
      "status": "won",
      "placed_at": "2026-03-09T14:30:00Z",
      "settled_at": "2026-03-10T17:00:00Z"
    }
  ]
}
```

### Record a bet

```
POST /api/v1/betting/record
Authorization: Bearer <token>
Content-Type: application/json

{
  "match_id": 5678,
  "outcome": "home_win",
  "stake": 25.00,
  "odds": 1.85,
  "bookmaker": "Bet365"
}
```

**Response 201:** Created bet object.

---

## Model Performance

### Current performance metrics

```
GET /api/v1/models/performance
Authorization: Bearer <token>
```

**Response 200:**
```json
{
  "model_version": "ensemble_v1.0",
  "period_days": 7,
  "total_predictions": 143,
  "correct_predictions": 89,
  "accuracy": 0.6224,
  "log_loss": 0.9814,
  "brier_score": 0.2217,
  "roi": 0.0731,
  "avg_confidence": 0.6850,
  "calibration_error": 0.0412,
  "last_evaluated": "2026-03-06T06:00:00Z"
}
```

### Drift status

```
GET /api/v1/models/drift-status
Authorization: Bearer <token>
```

**Response 200:**
```json
{
  "timestamp": "2026-03-06T06:00:00Z",
  "drift_score": 0.18,
  "features_with_drift": ["odds_movement", "home_form_5"],
  "total_features_checked": 47,
  "retraining_recommended": false
}
```

---

## Lottery

Draws are uniform and independent — nothing here forecasts numbers. The
decision-relevant endpoint is `/ev`; generation optimizes statistical profile
and jackpot-share avoidance (conditional EV), never win probability.

### Recent draws

```
GET /api/v1/lottery/draws?game=powerball&limit=20
Authorization: Bearer <token>
```

### Expected value (the honest product)

```
GET /api/v1/lottery/ev?game=powerball
GET /api/v1/lottery/ev?game=mega_millions&jackpot=500000000&state_tax=0.0685
Authorization: Bearer <token>
```

Omit `jackpot` to use the live next-draw estimate (advertised + actual cash
value); `422` when the live fetch fails — pass `jackpot` explicitly.

**Response 200 (abridged):**
```json
{
  "game": "powerball",
  "ticket_price": 2.0,
  "advertised_jackpot": 786000000,
  "cash_value": 341600000,
  "jackpot_odds": 292201338,
  "expected_co_winners": 0.23,
  "share_factor": 0.89,
  "ev_total": 0.73,
  "expected_loss_pct": 63.0,
  "breakeven_advertised_jackpot": null,
  "verdict": "Don't play on EV grounds: ...",
  "disclaimer": "Lottery draws are random and independent — ..."
}
```

### Tracked backtest ledger

```
GET /api/v1/lottery/lines?game=powerball&limit=48
Authorization: Bearer <token>
```

Lines the daily `lottery_pipeline` generated for upcoming draws (one per
strategy per draw), with settlement (`matched_main`, `matched_bonus`,
`prize_tier`, `settled_at`) once the target draw's numbers land. Hit rates
are expected to track pure chance — the ledger proves it honestly.

### Hot / cold / overdue analysis (entertainment)

```
GET /api/v1/lottery/analysis?game=powerball&num_draws=100
Authorization: Bearer <token>
```

### Generate combinations

```
POST /api/v1/lottery/recommendations?game=powerball&strategy=ev&num_sets=5
Authorization: Bearer <token>
```

Strategies: `blend | statistical | ev | hot | due | random`. `ev` biases
toward rarely-picked numbers (>31, no sequences) to reduce expected jackpot
splitting — it does NOT improve win odds. The response carries `warnings`
when draw history is too thin for the hot/due/profile statistics.

---

## WebSocket — Live Odds

```
ws://localhost:8000/ws/odds
```

**Authentication:** Pass the JWT in the `Authorization` header on the initial handshake, or as a query parameter:

```
ws://localhost:8000/ws/odds?token=<jwt>
```

**Message format (server → client):**
```json
{
  "type": "odds_update",
  "match_id": 5678,
  "bookmaker": "Bet365",
  "market": "1X2",
  "odds": {
    "home": 1.82,
    "draw": 3.55,
    "away": 4.30
  },
  "timestamp": "2026-03-09T14:01:23Z"
}
```

**Message types:** `odds_update`, `match_start`, `match_end`, `prediction_update`

---

## Error Responses

All errors follow RFC 7807:

```json
{
  "detail": "Human-readable error description",
  "status_code": 400,
  "error_code": "VALIDATION_ERROR"
}
```

| Status | Meaning |
|---|---|
| 400 | Bad request / validation error |
| 401 | Missing or invalid JWT |
| 403 | Authenticated but not authorised |
| 404 | Resource not found |
| 422 | Unprocessable entity (Pydantic validation) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service temporarily unavailable |

---

## Rate Limits

| User type | Limit |
|---|---|
| Authenticated | 100 requests / minute |
| Unauthenticated | 20 requests / minute |
| WebSocket connections | 10 concurrent per user |

Rate limit headers are returned on every response:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1709900460
```
