# Feature Engineering Service

Computes 250+ ML features across 7 categories for soccer match prediction.

## Architecture

Two-phase computation with parallel execution:
- **Phase 1**: 6 independent categories run in parallel via ThreadPoolExecutor
- **Phase 2**: Derived features computed from Phase 1 outputs

### Feature Categories

| Category | Prefix | Count | Description |
|----------|--------|-------|-------------|
| Team Performance | `team__` | ~130 | Form, goals, xG, shots across 4 windows |
| Head-to-Head | `h2h__` | 22 | Historical matchup patterns |
| Player Metrics | `player__` | 32 | Squad value, availability, key player form |
| Contextual | `ctx__` | 42 | Venue, rest days, referee, standings |
| Market Odds | `odds__` | 26 | Implied probabilities, movements, consensus |
| Temporal | `temporal__` | 15 | Day, month, season phase (cyclical encoding) |
| Derived | `derived__` | ~95 | Elo, Poisson, momentum, interactions |

### Feature Naming Convention

```
{category}__{subcategory}__{metric}__{window}
```

Examples:
- `team__home__form__wins__last5`
- `h2h__home_win_rate`
- `derived__poisson__home_win`

## Usage

```python
from src.core.config import FeatureConfig
from src.core.database import DatabaseManager
from src.orchestrator import RealTimeFeatureComputer
from redis import Redis

config = FeatureConfig()
db = DatabaseManager(config)
redis = Redis.from_url(config.redis_url)

computer = RealTimeFeatureComputer(config, db, redis)
features = computer.compute_features(match_id="<uuid>")
```

## Caching

Dual-layer cache:
- **Redis**: 1-hour TTL, fast reads
- **PostgreSQL**: 24-hour TTL, persistent

Cache key format: `features:{version}:{match_id}:{feature_set}`

## Testing

```bash
cd services/feature-engineering
pip install -r requirements.txt
pytest tests/ -v
```
