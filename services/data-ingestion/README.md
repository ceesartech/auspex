# Data Ingestion Service

Web scraping and data collection service for the betting recommendation system.

## Scrapers

| Scraper | Source | Data Type | Schedule |
|---------|--------|-----------|----------|
| Bet365 | bet365.com | Odds | Every minute |
| BetMGM | betmgm.com | Odds | Every minute |
| FBref | fbref.com | Match stats | Daily |
| Understat | understat.com | xG data | Daily |
| Transfermarkt | transfermarkt.com | Player data | Daily |
| ESPN | espn.com | NFL/Boxing/MMA | Daily |
| NHL API | nhle.com | NHL data | Daily |
| Tennis | atptour.com | Tennis data | Daily |
| Horse Racing | horseracingnation.com | Racing data | Daily |
| Powerball | powerball.com | Lottery results | Daily |
| Mega Millions | megamillions.com | Lottery results | Daily |
| Weather | open-meteo.com | Weather data | Daily |

## Running Tests

```bash
pytest services/data-ingestion/tests/ -v
```
