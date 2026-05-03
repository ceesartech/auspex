"""Weather data scraper for match conditions"""

import json
import logging
from typing import List, Dict
from datetime import datetime

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class WeatherScraper(BaseScraper):
    """Scrape weather data from Open-Meteo (free, no API key needed)"""

    API_URL = "https://api.open-meteo.com/v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scrape(self) -> int:
        """Scrape weather data for upcoming matches"""
        total = 0

        # Get upcoming matches that need weather data
        upcoming_matches = self._get_matches_needing_weather()

        for match in upcoming_matches:
            try:
                weather = self._fetch_weather(
                    latitude=match.get('latitude'),
                    longitude=match.get('longitude'),
                    date=match.get('match_date')
                )
                if weather:
                    self._save_weather(match['match_id'], weather)
                    total += 1
            except Exception as e:
                logger.error(f"Error fetching weather for match {match.get('match_id')}: {e}")
                continue

        return total

    def _get_matches_needing_weather(self) -> List[Dict]:
        """Get upcoming matches without weather data"""
        query = """
            SELECT m.id as match_id, m.match_date, m.venue,
                   v.latitude, v.longitude
            FROM matches m
            LEFT JOIN weather w ON w.match_id = m.id
            LEFT JOIN venues v ON v.name = m.venue
            WHERE m.status = 'scheduled'
                AND m.match_date > NOW()
                AND m.match_date < NOW() + INTERVAL '7 days'
                AND w.id IS NULL
                AND v.latitude IS NOT NULL
            LIMIT 50
        """
        result = self.db.execute_query(query, fetch=True)
        return result if result else []

    def _fetch_weather(self, latitude: float, longitude: float, date: str) -> Dict:
        """Fetch weather data from Open-Meteo API"""
        if not latitude or not longitude:
            return None

        url = (
            f"{self.API_URL}/forecast?"
            f"latitude={latitude}&longitude={longitude}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            f"wind_direction_10m,precipitation,visibility,pressure_msl,"
            f"weather_code"
            f"&start_date={date}&end_date={date}"
            f"&timezone=auto"
        )

        logger.info(f"Fetching weather for ({latitude}, {longitude}) on {date}")

        html = self._fetch_page(url)

        try:
            data = json.loads(html)
            hourly = data.get('hourly', {})

            if not hourly or not hourly.get('temperature_2m'):
                return None

            # Get weather at match time (approximate to nearest hour)
            # Default to afternoon (index 15 = 3pm)
            idx = 15 if len(hourly.get('temperature_2m', [])) > 15 else 0

            weather_code = hourly.get('weather_code', [0])[idx]
            condition = self._map_weather_code(weather_code)

            return {
                'temperature_celsius': hourly.get('temperature_2m', [None])[idx],
                'humidity_pct': hourly.get('relative_humidity_2m', [None])[idx],
                'wind_speed_kmh': hourly.get('wind_speed_10m', [None])[idx],
                'wind_direction': str(hourly.get('wind_direction_10m', [None])[idx]),
                'precipitation_mm': hourly.get('precipitation', [None])[idx],
                'weather_condition': condition,
                'visibility_km': hourly.get('visibility', [None])[idx],
                'pressure_hpa': hourly.get('pressure_msl', [None])[idx],
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse weather JSON: {e}")
            return None

    def _map_weather_code(self, code: int) -> str:
        """Map WMO weather code to condition string"""
        code_map = {
            0: 'clear',
            1: 'mainly_clear', 2: 'partly_cloudy', 3: 'overcast',
            45: 'fog', 48: 'depositing_rime_fog',
            51: 'light_drizzle', 53: 'moderate_drizzle', 55: 'dense_drizzle',
            56: 'freezing_drizzle', 57: 'freezing_drizzle',
            61: 'slight_rain', 63: 'moderate_rain', 65: 'heavy_rain',
            66: 'freezing_rain', 67: 'freezing_rain',
            71: 'slight_snow', 73: 'moderate_snow', 75: 'heavy_snow',
            77: 'snow_grains',
            80: 'slight_rain_shower', 81: 'moderate_rain_shower', 82: 'violent_rain_shower',
            85: 'slight_snow_shower', 86: 'heavy_snow_shower',
            95: 'thunderstorm', 96: 'thunderstorm_hail', 99: 'thunderstorm_heavy_hail',
        }
        return code_map.get(code, 'unknown')

    def _save_weather(self, match_id: str, weather: Dict):
        """Save weather data to database"""
        checksum = self._calculate_checksum({
            'match_id': str(match_id),
            **weather
        })

        if self._is_duplicate(checksum):
            return

        self.db.upsert(
            'weather',
            {
                'match_id': match_id,
                'temperature_celsius': weather.get('temperature_celsius'),
                'humidity_pct': weather.get('humidity_pct'),
                'wind_speed_kmh': weather.get('wind_speed_kmh'),
                'wind_direction': weather.get('wind_direction'),
                'precipitation_mm': weather.get('precipitation_mm'),
                'weather_condition': weather.get('weather_condition'),
                'visibility_km': weather.get('visibility_km'),
                'pressure_hpa': weather.get('pressure_hpa'),
                'source': 'open-meteo',
                'recorded_at': datetime.utcnow(),
            },
            conflict_columns=['match_id'],
            update_columns=[
                'temperature_celsius', 'humidity_pct', 'wind_speed_kmh',
                'wind_direction', 'precipitation_mm', 'weather_condition',
                'visibility_km', 'pressure_hpa', 'recorded_at'
            ]
        )
