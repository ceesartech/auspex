"""Airflow DAG for daily statistics scraping"""

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from redis import Redis
from services.data_ingestion.src.core.config import (
    FBrefConfig,
    ScraperConfig,
    TransfermarktConfig,
    UnderstatConfig,
)
from services.data_ingestion.src.core.database import DatabaseManager
from services.data_ingestion.src.scrapers.espn_scraper import ESPNScraper
from services.data_ingestion.src.scrapers.fbref_scraper import FBrefScraper
from services.data_ingestion.src.scrapers.horse_racing_scraper import HorseRacingScraper
from services.data_ingestion.src.scrapers.nhl_api_scraper import NHLAPIScraper
from services.data_ingestion.src.scrapers.tennis_scraper import TennisScraper
from services.data_ingestion.src.scrapers.transfermarkt_scraper import (
    TransfermarktScraper,
)
from services.data_ingestion.src.scrapers.understat_scraper import UnderstatScraper
from services.data_ingestion.src.scrapers.weather_scraper import WeatherScraper

logger = logging.getLogger(__name__)

default_args = {
    "owner": "betting-system",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "daily_stats_scraping",
    default_args=default_args,
    description="Daily scraping of match statistics from all sources",
    schedule_interval="0 6 * * *",  # 6 AM daily
    catchup=False,
    max_active_runs=1,
    tags=["stats", "daily", "scraping"],
)


def _create_scraper(config_class, scraper_class):
    """Helper to create a scraper with proper config, db, and redis"""
    config = config_class()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url)
    return scraper_class(config, db, redis)


def scrape_fbref():
    """Scrape match statistics from FBref"""
    scraper = _create_scraper(FBrefConfig, FBrefScraper)
    records = scraper.run()
    logger.info(f"Scraped {records} records from FBref")
    return records


def scrape_understat():
    """Scrape xG data from Understat"""
    scraper = _create_scraper(UnderstatConfig, UnderstatScraper)
    records = scraper.run()
    logger.info(f"Scraped {records} records from Understat")
    return records


def scrape_transfermarkt():
    """Scrape player data from Transfermarkt"""
    scraper = _create_scraper(TransfermarktConfig, TransfermarktScraper)
    records = scraper.run()
    logger.info(f"Scraped {records} records from Transfermarkt")
    return records


def scrape_espn():
    """Scrape NFL/Boxing/MMA data from ESPN"""
    scraper = _create_scraper(ScraperConfig, ESPNScraper)
    records = scraper.run()
    logger.info(f"Scraped {records} records from ESPN")
    return records


def scrape_nhl():
    """Scrape NHL data from official API"""
    scraper = _create_scraper(ScraperConfig, NHLAPIScraper)
    records = scraper.run()
    logger.info(f"Scraped {records} records from NHL API")
    return records


def scrape_tennis():
    """Scrape tennis data"""
    scraper = _create_scraper(ScraperConfig, TennisScraper)
    records = scraper.run()
    logger.info(f"Scraped {records} records from Tennis")
    return records


def scrape_horse_racing():
    """Scrape horse racing data"""
    scraper = _create_scraper(ScraperConfig, HorseRacingScraper)
    records = scraper.run()
    logger.info(f"Scraped {records} records from Horse Racing")
    return records


def scrape_weather():
    """Scrape weather data for upcoming matches"""
    scraper = _create_scraper(ScraperConfig, WeatherScraper)
    records = scraper.run()
    logger.info(f"Scraped {records} weather records")
    return records


# Tasks
fbref_task = PythonOperator(task_id="scrape_fbref", python_callable=scrape_fbref, dag=dag)
understat_task = PythonOperator(task_id="scrape_understat", python_callable=scrape_understat, dag=dag)
transfermarkt_task = PythonOperator(task_id="scrape_transfermarkt", python_callable=scrape_transfermarkt, dag=dag)
espn_task = PythonOperator(task_id="scrape_espn", python_callable=scrape_espn, dag=dag)
nhl_task = PythonOperator(task_id="scrape_nhl", python_callable=scrape_nhl, dag=dag)
tennis_task = PythonOperator(task_id="scrape_tennis", python_callable=scrape_tennis, dag=dag)
horse_racing_task = PythonOperator(task_id="scrape_horse_racing", python_callable=scrape_horse_racing, dag=dag)
weather_task = PythonOperator(task_id="scrape_weather", python_callable=scrape_weather, dag=dag)

# All stats scrapers run in parallel, weather runs after all
[fbref_task, understat_task, transfermarkt_task, espn_task, nhl_task, tennis_task, horse_racing_task] >> weather_task
