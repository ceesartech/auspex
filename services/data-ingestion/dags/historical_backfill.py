"""Airflow DAG for historical data backfill (10+ years)"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

from services.data_ingestion.src.core.config import (
    FBrefConfig, UnderstatConfig, ScraperConfig
)
from services.data_ingestion.src.core.database import DatabaseManager
from services.data_ingestion.src.scrapers.fbref_scraper import FBrefScraper
from services.data_ingestion.src.scrapers.understat_scraper import UnderstatScraper
from services.data_ingestion.src.scrapers.espn_scraper import ESPNScraper
from services.data_ingestion.src.scrapers.nhl_api_scraper import NHLAPIScraper
from services.data_ingestion.src.scrapers.tennis_scraper import TennisScraper
from services.data_ingestion.src.scrapers.lottery_scrapers import PowerballScraper, MegaMillionsScraper
from redis import Redis

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'betting-system',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'historical_backfill',
    default_args=default_args,
    description='One-time backfill of historical data (10+ years)',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=['backfill', 'historical']
)

def _create_scraper(config_class, scraper_class):
    """Helper to create a scraper with proper config, db, and redis"""
    config = config_class()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url)
    return scraper_class(config, db, redis)

def backfill_soccer_data():
    """Backfill historical soccer data from FBref (10+ years)"""
    config = FBrefConfig()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url)
    scraper = FBrefScraper(config, db, redis)

    total = 0
    current_year = datetime.now().year
    for year in range(current_year - 10, current_year + 1):
        try:
            # Override season for historical scraping
            for league in config.leagues:
                season = f"{year}-{year + 1}"
                league_url = f"{config.base_url}/en/comps/9/{season}/{season}-{league}-Stats"
                logger.info(f"Backfilling {league} {season}")
                count = scraper._scrape_league(league)
                total += count
        except Exception as e:
            logger.error(f"Error backfilling {year}: {e}")
            continue

    logger.info(f"Soccer backfill complete: {total} records")
    return total

def backfill_nfl_data():
    """Backfill historical NFL data from ESPN"""
    scraper = _create_scraper(ScraperConfig, ESPNScraper)
    records = scraper.run()
    logger.info(f"NFL backfill complete: {records} records")
    return records

def backfill_nhl_data():
    """Backfill historical NHL data"""
    scraper = _create_scraper(ScraperConfig, NHLAPIScraper)
    records = scraper.run()
    logger.info(f"NHL backfill complete: {records} records")
    return records

def backfill_tennis_data():
    """Backfill historical tennis data"""
    scraper = _create_scraper(ScraperConfig, TennisScraper)
    records = scraper.run()
    logger.info(f"Tennis backfill complete: {records} records")
    return records

def backfill_lottery_data():
    """Backfill historical lottery draw results"""
    config = ScraperConfig()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url)

    pb_scraper = PowerballScraper(config, db, redis)
    mm_scraper = MegaMillionsScraper(config, db, redis)

    pb_records = pb_scraper.run()
    mm_records = mm_scraper.run()

    total = pb_records + mm_records
    logger.info(f"Lottery backfill complete: {total} records")
    return total

# Tasks
soccer_task = PythonOperator(task_id='backfill_soccer', python_callable=backfill_soccer_data, dag=dag)
nfl_task = PythonOperator(task_id='backfill_nfl', python_callable=backfill_nfl_data, dag=dag)
nhl_task = PythonOperator(task_id='backfill_nhl', python_callable=backfill_nhl_data, dag=dag)
tennis_task = PythonOperator(task_id='backfill_tennis', python_callable=backfill_tennis_data, dag=dag)
lottery_task = PythonOperator(task_id='backfill_lottery', python_callable=backfill_lottery_data, dag=dag)

# All backfill tasks run in parallel
[soccer_task, nfl_task, nhl_task, tennis_task, lottery_task]
