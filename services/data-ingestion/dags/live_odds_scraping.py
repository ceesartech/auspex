"""Airflow DAG for live odds scraping"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

from services.data_ingestion.src.core.config import Bet365Config, BetMGMConfig
from services.data_ingestion.src.core.database import DatabaseManager
from services.data_ingestion.src.scrapers.bet365_scraper import Bet365Scraper
from services.data_ingestion.src.scrapers.betmgm_scraper import BetMGMScraper
from redis import Redis

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'betting-system',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'live_odds_scraping',
    default_args=default_args,
    description='Scrape live odds every minute',
    schedule_interval='*/1 * * * *',  # Every minute
    catchup=False,
    tags=['scraping', 'odds', 'live']
)

def scrape_bet365():
    """Scrape Bet365 odds"""
    config = Bet365Config()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url)

    scraper = Bet365Scraper(config, db, redis)
    records = scraper.run()

    logger.info(f"Scraped {records} odds from Bet365")
    return records

def scrape_betmgm():
    """Scrape BetMGM odds"""
    config = BetMGMConfig()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url)

    scraper = BetMGMScraper(config, db, redis)
    records = scraper.run()

    logger.info(f"Scraped {records} odds from BetMGM")
    return records

# Tasks
bet365_task = PythonOperator(
    task_id='scrape_bet365_odds',
    python_callable=scrape_bet365,
    dag=dag
)

betmgm_task = PythonOperator(
    task_id='scrape_betmgm_odds',
    python_callable=scrape_betmgm,
    dag=dag
)

# Run in parallel
[bet365_task, betmgm_task]
