"""Airflow DAG for lottery draw scraping"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

from services.data_ingestion.src.core.config import ScraperConfig
from services.data_ingestion.src.core.database import DatabaseManager
from services.data_ingestion.src.scrapers.lottery_scrapers import PowerballScraper, MegaMillionsScraper
from redis import Redis

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'betting-system',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'lottery_scraping',
    default_args=default_args,
    description='Scrape Powerball (Mon/Wed/Sat) and Mega Millions (Tue/Fri) results',
    schedule_interval='0 0 * * *',  # Daily at midnight
    catchup=False,
    max_active_runs=1,
    tags=['lottery', 'scraping']
)

def scrape_powerball():
    """Scrape Powerball draw results"""
    config = ScraperConfig()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url)

    scraper = PowerballScraper(config, db, redis)
    records = scraper.run()

    logger.info(f"Scraped {records} Powerball draws")
    return records

def scrape_megamillions():
    """Scrape Mega Millions draw results"""
    config = ScraperConfig()
    db = DatabaseManager(config)
    redis = Redis.from_url(config.redis_url)

    scraper = MegaMillionsScraper(config, db, redis)
    records = scraper.run()

    logger.info(f"Scraped {records} Mega Millions draws")
    return records

# Tasks
powerball_task = PythonOperator(
    task_id='scrape_powerball',
    python_callable=scrape_powerball,
    dag=dag
)

megamillions_task = PythonOperator(
    task_id='scrape_megamillions',
    python_callable=scrape_megamillions,
    dag=dag
)

# Run in parallel
[powerball_task, megamillions_task]
