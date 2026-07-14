#!/usr/bin/env python3
"""Validate project setup and configuration"""

import os
import sys
from pathlib import Path
import psycopg2
from redis import Redis
from dotenv import load_dotenv

def check_environment_variables():
    """Check if all required environment variables are set"""
    required_vars = [
        'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB',
        'DATABASE_URL', 'REDIS_URL',
        'AIRFLOW__CORE__FERNET_KEY', 'AIRFLOW__WEBSERVER__SECRET_KEY',
        'JWT_SECRET', 'USER_DOB'
    ]

    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"FAIL Missing environment variables: {', '.join(missing)}")
        return False

    print("PASS All required environment variables are set")
    return True

def check_database_connection():
    """Check PostgreSQL connection and schema"""
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        required_tables = ['leagues', 'teams', 'matches', 'odds', 'predictions']
        missing_tables = [t for t in required_tables if t not in tables]

        if missing_tables:
            print(f"FAIL Missing database tables: {', '.join(missing_tables)}")
            print("   Run: make db-migrate")
            return False

        cursor.close()
        conn.close()
        print(f"PASS Database connected ({len(tables)} tables found)")
        return True
    except Exception as e:
        print(f"FAIL Database connection failed: {e}")
        return False

def check_redis_connection():
    """Check Redis connection"""
    try:
        redis_client = Redis.from_url(os.getenv('REDIS_URL'))
        redis_client.ping()
        print("PASS Redis connected")
        return True
    except Exception as e:
        print(f"FAIL Redis connection failed: {e}")
        return False

def check_directory_structure():
    """Check if all required directories exist"""
    required_dirs = [
        'services/data-ingestion/src/core',
        'services/feature-engineering',
        'services/ml-models',
        'services/api',
        'data/raw',
        'data/processed',
        'data/models',
        'docs',
        'monitoring'
    ]

    project_root = Path(__file__).parent.parent
    missing = []

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing.append(dir_path)

    if missing:
        print(f"FAIL Missing directories: {', '.join(missing)}")
        return False

    print("PASS All required directories exist")
    return True

def check_required_files():
    """Check if critical files exist"""
    required_files = [
        'requirements.txt',
        'requirements-dev.txt',
        '.env.example',
        '.gitignore',
        'docker-compose.yml',
        'Makefile',
        'pytest.ini',
        'setup.py',
        'services/data-ingestion/db/migrations/001_create_schema.sql',
        'services/data-ingestion/db/migrations/002_create_indexes.sql',
        'services/data-ingestion/db/migrations/003_create_functions.sql',
        'services/data-ingestion/src/core/config.py',
        'services/data-ingestion/src/core/database.py',
    ]

    project_root = Path(__file__).parent.parent
    missing = []

    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing.append(file_path)

    if missing:
        print(f"FAIL Missing files: {', '.join(missing)}")
        return False

    print(f"PASS All {len(required_files)} required files exist")
    return True

def main():
    """Run all validation checks"""
    print("=" * 60)
    print("BETTING SYSTEM SETUP VALIDATION")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Environment Variables", check_environment_variables),
        ("Database Connection", check_database_connection),
        ("Redis Connection", check_redis_connection),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\nChecking {check_name}...")
        results.append(check_func())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"ALL CHECKS PASSED ({passed}/{total}) - Setup is complete!")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"SOME CHECKS FAILED ({passed}/{total}) - Please fix the issues above")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
