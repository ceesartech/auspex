#!/usr/bin/env python3
"""Print a model performance summary for the last 7 and 30 days."""

import os
import sys

# Allow importing from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import psycopg2
from redis import Redis

from monitoring.model_monitoring.performance_tracker import ModelPerformanceMonitor


def main() -> int:
    db = psycopg2.connect(os.environ['DATABASE_URL'])
    redis = Redis.from_url(os.environ['REDIS_URL'])
    monitor = ModelPerformanceMonitor(db, redis)

    for days, label in [(7, '7 Days'), (30, '30 Days')]:
        metrics = monitor.evaluate_recent_predictions(lookback_days=days)
        if metrics is None:
            print(f"No data available for the last {days} days")
            continue

        print()
        print('=' * 60)
        print(f'MODEL PERFORMANCE SUMMARY (Last {label})')
        print('=' * 60)
        print(f'Model Version     : {metrics.model_version}')
        print(f'Total Predictions : {metrics.total_predictions}')
        print(f'Correct           : {metrics.correct_predictions}')
        print(f'Accuracy          : {metrics.accuracy:.4f}  ({metrics.accuracy * 100:.2f}%)')
        print(f'Log Loss          : {metrics.log_loss_score:.4f}')
        print(f'Brier Score       : {metrics.brier_score:.4f}')
        print(f'ROI               : {metrics.roi:.4f}  ({metrics.roi * 100:.2f}%)')
        print(f'Avg Confidence    : {metrics.avg_confidence:.4f}')
        print(f'Calibration Error : {metrics.calibration_error:.4f}')
        print('=' * 60)

    issues = monitor.check_for_performance_issues()
    if issues:
        print('\nISSUES DETECTED:')
        for issue in issues:
            print(f'  - {issue}')
        return 1
    else:
        print('\nNo performance issues detected.')
        return 0


if __name__ == '__main__':
    sys.exit(main())
