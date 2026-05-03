#!/usr/bin/env python3
"""Manually trigger a model retraining job."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import psycopg2
from redis import Redis

from monitoring.model_monitoring.auto_retrainer import AutomatedRetrainer
from monitoring.model_monitoring.drift_detector import DriftDetector
from monitoring.model_monitoring.performance_tracker import ModelPerformanceMonitor


def main() -> int:
    parser = argparse.ArgumentParser(description='Trigger model retraining')
    parser.add_argument('--reason', default='Manual trigger', help='Reason for retraining')
    parser.add_argument('--auto', action='store_true', help='Only retrain if conditions are met')
    parser.add_argument(
        '--namespace', default=os.getenv('K8S_NAMESPACE', 'betting-system'),
        help='Kubernetes namespace',
    )
    args = parser.parse_args()

    db = psycopg2.connect(os.environ['DATABASE_URL'])
    redis = Redis.from_url(os.environ['REDIS_URL'])

    monitor = ModelPerformanceMonitor(db, redis)
    detector = DriftDetector(db, redis)
    retrainer = AutomatedRetrainer(db, redis, monitor, detector, namespace=args.namespace)

    if args.auto:
        result = retrainer.run()
    else:
        result = retrainer.trigger_retraining(args.reason)

    status = result.get('status', 'unknown')
    if status == 'triggered':
        print(f"Retraining triggered successfully.")
        print(f"  Retraining ID : {result.get('retraining_id')}")
        print(f"  Job name      : {result.get('job_name')}")
        print(f"  Reason        : {result.get('reason')}")
        return 0
    elif status == 'skipped':
        print(f"Retraining not required: {result.get('reason')}")
        return 0
    else:
        print(f"Unexpected status: {status}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
