#!/usr/bin/env python3
"""Validate that all monitoring components are healthy post-deployment."""

import os
import sys
import time
from typing import List, Tuple

import requests


CHECKS: List[Tuple[str, str]] = [
    ('Prometheus', os.getenv('PROMETHEUS_URL', 'http://prometheus:9090/-/healthy')),
    ('Grafana', os.getenv('GRAFANA_URL', 'http://grafana:3000/api/health')),
    ('Alertmanager', os.getenv('ALERTMANAGER_URL', 'http://alertmanager:9093/-/healthy')),
    ('Betting API', os.getenv('API_URL', 'http://betting-api:8000/health')),
    ('Loki', os.getenv('LOKI_URL', 'http://loki:3100/ready')),
    ('Pushgateway', os.getenv('PUSHGATEWAY_URL', 'http://pushgateway:9091/-/healthy')),
]

TIMEOUT = 5
RETRIES = 3
RETRY_DELAY = 2


def check_endpoint(name: str, url: str) -> Tuple[bool, str]:
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code < 400:
                return True, f'OK ({response.status_code})'
            return False, f'HTTP {response.status_code}'
        except requests.RequestException as exc:
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return False, str(exc)
    return False, 'Retries exhausted'


def main() -> int:
    print('Validating monitoring deployment...\n')
    failures = 0

    for name, url in CHECKS:
        ok, detail = check_endpoint(name, url)
        status = 'PASS' if ok else 'FAIL'
        print(f'  [{status}] {name:<20} {detail}')
        if not ok:
            failures += 1

    print()
    if failures:
        print(f'{failures} check(s) failed.')
        return 1

    print('All monitoring components healthy.')

    # Verify Prometheus targets
    prom_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
    try:
        resp = requests.get(f'{prom_url}/api/v1/targets', timeout=TIMEOUT)
        data = resp.json()
        active_targets = data.get('data', {}).get('activeTargets', [])
        down_targets = [t for t in active_targets if t.get('health') != 'up']
        print(f'\nPrometheus: {len(active_targets)} targets, {len(down_targets)} down')
        for t in down_targets:
            print(f'  DOWN: {t.get("labels", {}).get("job")} - {t.get("lastError")}')
        if down_targets:
            return 1
    except requests.RequestException as exc:
        print(f'Could not query Prometheus targets: {exc}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
