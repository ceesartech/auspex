"""Retry logic with exponential backoff"""

import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple
import random

logger = logging.getLogger(__name__)

def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None
):
    """Decorator for retrying with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise

                    # Calculate delay with jitter
                    delay = (backoff_factor ** retries) + random.uniform(0, 1)
                    logger.warning(
                        f"Attempt {retries}/{max_retries} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    if on_retry:
                        on_retry(retries, e)

                    time.sleep(delay)

            return None
        return wrapper
    return decorator
