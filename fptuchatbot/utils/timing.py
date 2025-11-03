"""
Performance timing utilities for monitoring execution time.
"""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from fptuchatbot.utils.logging import get_logger

logger = get_logger(__name__)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "operation", log: bool = True):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
            log: Whether to log the elapsed time
        """
        self.name = name
        self.log = log
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        """Start timer."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timer and optionally log."""
        self.elapsed = time.perf_counter() - self._start
        if self.log:
            logger.info(f"{self.name} completed in {self.elapsed:.4f}s")

    def __repr__(self) -> str:
        """String representation."""
        return f"Timer(name='{self.name}', elapsed={self.elapsed:.4f}s)"


@contextmanager
def timer(name: str = "operation") -> Generator[Timer, None, None]:
    """
    Context manager for timing operations.

    Usage:
        with timer("my_operation") as t:
            # do something
            pass
        print(f"Took {t.elapsed:.2f}s")

    Args:
        name: Name of the operation

    Yields:
        Timer instance
    """
    t = Timer(name, log=True)
    try:
        t.__enter__()
        yield t
    finally:
        t.__exit__()


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log execution time of a function.

    Usage:
        @log_execution_time
        def my_function():
            pass

    Args:
        func: Function to decorate

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} executed in {elapsed:.4f}s")

    return wrapper


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2.5s", "1m 30s", "1h 5m")
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

