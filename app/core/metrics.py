"""
Lightweight in-process metrics collection.

Provides simple counters and histograms without external dependencies.
In a production system these would be exported to Prometheus or Datadog.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Counter:
    """A monotonically increasing counter."""

    name: str
    _value: int = field(default=0, repr=False)

    def inc(self, amount: int = 1) -> None:
        self._value += amount

    @property
    def value(self) -> int:
        return self._value


@dataclass
class Histogram:
    """Records a distribution of observed values (e.g. latencies)."""

    name: str
    _observations: list[float] = field(default_factory=list, repr=False)

    def observe(self, value: float) -> None:
        self._observations.append(value)

    @property
    def count(self) -> int:
        return len(self._observations)

    @property
    def mean(self) -> float:
        if not self._observations:
            return 0.0
        return sum(self._observations) / len(self._observations)

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, pct: int) -> float:
        if not self._observations:
            return 0.0
        sorted_obs = sorted(self._observations)
        idx = int(len(sorted_obs) * pct / 100)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]

    def summary(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "mean_ms": round(self.mean, 2),
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
        }


class MetricsRegistry:
    """Central registry for all application metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._tags: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name=name)
        return self._counters[name]

    def histogram(self, name: str) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name=name)
        return self._histograms[name]

    def inc_tag(self, metric: str, tag: str) -> None:
        self._tags[metric][tag] += 1

    def snapshot(self) -> dict:
        return {
            "counters": {k: v.value for k, v in self._counters.items()},
            "histograms": {k: v.summary() for k, v in self._histograms.items()},
            "tags": {k: dict(v) for k, v in self._tags.items()},
        }


# Module-level singleton registry
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _registry


@contextmanager
def timed(metric_name: str) -> Generator[None, None, None]:
    """Context manager that records elapsed time (ms) into a histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _registry.histogram(metric_name).observe(elapsed_ms)
