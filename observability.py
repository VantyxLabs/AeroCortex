"""CloudWatch EMF metrics + structured logging (no-op when METRICS_ENABLED is not true)."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

_logger = logging.getLogger("aerocortex")


def metrics_enabled() -> bool:
    return (os.getenv("METRICS_ENABLED") or "false").strip().lower() in ("1", "true", "yes", "on")


def setup_logging() -> None:
    if not metrics_enabled():
        return
    try:
        from aws_lambda_powertools import Logger

        # Powertools configures the root logger for Lambda
        Logger(service="aerocortex")
    except Exception:
        pass


def emit_metric(name: str, value: float, unit: str = "Count", **dimensions: Any) -> None:
    if not metrics_enabled():
        return
    try:
        from aws_lambda_powertools import Metrics
        from aws_lambda_powertools.metrics import MetricUnit

        unit_map = {
            "Count": MetricUnit.Count,
            "Milliseconds": MetricUnit.Milliseconds,
            "Seconds": MetricUnit.Seconds,
        }
        metrics = Metrics(
            namespace=os.getenv("POWERTOOLS_METRICS_NAMESPACE") or "AeroCortex",
            service="aerocortex",
        )
        for k, v in dimensions.items():
            metrics.add_dimension(name=str(k), value=str(v))
        metrics.add_metric(name=name, unit=unit_map.get(unit, MetricUnit.Count), value=float(value))
        metrics.flush_metrics()
    except Exception:
        # Fallback: single_metric context
        try:
            from aws_lambda_powertools.metrics import single_metric, MetricUnit

            unit_map = {
                "Count": MetricUnit.Count,
                "Milliseconds": MetricUnit.Milliseconds,
            }
            with single_metric(
                name=name,
                unit=unit_map.get(unit, MetricUnit.Count),
                value=float(value),
                namespace=os.getenv("POWERTOOLS_METRICS_NAMESPACE") or "AeroCortex",
            ) as metric:
                for k, v in dimensions.items():
                    metric.add_dimension(name=str(k), value=str(v))
        except Exception as exc:
            _logger.debug("emit_metric skipped: %s", exc)
