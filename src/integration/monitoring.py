"""
Monitoring and Observability for RAG System

Provides metrics, logging, and tracing for the RAG pipeline.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available - using basic monitoring")

logger = logging.getLogger(__name__)


@dataclass
class SearchMetric:
    """Metrics for a search operation."""
    query: str
    search_type: str
    duration_ms: float
    result_count: int
    cache_hit: bool
    filters_used: int
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class IngestionMetric:
    """Metrics for ingestion operations."""
    operation: str  # 'add', 'update', 'delete'
    product_count: int
    duration_ms: float
    success: bool
    timestamp: datetime
    error: Optional[str] = None


class RAGMonitor:
    """
    Comprehensive monitoring for the RAG system.
    
    Features:
    - Search performance tracking
    - Ingestion monitoring
    - Error tracking
    - Cache performance
    - Custom metrics
    - Alerting thresholds
    """
    
    def __init__(
        self,
        brand_domain: str,
        enable_tracing: bool = True,
        metrics_export_interval: int = 60
    ):
        self.brand_domain = brand_domain
        self.enable_tracing = enable_tracing and OTEL_AVAILABLE
        
        # Metrics storage
        self.metrics_dir = Path(f"accounts/{brand_domain}/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics
        self.search_metrics = deque(maxlen=10000)
        self.ingestion_metrics = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        
        # Performance buckets
        self.latency_buckets = [50, 100, 200, 500, 1000, 2000, 5000]  # ms
        
        # Alerting thresholds
        self.alert_thresholds = {
            'search_latency_p95': 1000,  # ms
            'search_error_rate': 0.05,    # 5%
            'cache_hit_rate_min': 0.3,    # 30%
            'ingestion_error_rate': 0.1   # 10%
        }
        
        # Alert callbacks
        self.alert_handlers: List[Callable] = []
        
        # Initialize OpenTelemetry if available
        if self.enable_tracing:
            self._init_opentelemetry()
        
        # Background metrics aggregation
        self._start_background_tasks()
        
        logger.info(f"📊 Initialized RAG Monitor for {brand_domain}")
    
    @contextmanager
    def track_search(
        self,
        query: str,
        search_type: str = "hybrid",
        cache_hit: bool = False,
        filters_used: int = 0
    ):
        """
        Context manager to track search operations.
        
        Usage:
            with monitor.track_search(query, "hybrid") as tracker:
                results = search_engine.search(query)
                tracker.set_result_count(len(results))
        """
        start_time = time.time()
        tracker = SearchTracker()
        
        try:
            yield tracker
            
            # Record success
            duration_ms = (time.time() - start_time) * 1000
            
            metric = SearchMetric(
                query=query[:100],  # Truncate for storage
                search_type=search_type,
                duration_ms=duration_ms,
                result_count=tracker.result_count,
                cache_hit=cache_hit,
                filters_used=filters_used,
                timestamp=datetime.now()
            )
            
            self._record_search_metric(metric)
            
        except Exception as e:
            # Record error
            duration_ms = (time.time() - start_time) * 1000
            
            metric = SearchMetric(
                query=query[:100],
                search_type=search_type,
                duration_ms=duration_ms,
                result_count=0,
                cache_hit=cache_hit,
                filters_used=filters_used,
                timestamp=datetime.now(),
                error=str(e)
            )
            
            self._record_search_metric(metric)
            self.error_counts['search_error'] += 1
            
            raise
    
    def track_ingestion(
        self,
        operation: str,
        product_count: int,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Track ingestion operations.
        """
        metric = IngestionMetric(
            operation=operation,
            product_count=product_count,
            duration_ms=duration_ms,
            success=success,
            timestamp=datetime.now(),
            error=error
        )
        
        self.ingestion_metrics.append(metric)
        
        if not success:
            self.error_counts['ingestion_error'] += 1
        
        # Check thresholds
        self._check_ingestion_alerts()
    
    def record_cache_metrics(self, cache_stats: Dict[str, Any]) -> None:
        """
        Record cache performance metrics.
        """
        # Extract key metrics
        query_hit_rate = cache_stats.get('query_hit_rate', 0)
        
        # Check cache hit rate threshold
        if query_hit_rate < self.alert_thresholds['cache_hit_rate_min']:
            self._trigger_alert(
                'low_cache_hit_rate',
                f"Cache hit rate {query_hit_rate:.2%} below threshold",
                severity='warning'
            )
    
    def get_search_statistics(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get search performance statistics.
        
        Args:
            time_window: Time window to analyze (default: all data)
        """
        # Filter by time window
        if time_window:
            cutoff = datetime.now() - time_window
            metrics = [m for m in self.search_metrics if m.timestamp > cutoff]
        else:
            metrics = list(self.search_metrics)
        
        if not metrics:
            return {'error': 'No metrics available'}
        
        # Calculate statistics
        latencies = [m.duration_ms for m in metrics if m.error is None]
        
        if not latencies:
            return {'error': 'No successful searches'}
        
        latencies.sort()
        
        # Calculate percentiles
        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]
        
        # Count errors
        error_count = sum(1 for m in metrics if m.error is not None)
        error_rate = error_count / len(metrics) if metrics else 0
        
        # Cache statistics
        cache_hits = sum(1 for m in metrics if m.cache_hit)
        cache_hit_rate = cache_hits / len(metrics) if metrics else 0
        
        # Results by search type
        by_type = defaultdict(list)
        for m in metrics:
            if m.error is None:
                by_type[m.search_type].append(m.duration_ms)
        
        type_stats = {}
        for search_type, type_latencies in by_type.items():
            type_stats[search_type] = {
                'count': len(type_latencies),
                'avg_ms': sum(type_latencies) / len(type_latencies),
                'p95_ms': percentile(sorted(type_latencies), 95)
            }
        
        stats = {
            'total_searches': len(metrics),
            'successful_searches': len(latencies),
            'error_rate': error_rate,
            'cache_hit_rate': cache_hit_rate,
            'latency': {
                'min_ms': min(latencies),
                'avg_ms': sum(latencies) / len(latencies),
                'p50_ms': percentile(latencies, 50),
                'p95_ms': percentile(latencies, 95),
                'p99_ms': percentile(latencies, 99),
                'max_ms': max(latencies)
            },
            'by_search_type': type_stats,
            'avg_result_count': sum(m.result_count for m in metrics if m.error is None) / len(latencies),
            'time_window': str(time_window) if time_window else 'all_time'
        }
        
        # Check alerts
        if stats['latency']['p95_ms'] > self.alert_thresholds['search_latency_p95']:
            self._trigger_alert(
                'high_search_latency',
                f"P95 latency {stats['latency']['p95_ms']:.0f}ms exceeds threshold",
                severity='warning'
            )
        
        if error_rate > self.alert_thresholds['search_error_rate']:
            self._trigger_alert(
                'high_error_rate',
                f"Search error rate {error_rate:.2%} exceeds threshold",
                severity='critical'
            )
        
        return stats
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """
        Get ingestion performance statistics.
        """
        if not self.ingestion_metrics:
            return {'error': 'No ingestion metrics available'}
        
        # Group by operation
        by_operation = defaultdict(list)
        for m in self.ingestion_metrics:
            by_operation[m.operation].append(m)
        
        stats = {
            'total_operations': len(self.ingestion_metrics),
            'by_operation': {}
        }
        
        for operation, metrics in by_operation.items():
            successful = [m for m in metrics if m.success]
            failed = [m for m in metrics if not m.success]
            
            op_stats = {
                'total': len(metrics),
                'successful': len(successful),
                'failed': len(failed),
                'error_rate': len(failed) / len(metrics) if metrics else 0,
                'total_products': sum(m.product_count for m in metrics),
                'avg_products_per_op': sum(m.product_count for m in metrics) / len(metrics) if metrics else 0
            }
            
            if successful:
                latencies = [m.duration_ms for m in successful]
                op_stats['latency'] = {
                    'avg_ms': sum(latencies) / len(latencies),
                    'max_ms': max(latencies)
                }
            
            stats['by_operation'][operation] = op_stats
        
        return stats
    
    def export_metrics(self, format: str = "json") -> Path:
        """
        Export metrics to file.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            export_path = self.metrics_dir / f"metrics_{timestamp}.json"
            
            data = {
                'brand': self.brand_domain,
                'exported_at': datetime.now().isoformat(),
                'search_metrics': [asdict(m) for m in self.search_metrics],
                'ingestion_metrics': [asdict(m) for m in self.ingestion_metrics],
                'error_counts': dict(self.error_counts),
                'statistics': {
                    'search': self.get_search_statistics(),
                    'ingestion': self.get_ingestion_statistics()
                }
            }
            
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"📊 Exported metrics to {export_path}")
        return export_path
    
    def add_alert_handler(self, handler: Callable[[str, str, str], None]) -> None:
        """
        Add alert handler function.
        
        Handler signature: (alert_type: str, message: str, severity: str) -> None
        """
        self.alert_handlers.append(handler)
    
    def _record_search_metric(self, metric: SearchMetric) -> None:
        """Record search metric and check thresholds."""
        self.search_metrics.append(metric)
        
        # OpenTelemetry recording if available
        if self.enable_tracing and hasattr(self, 'search_counter'):
            labels = {
                'search_type': metric.search_type,
                'cache_hit': str(metric.cache_hit),
                'has_error': str(metric.error is not None)
            }
            self.search_counter.add(1, labels)
            self.search_latency.record(metric.duration_ms, labels)
    
    def _check_ingestion_alerts(self) -> None:
        """Check ingestion metrics against thresholds."""
        # Calculate recent error rate
        recent = list(self.ingestion_metrics)[-100:]  # Last 100 operations
        if recent:
            error_rate = sum(1 for m in recent if not m.success) / len(recent)
            
            if error_rate > self.alert_thresholds['ingestion_error_rate']:
                self._trigger_alert(
                    'high_ingestion_error_rate',
                    f"Ingestion error rate {error_rate:.2%} exceeds threshold",
                    severity='critical'
                )
    
    def _trigger_alert(self, alert_type: str, message: str, severity: str = 'warning') -> None:
        """Trigger an alert."""
        logger.warning(f"🚨 Alert [{severity}] {alert_type}: {message}")
        
        # Call registered handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_type, message, severity)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _init_opentelemetry(self) -> None:
        """Initialize OpenTelemetry instrumentation."""
        try:
            # Setup tracing
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(__name__)
            
            # Setup metrics
            metrics.set_meter_provider(MeterProvider())
            meter = metrics.get_meter(__name__)
            
            # Create metrics
            self.search_counter = meter.create_counter(
                name="rag_searches_total",
                description="Total number of RAG searches",
                unit="1"
            )
            
            self.search_latency = meter.create_histogram(
                name="rag_search_duration_ms",
                description="RAG search duration in milliseconds",
                unit="ms"
            )
            
            logger.info("✅ OpenTelemetry instrumentation initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry: {e}")
            self.enable_tracing = False
    
    def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Periodic metrics export
        def export_loop():
            while True:
                time.sleep(3600)  # Every hour
                try:
                    self.export_metrics()
                except Exception as e:
                    logger.error(f"Failed to export metrics: {e}")
        
        thread = threading.Thread(target=export_loop, daemon=True)
        thread.start()


class SearchTracker:
    """Helper class for tracking search operations."""
    
    def __init__(self):
        self.result_count = 0
        self.metadata = {}
    
    def set_result_count(self, count: int) -> None:
        """Set the number of results returned."""
        self.result_count = count
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to track."""
        self.metadata[key] = value


class PerformanceProfiler:
    """
    Simple performance profiler for debugging.
    """
    
    def __init__(self):
        self.timings = defaultdict(list)
    
    @contextmanager
    def profile(self, operation: str):
        """Profile an operation."""
        start = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start) * 1000
            self.timings[operation].append(duration)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary."""
        summary = {}
        
        for operation, timings in self.timings.items():
            if timings:
                summary[operation] = {
                    'count': len(timings),
                    'total_ms': sum(timings),
                    'avg_ms': sum(timings) / len(timings),
                    'min_ms': min(timings),
                    'max_ms': max(timings)
                }
        
        return summary