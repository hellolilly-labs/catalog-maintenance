"""
Search Performance Tracking Module

This module provides utilities for tracking and analyzing search performance
metrics in the voice assistant. It integrates with the enhanced RAG system
to monitor query effectiveness, user satisfaction, and system performance.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Container for search performance metrics."""
    query: str
    enhanced_query: str
    search_type: str  # 'hybrid', 'rag', 'llm'
    dense_weight: Optional[float] = None
    sparse_weight: Optional[float] = None
    filters_applied: int = 0
    results_returned: int = 0
    response_time_ms: float = 0.0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    account: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    # Result quality metrics
    top_result_score: Optional[float] = None
    average_score: Optional[float] = None
    filter_match_rate: Optional[float] = None
    
    # User interaction metrics
    results_clicked: List[str] = field(default_factory=list)
    time_to_first_click: Optional[float] = None
    conversation_continued: bool = False
    user_satisfaction: Optional[str] = None  # 'positive', 'neutral', 'negative'


class SearchPerformanceTracker:
    """
    Tracks and analyzes search performance over time.
    
    This class provides methods to:
    - Record search metrics
    - Analyze performance trends
    - Identify optimization opportunities
    - Generate performance reports
    """
    
    def __init__(self, max_history_size: int = 1000, analytics_window_size: int = 100):
        """
        Initialize the performance tracker.
        
        Args:
            max_history_size: Maximum number of searches to keep in history
            analytics_window_size: Window size for rolling analytics
        """
        self.max_history_size = max_history_size
        self.analytics_window_size = analytics_window_size
        
        # Search history
        self._search_history: deque = deque(maxlen=max_history_size)
        
        # Performance aggregates by account
        self._account_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_searches': 0,
            'avg_response_time': 0.0,
            'avg_results_returned': 0.0,
            'search_types': defaultdict(int),
            'filter_usage': defaultdict(int),
            'satisfaction_scores': defaultdict(int)
        })
        
        # Real-time performance monitoring
        self._recent_response_times: deque = deque(maxlen=analytics_window_size)
        self._recent_result_counts: deque = deque(maxlen=analytics_window_size)
        self._recent_satisfaction: deque = deque(maxlen=analytics_window_size)
    
    def track_search(self, metrics: SearchMetrics) -> None:
        """
        Record a search event with its metrics.
        
        Args:
            metrics: SearchMetrics object containing search details
        """
        # Add to history
        self._search_history.append(metrics)
        
        # Update real-time monitoring
        self._recent_response_times.append(metrics.response_time_ms)
        self._recent_result_counts.append(metrics.results_returned)
        
        # Update account-specific metrics
        if metrics.account:
            account_stats = self._account_metrics[metrics.account]
            account_stats['total_searches'] += 1
            
            # Update rolling averages
            n = account_stats['total_searches']
            account_stats['avg_response_time'] = (
                (account_stats['avg_response_time'] * (n - 1) + metrics.response_time_ms) / n
            )
            account_stats['avg_results_returned'] = (
                (account_stats['avg_results_returned'] * (n - 1) + metrics.results_returned) / n
            )
            
            # Track search type distribution
            account_stats['search_types'][metrics.search_type] += 1
            
            # Track filter usage
            if metrics.filters_applied > 0:
                account_stats['filter_usage']['with_filters'] += 1
            else:
                account_stats['filter_usage']['without_filters'] += 1
        
        # Log performance warnings
        if metrics.response_time_ms > 2000:  # 2 seconds
            logger.warning(
                f"Slow search detected: {metrics.response_time_ms}ms for query '{metrics.query}' "
                f"(type: {metrics.search_type}, account: {metrics.account})"
            )
        
        if metrics.results_returned == 0:
            logger.warning(
                f"No results returned for query '{metrics.query}' "
                f"(enhanced: '{metrics.enhanced_query}', filters: {metrics.filters_applied})"
            )
    
    def track_user_interaction(
        self,
        session_id: str,
        product_clicked: Optional[str] = None,
        satisfaction: Optional[str] = None
    ) -> None:
        """
        Track user interactions with search results.
        
        Args:
            session_id: Session identifier
            product_clicked: Product ID if user clicked a result
            satisfaction: User satisfaction indicator
        """
        # Find the most recent search for this session
        for metrics in reversed(self._search_history):
            if metrics.session_id == session_id:
                if product_clicked and product_clicked not in metrics.results_clicked:
                    metrics.results_clicked.append(product_clicked)
                    if metrics.time_to_first_click is None:
                        metrics.time_to_first_click = time.time() - metrics.timestamp
                
                if satisfaction:
                    metrics.user_satisfaction = satisfaction
                    self._recent_satisfaction.append(satisfaction)
                    if metrics.account:
                        self._account_metrics[metrics.account]['satisfaction_scores'][satisfaction] += 1
                
                break
    
    def get_performance_summary(self, account: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a performance summary.
        
        Args:
            account: Optional account to filter by
            
        Returns:
            Dictionary with performance statistics
        """
        if account:
            # Account-specific summary
            if account not in self._account_metrics:
                return {"error": f"No data for account {account}"}
            
            stats = self._account_metrics[account]
            return {
                "account": account,
                "total_searches": stats['total_searches'],
                "avg_response_time_ms": round(stats['avg_response_time'], 2),
                "avg_results_returned": round(stats['avg_results_returned'], 2),
                "search_type_distribution": dict(stats['search_types']),
                "filter_usage": dict(stats['filter_usage']),
                "satisfaction_distribution": dict(stats['satisfaction_scores'])
            }
        else:
            # Global summary
            total_searches = len(self._search_history)
            if total_searches == 0:
                return {"error": "No search data available"}
            
            # Calculate recent performance
            avg_response_time = (
                sum(self._recent_response_times) / len(self._recent_response_times)
                if self._recent_response_times else 0
            )
            avg_results = (
                sum(self._recent_result_counts) / len(self._recent_result_counts)
                if self._recent_result_counts else 0
            )
            
            # Satisfaction analysis
            satisfaction_counts = defaultdict(int)
            for sat in self._recent_satisfaction:
                satisfaction_counts[sat] += 1
            
            return {
                "total_searches": total_searches,
                "recent_avg_response_time_ms": round(avg_response_time, 2),
                "recent_avg_results_returned": round(avg_results, 2),
                "recent_satisfaction": dict(satisfaction_counts),
                "accounts_tracked": list(self._account_metrics.keys())
            }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Analyze performance data and provide optimization recommendations.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Analyze response times
        if self._recent_response_times:
            avg_response_time = sum(self._recent_response_times) / len(self._recent_response_times)
            if avg_response_time > 1500:  # 1.5 seconds
                recommendations.append({
                    "type": "performance",
                    "severity": "high",
                    "message": f"Average response time ({avg_response_time:.0f}ms) exceeds target (1500ms)",
                    "suggestion": "Consider increasing cache size or optimizing query processing"
                })
        
        # Analyze result effectiveness
        no_result_queries = [
            m for m in list(self._search_history)[-100:]  # Last 100 searches
            if m.results_returned == 0
        ]
        if len(no_result_queries) > 10:  # More than 10% no results
            recommendations.append({
                "type": "relevance",
                "severity": "medium",
                "message": f"{len(no_result_queries)} queries returned no results in last 100 searches",
                "suggestion": "Review query enhancement logic and filter extraction"
            })
        
        # Analyze search type distribution
        for account, stats in self._account_metrics.items():
            if stats['total_searches'] > 50:  # Enough data
                search_types = stats['search_types']
                if 'llm' in search_types and search_types['llm'] > stats['total_searches'] * 0.5:
                    recommendations.append({
                        "type": "configuration",
                        "severity": "low",
                        "message": f"Account {account} using LLM search >50% of time",
                        "suggestion": f"Consider enabling RAG index for {account}"
                    })
        
        return recommendations
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file for analysis.
        
        Args:
            filepath: Path to save the metrics
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": self.get_performance_summary(),
            "account_metrics": dict(self._account_metrics),
            "recent_searches": [
                asdict(m) for m in list(self._search_history)[-100:]
            ],
            "recommendations": self.get_optimization_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported search metrics to {filepath}")


# Global tracker instance
_performance_tracker: Optional[SearchPerformanceTracker] = None


def get_performance_tracker() -> SearchPerformanceTracker:
    """Get the global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = SearchPerformanceTracker()
    return _performance_tracker


def track_search_performance(
    query: str,
    enhanced_query: str,
    search_type: str,
    results: List[Any],
    start_time: float,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    account: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None
) -> SearchMetrics:
    """
    Convenience function to track search performance.
    
    Returns:
        SearchMetrics object that was tracked
    """
    end_time = time.time()
    response_time_ms = (end_time - start_time) * 1000
    
    # Calculate result quality metrics
    scores = [r.get('score', 0) for r in results if isinstance(r, dict)]
    top_score = max(scores) if scores else None
    avg_score = sum(scores) / len(scores) if scores else None
    
    # Create metrics object
    metrics = SearchMetrics(
        query=query,
        enhanced_query=enhanced_query,
        search_type=search_type,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        filters_applied=len(filters) if filters else 0,
        results_returned=len(results),
        response_time_ms=response_time_ms,
        user_id=user_id,
        session_id=session_id,
        account=account,
        top_result_score=top_score,
        average_score=avg_score
    )
    
    # Track in global tracker
    tracker = get_performance_tracker()
    tracker.track_search(metrics)
    
    return metrics