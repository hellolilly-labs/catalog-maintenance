"""
Base Agent Interface for Multi-Agent System

Provides the foundational interface that all specialized agents implement,
ensuring consistent communication patterns and LiveKit compatibility.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class AgentInsight:
    """Standardized insight response from any agent"""
    agent_name: str
    confidence_score: float  # 0.0 to 1.0
    timestamp: datetime
    insights: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: float


@dataclass
class AgentPerformanceMetrics:
    """Performance tracking for agent optimization"""
    agent_name: str
    avg_processing_time_ms: float
    success_rate: float
    insight_utilization_rate: float
    error_count: int
    last_updated: datetime


class BaseAgent(ABC):
    """
    Abstract base class for all multi-agent system agents.
    
    Designed for real-time processing with LiveKit integration considerations:
    - Sub-200ms response times for real-time voice interactions
    - Async processing for parallel agent coordination
    - Graceful error handling for robust real-time systems
    - Performance monitoring for optimization
    """
    
    def __init__(self, agent_name: str, max_processing_time_ms: float = 200):
        self.agent_name = agent_name
        self.max_processing_time_ms = max_processing_time_ms
        self.performance_metrics = AgentPerformanceMetrics(
            agent_name=agent_name,
            avg_processing_time_ms=0.0,
            success_rate=1.0,
            insight_utilization_rate=0.0,
            error_count=0,
            last_updated=datetime.now()
        )
        self._total_processing_time = 0.0
        self._total_requests = 0
        self._successful_requests = 0
        
        logger.info(f"🤖 Initialized {agent_name} agent with {max_processing_time_ms}ms max processing time")
    
    @abstractmethod
    async def analyze_real_time(self, message: str, context: 'ConversationContext') -> AgentInsight:
        """
        Core method that all agents must implement for real-time analysis.
        
        Args:
            message: Customer's latest message
            context: Full conversation context and history
            
        Returns:
            AgentInsight: Standardized insight response
            
        Requirements:
            - Must complete within self.max_processing_time_ms
            - Must handle all exceptions gracefully
            - Must return valid AgentInsight even on partial failures
        """
        pass
    
    async def analyze_with_timeout(self, message: str, context: 'ConversationContext') -> Optional[AgentInsight]:
        """
        Wrapper that enforces timeout and tracks performance metrics.
        Designed for LiveKit real-time requirements.
        """
        start_time = time.time()
        
        try:
            # Enforce timeout for real-time processing
            insight = await asyncio.wait_for(
                self.analyze_real_time(message, context),
                timeout=self.max_processing_time_ms / 1000.0
            )
            
            # Track performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            insight.processing_time_ms = processing_time_ms
            
            self._update_performance_metrics(processing_time_ms, success=True)
            
            logger.debug(f"✅ {self.agent_name} completed analysis in {processing_time_ms:.1f}ms")
            return insight
            
        except asyncio.TimeoutError:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time_ms, success=False)
            
            logger.warning(f"⏰ {self.agent_name} timed out after {processing_time_ms:.1f}ms")
            return self._create_fallback_insight(message, "timeout")
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time_ms, success=False)
            
            logger.error(f"❌ {self.agent_name} failed: {e}")
            return self._create_fallback_insight(message, f"error: {str(e)}")
    
    def _create_fallback_insight(self, message: str, reason: str) -> AgentInsight:
        """Create a minimal fallback insight when agent fails"""
        return AgentInsight(
            agent_name=self.agent_name,
            confidence_score=0.0,
            timestamp=datetime.now(),
            insights={"fallback_reason": reason},
            recommendations=[],
            metadata={"is_fallback": True, "original_message": message},
            processing_time_ms=self.max_processing_time_ms
        )
    
    def _update_performance_metrics(self, processing_time_ms: float, success: bool):
        """Update performance tracking for optimization"""
        self._total_requests += 1
        self._total_processing_time += processing_time_ms
        
        if success:
            self._successful_requests += 1
        else:
            self.performance_metrics.error_count += 1
        
        # Update running averages
        self.performance_metrics.avg_processing_time_ms = (
            self._total_processing_time / self._total_requests
        )
        self.performance_metrics.success_rate = (
            self._successful_requests / self._total_requests
        )
        self.performance_metrics.last_updated = datetime.now()
    
    def get_performance_metrics(self) -> AgentPerformanceMetrics:
        """Get current performance metrics for monitoring"""
        return self.performance_metrics
    
    def is_healthy(self) -> bool:
        """Check if agent is performing within acceptable parameters"""
        return (
            self.performance_metrics.success_rate >= 0.95 and
            self.performance_metrics.avg_processing_time_ms <= self.max_processing_time_ms and
            self.performance_metrics.error_count < 10
        )
    
    @abstractmethod
    def get_agent_description(self) -> str:
        """Return a description of what this agent does"""
        pass
    
    def reset_performance_metrics(self):
        """Reset performance tracking (useful for testing/debugging)"""
        self.performance_metrics = AgentPerformanceMetrics(
            agent_name=self.agent_name,
            avg_processing_time_ms=0.0,
            success_rate=1.0,
            insight_utilization_rate=0.0,
            error_count=0,
            last_updated=datetime.now()
        )
        self._total_processing_time = 0.0
        self._total_requests = 0
        self._successful_requests = 0
        
        logger.info(f"🔄 Reset performance metrics for {self.agent_name}")