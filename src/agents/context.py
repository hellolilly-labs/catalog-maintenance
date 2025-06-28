"""
Conversation Context Management for Multi-Agent System

Manages conversation state, history, and context that flows between agents
and enhances real-time decision making. Optimized for LiveKit real-time
communication patterns.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

from .base_agent import AgentInsight

logger = logging.getLogger(__name__)


class ConversationStage(Enum):
    """Customer conversation journey stages"""
    AWARENESS = "awareness"           # Just learning about brand/products
    INTEREST = "interest"            # Showing interest in specific products
    CONSIDERATION = "consideration"   # Comparing options, asking detailed questions
    DECISION = "decision"            # Ready to purchase, discussing specifics
    POST_PURCHASE = "post_purchase"  # Support, additional purchases


class CustomerIntent(Enum):
    """Customer intent classification"""
    BROWSING = "browsing"           # General exploration
    RESEARCHING = "researching"     # Gathering information
    COMPARING = "comparing"         # Comparing products/options
    BUYING = "buying"              # Ready to purchase
    SUPPORT = "support"            # Need help/support


@dataclass
class CustomerProfile:
    """Customer profile built over conversation"""
    customer_id: Optional[str] = None
    
    # Demographics & psychographics (inferred)
    estimated_age_range: Optional[str] = None
    lifestyle_indicators: List[str] = field(default_factory=list)
    personality_traits: List[str] = field(default_factory=list)
    
    # Preferences & behavior
    communication_style: Optional[str] = None  # technical, casual, formal, etc.
    decision_making_style: Optional[str] = None  # analytical, intuitive, social, etc.
    price_sensitivity: Optional[str] = None  # high, medium, low
    brand_affinity_score: float = 0.0  # 0.0 to 1.0
    
    # Purchase context
    urgency_level: Optional[str] = None  # high, medium, low
    budget_range: Optional[str] = None
    purchase_purpose: Optional[str] = None  # self, gift, business, etc.
    
    # Conversation patterns
    preferred_response_length: Optional[str] = None  # short, medium, detailed
    technical_depth_preference: Optional[str] = None  # basic, intermediate, expert
    question_asking_frequency: float = 0.0
    
    # Updated over time
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationMetrics:
    """Real-time conversation performance metrics"""
    total_messages: int = 0
    customer_messages: int = 0
    agent_messages: int = 0
    
    # Engagement metrics
    avg_response_time_seconds: float = 0.0
    customer_engagement_score: float = 0.0  # 0.0 to 1.0
    conversation_satisfaction_score: float = 0.0  # 0.0 to 1.0
    
    # Conversation flow
    questions_asked_by_customer: int = 0
    questions_asked_by_agent: int = 0
    topic_changes: int = 0
    
    # Business metrics
    products_mentioned: int = 0
    products_shown_interest: int = 0
    purchase_signals: int = 0
    objections_raised: int = 0
    
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationContext:
    """
    Complete conversation context for multi-agent analysis.
    Designed for real-time processing and LiveKit integration.
    """
    
    # Basic conversation info
    conversation_id: str
    brand_domain: str
    started_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Message history (limited for performance)
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history_length: int = 50  # Keep recent context, drop old messages
    
    # Customer info
    customer_profile: CustomerProfile = field(default_factory=CustomerProfile)
    current_intent: Optional[CustomerIntent] = None
    conversation_stage: ConversationStage = ConversationStage.AWARENESS
    
    # Real-time state
    last_customer_message: Optional[str] = None
    last_agent_response: Optional[str] = None
    response_in_progress: bool = False
    
    # Agent insights history (recent only for performance)
    recent_agent_insights: List[AgentInsight] = field(default_factory=list)
    max_insights_history: int = 20
    
    # Context variables for personalization
    mentioned_products: List[str] = field(default_factory=list)
    expressed_interests: List[str] = field(default_factory=list)
    stated_preferences: Dict[str, Any] = field(default_factory=dict)
    budget_indicators: List[str] = field(default_factory=list)
    
    # Performance tracking
    metrics: ConversationMetrics = field(default_factory=ConversationMetrics)
    
    # LiveKit specific context
    livekit_room_id: Optional[str] = None
    audio_enabled: bool = False
    video_enabled: bool = False
    
    def add_message(self, sender: str, content: str, message_type: str = "text", metadata: Optional[Dict] = None):
        """Add message to conversation history with automatic cleanup"""
        message = {
            "sender": sender,
            "content": content,
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.message_history.append(message)
        
        # Keep recent history only for performance
        if len(self.message_history) > self.max_history_length:
            self.message_history = self.message_history[-self.max_history_length:]
        
        # Update state
        if sender == "customer":
            self.last_customer_message = content
            self.metrics.customer_messages += 1
        else:
            self.last_agent_response = content
            self.metrics.agent_messages += 1
        
        self.metrics.total_messages += 1
        self.last_updated = datetime.now()
        
        logger.debug(f"📝 Added {sender} message to conversation {self.conversation_id}")
    
    def add_agent_insight(self, insight: AgentInsight):
        """Add agent insight to recent history with cleanup"""
        self.recent_agent_insights.append(insight)
        
        # Keep recent insights only
        if len(self.recent_agent_insights) > self.max_insights_history:
            self.recent_agent_insights = self.recent_agent_insights[-self.max_insights_history:]
        
        logger.debug(f"🧠 Added {insight.agent_name} insight to conversation {self.conversation_id}")
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages for context"""
        return self.message_history[-count:]
    
    def get_customer_messages_only(self, count: int = 5) -> List[str]:
        """Get recent customer messages for analysis"""
        customer_messages = [
            msg["content"] for msg in self.message_history 
            if msg["sender"] == "customer"
        ]
        return customer_messages[-count:]
    
    def update_customer_profile(self, updates: Dict[str, Any]):
        """Update customer profile with new information"""
        for key, value in updates.items():
            if hasattr(self.customer_profile, key):
                setattr(self.customer_profile, key, value)
        
        self.customer_profile.last_updated = datetime.now()
        logger.debug(f"👤 Updated customer profile for conversation {self.conversation_id}")
    
    def update_conversation_stage(self, new_stage: ConversationStage):
        """Update conversation stage with logging"""
        if new_stage != self.conversation_stage:
            logger.info(f"🔄 Conversation {self.conversation_id} stage: {self.conversation_stage.value} → {new_stage.value}")
            self.conversation_stage = new_stage
            self.last_updated = datetime.now()
    
    def update_customer_intent(self, new_intent: CustomerIntent):
        """Update customer intent with logging"""
        if new_intent != self.current_intent:
            logger.info(f"🎯 Conversation {self.conversation_id} intent: {self.current_intent} → {new_intent.value}")
            self.current_intent = new_intent
            self.last_updated = datetime.now()
    
    def get_conversation_duration(self) -> timedelta:
        """Get total conversation duration"""
        return datetime.now() - self.started_at
    
    def is_conversation_stale(self, minutes: int = 30) -> bool:
        """Check if conversation has been inactive"""
        return (datetime.now() - self.last_updated) > timedelta(minutes=minutes)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of conversation context for agent analysis"""
        return {
            "conversation_id": self.conversation_id,
            "brand_domain": self.brand_domain,
            "duration_minutes": self.get_conversation_duration().total_seconds() / 60,
            "message_count": len(self.message_history),
            "conversation_stage": self.conversation_stage.value,
            "customer_intent": self.current_intent.value if self.current_intent else None,
            "customer_profile_summary": {
                "communication_style": self.customer_profile.communication_style,
                "decision_making_style": self.customer_profile.decision_making_style,
                "price_sensitivity": self.customer_profile.price_sensitivity,
                "brand_affinity_score": self.customer_profile.brand_affinity_score,
                "urgency_level": self.customer_profile.urgency_level
            },
            "recent_topics": {
                "mentioned_products": self.mentioned_products[-5:] if self.mentioned_products else [],
                "expressed_interests": self.expressed_interests[-5:] if self.expressed_interests else [],
                "stated_preferences": self.stated_preferences
            },
            "livekit_context": {
                "room_id": self.livekit_room_id,
                "audio_enabled": self.audio_enabled,
                "video_enabled": self.video_enabled
            }
        }


@dataclass 
class EnhancedContext:
    """
    Enhanced context that combines conversation context with multi-agent insights.
    This is what gets passed to the primary sales agent for response generation.
    """
    
    # Base conversation context
    conversation_context: ConversationContext
    
    # Multi-agent insights
    psychology_insights: Optional[AgentInsight] = None
    product_insights: Optional[AgentInsight] = None  
    sales_insights: Optional[AgentInsight] = None
    brand_insights: Optional[AgentInsight] = None
    conversation_insights: Optional[AgentInsight] = None
    market_insights: Optional[AgentInsight] = None
    
    # Fusion metadata
    total_processing_time_ms: float = 0.0
    insights_used: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    fallback_insights: int = 0
    
    # Real-time optimizations
    recommended_response_length: Optional[str] = None
    suggested_tone: Optional[str] = None
    priority_topics: List[str] = field(default_factory=list)
    immediate_opportunities: List[str] = field(default_factory=list)
    
    def get_valid_insights(self) -> List[AgentInsight]:
        """Get all non-fallback insights"""
        insights = []
        for insight in [
            self.psychology_insights, self.product_insights, self.sales_insights,
            self.brand_insights, self.conversation_insights, self.market_insights
        ]:
            if insight and not insight.metadata.get("is_fallback", False):
                insights.append(insight)
        return insights
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence based on valid insights"""
        valid_insights = self.get_valid_insights()
        if not valid_insights:
            return 0.0
        
        total_confidence = sum(insight.confidence_score for insight in valid_insights)
        return total_confidence / len(valid_insights)