"""
Agent Communication Hub - Central Coordination for Multi-Agent System

Orchestrates real-time communication between all agents and manages the 
fusion of insights for enhanced customer responses. Optimized for LiveKit
real-time communication requirements.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Type

from .base_agent import BaseAgent, AgentInsight
from .context import ConversationContext, EnhancedContext
from .prompt_generator import create_prompt_generator
from .product_intelligence_agent import create_product_intelligence_agent

logger = logging.getLogger(__name__)


class AgentCommunicationHub:
    """
    Central hub for real-time agent coordination and insight fusion.
    
    Responsibilities:
    - Broadcast customer messages to all active agents simultaneously
    - Collect and fuse agent insights in real-time
    - Manage agent lifecycle and performance monitoring
    - Provide enhanced context to primary sales agent
    - Ensure sub-200ms processing for LiveKit real-time requirements
    """
    
    def __init__(self, max_total_processing_time_ms: float = 1000):
        self.max_total_processing_time_ms = max_total_processing_time_ms
        self.active_agents: Dict[str, BaseAgent] = {}
        self.agent_performance_history: Dict[str, List[float]] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Prompt generator for Langfuse integration
        self.prompt_generator = create_prompt_generator()
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.avg_processing_time_ms = 0.0
        
        logger.info(f"🎯 Initialized AgentCommunicationHub with {max_total_processing_time_ms}ms max processing time")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the communication hub"""
        self.active_agents[agent.agent_name] = agent
        self.agent_performance_history[agent.agent_name] = []
        
        logger.info(f"✅ Registered agent: {agent.agent_name}")
        logger.debug(f"   Description: {agent.get_agent_description()}")
    
    def unregister_agent(self, agent_name: str):
        """Remove an agent from the communication hub"""
        if agent_name in self.active_agents:
            del self.active_agents[agent_name]
            logger.info(f"❌ Unregistered agent: {agent_name}")
    
    def get_conversation_context(self, conversation_id: str, brand_domain: str) -> ConversationContext:
        """Get or create conversation context"""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = ConversationContext(
                conversation_id=conversation_id,
                brand_domain=brand_domain
            )
            logger.info(f"🆕 Created new conversation context: {conversation_id}")
        
        return self.conversation_contexts[conversation_id]
    
    async def process_customer_message(
        self, 
        message: str, 
        conversation_id: str, 
        brand_domain: str,
        livekit_room_id: Optional[str] = None,
        audio_enabled: bool = False,
        video_enabled: bool = False
    ) -> EnhancedContext:
        """
        Process customer message through all agents and return enhanced context.
        
        This is the core method that:
        1. Updates conversation context
        2. Broadcasts message to all agents simultaneously  
        3. Collects insights within time budget
        4. Fuses insights into enhanced context
        5. Returns enhanced context for primary sales agent
        """
        
        start_time = time.time()
        self.total_requests += 1
        
        # Get/update conversation context
        context = self.get_conversation_context(conversation_id, brand_domain)
        context.add_message("customer", message)
        
        # Update LiveKit context if provided
        if livekit_room_id:
            context.livekit_room_id = livekit_room_id
            context.audio_enabled = audio_enabled
            context.video_enabled = video_enabled
        
        logger.info(f"🔄 Processing message in conversation {conversation_id}: '{message[:50]}...'")
        
        try:
            # Broadcast message to all agents simultaneously
            agent_insights = await self._broadcast_to_agents(message, context)
            
            # Fuse insights into enhanced context
            enhanced_context = await self._fuse_agent_insights(
                agent_insights, context, start_time
            )
            
            # Update performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time_ms, success=True)
            
            logger.info(f"✅ Processed message in {processing_time_ms:.1f}ms with {len(agent_insights)} insights")
            
            return enhanced_context
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time_ms, success=False)
            
            logger.error(f"❌ Failed to process message: {e}")
            
            # Return minimal enhanced context on failure
            return EnhancedContext(
                conversation_context=context,
                total_processing_time_ms=processing_time_ms,
                fallback_insights=len(self.active_agents),
                confidence_score=0.0
            )
    
    async def generate_enhanced_prompt(
        self,
        message: str,
        conversation_id: str,
        brand_domain: str,
        brand_intelligence: Optional[Dict[str, Any]] = None,
        livekit_room_id: Optional[str] = None,
        audio_enabled: bool = False,
        video_enabled: bool = False
    ) -> Dict[str, Any]:
        """
        Generate an enhanced prompt for the Conversation Engine.
        
        This is the main method that external systems should call to get
        intelligent, context-aware prompts stored in Langfuse.
        
        Returns:
            Dict containing prompt key, enhanced prompt, and metadata
        """
        
        # Get enhanced context from multi-agent analysis
        enhanced_context = await self.process_customer_message(
            message=message,
            conversation_id=conversation_id,
            brand_domain=brand_domain,
            livekit_room_id=livekit_room_id,
            audio_enabled=audio_enabled,
            video_enabled=video_enabled
        )
        
        # Generate enhanced prompt with all agent insights
        prompt_result = await self.prompt_generator.generate_enhanced_prompt(
            enhanced_context=enhanced_context,
            message=message,
            brand_intelligence=brand_intelligence
        )
        
        logger.info(f"🎯 Generated enhanced prompt for conversation {conversation_id}: {prompt_result['prompt_key']}")
        
        return prompt_result
    
    async def _broadcast_to_agents(
        self, 
        message: str, 
        context: ConversationContext
    ) -> Dict[str, Optional[AgentInsight]]:
        """
        Broadcast message to all agents simultaneously and collect insights.
        Uses asyncio.gather for parallel processing to minimize latency.
        """
        
        if not self.active_agents:
            logger.warning("⚠️ No active agents registered")
            return {}
        
        # Create tasks for all agents
        agent_tasks = {}
        for agent_name, agent in self.active_agents.items():
            if agent.is_healthy():
                agent_tasks[agent_name] = agent.analyze_with_timeout(message, context)
            else:
                logger.warning(f"⚠️ Skipping unhealthy agent: {agent_name}")
        
        # Execute all agent tasks in parallel
        try:
            results = await asyncio.gather(*agent_tasks.values(), return_exceptions=True)
            
            # Map results back to agent names
            agent_insights = {}
            for (agent_name, _), result in zip(agent_tasks.items(), results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Agent {agent_name} failed: {result}")
                    agent_insights[agent_name] = None
                else:
                    agent_insights[agent_name] = result
                    
                    # Store insight in conversation context
                    if result:
                        context.add_agent_insight(result)
            
            return agent_insights
            
        except Exception as e:
            logger.error(f"❌ Failed to broadcast to agents: {e}")
            return {}
    
    async def _fuse_agent_insights(
        self, 
        agent_insights: Dict[str, Optional[AgentInsight]], 
        context: ConversationContext,
        start_time: float
    ) -> EnhancedContext:
        """
        Fuse agent insights into a coherent enhanced context.
        
        This method combines insights from different agents and provides
        real-time recommendations for response optimization.
        """
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Extract insights by agent type
        psychology_insight = agent_insights.get("psychology_agent")
        product_insight = agent_insights.get("product_intelligence_agent") 
        sales_insight = agent_insights.get("sales_strategy_agent")
        brand_insight = agent_insights.get("brand_authenticity_agent")
        conversation_insight = agent_insights.get("conversation_intelligence_agent")
        market_insight = agent_insights.get("market_intelligence_agent")
        
        # Count valid vs fallback insights
        valid_insights = [i for i in agent_insights.values() if i and not i.metadata.get("is_fallback")]
        fallback_insights = len(agent_insights) - len(valid_insights)
        
        # Calculate overall confidence
        if valid_insights:
            confidence_score = sum(i.confidence_score for i in valid_insights) / len(valid_insights)
        else:
            confidence_score = 0.0
        
        # Generate real-time recommendations
        recommendations = await self._generate_real_time_recommendations(
            agent_insights, context
        )
        
        enhanced_context = EnhancedContext(
            conversation_context=context,
            psychology_insights=psychology_insight,
            product_insights=product_insight,
            sales_insights=sales_insight,
            brand_insights=brand_insight,
            conversation_insights=conversation_insight,
            market_insights=market_insight,
            total_processing_time_ms=processing_time_ms,
            insights_used=[name for name, insight in agent_insights.items() if insight],
            confidence_score=confidence_score,
            fallback_insights=fallback_insights,
            **recommendations
        )
        
        logger.debug(f"🔗 Fused {len(valid_insights)} valid insights with {confidence_score:.2f} confidence")
        
        return enhanced_context
    
    async def _generate_real_time_recommendations(
        self, 
        agent_insights: Dict[str, Optional[AgentInsight]], 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Generate real-time recommendations based on agent insights.
        
        This analyzes all agent insights to provide immediate guidance
        for response optimization, tone adaptation, and conversation flow.
        """
        
        recommendations = {
            "recommended_response_length": "medium",
            "suggested_tone": "professional",
            "priority_topics": [],
            "immediate_opportunities": []
        }
        
        # Analyze psychology insights for communication adaptation
        psychology = agent_insights.get("psychology_agent")
        if psychology and psychology.insights:
            if psychology.insights.get("preferred_communication_style") == "technical":
                recommendations["recommended_response_length"] = "detailed"
                recommendations["suggested_tone"] = "technical"
            elif psychology.insights.get("emotional_state") == "excited":
                recommendations["suggested_tone"] = "enthusiastic"
            elif psychology.insights.get("urgency_level") == "high":
                recommendations["immediate_opportunities"].append("urgency_response")
        
        # Analyze product insights for topic prioritization
        product = agent_insights.get("product_intelligence_agent")
        if product and product.insights:
            if product.insights.get("priority_products"):
                recommendations["priority_topics"].extend(
                    product.insights["priority_products"][:3]
                )
            if product.insights.get("upsell_opportunities"):
                recommendations["immediate_opportunities"].append("upsell_mention")
        
        # Analyze sales insights for strategy adaptation
        sales = agent_insights.get("sales_strategy_agent")
        if sales and sales.insights:
            if sales.insights.get("buying_signals"):
                recommendations["immediate_opportunities"].append("closing_opportunity")
            if sales.insights.get("objection_signals"):
                recommendations["immediate_opportunities"].append("preemptive_objection_handling")
        
        # Analyze conversation insights for flow optimization
        conversation = agent_insights.get("conversation_intelligence_agent")
        if conversation and conversation.insights:
            if conversation.insights.get("optimal_response_length"):
                recommendations["recommended_response_length"] = conversation.insights["optimal_response_length"]
            if conversation.insights.get("engagement_level") == "low":
                recommendations["immediate_opportunities"].append("engagement_boost")
        
        return recommendations
    
    def _update_performance_metrics(self, processing_time_ms: float, success: bool):
        """Update hub performance metrics"""
        if success:
            self.successful_requests += 1
        
        # Update running average
        total_time = self.avg_processing_time_ms * (self.total_requests - 1) + processing_time_ms
        self.avg_processing_time_ms = total_time / self.total_requests
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health and performance metrics"""
        agent_health = {}
        for agent_name, agent in self.active_agents.items():
            metrics = agent.get_performance_metrics()
            agent_health[agent_name] = {
                "is_healthy": agent.is_healthy(),
                "success_rate": metrics.success_rate,
                "avg_processing_time_ms": metrics.avg_processing_time_ms,
                "error_count": metrics.error_count
            }
        
        return {
            "hub_health": {
                "total_requests": self.total_requests,
                "success_rate": self.successful_requests / max(1, self.total_requests),
                "avg_processing_time_ms": self.avg_processing_time_ms,
                "active_agents": len(self.active_agents),
                "healthy_agents": sum(1 for agent in self.active_agents.values() if agent.is_healthy())
            },
            "agent_health": agent_health,
            "active_conversations": len(self.conversation_contexts)
        }
    
    def cleanup_stale_conversations(self, max_age_minutes: int = 60):
        """Clean up stale conversation contexts to prevent memory leaks"""
        stale_conversations = []
        
        for conv_id, context in self.conversation_contexts.items():
            if context.is_conversation_stale(max_age_minutes):
                stale_conversations.append(conv_id)
        
        for conv_id in stale_conversations:
            del self.conversation_contexts[conv_id]
            logger.info(f"🧹 Cleaned up stale conversation: {conv_id}")
        
        if stale_conversations:
            logger.info(f"🧹 Cleaned up {len(stale_conversations)} stale conversations")