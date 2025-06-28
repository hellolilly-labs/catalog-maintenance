"""
Multi-Agent Intelligent Product Discovery & Sales Agent System

This package implements a revolutionary multi-agent architecture where specialized
AI agents work in real-time to create sophisticated product discovery and sales
experiences with LiveKit integration for real-time communications.

Architecture Overview:
- Primary Sales Agent: Customer-facing agent with expert knowledge
- Background Intelligence Agents: 6 specialized agents providing real-time insights
- Communication Hub: Real-time coordination and insight fusion
- LiveKit Integration: Optimized for real-time voice/video interactions

Key Components:
- base_agent.py: Abstract base class for all agents
- communication_hub.py: Central coordination and insight fusion
- primary_sales_agent.py: Customer-facing expert sales agent
- psychology_agent.py: Customer psychology and intent analysis
- product_intelligence_agent.py: Real-time product recommendations
- sales_strategy_agent.py: Sales optimization and objection handling
- brand_authenticity_agent.py: Brand voice consistency
- conversation_agent.py: Conversation flow optimization
"""

from .base_agent import BaseAgent
from .communication_hub import AgentCommunicationHub
from .context import ConversationContext, EnhancedContext

__all__ = [
    'BaseAgent',
    'AgentCommunicationHub', 
    'ConversationContext',
    'EnhancedContext'
]