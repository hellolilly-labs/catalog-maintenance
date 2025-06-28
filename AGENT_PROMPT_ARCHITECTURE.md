# 🎯 Multi-Agent Prompt Generation Architecture

## Core Concept

The Multi-Agent System generates **intelligent, context-aware prompts** that are:
1. **Stored in Langfuse** using our PromptManager
2. **Consumed by your Conversation Engine** (separate project)
3. **Optimized in real-time** based on customer psychology and context

## Architecture Flow

```
Customer Message
      ↓
Multi-Agent Analysis (Psychology, Product, Sales, Brand, etc.)
      ↓
Enhanced Context + Real-Time Insights
      ↓
Intelligent Prompt Generation
      ↓
Langfuse Storage (via PromptManager)
      ↓
Your Conversation Engine → Customer Response
```

## Agent Output: Enhanced Prompts

Each agent contributes to an **Enhanced Prompt** with:

### 1. **Base Prompt Template** (from brand research)
```
You are a {brand_name} expert sales agent helping customers with {product_category} products...
```

### 2. **Real-Time Psychology Enhancement**
```
CUSTOMER PSYCHOLOGY CONTEXT:
- Emotional state: {emotional_state} (confidence: {confidence})
- Communication preference: {communication_style}
- Decision-making style: {decision_style}
- Price sensitivity: {price_sensitivity}
- Urgency level: {urgency_level}

ADAPT YOUR RESPONSE TO:
- {psychology_recommendations}
```

### 3. **Product Intelligence Enhancement**
```
PRIORITY PRODUCTS FOR THIS CUSTOMER:
1. {product_1} - {why_relevant}
2. {product_2} - {why_relevant}
3. {product_3} - {why_relevant}

UPSELL OPPORTUNITIES:
- {upsell_suggestions}

COMPETITIVE ADVANTAGES TO HIGHLIGHT:
- {competitive_edges}
```

### 4. **Sales Strategy Enhancement**
```
SALES APPROACH:
- Strategy: {sales_approach}
- Buying signals detected: {buying_signals}
- Potential objections: {potential_objections}
- Closing opportunities: {closing_opportunities}

IMMEDIATE TACTICS:
- {sales_tactics}
```

### 5. **Brand Authenticity Enhancement**
```
BRAND VOICE REQUIREMENTS:
- Tone: {brand_tone}
- Voice characteristics: {voice_traits}
- Values to emphasize: {brand_values}
- Story opportunities: {brand_stories}

BRAND COMPLIANCE:
- {authenticity_requirements}
```

### 6. **Conversation Flow Enhancement**
```
RESPONSE OPTIMIZATION:
- Optimal length: {response_length}
- Technical depth: {technical_level}
- Engagement tactics: {engagement_strategies}
- Next conversation moves: {conversation_flow}
```

## Langfuse Prompt Structure

### Prompt Naming Convention:
```
liddy/conversation/{brand_domain}/{agent_type}_{conversation_stage}_{customer_intent}
```

Examples:
- `liddy/conversation/specialized.com/enhanced_prompt_consideration_comparing`
- `liddy/conversation/gucci.com/enhanced_prompt_decision_buying`
- `liddy/conversation/startup.com/enhanced_prompt_awareness_browsing`

### Prompt Variables:
```json
{
  "brand_context": {
    "brand_name": "Specialized",
    "brand_domain": "specialized.com",
    "brand_voice": "technical expert, performance-focused",
    "core_values": ["innovation", "performance", "authenticity"]
  },
  "customer_psychology": {
    "emotional_state": "excited",
    "communication_style": "technical", 
    "decision_style": "analytical",
    "price_sensitivity": "low",
    "urgency_level": "medium"
  },
  "product_intelligence": {
    "priority_products": [...],
    "upsell_opportunities": [...],
    "competitive_advantages": [...]
  },
  "sales_strategy": {
    "approach": "consultative",
    "buying_signals": [...],
    "objection_handling": [...]
  },
  "conversation_optimization": {
    "response_length": "detailed",
    "technical_depth": "expert",
    "engagement_tactics": [...]
  }
}
```