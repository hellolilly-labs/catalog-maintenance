"""
User Context Enhancement for Search Service

This module shows how to integrate UserState into query enhancement.
"""

from typing import Optional, Dict, Any, List
import time
from liddy.model import UserState

def format_user_context_for_prompt(
    user_state: Optional[UserState], 
    **kwargs
) -> str:
    """
    Format UserState into a context string for the query enhancement prompt.
    
    Args:
        user_state: The UserState object containing user information
        **kwargs: Additional context like recent_products, search_history, etc.
        
    Returns:
        Formatted string with user context for the prompt
    """
    if not user_state:
        return "No user context available"
    
    context_parts = []
    
    # 1. Basic User Information
    if user_state.account:
        context_parts.append(f"- Account: {user_state.account}")
    
    # 2. Sentiment Analysis - Understanding user mood/engagement
    if user_state.sentiment_analysis and user_state.sentiment_analysis.sentiments:
        latest_sentiment = user_state.sentiment_analysis.sentiments[-1]
        
        # Trust level affects how confident we should be in recommendations
        trust_level = latest_sentiment.sentiment.trustLevel.value
        context_parts.append(f"- Trust level: {trust_level}")
        
        # Engagement level affects how detailed our responses should be
        engagement_level = latest_sentiment.sentiment.engagementLevel.value
        context_parts.append(f"- Engagement level: {engagement_level}")
        
        # Key observations help understand user behavior
        if latest_sentiment.sentiment.keyObservations:
            obs = ", ".join(latest_sentiment.sentiment.keyObservations[:2])
            context_parts.append(f"- Recent behavior: {obs}")
        
        # Emotional tones can guide query enhancement
        if latest_sentiment.sentiment.fromEmotionalTones:
            dominant_emotion = max(
                latest_sentiment.sentiment.fromEmotionalTones,
                key=lambda x: x.weight
            )
            context_parts.append(f"- Emotional state: {dominant_emotion.value}")
    
    # 3. Communication Style Preferences
    if user_state.communication_directive and user_state.communication_directive.formality:
        formality_score = user_state.communication_directive.formality.score
        if formality_score < 3:
            context_parts.append("- Communication style: Casual, friendly")
        elif formality_score > 7:
            context_parts.append("- Communication style: Formal, professional")
        else:
            context_parts.append("- Communication style: Balanced")
    
    # 4. Session Context - Where they are in their journey
    if user_state.interaction_start_time:
        session_duration = time.time() - user_state.interaction_start_time
        if session_duration < 60:
            context_parts.append("- Session stage: Just started browsing")
        elif session_duration < 300:
            context_parts.append("- Session stage: Actively exploring")
        elif session_duration < 900:
            context_parts.append("- Session stage: Deep research mode")
        else:
            context_parts.append("- Session stage: Extended session (may need help deciding)")
    
    # 5. Previous Session Information
    if user_state.conversation_exit_state:
        exit_state = user_state.conversation_exit_state
        if exit_state.transcript_summary:
            context_parts.append(f"- Previous interest: {exit_state.transcript_summary[:100]}...")
        if exit_state.resumption_message:
            context_parts.append(f"- Returning user context: {exit_state.resumption_message}")
    
    # 6. Additional Context from kwargs
    # Recent products viewed
    if 'recent_products' in kwargs and kwargs['recent_products']:
        recent_products = kwargs['recent_products'][-3:]  # Last 3 products
        product_info = []
        for p in recent_products:
            name = p.get('name', 'Unknown')
            price = p.get('price', 0)
            product_info.append(f"{name} (${price})")
        context_parts.append(f"- Recently viewed: {', '.join(product_info)}")
        
        # Analyze price range interest
        prices = [p.get('price', 0) for p in recent_products if p.get('price')]
        if prices:
            avg_price = sum(prices) / len(prices)
            if avg_price < 100:
                context_parts.append("- Price sensitivity: Budget-conscious")
            elif avg_price > 1000:
                context_parts.append("- Price sensitivity: Premium shopper")
    
    # Search history
    if 'search_history' in kwargs and kwargs['search_history']:
        recent_searches = kwargs['search_history'][-3:]  # Last 3 searches
        context_parts.append(f"- Recent searches: {', '.join(recent_searches)}")
        
        # Identify patterns in searches
        if any('sale' in s.lower() or 'discount' in s.lower() for s in recent_searches):
            context_parts.append("- Looking for: Deals and discounts")
    
    # Cart information
    if 'cart_items' in kwargs and kwargs['cart_items']:
        cart_value = sum(item.get('price', 0) * item.get('quantity', 1) 
                        for item in kwargs['cart_items'])
        context_parts.append(f"- Cart value: ${cart_value:.2f}")
        context_parts.append(f"- Cart items: {len(kwargs['cart_items'])}")
    
    # Browse history patterns
    if 'category_views' in kwargs:
        top_categories = sorted(
            kwargs['category_views'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:2]
        if top_categories:
            cats = [f"{cat[0]} ({cat[1]} views)" for cat in top_categories]
            context_parts.append(f"- Interested in: {', '.join(cats)}")
    
    return "\n".join(context_parts) if context_parts else "No user context available"


# Example data points that would be useful for query enhancement:
USEFUL_USER_CONTEXT_DATA = {
    "user_state": {
        "user_id": "Unique identifier",
        "sentiment_analysis": "Trust/engagement levels, emotional state",
        "communication_directive": "Formality preferences",
        "interaction_start_time": "Session duration",
        "conversation_exit_state": "Previous session summary"
    },
    "behavioral_data": {
        "recent_products": "Products viewed with prices and categories",
        "search_history": "Previous search queries",
        "cart_items": "Current cart contents",
        "category_views": "Categories browsed and frequency",
        "filter_usage": "Commonly used filters",
        "price_range_viewed": "Min/max prices of viewed products"
    },
    "preference_indicators": {
        "brand_affinity": "Preferred brands from history",
        "feature_preferences": "Features frequently searched/filtered",
        "purchase_history": "Past purchases for pattern analysis",
        "abandoned_carts": "Products almost purchased",
        "comparison_behavior": "Products compared side-by-side"
    },
    "contextual_data": {
        "time_of_day": "Shopping patterns by time",
        "device_type": "Mobile vs desktop behavior",
        "referral_source": "How they arrived at the site",
        "geographic_location": "Regional preferences",
        "seasonal_context": "Time of year considerations"
    }
}