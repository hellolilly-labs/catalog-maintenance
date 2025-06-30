# UserState RAG Enhancements

## Overview

The existing UserState system is comprehensive but needs RAG-specific enhancements to optimize product search and discovery. Rather than creating a new user state management system, we'll extend the existing `UserState` model with fields that improve RAG performance.

## Proposed UserState Enhancements

### 1. Search Preferences (for Query Optimization)

```python
@dataclass
class SearchPreferences:
    """User's search preferences for RAG optimization."""
    preferred_brands: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)
    price_range: Optional[Tuple[float, float]] = None  # (min, max)
    size_preferences: Dict[str, str] = field(default_factory=dict)  # {"shoes": "10", "shirts": "L"}
    style_preferences: List[str] = field(default_factory=list)  # ["modern", "classic", "minimalist"]
    excluded_materials: List[str] = field(default_factory=list)  # ["leather", "wool"]
    
    # Search behavior
    prefers_exact_matches: bool = False  # Adjust dense/sparse weights
    values_sustainability: bool = False
    values_luxury: bool = False
    values_budget_friendly: bool = False
```

### 2. Search History (for Context)

```python
@dataclass
class SearchQuery:
    """Individual search query with results."""
    query: str
    timestamp: float
    filters_applied: Dict[str, Any]
    result_count: int
    clicked_results: List[str] = field(default_factory=list)  # Product IDs
    search_type: str = "hybrid"  # "exact", "semantic", "hybrid"
    satisfaction_score: Optional[float] = None  # 0-1, based on clicks/time

@dataclass
class SearchHistory:
    """User's search history for pattern analysis."""
    recent_queries: List[SearchQuery] = field(default_factory=list)  # Last 20
    frequent_terms: Dict[str, int] = field(default_factory=dict)  # Term -> count
    successful_queries: List[str] = field(default_factory=list)  # Led to purchases
    category_interests: Dict[str, float] = field(default_factory=dict)  # Category -> interest score
```

### 3. Product Interactions (for Personalization)

```python
@dataclass
class ProductInteraction:
    """Track how users interact with products."""
    product_id: str
    timestamp: float
    interaction_type: str  # "view", "click", "add_to_cart", "purchase", "save"
    duration: Optional[float] = None  # Time spent viewing
    source: str = "search"  # "search", "recommendation", "browse"
    query: Optional[str] = None  # Search query that led to this

@dataclass
class ProductPreferences:
    """Learned product preferences."""
    viewed_products: List[ProductInteraction] = field(default_factory=list)
    saved_products: List[str] = field(default_factory=list)  # Product IDs
    purchased_products: List[str] = field(default_factory=list)
    
    # Inferred preferences
    preferred_price_points: List[float] = field(default_factory=list)
    preferred_features: Dict[str, float] = field(default_factory=dict)  # Feature -> weight
    brand_affinity: Dict[str, float] = field(default_factory=dict)  # Brand -> score
```

### 4. Enhanced UserState

```python
@dataclass
class UserState:
    # ... existing fields ...
    
    # RAG-specific additions
    search_preferences: Optional[SearchPreferences] = None
    search_history: Optional[SearchHistory] = None
    product_preferences: Optional[ProductPreferences] = None
    
    # RAG performance tracking
    avg_search_satisfaction: float = 0.0
    total_searches: int = 0
    successful_searches: int = 0
    
    # Cache keys for fast lookup
    embedding_cache_keys: List[str] = field(default_factory=list)  # Recent embeddings
    
    def get_rag_context(self) -> Dict[str, Any]:
        """Get context for RAG query optimization."""
        return {
            'preferred_brands': self.search_preferences.preferred_brands if self.search_preferences else [],
            'price_range': self.search_preferences.price_range if self.search_preferences else None,
            'recent_searches': [q.query for q in self.search_history.recent_queries[-5:]] if self.search_history else [],
            'category_interests': self.search_history.category_interests if self.search_history else {},
            'search_style': 'exact' if self.search_preferences and self.search_preferences.prefers_exact_matches else 'balanced'
        }
```

## Integration with RAG System

### 1. Update Search Service

```python
# In search_service.py
@staticmethod
async def enhance_product_query_with_filters(
    query: str,
    user_state: UserState,  # Now includes RAG enhancements
    chat_ctx: llm.ChatContext,
    account: str,
    product_knowledge: str = ""
) -> Tuple[str, Dict[str, Any]]:
    # Use RAG context from user state
    rag_context = user_state.get_rag_context()
    
    # Add user preferences to prompt
    preferences_context = f"""
User Preferences:
- Preferred Brands: {', '.join(rag_context['preferred_brands'])}
- Price Range: ${rag_context['price_range'][0]}-${rag_context['price_range'][1]} if available
- Recent Searches: {', '.join(rag_context['recent_searches'])}
- Category Interests: {json.dumps(rag_context['category_interests'])}
"""
    
    # Rest of the enhancement logic...
```

### 2. Update Sample Assistant

```python
# In sample_assistant.py
async def product_search(self, query: str):
    # Get user's RAG context
    rag_context = self.session.userdata.get_rag_context()
    
    # Determine search strategy based on user preferences
    if rag_context.get('search_style') == 'exact':
        # User prefers exact matches - adjust weights
        dense_weight = 0.3
        sparse_weight = 0.7
    else:
        dense_weight = None  # Use auto-determined weights
        sparse_weight = None
    
    # Enhanced search with user context
    results = await SearchService.search_products_rag_with_filters(
        query=query,
        user_state=self.session.userdata,
        filters=extracted_filters,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight
    )
```

### 3. Track Search Performance

```python
# After search results are returned
def track_search_performance(user_state: UserState, query: str, results: List[Dict], clicked: List[str]):
    if not user_state.search_history:
        user_state.search_history = SearchHistory()
    
    # Add to history
    search_query = SearchQuery(
        query=query,
        timestamp=time.time(),
        filters_applied=filters,
        result_count=len(results),
        clicked_results=clicked,
        satisfaction_score=len(clicked) / len(results) if results else 0
    )
    
    user_state.search_history.recent_queries.append(search_query)
    
    # Update frequent terms
    for term in query.lower().split():
        user_state.search_history.frequent_terms[term] = \
            user_state.search_history.frequent_terms.get(term, 0) + 1
    
    # Update satisfaction metrics
    user_state.total_searches += 1
    if clicked:
        user_state.successful_searches += 1
    user_state.avg_search_satisfaction = user_state.successful_searches / user_state.total_searches
```

## Benefits

1. **Personalized Search**: Weights and filters adapted to user preferences
2. **Better Query Understanding**: Context from past searches improves interpretation
3. **Adaptive Strategy**: Switch between exact/semantic search based on user behavior
4. **Performance Tracking**: Monitor and improve search quality per user
5. **Seamless Integration**: Extends existing system rather than replacing it

## Implementation Steps

1. **Extend UserState model** with RAG fields
2. **Update Redis serialization** to handle new fields
3. **Modify search_service.py** to use user context
4. **Add tracking logic** to sample_assistant.py
5. **Create migration script** for existing users

## Backward Compatibility

The enhancements are optional fields, so existing code will continue to work. We check for field existence before using:

```python
if user_state.search_preferences:
    # Use preferences
else:
    # Use defaults
```

This approach maintains compatibility while adding powerful RAG optimization capabilities.