# Catalog Labels Caching System

## Overview

The Catalog Labels Caching System provides intelligent session-based caching of product catalog labels for the voice assistant service. This optimizes search performance and enables dynamic system prompt generation by preloading and caching product category information during Assistant initialization.

## Key Features

### 🚀 Session-Based Caching
- **In-Memory Storage**: Labels cached in memory for the duration of the stateful session
- **No External Dependencies**: Uses simple Python dictionaries - perfect for short-lived voice sessions
- **Automatic Management**: Cache automatically managed with the Assistant lifecycle

### 🏷️ Intelligent Preloading
- **Parallel Initialization**: Labels loaded in parallel with product catalog during Assistant startup
- **Performance Monitoring**: Comprehensive timing and statistics tracking
- **Graceful Degradation**: Search continues to function even if label loading fails

### 🤖 System Prompt Integration
- **Dynamic Context**: Automatically formats labels for AI assistant system prompts
- **Category Intelligence**: Provides structured overview of available product categories
- **Usage Guidelines**: Includes intelligent filtering suggestions for the AI

## Architecture

### SearchService Enhancements

```python
class SearchService:
    # Session-based cache storage
    _catalog_labels_cache: Dict[str, Dict[str, Set[str]]] = {}
    _catalog_labels_loaded: Dict[str, bool] = {}
```

### Key Methods

#### Preloading & Loading
```python
# Preload labels during Assistant initialization
await SearchService.preload_catalog_labels(account)

# Load with caching support and force reload option
labels = await SearchService.load_catalog_labels(account, force_reload=False)
```

#### Cache Access
```python
# Fast cache retrieval (no database hit)
labels = SearchService.get_cached_catalog_labels(account)

# Check loading status
is_loaded = SearchService.is_catalog_labels_loaded(account)
```

#### System Prompt Generation
```python
# Generate formatted context for AI system prompts
prompt_section = SearchService.format_catalog_labels_for_system_prompt(account)
```

#### Cache Management
```python
# Clear specific account cache
SearchService.clear_catalog_labels_cache(account)

# Clear all cached labels
SearchService.clear_catalog_labels_cache()
```

## Assistant Integration

### Initialization Flow

1. **Phase 2a**: Catalog labels preloading task created during Assistant `__init__`
2. **Parallel Loading**: Runs alongside product catalog loading
3. **Performance Tracking**: Timing and statistics logged
4. **System Prompt Enhancement**: Labels automatically integrated into AI context

### Usage in Assistant

```python
class Assistant(Agent):
    def __init__(self, ...):
        # Preloading happens automatically
        if account:
            self._catalog_labels_task = asyncio.create_task(
                self._preload_catalog_labels(account)
            )
    
    async def product_search(self, ...):
        # Ensure labels are loaded before search
        await self.ensure_catalog_labels_loaded()
        # Search now uses cached labels for intelligence
```

## Performance Characteristics

### Startup Performance
- **Target**: Labels loaded in <1s during Assistant initialization
- **Parallel**: Runs alongside product loading (doesn't add to startup time)
- **Monitoring**: Comprehensive timing and efficiency metrics

### Search Performance  
- **Cache Hits**: Sub-millisecond retrieval from memory cache
- **Intelligent Filtering**: Enhanced semantic search using preloaded categories
- **No Database Overhead**: Repeated searches don't trigger database calls

### Memory Usage
- **Efficient Storage**: Labels stored as Python sets for deduplication
- **Session Lifecycle**: Memory automatically freed when Assistant terminates
- **Monitoring**: Memory usage tracked and logged

## System Prompt Enhancement

### Generated Context Format

```markdown
# PRODUCT CATEGORIES AND LABELS

Use these categories to understand customer intent and make precise recommendations:

**Use Cases**: racing, commuting, training, mountain biking, road cycling
**Materials**: carbon fiber, aluminum, steel, titanium (+3 more)
**Features**: lightweight, aerodynamic, durable, comfortable (+7 more)
**Style Attributes**: modern, classic, aggressive, sleek (+2 more)
**Technical Specs**: weight: 15kg, frame: 56cm, gears: 11-speed (+5 more)
**Target Audience**: professional, beginner, intermediate (+1 more)

**Usage Guidelines:**
- Match customer language to appropriate categories (e.g., 'beginner' → skill_level)
- Use multiple categories for precise filtering (e.g., use_cases + materials)
- Consider category combinations for better recommendations
- Explain category relevance when making suggestions
```

### AI Benefits

1. **Category Awareness**: AI understands available product attributes
2. **Intelligent Matching**: Can map customer language to specific categories
3. **Contextual Recommendations**: Uses category knowledge for better suggestions
4. **Filter Guidance**: Knows what filtering options are available

## Implementation Benefits

### For Voice Assistants
- **Faster Search**: Cached labels enable instant semantic filtering
- **Smarter Responses**: AI has structured understanding of product catalog
- **Better UX**: More intelligent product recommendations and filtering

### For Developers  
- **Simple API**: Clean interface with automatic caching
- **Performance Monitoring**: Built-in metrics and timing
- **Graceful Degradation**: Works even if label extraction fails

### For System Performance
- **Stateful Optimization**: Perfect for short-lived voice sessions
- **Memory Efficient**: Optimized storage with automatic cleanup
- **No External Dependencies**: No Redis/cache service complexity

## Usage Examples

### Basic Usage in SearchService

```python
# Preload during initialization
success = await SearchService.preload_catalog_labels("specialized.com")

# Use cached labels for search enhancement  
labels = SearchService.get_cached_catalog_labels("specialized.com")
if labels:
    semantic_filters = SearchService.extract_semantic_filters(query, labels)
```

### Assistant Integration

```python
# Labels automatically preloaded during Assistant creation
assistant = Assistant(ctx=ctx, account="specialized.com", ...)

# Get formatted context for system prompts
context = assistant.get_catalog_intelligence_context()

# Ensure labels are ready before search operations
await assistant.ensure_catalog_labels_loaded()
```

### System Prompt Integration

```python
# Format labels for AI context
prompt_section = SearchService.format_catalog_labels_for_system_prompt(account)

# Add to system prompt (if prompt manager supports dynamic context)
if hasattr(prompt_manager, 'add_dynamic_context'):
    prompt_manager.add_dynamic_context('product_categories', prompt_section)
```

## Monitoring & Debugging

### Performance Metrics

```python
# Get comprehensive statistics
stats = SearchService.get_category_summary_stats(account)
print(f"Categories: {stats['categories']}")
print(f"Total labels: {stats['total_labels']}")
print(f"Sample categories: {list(stats['category_details'].keys())}")
```

### Assistant Performance

```python
# Get full Assistant performance summary
summary = await assistant.get_performance_summary()
print(summary)  # Includes label loading performance
```

### Logging

The system provides comprehensive logging at key points:
- **Preload Success**: `🚀 Preloaded catalog labels for specialized.com in 0.234s (5 categories, 47 labels)`
- **Cache Usage**: `Using cached catalog labels for specialized.com`
- **Performance Warnings**: `⚠️ Catalog labels loading timeout for account specialized.com`

## Testing

Run the comprehensive test suite:

```bash
python test_catalog_labels_caching.py
```

Tests cover:
- ✅ Preloading functionality
- ✅ Cache retrieval performance
- ✅ System prompt generation
- ✅ Cache management
- ✅ Assistant integration

## Best Practices

### Implementation
1. **Always Preload**: Call `preload_catalog_labels()` during Assistant initialization
2. **Use Cache First**: Check cached labels before triggering database loads
3. **Handle Failures Gracefully**: Search should work even if label loading fails
4. **Monitor Performance**: Log timing and efficiency metrics

### Performance
1. **Parallel Loading**: Load labels alongside other initialization tasks
2. **Session Lifecycle**: Clear cache when sessions end to manage memory
3. **Timeout Handling**: Use reasonable timeouts for label loading operations

### AI Integration
1. **System Prompt Enhancement**: Include formatted labels in AI context
2. **Category Awareness**: Train AI to use category knowledge effectively
3. **Filter Intelligence**: Use labels for smarter semantic search

## Future Enhancements

### Potential Improvements
- **Label Quality Scoring**: Rank categories by usage frequency
- **Dynamic Updates**: Refresh labels when catalog changes
- **Cross-Account Intelligence**: Share common categories across accounts
- **ML Enhancement**: Use labels for search result ranking

### Integration Opportunities  
- **Analytics**: Track which categories are most useful for searches
- **Personalization**: Adapt category emphasis based on user preferences
- **A/B Testing**: Compare search effectiveness with/without label intelligence

---

**Key Takeaway**: This caching system transforms static product catalogs into intelligent, category-aware search systems that enable smarter AI-driven product discovery and recommendations. 