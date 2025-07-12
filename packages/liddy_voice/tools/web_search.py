"""
Web Search tool for voice assistant
Provides simple web search functionality for competitor product searches
"""

import logging
from typing import List, Dict, Any, Optional
from liddy_intelligence.web_search import get_web_search_engine

logger = logging.getLogger(__name__)

async def search_web(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for information
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results with title, url, and content
    """
    try:
        search_engine = get_web_search_engine()
        
        # Check if search is available
        if not search_engine.is_available():
            logger.warning("No web search providers available")
            return []
        
        # Perform search
        results = await search_engine.search(
            query=query,
            max_results=num_results
        )
        
        # Convert results to simple format
        search_results = []
        for result in results.get("results", []):
            search_results.append({
                "title": result.title,
                "url": result.url,
                "content": result.content,
                "score": result.score
            })
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return []