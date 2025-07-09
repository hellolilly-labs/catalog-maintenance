"""
Web Search Data Source Strategy

Implements web search-based data gathering using the existing web search infrastructure.
Extracts and centralizes the common web search patterns from researcher implementations.
"""

import logging
from typing import List, Dict, Any
from .base import DataSource, DataGatheringContext, DataGatheringResult

logger = logging.getLogger(__name__)


class WebSearchDataSource(DataSource):
    """
    Data source strategy for web search-based data gathering.
    
    Centralizes the common web search logic that was previously duplicated
    across multiple researcher implementations.
    """
    
    def __init__(self):
        self._web_search_engine = None
        self._engine_initialized = False
    
    async def _get_web_search_engine(self):
        """Lazy initialization of web search engine"""
        if not self._engine_initialized:
            try:
                from src.web_search import get_web_search_engine
                self._web_search_engine = get_web_search_engine()
                self._engine_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize web search engine: {e}")
                self._web_search_engine = None
                self._engine_initialized = True
        return self._web_search_engine
    
    def is_available(self) -> bool:
        """Check if web search is available"""
        # Import check without initialization
        try:
            from src.web_search import get_web_search_engine
            return True
        except ImportError:
            return False
    
    async def gather(self, queries: List[Any], context: DataGatheringContext) -> DataGatheringResult:
        """
        Gather data using web search queries.
        
        Args:
            queries: List of search queries to execute
            context: Context information including brand domain and research phase
            
        Returns:
            DataGatheringResult with search results and metadata
        """
        web_search = await self._get_web_search_engine()
        
        if not web_search or not web_search.is_available():
            logger.warning("Web search engine not available, returning empty results")
            return DataGatheringResult(
                results=[],
                sources=[],
                successful_searches=0,
                failed_searches=len(queries),
                metadata={"error": "Web search engine not available"}
            )
        
        all_results = []
        detailed_sources = []
        successful_searches = 0
        failed_searches = 0
        ssl_errors = 0
        
        # Execute search queries
        for query_idx, query in enumerate(queries):
            try:
                # Handle both string queries and dictionary queries
                if isinstance(query, dict):
                    query_string = query.get("query", "")
                    max_results = query.get("max_results", 3)
                    include_domains = query.get("include_domains", None)
                    search_results = await web_search.search(query_string, max_results=max_results, include_domains=include_domains)
                else:
                    query_string = str(query)
                    max_results = 3
                    search_results = await web_search.search(query_string)
                
                if search_results.get("results"):
                    successful_searches += 1
                    
                    # Process results (limited by max_results)
                    for result_idx, result in enumerate(search_results["results"][:max_results]):
                        # Convert SearchResult object to dictionary
                        result_dict = {
                            "title": result.title,
                            "url": result.url,
                            "content": result.content,
                            "snippet": result.content,  # Use content as snippet
                            "score": result.score,
                            "published_date": result.published_date
                        }
                        
                        # Add search context to result
                        result_dict["source_query"] = query if isinstance(query, str) else query_string
                        result_dict["source_type"] = context.researcher_name
                        result_dict["query_index"] = query_idx
                        result_dict["result_index"] = result_idx
                        all_results.append(result_dict)
                        
                        # Create detailed source record
                        source_record = {
                            "source_id": f"query_{query_idx}_result_{result_idx}",
                            "title": result_dict.get("title", ""),
                            "url": result_dict.get("url", ""),
                            "snippet": result_dict.get("snippet", ""),
                            "search_query": query_string,
                            "search_score": result_dict.get("score", 0.0),
                            "published_date": result_dict.get("published_date"),
                            "collection_timestamp": "2024-12-20T12:00:00Z",  # TODO: Use actual timestamp
                            "search_provider": "tavily",
                            "result_rank": result_idx + 1,
                            "query_rank": query_idx + 1
                        }
                        detailed_sources.append(source_record)
                        
                else:
                    failed_searches += 1
                    logger.warning(f"No results found for query: {query_string}")
                    
            except Exception as e:
                failed_searches += 1
                
                # Check for SSL errors specifically
                if "ssl" in str(e).lower() or "certificate" in str(e).lower():
                    ssl_errors += 1
                    logger.warning(f"SSL error for query '{query_string}': {e}")
                else:
                    logger.error(f"Search error for query '{query_string}': {e}")
        
        # Log search statistics
        total_queries = len(queries)
        logger.info(f"Web search completed: {successful_searches}/{total_queries} queries successful, "
                   f"{failed_searches} failed, {ssl_errors} SSL errors")
        
        return DataGatheringResult(
            results=all_results,
            sources=detailed_sources,
            successful_searches=successful_searches,
            failed_searches=failed_searches,
            ssl_errors=ssl_errors,
            metadata={
                "search_provider": "tavily",
                "total_queries": total_queries,
                "results_per_query": 3,
                "search_timestamp": "2024-12-20T12:00:00Z"  # TODO: Use actual timestamp
            }
        )