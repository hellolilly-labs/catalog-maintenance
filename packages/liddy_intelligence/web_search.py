"""
Web Search Integration for Brand Intelligence

Implements multi-provider web search for brand research and vertical detection.
Supports Tavily API, Google Search API, and fallback methods.
"""

import asyncio
import logging
import json
import ssl
import hashlib
from typing import Dict, List, Optional, Any
import aiohttp
import certifi
from tavily import TavilyClient
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import os
from urllib.parse import urlparse

from configs.settings import get_settings
# from src.redis_client import get_redis_client  # TODO: Implement redis support

logger = logging.getLogger(__name__)

# Cache settings
SEARCH_CACHE_TTL = 2 * 24 * 60 * 60  # 2 days in seconds
SEARCH_CACHE_PREFIX = "tavily_search:"

@dataclass 
class SearchResult:
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None

@dataclass
class CrawlResult:
    """Result from Tavily Crawl endpoint"""
    base_url: str
    results: List[Dict[str, Any]]
    response_time: float
    total_pages: int
    
    @property
    def urls(self) -> List[str]:
        """Extract URLs from crawl results"""
        return [result.get('url', '') for result in self.results]
    
    @property
    def content_by_url(self) -> Dict[str, str]:
        """Get content mapped by URL"""
        return {
            result.get('url', ''): result.get('raw_content', '')
            for result in self.results
        }

@dataclass
class SitemapResult:
    """Result from Tavily Map endpoint"""
    base_url: str
    urls: List[str]
    response_time: float
    
    @property
    def total_pages(self) -> int:
        return len(self.urls)

class WebSearchProvider(ABC):
    """Abstract base class for web search providers"""
    
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        pass
    
    async def crawl_site(self, base_url: str, instructions: str = "") -> Optional[CrawlResult]:
        """Crawl a website (if supported by provider)"""
        raise NotImplementedError("Crawl not supported by this provider")
    
    async def map_site(self, base_url: str) -> Optional[SitemapResult]:
        """Get sitemap of a website (if supported by provider)"""
        raise NotImplementedError("Site mapping not supported by this provider")


def create_ssl_context() -> ssl.SSLContext:
    """Create SSL context using certifi certificates to fix SSL verification issues"""
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        logger.debug(f"Created SSL context with certifi bundle: {certifi.where()}")
        return context
    except Exception as e:
        logger.warning(f"Failed to create SSL context with certifi: {e}")
        # Fallback to default context (may still fail)
        return ssl.create_default_context()


def _generate_cache_key(query: str, search_params: Dict[str, Any]) -> str:
    """Generate a unique cache key based on query and parameters"""
    # Create a deterministic string from query and parameters
    cache_data = {
        "query": query,
        "params": search_params
    }
    
    # Sort parameters to ensure consistent key generation
    cache_string = json.dumps(cache_data, sort_keys=True)
    
    # Create hash of the cache string
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
    
    return f"{SEARCH_CACHE_PREFIX}{cache_hash}"


def _serialize_search_results(results: List[SearchResult]) -> str:
    """Serialize search results for caching"""
    try:
        serializable_results = [asdict(result) for result in results]
        return json.dumps(serializable_results)
    except Exception as e:
        logger.warning(f"Failed to serialize search results: {e}")
        return "[]"


def _deserialize_search_results(cached_data: str) -> List[SearchResult]:
    """Deserialize cached search results"""
    try:
        data = json.loads(cached_data)
        return [SearchResult(**item) for item in data]
    except Exception as e:
        logger.warning(f"Failed to deserialize cached search results: {e}")
        return []


class TavilySearchProvider(WebSearchProvider):
    """Tavily AI-optimized search provider using official tavily-python library with Redis caching"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_settings().TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com"
        self.client = TavilyClient(api_key=api_key)
        self.redis_client = None  # get_redis_client()  # TODO: Re-enable redis when needed
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Search using official Tavily library with Redis caching"""
        try:
            # Prepare search parameters for the official library
            search_params = {
                "search_depth": kwargs.get("search_depth", "advanced"),
                "topic": kwargs.get("topic", "general"),
                "include_answer": kwargs.get("include_answer", True),
                "include_raw_content": kwargs.get("include_raw_content", False),
                "include_images": kwargs.get("include_images", False),
                "max_results": kwargs.get("max_results", 10),
                "include_domains": kwargs.get("include_domains"),
                "exclude_domains": kwargs.get("exclude_domains"),
                "time_range": kwargs.get("time_range"),
                "days": kwargs.get("days", 0)
            }
            
            # Remove None values for consistent cache keys
            search_params = {k: v for k, v in search_params.items() if v is not None}
            
            # Generate cache key
            cache_key = _generate_cache_key(query, search_params)
            
            # Try to get from cache first
            if self.redis_client:
                try:
                    cached_results = self.redis_client.get(cache_key)
                    if cached_results:
                        logger.info(f"🔄 Cache HIT for query: {query[:50]}...")
                        return _deserialize_search_results(cached_results)
                except Exception as e:
                    logger.warning(f"Redis cache read failed: {e}")
            
            # Cache miss or Redis unavailable - make API call
            logger.info(f"🔍 Cache MISS - API call for query: {query[:50]}...")
            
            # Add query to search params for API call
            api_params = {"query": query, **search_params}
            
            # Run the synchronous client method in a thread pool to make it async
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: self.client.search(**api_params)
            )
            
            results = self._parse_search_results(data)
            
            # Cache the results if Redis is available
            if self.redis_client and results:
                try:
                    cached_data = _serialize_search_results(results)
                    self.redis_client.setex(cache_key, SEARCH_CACHE_TTL, cached_data)
                    logger.debug(f"💾 Cached {len(results)} results for 2 days")
                except Exception as e:
                    logger.warning(f"Redis cache write failed: {e}")
            
            return results
                        
        except Exception as e:
            logger.error(f"Error in Tavily search: {e}")
            return []
    
    async def crawl_site(self, base_url: str, instructions: str = "") -> Optional[CrawlResult]:
        """Crawl a website using official Tavily library"""
        try:
            # Run the synchronous client method in a thread pool to make it async
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: self.client.crawl(url=base_url, instructions=instructions)
            )
            
            return CrawlResult(
                base_url=data.get("base_url", base_url),
                results=data.get("results", []),
                response_time=data.get("response_time", 0.0),
                total_pages=len(data.get("results", []))
            )
                        
        except Exception as e:
            logger.error(f"Error in Tavily crawl: {e}")
            return None
    
    async def map_site(self, base_url: str) -> Optional[SitemapResult]:
        """Get sitemap using official Tavily library"""
        try:
            # Run the synchronous client method in a thread pool to make it async
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: self.client.map(url=base_url)
            )
            
            return SitemapResult(
                base_url=data.get("base_url", base_url),
                urls=data.get("results", []),
                response_time=data.get("response_time", 0.0)
            )
                        
        except Exception as e:
            print(f"Error in Tavily map: {e}")
            return None
    
    async def comprehensive_brand_research(
        self, 
        brand_domain: str,
        research_focus: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive brand research using Crawl + Map + Search
        
        Args:
            brand_domain: Brand website domain (e.g., "specialized.com")
            research_focus: List of research focuses like ["company history", "mission", "values"]
        
        Returns:
            Comprehensive research results
        """
        if research_focus is None:
            research_focus = [
                "company history and background",
                "mission, vision, and values", 
                "brand story and founding",
                "leadership team and culture",
                "business model and strategy"
            ]
        
        results = {
            "brand_domain": brand_domain,
            "research_focus": research_focus,
            "sitemap": None,
            "crawl_results": {},
            "search_results": {},
            "synthesis": {}
        }
        
        try:
            # Step 1: Get sitemap to understand site structure
            print(f"🗺️ Mapping site structure for {brand_domain}...")
            sitemap = await self.map_site(f"https://{brand_domain}")
            if sitemap:
                results["sitemap"] = {
                    "total_pages": sitemap.total_pages,
                    "key_sections": self._categorize_urls(sitemap.urls),
                    "response_time": sitemap.response_time
                }
                print(f"   Found {sitemap.total_pages} pages")
            
            # Step 2: Targeted crawling for each research focus
            for focus in research_focus:
                print(f"🕷️ Crawling for: {focus}...")
                
                instructions = f"Find all pages related to {focus}. Focus on pages like About Us, Our Story, Company History, Mission, Values, Leadership, and similar sections."
                
                crawl_result = await self.crawl_site(f"https://{brand_domain}", instructions)
                if crawl_result:
                    results["crawl_results"][focus] = {
                        "total_pages": crawl_result.total_pages,
                        "urls": crawl_result.urls,
                        "content": crawl_result.content_by_url,
                        "response_time": crawl_result.response_time
                    }
                    print(f"   Found {crawl_result.total_pages} relevant pages")
            
            # Step 3: Supplementary web searches for external validation
            for focus in research_focus:
                search_query = f'"{brand_domain.replace(".com", "")}" {focus} company profile'
                print(f"🔍 External search for: {focus}...")
                
                search_results = await self.search(
                    search_query,
                    search_depth="basic",
                    max_results=5,
                    exclude_domains=[brand_domain]  # Get external sources
                )
                
                if search_results:
                    results["search_results"][focus] = [
                        {
                            "title": r.title,
                            "url": r.url,
                            "content": r.content,
                            "score": r.score
                        }
                        for r in search_results[:3]  # Top 3 external sources
                    ]
                    print(f"   Found {len(search_results)} external sources")
            
            # Step 4: Generate synthesis summary
            total_pages_crawled = sum(
                data["total_pages"] for data in results["crawl_results"].values()
            )
            total_external_sources = sum(
                len(data) for data in results["search_results"].values()
            )
            
            results["synthesis"] = {
                "total_pages_crawled": total_pages_crawled,
                "total_external_sources": total_external_sources,
                "research_coverage": list(results["crawl_results"].keys()),
                "data_quality": "comprehensive" if total_pages_crawled > 10 else "moderate"
            }
            
            print(f"✅ Research complete: {total_pages_crawled} pages + {total_external_sources} external sources")
            
        except Exception as e:
            print(f"Error in comprehensive research: {e}")
            results["error"] = str(e)
        
        return results
    
    def _categorize_urls(self, urls: List[str]) -> Dict[str, List[str]]:
        """Categorize URLs into likely sections"""
        categories = {
            "about": [],
            "products": [],
            "company": [],
            "news": [],
            "support": [],
            "other": []
        }
        
        for url in urls:
            url_lower = url.lower()
            if any(term in url_lower for term in ["about", "story", "history", "mission", "values"]):
                categories["about"].append(url)
            elif any(term in url_lower for term in ["product", "bike", "gear", "shop", "catalog"]):
                categories["products"].append(url)
            elif any(term in url_lower for term in ["company", "team", "leadership", "careers"]):
                categories["company"].append(url)
            elif any(term in url_lower for term in ["news", "blog", "press", "media"]):
                categories["news"].append(url)
            elif any(term in url_lower for term in ["support", "help", "contact", "faq"]):
                categories["support"].append(url)
            else:
                categories["other"].append(url)
        
        return {k: v for k, v in categories.items() if v}  # Remove empty categories
    
    def _parse_search_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                published_date=item.get("published_date")
            ))
        return results


class GoogleSearchProvider(WebSearchProvider):
    """Google Custom Search API provider"""
    
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        max_results = kwargs.get('max_results', 10)
        try:
            # Create SSL context with certifi certificates
            ssl_context = create_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                params = {
                    "key": self.api_key,
                    "cx": self.search_engine_id,
                    "q": query,
                    "num": min(max_results, 10)  # Google API limit
                }
                
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        for item in data.get("items", []):
                            results.append(SearchResult(
                                title=item.get("title", ""),
                                url=item.get("link", ""),
                                content=item.get("snippet", ""),
                                score=1.0  # Google doesn't provide scores
                            ))
                        
                        return results
                    else:
                        logger.error(f"Google search failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error in Google search: {e}")
            return []


class BrandWebSearchEngine:
    """
    Multi-provider web search engine for brand research
    
    Automatically selects the best available search provider
    and provides brand-specific search capabilities.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.providers = self._initialize_providers()
    
    def _initialize_providers(self) -> List[WebSearchProvider]:
        """Initialize available search providers based on configuration"""
        providers = []
        
        # Tavily (preferred for AI research)
        if self.settings.TAVILY_API_KEY:
            providers.append(TavilySearchProvider(self.settings.TAVILY_API_KEY))
            logger.info("Initialized Tavily search provider")
        
        # Google Custom Search (backup)
        if self.settings.GOOGLE_SEARCH_API_KEY and self.settings.GOOGLE_SEARCH_ENGINE_ID:
            providers.append(GoogleSearchProvider(
                self.settings.GOOGLE_SEARCH_API_KEY,
                self.settings.GOOGLE_SEARCH_ENGINE_ID
            ))
            logger.info("Initialized Google search provider")
        
        if not providers:
            logger.warning("No web search providers configured")
        
        return providers
    
    async def search(self, query: str, max_results: int = 10, include_domains: List[str] = None) -> Dict[str, Any]:
        """
        General search method using the best available provider
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dict with search results
        """
        if not self.providers:
            logger.warning("No search providers available")
            return {"results": [], "error": "No search providers configured"}
        
        provider = self.providers[0]
        
        try:
            results = await provider.search(query, max_results=max_results, include_domains=include_domains)
            return {
                "query": query,
                "total_results": len(results),
                "results": results,
                "provider_used": provider.__class__.__name__
            }
        except Exception as e:
            logger.error(f"Error in search for '{query}': {e}")
            return {"results": [], "error": str(e)}
    
    async def search_brand_info(self, brand_domain: str) -> Dict[str, Any]:
        """
        Search for comprehensive brand information using direct vertical questions
        
        Args:
            brand_domain: Brand domain (e.g., "specialized.com")
            
        Returns:
            Dict with search results and analysis
        """
        if not self.providers:
            logger.warning("No search providers available")
            return {"results": [], "analysis": "No search providers configured"}
        
        # ENHANCED: Direct vertical detection queries (addressing user feedback)
        search_queries = [
            f"What is the primary business vertical that {brand_domain} operates in",
            f"{brand_domain} company main industry sector core business focus",
            f"What does {brand_domain} primarily manufacture and sell",
            f"{brand_domain} business type industry classification primary market",
            f"site:{brand_domain} about company industry business vertical",
            f"{brand_domain} company profile what industry are they in"
        ]
        
        all_results = []
        
        # Search with the first available provider
        provider = self.providers[0]
        
        for query in search_queries:
            try:
                results = await provider.search(query, max_results=6)  # More results per query
                for result in results:
                    result["query"] = query
                    result["query_type"] = self._classify_query_type(query)
                    all_results.append(result)
                
                # Small delay between queries to be respectful
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
        
        return {
            "brand_domain": brand_domain,
            "total_results": len(all_results),
            "results": all_results,
            "queries_used": search_queries,
            "provider_used": provider.__class__.__name__,
            "search_strategy": "direct_vertical_questions"
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of search query for better analysis"""
        query_lower = query.lower()
        
        if "what is" in query_lower and "vertical" in query_lower:
            return "direct_vertical_question"
        elif "what does" in query_lower and "primarily" in query_lower:
            return "primary_business_question"  
        elif "site:" in query_lower:
            return "domain_specific_search"
        elif "company profile" in query_lower:
            return "company_profile_search"
        elif "industry" in query_lower and "sector" in query_lower:
            return "industry_classification_search"
        else:
            return "general_brand_search"
    
    async def search_domain_analysis(self, brand_domain: str) -> Optional[str]:
        """
        Simple domain-based brand analysis without API calls
        
        Args:
            brand_domain: Brand domain to analyze
            
        Returns:
            Likely vertical based on domain analysis
        """
        # Domain pattern analysis (fallback when no APIs available)
        domain_patterns = {
            "cycling": ["bike", "cycle", "specialized", "trek", "giant", "cannondale"],
            "fashion": ["fashion", "style", "wear", "clothing", "apparel"],
            "beauty": ["beauty", "cosmetic", "skin", "makeup", "sephora"],
            "electronics": ["tech", "digital", "electronic", "apple", "samsung"],
            "sports": ["sport", "athletic", "fitness", "nike", "adidas"],
            "outdoor": ["outdoor", "hiking", "camping", "patagonia", "rei"],
            "automotive": ["auto", "car", "motor", "bmw", "ford", "tesla"],
            "home": ["home", "furniture", "decor", "ikea", "wayfair"]
        }
        
        domain_lower = brand_domain.lower()
        
        for vertical, keywords in domain_patterns.items():
            if any(keyword in domain_lower for keyword in keywords):
                logger.info(f"Domain pattern analysis suggests '{vertical}' for {brand_domain}")
                return vertical
        
        return None
    
    def is_available(self) -> bool:
        """Check if any search providers are available"""
        return len(self.providers) > 0
    
    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of available providers"""
        return {
            "tavily": bool(self.settings.TAVILY_API_KEY),
            "google": bool(self.settings.GOOGLE_SEARCH_API_KEY and self.settings.GOOGLE_SEARCH_ENGINE_ID),
            "total_providers": len(self.providers)
        }


# Factory function
def get_web_search_engine() -> BrandWebSearchEngine:
    """Get configured web search engine"""
    return BrandWebSearchEngine() 