"""
Product Catalog Data Source Strategy

Implements product catalog-based data gathering for research phases that need
product information. Currently used primarily by ProductStyleResearcher.
"""

import logging
from typing import List, Dict, Any
from .base import DataSource, DataGatheringContext, DataGatheringResult

logger = logging.getLogger(__name__)


class ProductCatalogDataSource(DataSource):
    """
    Data source strategy for product catalog-based data gathering.
    
    Handles integration with product catalog systems to gather product-specific
    information for research phases that need product context.
    """
    
    def __init__(self):
        self._product_manager = None
        self._manager_initialized = False
    
    async def _get_product_manager(self):
        """Lazy initialization of product manager"""
        if not self._manager_initialized:
            try:
                from src.models.product_manager import get_product_manager
                self._product_manager = get_product_manager()
                self._manager_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize product manager: {e}")
                self._product_manager = None
                self._manager_initialized = True
        return self._product_manager
    
    def is_available(self) -> bool:
        """Check if product catalog is available"""
        try:
            from src.models.product_manager import get_product_manager
            return True
        except ImportError:
            return False
    
    async def gather(self, queries: List[str], context: DataGatheringContext) -> DataGatheringResult:
        """
        Gather data from product catalog.
        
        Args:
            queries: List of product-related queries (may be product IDs, categories, etc.)
            context: Context information including brand domain
            
        Returns:
            DataGatheringResult with product data and metadata
        """
        product_manager = await self._get_product_manager()
        
        if not product_manager:
            logger.warning("Product manager not available, returning empty results")
            return DataGatheringResult(
                results=[],
                sources=[],
                successful_searches=0,
                failed_searches=len(queries),
                metadata={"error": "Product manager not available"}
            )
        
        all_results = []
        detailed_sources = []
        successful_searches = 0
        failed_searches = 0
        
        try:
            # Get products for the brand
            products = await product_manager.get_products_for_brand(context.brand_domain)
            
            if products:
                successful_searches = 1  # Consider this one successful operation
                
                # Process products into result format
                for product_idx, product in enumerate(products[:20]):  # Limit to top 20 products
                    # Convert product to result format
                    result_dict = {
                        "title": product.get("name", "Unknown Product"),
                        "url": product.get("url", ""),
                        "content": self._format_product_content(product),
                        "snippet": product.get("description", "")[:200],
                        "score": 1.0,  # Products are all equally relevant
                        "published_date": None
                    }
                    
                    # Add product-specific context
                    result_dict["source_query"] = f"products_{context.brand_domain}"
                    result_dict["source_type"] = f"{context.researcher_name}_catalog"
                    result_dict["query_index"] = 0
                    result_dict["result_index"] = product_idx
                    result_dict["product_id"] = product.get("id")
                    all_results.append(result_dict)
                    
                    # Create detailed source record
                    source_record = {
                        "source_id": f"product_{product_idx}",
                        "title": result_dict.get("title", ""),
                        "url": result_dict.get("url", ""),
                        "snippet": result_dict.get("snippet", ""),
                        "search_query": f"products for {context.brand_domain}",
                        "search_score": 1.0,
                        "published_date": None,
                        "collection_timestamp": "2024-12-20T12:00:00Z",  # TODO: Use actual timestamp
                        "search_provider": "product_catalog",
                        "result_rank": product_idx + 1,
                        "query_rank": 1,
                        "product_id": product.get("id")
                    }
                    detailed_sources.append(source_record)
            else:
                failed_searches = 1
                logger.warning(f"No products found for brand: {context.brand_domain}")
                
        except Exception as e:
            failed_searches = 1
            logger.error(f"Error gathering product catalog data for {context.brand_domain}: {e}")
        
        logger.info(f"Product catalog data gathering completed: {successful_searches} successful, "
                   f"{failed_searches} failed")
        
        return DataGatheringResult(
            results=all_results,
            sources=detailed_sources,
            successful_searches=successful_searches,
            failed_searches=failed_searches,
            ssl_errors=0,
            metadata={
                "search_provider": "product_catalog",
                "total_products": len(all_results),
                "brand_domain": context.brand_domain,
                "collection_timestamp": "2024-12-20T12:00:00Z"  # TODO: Use actual timestamp
            }
        )
    
    def _format_product_content(self, product: Dict[str, Any]) -> str:
        """Format product data into content string for analysis"""
        content_parts = []
        
        if product.get("name"):
            content_parts.append(f"Product: {product['name']}")
        
        if product.get("description"):
            content_parts.append(f"Description: {product['description']}")
        
        if product.get("category"):
            content_parts.append(f"Category: {product['category']}")
        
        if product.get("price"):
            content_parts.append(f"Price: {product['price']}")
        
        if product.get("features"):
            if isinstance(product["features"], list):
                content_parts.append(f"Features: {', '.join(product['features'])}")
            else:
                content_parts.append(f"Features: {product['features']}")
        
        return " | ".join(content_parts)