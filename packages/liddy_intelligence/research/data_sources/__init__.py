"""
Data Source Strategy Pattern for Research Pipeline

This package implements the Strategy pattern for data gathering in research phases.
Each data source strategy handles a specific type of data collection, making the
research pipeline more modular and testable.

Usage:
    from liddy_intelligence.research.data_sources import WebSearchDataSource, ProductCatalogDataSource
    
    web_source = WebSearchDataSource()
    data = await web_source.gather(queries, context)
"""

from liddy_intelligence.research.data_sources.base import DataSource, DataGatheringContext, DataGatheringResult
from liddy_intelligence.research.data_sources.web_search import WebSearchDataSource
from liddy_intelligence.research.data_sources.product_catalog import ProductCatalogDataSource

__all__ = ['DataSource', 'DataGatheringContext', 'DataGatheringResult', 'WebSearchDataSource', 'ProductCatalogDataSource']