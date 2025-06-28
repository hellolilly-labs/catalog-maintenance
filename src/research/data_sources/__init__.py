"""
Data Source Strategy Pattern for Research Pipeline

This package implements the Strategy pattern for data gathering in research phases.
Each data source strategy handles a specific type of data collection, making the
research pipeline more modular and testable.

Usage:
    from src.research.data_sources import WebSearchDataSource, ProductCatalogDataSource
    
    web_source = WebSearchDataSource()
    data = await web_source.gather(queries, context)
"""

from .base import DataSource
from .web_search import WebSearchDataSource
from .product_catalog import ProductCatalogDataSource

__all__ = ['DataSource', 'WebSearchDataSource', 'ProductCatalogDataSource']