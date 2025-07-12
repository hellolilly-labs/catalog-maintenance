"""
Search Module

Provides unified search functionality with multiple backend support.
"""

from .base import BaseRAG, SearchResult
from .pinecone import PineconeRAG, get_search_pinecone
from .service import SearchService, SearchMetrics

__all__ = [
    'BaseRAG',
    'SearchResult', 
    'PineconeRAG',
    'get_search_pinecone',
    'SearchService',
    'SearchMetrics'
]