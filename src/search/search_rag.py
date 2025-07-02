"""
SearchRAG - Backward Compatibility Module

This module provides backward compatibility for code that imports from search_rag.
It exports the SearchRAG interface and the default Pinecone implementation.
"""

from .base import BaseRAG, SearchResult
from .search_pinecone import PineconeRAG, get_search_pinecone

# For backward compatibility
get_search_rag = get_search_pinecone

__all__ = ['BaseRAG', 'SearchResult', 'PineconeRAG', 'get_search_rag', 'get_search_pinecone']