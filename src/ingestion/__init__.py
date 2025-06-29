"""
Ingestion module for RAG system
"""

from .universal_product_processor import UniversalProductProcessor
from .pinecone_ingestion import PineconeIngestion

__all__ = ['UniversalProductProcessor', 'PineconeIngestion']