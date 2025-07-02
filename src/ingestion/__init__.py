"""
Ingestion module for RAG system
"""

from .universal_product_processor import UniversalProductProcessor
from .sparse_embeddings import SparseEmbeddingGenerator

# Conditionally import PineconeIngestion to avoid import errors
try:
    from .pinecone_ingestion import PineconeIngestion
    __all__ = ['UniversalProductProcessor', 'PineconeIngestion', 'SparseEmbeddingGenerator']
except ImportError:
    __all__ = ['UniversalProductProcessor', 'SparseEmbeddingGenerator']