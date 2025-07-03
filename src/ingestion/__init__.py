"""
Ingestion module for RAG system

This module has been migrated to the monorepo structure.
Please update your imports to use:
  from liddy_intelligence.ingestion.core import UniversalProductProcessor, SparseEmbeddingGenerator, PineconeIngestion
"""

# Backward compatibility imports
from liddy_intelligence.ingestion.core.universal_product_processor import UniversalProductProcessor
from liddy_intelligence.ingestion.core.sparse_embeddings import SparseEmbeddingGenerator

# Conditionally import PineconeIngestion to avoid import errors
try:
    from liddy_intelligence.ingestion.core.pinecone_ingestion import PineconeIngestion
    __all__ = ['UniversalProductProcessor', 'PineconeIngestion', 'SparseEmbeddingGenerator']
except ImportError:
    __all__ = ['UniversalProductProcessor', 'SparseEmbeddingGenerator']