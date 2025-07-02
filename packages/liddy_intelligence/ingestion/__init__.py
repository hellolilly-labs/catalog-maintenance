"""
Ingestion module for RAG system
"""

from liddy_intelligence.ingestion.universal_product_processor import UniversalProductProcessor
from liddy_intelligence.ingestion.sparse_embeddings import SparseEmbeddingGenerator

# Conditionally import PineconeIngestion to avoid import errors
try:
    from liddy_intelligence.ingestion.pinecone_ingestion import PineconeIngestion
    __all__ = ['UniversalProductProcessor', 'PineconeIngestion', 'SparseEmbeddingGenerator']
except ImportError:
    __all__ = ['UniversalProductProcessor', 'SparseEmbeddingGenerator']