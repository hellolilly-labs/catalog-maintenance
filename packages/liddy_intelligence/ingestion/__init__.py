"""
Product Catalog Ingestion Package

This package provides comprehensive product catalog ingestion capabilities:

Core Components:
- UniversalProductProcessor: Handles any product format
- SeparateIndexIngestion: Manages separate dense/sparse indexes
- PineconeIngestion: Legacy single-index ingestion
- SparseEmbeddingGenerator: Creates sparse embeddings
- STTVocabularyExtractor: Extracts vocabulary for speech recognition

Scripts:
- ingest_catalog.py: Main ingestion script
- pre_generate_descriptors.py: Pre-generate product descriptors
"""

# Core imports
from .core.universal_product_processor import UniversalProductProcessor
from .core.sparse_embeddings import SparseEmbeddingGenerator
from .core.separate_index_ingestion import SeparateIndexIngestion
from .core.stt_vocabulary_extractor import STTVocabularyExtractor

# Try to import PineconeIngestion (may fail if pinecone not installed)
try:
    from .core.pinecone_ingestion import PineconeIngestion
    __all__ = [
        'UniversalProductProcessor',
        'SparseEmbeddingGenerator', 
        'SeparateIndexIngestion',
        'STTVocabularyExtractor',
        'PineconeIngestion'
    ]
except ImportError:
    __all__ = [
        'UniversalProductProcessor',
        'SparseEmbeddingGenerator',
        'SeparateIndexIngestion', 
        'STTVocabularyExtractor'
    ]