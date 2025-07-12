"""
Core ingestion modules for product catalog processing.

This package contains the core components for ingesting product catalogs:
- Universal product processing
- Pinecone vector database integration
- Sparse embedding generation
- STT vocabulary extraction
"""

from .universal_product_processor import UniversalProductProcessor
from .sparse_embeddings import SparseEmbeddingGenerator
from .separate_index_ingestion import SeparateIndexIngestion
from .stt_vocabulary_extractor import STTVocabularyExtractor

# Optional imports
try:
    from .pinecone_ingestion import PineconeIngestion
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