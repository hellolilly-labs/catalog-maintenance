"""
Sparse Embedding Generation using rank-bm25

Implements BM25-based sparse embeddings using the proven rank-bm25 library
while maintaining custom brand-specific term weighting and Pinecone compatibility.
"""

import re
import json
import logging
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

from ..models.product import Product
from ..storage import get_account_storage_provider

logger = logging.getLogger(__name__)


class SparseEmbeddingGenerator:
    """
    Generates sparse embeddings for products using rank-bm25 library.
    
    This approach creates keyword-based representations that complement
    dense embeddings for improved search accuracy, especially for:
    - Brand names
    - Model numbers
    - Technical specifications
    - Exact phrase matching
    """
    
    def __init__(self, brand_domain: str, max_features: int = 50000):
        self.brand_domain = brand_domain
        self.max_features = max_features
        
        # Initialize storage provider
        self.storage = get_account_storage_provider()
        
        # Brand-specific term weights
        self.term_importance = self._load_term_importance()
        
        # BM25 instance (initialized when vocabulary is built)
        self.bm25 = None
        self.corpus_tokens = []
        
        # Vocabulary mappings
        self.vocabulary = {}
        self.inverse_vocabulary = {}
        
        # Vocabulary will be loaded when needed
        self._vocabulary_loaded = False
        
        logger.info(f"📊 Initialized BM25-based Sparse Embedding Generator for {brand_domain}")
    
    async def build_vocabulary(self, products: List[Product]) -> None:
        """
        Build vocabulary and BM25 model from product catalog.
        
        Args:
            products: List of Product objects
        """
        logger.info(f"🔨 Building vocabulary from {len(products)} products")
        
        # Try to load existing vocabulary first
        if not self._vocabulary_loaded:
            await self._load_vocabulary()
            self._vocabulary_loaded = True
        
        # Convert products to tokenized documents
        self.corpus_tokens = []
        all_tokens = set()
        
        for product in products:
            tokens = self._extract_and_weight_tokens(product)
            # For BM25, we need just the tokens (not weights yet)
            doc_tokens = []
            for token, weight in tokens:
                # Repeat tokens based on weight (simple but effective)
                repeat_count = max(1, int(weight))
                doc_tokens.extend([token] * repeat_count)
            
            self.corpus_tokens.append(doc_tokens)
            all_tokens.update(doc_tokens)
        
        # Build vocabulary from unique tokens
        # Sort for consistent ordering
        sorted_tokens = sorted(all_tokens)
        
        # Limit to max_features if necessary
        if len(sorted_tokens) > self.max_features:
            # Count token frequencies across all documents
            token_counts = Counter()
            for doc_tokens in self.corpus_tokens:
                token_counts.update(doc_tokens)
            
            # Keep most common tokens
            most_common = token_counts.most_common(self.max_features)
            sorted_tokens = sorted([token for token, _ in most_common])
        
        # Create vocabulary mappings
        self.vocabulary = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.inverse_vocabulary = {idx: token for token, idx in self.vocabulary.items()}
        
        # Filter corpus tokens to only include vocabulary terms
        filtered_corpus = []
        for doc_tokens in self.corpus_tokens:
            filtered_tokens = [t for t in doc_tokens if t in self.vocabulary]
            filtered_corpus.append(filtered_tokens)
        
        self.corpus_tokens = filtered_corpus
        
        # Initialize BM25 model
        self.bm25 = BM25Okapi(
            self.corpus_tokens,
            k1=1.2,  # term frequency saturation
            b=0.75,  # length normalization
            epsilon=0.25  # floor value for IDF
        )
        
        logger.info(f"✅ Built vocabulary with {len(self.vocabulary)} terms")
        logger.info(f"   Average document length: {np.mean([len(doc) for doc in self.corpus_tokens]):.1f} tokens")
        
        # Save vocabulary and model data
        await self._save_vocabulary()
    
    def generate_sparse_embedding(
        self,
        product: Product
    ) -> Dict[str, List[float]]:
        """
        Generate sparse embedding for a single product.
        
        Args:
            product: Product object
            
        Returns:
            Dictionary with 'indices' and 'values' for sparse representation
        """
        if not self.bm25:
            logger.warning("BM25 model not initialized. Building vocabulary first...")
            return {'indices': [], 'values': []}
        
        # Extract tokens with weights for the query product
        query_tokens_weighted = self._extract_and_weight_tokens(product)
        
        # Create query document with repeated tokens based on weights
        query_doc = []
        for token, weight in query_tokens_weighted:
            if token in self.vocabulary:
                repeat_count = max(1, int(weight))
                query_doc.extend([token] * repeat_count)
        
        # For sparse embeddings, we want the token-level scores
        # not document-level scores, so we calculate them directly
        token_scores = {}
        
        # Calculate importance scores for each unique token in the query
        for token in set(query_doc):
            if token in self.vocabulary:
                # Get IDF score from BM25 model
                if hasattr(self.bm25, 'idf') and token in self.bm25.idf:
                    idf_score = self.bm25.idf[token]
                else:
                    # Fallback: calculate IDF manually
                    doc_freq = sum(1 for doc in self.corpus_tokens if token in doc)
                    idf_score = np.log((len(self.corpus_tokens) - doc_freq + 0.5) / (doc_freq + 0.5))
                
                # Apply term frequency from the query
                tf = query_doc.count(token)
                # BM25 term frequency saturation
                tf_component = (tf * (self.bm25.k1 + 1)) / (tf + self.bm25.k1)
                
                # Combine IDF, TF, and brand-specific importance
                brand_boost = self.term_importance.get(token, 1.0)
                token_scores[token] = idf_score * tf_component * brand_boost
        
        # Convert to sparse vector format
        indices = []
        values = []
        
        for token, score in token_scores.items():
            idx = self.vocabulary[token]
            indices.append(idx)
            values.append(score)
        
        # Sort by index for consistency
        if indices:
            sorted_pairs = sorted(zip(indices, values))
            indices, values = zip(*sorted_pairs)
            indices = list(indices)
            values = list(values)
            
            # Normalize values to [0, 1] range
            max_val = max(values)
            if max_val > 0:
                values = [v / max_val for v in values]
        
        return {
            'indices': indices,
            'values': values
        }
    
    def _extract_and_weight_tokens(
        self,
        product: Product
    ) -> List[Tuple[str, float]]:
        """
        Extract tokens with their importance weights from a product.
        
        Returns:
            List of (token, weight) tuples
        """
        weighted_tokens = []
        
        # Brand (highest importance: 10.0)
        if product.brand:
            brand_lower = product.brand.lower()
            weighted_tokens.append((brand_lower, 10.0))
            for token in self._tokenize(brand_lower):
                weighted_tokens.append((token, 8.0))
        
        # Product name (weight: 8.0)
        if product.name:
            name_lower = product.name.lower()
            # Full name gets high weight
            weighted_tokens.append((name_lower, 8.0))
            
            # Individual tokens
            for token in self._tokenize(name_lower):
                weighted_tokens.append((token, 6.0))
            
            # Model numbers get very high weight
            for model in self._extract_model_numbers(product.name):
                weighted_tokens.append((model, 9.0))
        
        # Categories (weight: 5.0)
        for category in product.categories:
            if category:
                cat_lower = category.lower()
                weighted_tokens.append((cat_lower, 5.0))
                for token in self._tokenize(cat_lower):
                    weighted_tokens.append((token, 4.0))
        
        # Price range (weight: 3.0)
        # Handle price as string (may have $ and commas)
        price_str = product.salePrice or product.originalPrice
        if price_str:
            try:
                # Remove $ and commas, convert to float
                price = float(price_str.replace('$', '').replace(',', ''))
                if price > 0:
                    price_range = self._get_price_range(price)
                    weighted_tokens.append((price_range, 3.0))
            except (ValueError, AttributeError):
                pass
        
        # Search keywords (weight: 4.0)
        if product.search_keywords:
            for keyword in product.search_keywords:
                keyword_lower = keyword.lower()
                # Add the full phrase (important for multi-word searches)
                weighted_tokens.append((keyword_lower, 4.0))
                # Also tokenize for partial matches
                for token in self._tokenize(keyword_lower):
                    if len(token) > 2:  # Skip very short tokens
                        weighted_tokens.append((token, 3.0))
        
        # Product labels (weight: 3.0)
        if product.product_labels:
            for _, values in product.product_labels.items():
                if isinstance(values, str):
                    value_lower = values.lower()
                    # Add the full label phrase
                    weighted_tokens.append((value_lower, 3.0))
                    # Also tokenize for partial matches
                    for token in self._tokenize(value_lower):
                        if len(token) > 2:
                            weighted_tokens.append((token, 2.0))
                elif isinstance(values, list):
                    for v in values:
                        if isinstance(v, str):
                            v_lower = v.lower()
                            # Add the full label phrase
                            weighted_tokens.append((v_lower, 3.0))
                            # Also tokenize for partial matches
                            for token in self._tokenize(v_lower):
                                if len(token) > 2:
                                    weighted_tokens.append((token, 2.0))
        
        # Key selling points (weight: 2.0)
        if product.key_selling_points:
            for point in product.key_selling_points:
                for token in self._tokenize(point.lower()):
                    if len(token) > 3:  # Skip short words
                        weighted_tokens.append((token, 2.0))
        
        # Highlights (weight: 2.5) - these are often important features
        if product.highlights:
            for highlight in product.highlights:
                for token in self._tokenize(highlight.lower()):
                    if len(token) > 3:
                        weighted_tokens.append((token, 2.5))
        
        # Specifications (weight: 2.0)
        if product.specifications:
            for spec_key, spec_value in product.specifications.items():
                # Add the spec key
                for token in self._tokenize(spec_key.lower()):
                    weighted_tokens.append((token, 2.0))
                # Add the spec value if it's a string
                if isinstance(spec_value, str):
                    for token in self._tokenize(spec_value.lower()):
                        if len(token) > 2:  # Even shorter threshold for spec values
                            weighted_tokens.append((token, 2.0))
        
        # Colors (weight: 2.0)
        if product.colors:
            for color in product.colors:
                if isinstance(color, str):
                    weighted_tokens.append((color.lower(), 2.0))
                elif isinstance(color, dict) and 'name' in color:
                    weighted_tokens.append((color['name'].lower(), 2.0))
        
        # Sizes (weight: 1.5)
        if product.sizes:
            for size in product.sizes:
                weighted_tokens.append((size.lower(), 1.5))
        
        # Description (weight: 1.0)
        if product.description:
            desc_tokens = self._tokenize(product.description.lower())
            for token in desc_tokens[:50]:  # Limit description tokens
                weighted_tokens.append((token, 1.0))
        
        # Voice summary (weight: 1.5) - concise, high-value content
        if product.voice_summary:
            for token in self._tokenize(product.voice_summary.lower()):
                if len(token) > 3:
                    weighted_tokens.append((token, 1.5))
        
        return weighted_tokens
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        """
        # Remove special characters but keep hyphens and periods in model numbers
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Filter and clean tokens
        cleaned_tokens = []
        for token in tokens:
            # Remove trailing punctuation
            token = token.strip('.-')
            
            # Keep if meaningful length
            if len(token) >= 2:
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def _extract_model_numbers(self, text: str) -> List[str]:
        """
        Extract model numbers and codes from text.
        """
        model_patterns = [
            r'\b[A-Z0-9]{2,}[\-_]?[A-Z0-9]+\b',  # ABC-123, XY_456
            r'\b[A-Z]{2,}\d{2,}\b',              # ABC123
            r'\b\d{3,}[A-Z]+\b',                 # 123ABC
            r'\bv\d+\.\d+\b',                    # v1.0, v2.5
            r'\b[A-Z]+[\-]\d+\b',                # SL-7, XR-12
        ]
        
        models = []
        for pattern in model_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            models.extend([m.lower() for m in matches])
        
        return list(set(models))
    
    def _get_price_range(self, price: float) -> str:
        """
        Convert price to searchable range term.
        """
        if price < 50:
            return "budget"
        elif price < 150:
            return "affordable"
        elif price < 500:
            return "midrange"
        elif price < 1500:
            return "premium"
        else:
            return "luxury"
    
    def _load_term_importance(self) -> Dict[str, float]:
        """
        Load brand-specific term importance weights dynamically.
        """
        importance = {}
        
        # Instead of hardcoding, we could load from a config file
        # For now, we'll do basic pattern matching
        brand_lower = self.brand_domain.lower()
        
        # Extract brand name from domain
        brand_name = brand_lower.split('.')[0]
        
        # Always boost the brand name itself
        importance[brand_name] = 2.0
        
        # Common patterns that indicate importance
        # These could be loaded from brand research in the future
        
        return importance
    
    async def _save_vocabulary(self) -> None:
        """Save vocabulary and BM25 data using storage manager."""
        # Calculate document frequencies for storage
        doc_frequencies = {}
        for token in self.vocabulary:
            doc_freq = sum(1 for doc in self.corpus_tokens if token in doc)
            doc_frequencies[token] = doc_freq
        
        vocab_data = {
            'vocabulary': self.vocabulary,
            'document_frequencies': doc_frequencies,
            'total_documents': len(self.corpus_tokens),
            'max_features': self.max_features,
            'average_doc_length': float(np.mean([len(doc) for doc in self.corpus_tokens])),
            'term_importance': self.term_importance,
            'bm25_params': {
                'k1': self.bm25.k1,
                'b': self.bm25.b,
                'epsilon': self.bm25.epsilon
            }
        }
        
        # Save using storage manager
        success = await self.storage.write_file(
            account=self.brand_domain,
            file_path="ingestion/sparse_vocabulary_bm25.json",
            content=json.dumps(vocab_data, indent=2),
            content_type="application/json"
        )
        
        if success:
            logger.info(f"💾 Saved BM25 vocabulary for {self.brand_domain}")
        else:
            logger.error(f"Failed to save BM25 vocabulary for {self.brand_domain}")
    
    async def _load_vocabulary(self) -> None:
        """Load vocabulary and reinitialize BM25 if exists."""
        try:
            # Load using storage manager
            content = await self.storage.load_content(
                brand_domain=self.brand_domain,
                content_type="ingestion/sparse_vocabulary_bm25"
            )
            
            if content:
                vocab_data = json.loads(content)
                
                self.vocabulary = vocab_data['vocabulary']
                self.inverse_vocabulary = {int(idx): term for term, idx in self.vocabulary.items()}
                self.term_importance.update(vocab_data.get('term_importance', {}))
                
                # Note: We can't fully restore BM25 without the corpus
                # This would require also saving corpus tokens or recalculating from products
                
                logger.info(f"📖 Loaded vocabulary with {len(self.vocabulary)} terms")
                logger.info("   Note: BM25 model will be rebuilt when build_vocabulary is called")
            else:
                logger.info("No existing vocabulary found - will build from catalog")
                
        except Exception as e:
            logger.warning(f"Failed to load vocabulary: {e}")
            logger.info("Will build new vocabulary from catalog")