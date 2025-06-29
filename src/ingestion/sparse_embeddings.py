"""
Sparse Embedding Generation for Hybrid Search

Implements BM25-inspired sparse embeddings for keyword precision in product search.
"""

import re
import math
import json
import logging
from typing import Dict, List, Tuple, Set, Any
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class SparseEmbeddingGenerator:
    """
    Generates sparse embeddings for products using BM25-inspired scoring.
    
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
        
        # Brand-specific term weights
        self.term_importance = self._load_term_importance()
        
        # Initialize vocabulary (can be persisted)
        self.vocabulary = {}
        self.inverse_vocabulary = {}
        self.next_index = 0
        
        # Document frequency for IDF calculation
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        
        # BM25 parameters
        self.k1 = 1.2  # Term frequency saturation
        self.b = 0.75   # Length normalization
        
        logger.info(f"📊 Initialized Sparse Embedding Generator for {brand_domain}")
    
    def build_vocabulary(self, products: List[Dict[str, Any]]) -> None:
        """
        Build vocabulary from product catalog for consistent indexing.
        """
        logger.info(f"🔨 Building vocabulary from {len(products)} products")
        
        # Extract all terms from products
        all_terms = []
        for product in products:
            terms = self._extract_product_terms(product)
            all_terms.extend(terms)
        
        # Count term frequencies
        term_counts = Counter(all_terms)
        
        # Select top features
        top_terms = [term for term, _ in term_counts.most_common(self.max_features)]
        
        # Build vocabulary mapping
        self.vocabulary = {term: idx for idx, term in enumerate(top_terms)}
        self.inverse_vocabulary = {idx: term for term, idx in self.vocabulary.items()}
        
        # Calculate document frequencies
        for product in products:
            unique_terms = set(self._extract_product_terms(product))
            for term in unique_terms:
                if term in self.vocabulary:
                    self.document_frequencies[term] += 1
        
        self.total_documents = len(products)
        
        logger.info(f"✅ Built vocabulary with {len(self.vocabulary)} terms")
        
        # Save vocabulary for persistence
        self._save_vocabulary()
    
    def generate_sparse_embedding(
        self,
        product: Dict[str, Any],
        enhanced_data: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Generate sparse embedding for a single product.
        
        Returns:
            Dictionary with 'indices' and 'values' for sparse representation
        """
        # Extract and weight terms
        term_weights = self._calculate_term_weights(product, enhanced_data)
        
        # Convert to sparse vector format
        indices = []
        values = []
        
        for term, weight in term_weights.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                indices.append(idx)
                values.append(weight)
        
        # Sort by index for consistency
        sorted_pairs = sorted(zip(indices, values))
        if sorted_pairs:
            indices, values = zip(*sorted_pairs)
            indices = list(indices)
            values = list(values)
        else:
            indices = []
            values = []
        
        # Normalize values
        if values:
            max_val = max(values)
            if max_val > 0:
                values = [v / max_val for v in values]
        
        return {
            'indices': indices,
            'values': values
        }
    
    def _extract_product_terms(self, product: Dict[str, Any]) -> List[str]:
        """
        Extract all relevant terms from a product.
        """
        terms = []
        
        # Extract from universal fields
        universal = product.get('universal_fields', {})
        
        # Brand (highest importance)
        if universal.get('brand'):
            brand = universal['brand'].lower()
            terms.extend(self._tokenize(brand))
            terms.append(brand)  # Also keep full brand
        
        # Product name
        if universal.get('name'):
            name = universal['name'].lower()
            terms.extend(self._tokenize(name))
            # Also extract model numbers/codes
            model_numbers = self._extract_model_numbers(name)
            terms.extend(model_numbers)
        
        # Categories
        for category in universal.get('category', []):
            terms.extend(self._tokenize(category.lower()))
        
        # Description
        if universal.get('description'):
            desc_terms = self._tokenize(universal['description'].lower())
            terms.extend(desc_terms[:50])  # Limit description terms
        
        # Search keywords
        for keyword in product.get('search_keywords', []):
            terms.append(keyword.lower())
        
        # Filter metadata
        for key, value in product.get('filter_metadata', {}).items():
            if isinstance(value, str):
                terms.extend(self._tokenize(value.lower()))
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        terms.extend(self._tokenize(v.lower()))
        
        # Key selling points
        for point in product.get('key_selling_points', []):
            point_terms = self._tokenize(point.lower())
            terms.extend(point_terms[:10])  # Limit per point
        
        return terms
    
    def _calculate_term_weights(
        self,
        product: Dict[str, Any],
        enhanced_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate BM25-inspired weights for terms.
        """
        term_frequencies = defaultdict(float)
        universal = product.get('universal_fields', {})
        
        # Brand terms (weight: 10.0)
        if universal.get('brand'):
            brand = universal['brand'].lower()
            term_frequencies[brand] += 10.0
            for token in self._tokenize(brand):
                term_frequencies[token] += 8.0
        
        # Product name terms (weight: 8.0)
        if universal.get('name'):
            name = universal['name'].lower()
            # Full name gets high weight
            term_frequencies[name] += 8.0
            
            # Individual tokens
            for token in self._tokenize(name):
                term_frequencies[token] += 6.0
            
            # Model numbers get very high weight
            for model in self._extract_model_numbers(name):
                term_frequencies[model] += 9.0
        
        # Category terms (weight: 5.0)
        for category in universal.get('category', []):
            cat_lower = category.lower()
            term_frequencies[cat_lower] += 5.0
            for token in self._tokenize(cat_lower):
                term_frequencies[token] += 4.0
        
        # Price range indicators
        price = universal.get('price', 0)
        if price > 0:
            price_range = self._get_price_range(price)
            term_frequencies[price_range] += 3.0
        
        # Search keywords (weight: 4.0)
        for keyword in product.get('search_keywords', []):
            term_frequencies[keyword.lower()] += 4.0
        
        # Filter values (weight: 3.0)
        for key, value in product.get('filter_metadata', {}).items():
            if isinstance(value, str):
                term_frequencies[value.lower()] += 3.0
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        term_frequencies[v.lower()] += 3.0
        
        # Enhanced descriptor key terms (weight: 2.0)
        if enhanced_data.get('key_features'):
            for feature in enhanced_data['key_features']:
                for token in self._tokenize(feature.lower()):
                    if len(token) > 3:  # Skip short words
                        term_frequencies[token] += 2.0
        
        # Apply BM25 scoring
        weighted_terms = {}
        doc_length = sum(term_frequencies.values())
        avg_doc_length = 100  # Estimated average
        
        for term, freq in term_frequencies.items():
            # IDF component
            df = self.document_frequencies.get(term, 1)
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            
            # BM25 term frequency component
            tf_component = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
            )
            
            # Final weight
            weight = idf * tf_component
            
            # Apply brand-specific importance boost
            if term in self.term_importance:
                weight *= self.term_importance[term]
            
            weighted_terms[term] = weight
        
        return weighted_terms
    
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
        Load brand-specific term importance weights.
        """
        # Default importance for different term types
        importance = {}
        
        # Brand-specific terms get boosted
        if "specialized" in self.brand_domain.lower():
            importance.update({
                "specialized": 2.0,
                "sworks": 2.0,
                "turbo": 1.5,
                "tarmac": 1.5,
                "roubaix": 1.5,
                "diverge": 1.5,
                "stumpjumper": 1.5,
                "carbon": 1.3,
                "di2": 1.3,
                "etap": 1.3,
            })
        elif "balenciaga" in self.brand_domain.lower():
            importance.update({
                "balenciaga": 2.0,
                "cagole": 1.5,
                "hourglass": 1.5,
                "track": 1.5,
                "triple s": 1.5,
                "leather": 1.3,
                "designer": 1.3,
            })
        elif "sundayriley" in self.brand_domain.lower():
            importance.update({
                "sunday riley": 2.0,
                "good genes": 1.5,
                "luna": 1.5,
                "ceo": 1.5,
                "ice": 1.5,
                "serum": 1.3,
                "retinoid": 1.3,
                "vitamin c": 1.3,
                "lactic acid": 1.3,
            })
        
        return importance
    
    def _save_vocabulary(self) -> None:
        """
        Save vocabulary to disk for persistence.
        """
        vocab_path = Path(f"accounts/{self.brand_domain}/sparse_vocabulary.json")
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            'vocabulary': self.vocabulary,
            'document_frequencies': dict(self.document_frequencies),
            'total_documents': self.total_documents,
            'term_importance': self.term_importance
        }
        
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"💾 Saved vocabulary to {vocab_path}")
    
    def load_vocabulary(self) -> bool:
        """
        Load vocabulary from disk if available.
        
        Returns:
            True if vocabulary was loaded, False otherwise
        """
        vocab_path = Path(f"accounts/{self.brand_domain}/sparse_vocabulary.json")
        
        if not vocab_path.exists():
            return False
        
        try:
            with open(vocab_path) as f:
                vocab_data = json.load(f)
            
            self.vocabulary = vocab_data['vocabulary']
            self.inverse_vocabulary = {idx: term for term, idx in self.vocabulary.items()}
            self.document_frequencies = defaultdict(int, vocab_data['document_frequencies'])
            self.total_documents = vocab_data['total_documents']
            self.term_importance.update(vocab_data.get('term_importance', {}))
            
            logger.info(f"📂 Loaded vocabulary with {len(self.vocabulary)} terms")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load vocabulary: {e}")
            return False