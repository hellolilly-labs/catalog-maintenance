"""
STT Vocabulary Extractor for AssemblyAI Word Boosting

Extracts brand-specific vocabulary from product catalogs to improve
speech-to-text accuracy for technical terms, product names, and model numbers.
"""

import re
import json
import logging
from typing import List, Set, Dict, Any
from collections import Counter

from liddy.models.product import Product
from liddy.storage import get_account_storage_provider

logger = logging.getLogger(__name__)


class STTVocabularyExtractor:
    """
    Extracts and manages brand-specific vocabulary for STT optimization.
    
    Features:
    - Extracts unique product names, model numbers, and technical terms
    - Filters out common words to focus on brand-specific terminology
    - Maintains vocabulary within AssemblyAI's 2,500 character limit
    - Stores vocabulary per brand for easy updates
    """
    
    # Common words to filter out (expand as needed)
    COMMON_WORDS = {
        # Articles & determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        # Prepositions
        'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about',
        # Conjunctions
        'and', 'or', 'but', 'so', 'because', 'if', 'when', 'while',
        # Common verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should',
        # Common adjectives
        'new', 'good', 'best', 'great', 'high', 'low', 'large', 'small',
        # Generic product terms (customize per industry)
        'product', 'item', 'model', 'version', 'edition',
        # Common bike terms (we might want to keep some of these)
        'bike', 'bicycle', 'frame', 'wheel', 'tire',
    }
    
    # Minimum word length to consider (filters out single letters)
    MIN_WORD_LENGTH = 2
    
    # Maximum vocabulary size in characters (AssemblyAI limit)
    MAX_VOCAB_SIZE = 2500
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.storage = get_account_storage_provider()
        
        # Track term frequencies to prioritize important terms
        self.term_frequencies = Counter()
        
        # Store extracted vocabulary
        self.vocabulary = set()
        
        logger.info(f"🎤 Initialized STT Vocabulary Extractor for {brand_domain}")
    
    def extract_from_catalog(self, products: List[Product]) -> None:
        """
        Extract vocabulary from product catalog.
        
        Args:
            products: List of Product objects
        """
        logger.info(f"📝 Extracting STT vocabulary from {len(products)} products")
        
        # Clear previous data
        self.term_frequencies.clear()
        self.vocabulary.clear()
        
        for product in products:
            self._extract_from_product(product)
        
        # Build final vocabulary
        self._build_vocabulary()
        
        logger.info(f"✅ Extracted {len(self.vocabulary)} unique terms")
    
    def _extract_from_product(self, product: Product) -> None:
        """Extract vocabulary from a single product."""
        
        # 1. Product name (highest priority)
        if product.name:
            # Full product name
            self._add_term(product.name, weight=10)
            
            # Extract model numbers
            model_numbers = self._extract_model_numbers(product.name)
            for model in model_numbers:
                self._add_term(model, weight=15)  # Model numbers get highest weight
        
        # 2. Brand name
        if product.brand:
            self._add_term(product.brand, weight=8)
        
        # 3. Search keywords (these are curated, so high value)
        if product.search_keywords:
            for keyword in product.search_keywords:
                self._add_term(keyword, weight=6)
        
        # 4. Product labels (category-specific terms)
        if product.product_labels:
            for category, values in product.product_labels.items():
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str):
                            self._add_term(value, weight=5)
                elif isinstance(values, str):
                    self._add_term(values, weight=5)
        
        # 5. Categories
        for category in product.categories:
            # Skip generic categories
            if category.lower() not in ['general', 'other', 'misc']:
                self._add_term(category, weight=4)
        
        # 6. Key selling points (might contain technical terms)
        if product.key_selling_points:
            for point in product.key_selling_points:
                # Extract technical terms from selling points
                technical_terms = self._extract_technical_terms(point)
                for term in technical_terms:
                    self._add_term(term, weight=3)
        
        # 7. Specifications (technical terms)
        if product.specifications:
            for spec_key, spec_value in product.specifications.items():
                # Specification names often contain technical terms
                if not self._is_common_spec_key(spec_key):
                    self._add_term(spec_key, weight=3)
                
                # Specification values (if they're model names, etc.)
                if isinstance(spec_value, str) and len(spec_value) < 50:
                    if self._looks_like_model_or_technical_term(spec_value):
                        self._add_term(spec_value, weight=3)
        
        # 8. Colors (brand-specific color names)
        if product.colors:
            for color in product.colors:
                if isinstance(color, str):
                    # Skip basic colors
                    if color.lower() not in ['red', 'blue', 'green', 'black', 'white', 'gray', 'grey']:
                        self._add_term(color, weight=2)
                elif isinstance(color, dict) and 'name' in color:
                    color_name = color['name']
                    if color_name.lower() not in ['red', 'blue', 'green', 'black', 'white', 'gray', 'grey']:
                        self._add_term(color_name, weight=2)
    
    def _add_term(self, term: str, weight: int = 1) -> None:
        """Add a term with its weight to the frequency counter."""
        if not term or not isinstance(term, str):
            return
        
        # Clean the term
        term = term.strip()
        
        # Skip if too short
        if len(term) < self.MIN_WORD_LENGTH:
            return
        
        # Skip URLs and email addresses
        if term.startswith(('http://', 'https://', 'www.')) or '@' in term:
            return
        
        # Add to frequency counter
        self.term_frequencies[term] += weight
        
        # Also add individual words from phrases
        words = self._tokenize(term)
        for word in words:
            if len(word) >= self.MIN_WORD_LENGTH and word.lower() not in self.COMMON_WORDS:
                self.term_frequencies[word] += weight // 2  # Individual words get half weight
    
    def _build_vocabulary(self) -> None:
        """Build final vocabulary from frequency data."""
        # Sort terms by frequency (descending)
        sorted_terms = sorted(self.term_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary within size limit
        current_size = 0
        for term, freq in sorted_terms:
            # Skip common words unless they have very high frequency (brand names)
            if term.lower() in self.COMMON_WORDS and freq < 20:
                continue
            
            # Check if adding this term would exceed limit
            term_size = len(term) + 1  # +1 for space separator
            if current_size + term_size > self.MAX_VOCAB_SIZE:
                break
            
            self.vocabulary.add(term)
            current_size += term_size
        
        logger.info(f"📊 Vocabulary size: {current_size} characters ({len(self.vocabulary)} terms)")
    
    def _extract_model_numbers(self, text: str) -> List[str]:
        """Extract model numbers and codes from text."""
        model_patterns = [
            r'\b[A-Z0-9]{2,}[\-_]?[A-Z0-9]+\b',  # ABC-123, XY_456
            r'\b[A-Z]{2,}\d{2,}\b',              # ABC123
            r'\b\d{3,}[A-Z]+\b',                 # 123ABC
            r'\bv\d+\.\d+\b',                    # v1.0, v2.5
            r'\b[A-Z]+[\-]\d+\b',                # SL-7, XR-12
            r'\bSL\d+\b',                         # SL7, SL8
            r'\b[A-Z]\d{3,}\b',                  # S1000, X5000
        ]
        
        models = []
        for pattern in model_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            models.extend(matches)
        
        return list(set(models))
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        technical_terms = []
        
        # Look for terms with specific patterns
        patterns = [
            r'\b\w+[-]\w+\b',          # hyphenated terms
            r'\b[A-Z]{2,}\b',          # acronyms
            r'\b\d+\w+\b',             # measurements (11-speed, 29er)
            r'\b\w+\d+\b',             # model designations
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            technical_terms.extend(matches)
        
        return technical_terms
    
    def _looks_like_model_or_technical_term(self, text: str) -> bool:
        """Check if text looks like a model number or technical term."""
        # Has mix of letters and numbers
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        # Common patterns for technical terms
        if has_letters and has_numbers:
            return True
        
        # All caps (likely acronym)
        if text.isupper() and len(text) > 1:
            return True
        
        # Contains hyphen or underscore
        if '-' in text or '_' in text:
            return True
        
        return False
    
    def _is_common_spec_key(self, key: str) -> bool:
        """Check if a specification key is too common to include."""
        common_keys = {
            'weight', 'height', 'width', 'length', 'size', 
            'color', 'material', 'price', 'quantity'
        }
        return key.lower() in common_keys
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove special characters but keep hyphens and periods in model numbers
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Clean tokens
        cleaned_tokens = []
        for token in tokens:
            token = token.strip('.-')
            if token:
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    async def save_vocabulary(self) -> bool:
        """Save vocabulary to storage."""
        vocab_data = {
            'brand_domain': self.brand_domain,
            'vocabulary': sorted(list(self.vocabulary)),  # Convert set to sorted list
            'vocabulary_string': ' '.join(sorted(self.vocabulary)),  # Ready-to-use string
            'term_count': len(self.vocabulary),
            'character_count': len(' '.join(self.vocabulary)),
            'metadata': {
                'extractor_version': '1.0',
                'max_vocab_size': self.MAX_VOCAB_SIZE,
                'common_words_filtered': len(self.COMMON_WORDS)
            }
        }
        
        success = await self.storage.write_file(
            account=self.brand_domain,
            file_path="stt/vocabulary.json",
            content=json.dumps(vocab_data, indent=2),
            content_type="application/json"
        )
        
        if success:
            logger.info(f"💾 Saved STT vocabulary for {self.brand_domain}")
        else:
            logger.error(f"Failed to save STT vocabulary for {self.brand_domain}")
        
        return success
    
    async def load_vocabulary(self) -> Dict[str, Any]:
        """Load vocabulary from storage."""
        try:
            content = await self.storage.read_file(
                account=self.brand_domain,
                file_path="stt/vocabulary.json"
            )
            
            if content:
                vocab_data = json.loads(content)
                self.vocabulary = set(vocab_data.get('vocabulary', []))
                logger.info(f"📖 Loaded STT vocabulary with {len(self.vocabulary)} terms")
                return vocab_data
            
        except Exception as e:
            logger.warning(f"Failed to load STT vocabulary: {e}")
        
        return {}
    
    def get_vocabulary_string(self) -> str:
        """Get vocabulary as a space-separated string for AssemblyAI."""
        return ' '.join(sorted(self.vocabulary))[:self.MAX_VOCAB_SIZE]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        vocab_string = self.get_vocabulary_string()
        return {
            'term_count': len(self.vocabulary),
            'character_count': len(vocab_string),
            'top_terms': [term for term, _ in self.term_frequencies.most_common(20)],
            'coverage_percentage': (len(vocab_string) / self.MAX_VOCAB_SIZE) * 100
        }