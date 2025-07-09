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

# Try to import wordfreq for better common word detection
try:
    from wordfreq import top_n_list, zipf_frequency
    WORDFREQ_AVAILABLE = True
    logger.info("wordfreq library available for enhanced common word filtering")
except ImportError:
    WORDFREQ_AVAILABLE = False
    logger.warning("wordfreq library not available, falling back to basic filtering")


class STTVocabularyExtractor:
    """
    Extracts and manages brand-specific vocabulary for STT optimization.
    
    Features:
    - Extracts unique product names, model numbers, and technical terms
    - Filters out common words to focus on brand-specific terminology
    - Maintains vocabulary within AssemblyAI's 2,500 character limit
    - Stores vocabulary per brand for easy updates
    """
    
    # Common words to filter out - expanded to focus on non-standard terms only
    COMMON_WORDS = {
        # Articles & determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'all', 'any', 'each', 'every',
        # Prepositions
        'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 'under', 'over',
        'through', 'between', 'into', 'during', 'before', 'after', 'above', 'below',
        # Conjunctions
        'and', 'or', 'but', 'so', 'because', 'if', 'when', 'while', 'although', 'since',
        # Common verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'shall',
        'get', 'make', 'take', 'give', 'go', 'come', 'see', 'know', 'think', 'use',
        # Common adjectives
        'new', 'good', 'best', 'great', 'high', 'low', 'large', 'small', 'big', 'old',
        'long', 'short', 'easy', 'hard', 'fast', 'slow', 'light', 'heavy', 'dark',
        # Common nouns (standard English)
        'product', 'item', 'model', 'version', 'edition', 'type', 'style', 'design',
        'color', 'size', 'shape', 'material', 'feature', 'benefit', 'quality',
        # Common bike/cycling terms that STT handles well
        'bike', 'bicycle', 'cycling', 'rider', 'frame', 'wheel', 'tire', 'seat', 'pedal',
        'handlebar', 'gear', 'brake', 'chain', 'fork', 'stem', 'saddle',
        # Common skincare terms that STT handles well
        'cream', 'serum', 'lotion', 'mask', 'cleanser', 'moisturizer', 'treatment',
        'skin', 'face', 'body', 'hair', 'eye', 'lip', 'hand', 'foot',
        # Common supplement terms
        'supplement', 'vitamin', 'mineral', 'capsule', 'tablet', 'powder', 'liquid',
        'daily', 'health', 'wellness', 'nutrition', 'dietary',
        # Numbers and basic units
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'inch', 'inches', 'foot', 'feet', 'pound', 'pounds', 'ounce', 'ounces',
        # Other common English words
        'made', 'designed', 'created', 'built', 'crafted', 'developed',
        'premium', 'standard', 'basic', 'advanced', 'professional',
        'available', 'includes', 'features', 'provides', 'offers'
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
        
        # Load top common English words if wordfreq is available
        self._common_words_set = set()
        if WORDFREQ_AVAILABLE:
            # Get top 5000 most common English words
            # These are words that STT/TTS engines handle extremely well
            try:
                top_common = top_n_list('en', 5000)
                self._common_words_set = set(top_common)
                logger.info(f"Loaded {len(self._common_words_set)} common words from wordfreq")
            except Exception as e:
                logger.warning(f"Failed to load wordfreq data: {e}")
                self._common_words_set = set()
        
        # Add our manual common words list to supplement wordfreq
        self._common_words_set.update(self.COMMON_WORDS)
        
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
                
                # Extract brand-specific terms
                brand_terms = self._extract_brand_specific_terms(point)
                for term in brand_terms:
                    self._add_term(term, weight=8)
        
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
        
        # Early filtering of very common words to save processing
        if weight < 10 and self._is_common_english_term(term):
            return
        
        # Add to frequency counter
        self.term_frequencies[term] += weight
        
        # Also add individual words from phrases
        words = self._tokenize(term)
        for word in words:
            if len(word) >= self.MIN_WORD_LENGTH:
                # Check if word is common before adding
                if not self._is_common_english_term(word):
                    self.term_frequencies[word] += weight // 2  # Individual words get half weight
    
    def _build_vocabulary(self) -> None:
        """Build final vocabulary from frequency data, focusing on non-standard terms."""
        # Sort terms by frequency (descending)
        sorted_terms = sorted(self.term_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        # Track statistics
        common_filtered = 0
        non_standard_filtered = 0
        
        # Build vocabulary within size limit
        current_size = 0
        for term, freq in sorted_terms:
            # Skip if it's a common English word
            if self._is_common_english_term(term):
                common_filtered += 1
                continue
            
            # Skip if it doesn't meet our criteria for non-standard terms
            if not self._is_non_standard_term(term):
                non_standard_filtered += 1
                continue
            
            # Check if adding this term would exceed limit
            term_size = len(term) + 1  # +1 for space separator
            if current_size + term_size > self.MAX_VOCAB_SIZE:
                break
            
            self.vocabulary.add(term)
            current_size += term_size
        
        logger.info(f"📊 Vocabulary statistics:")
        logger.info(f"   Total unique terms: {len(self.term_frequencies)}")
        logger.info(f"   Common words filtered: {common_filtered}")
        logger.info(f"   Non-technical terms filtered: {non_standard_filtered}")
        logger.info(f"   Final vocabulary: {len(self.vocabulary)} terms ({current_size} characters)")
        
        # Log top 20 terms for debugging
        if self.vocabulary:
            logger.info(f"   Top terms: {', '.join(list(self.vocabulary)[:20])}")
    
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
            r'\bTarmac\s+SL\d+\b',               # Tarmac SL7
            r'\b[A-Z][a-z]+[A-Z][a-z]+\b',       # CamelCase names
        ]
        
        models = []
        for pattern in model_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            models.extend(matches)
        
        # Also extract specific brand model names that might not match patterns
        brand_specific_models = self._extract_brand_specific_terms(text)
        models.extend(brand_specific_models)
        
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
    
    def _extract_brand_specific_terms(self, text: str) -> List[str]:
        """Extract brand-specific terms that need STT assistance."""
        terms = []
        
        # Chemical and scientific terms
        chemical_patterns = [
            r'\b\w+phthalate\b',        # Phthalate compounds
            r'\b\w+peptide\b',          # Peptides
            r'\b\w+glycol\b',           # Glycols
            r'\b\w+amine\b',            # Amines
            r'\b\w+oxide\b',            # Oxides
            r'\bhydroxy\w+\b',          # Hydroxy compounds
            r'\bretino\w+\b',           # Retinoids
            r'\b\w+ceramide\b',         # Ceramides
            r'\bniacinamide\b',         # Specific compounds
            r'\bhyaluronic\b',
            r'\bsalicylic\b',
            r'\bglyco\w+\b',            # Glycolic, etc.
            r'\blactic\s+acid\b',
            r'\bascorbic\b',
            r'\btocopherol\b',          # Vitamin E forms
            r'\b[A-Z]+[a-z]*\+\b',      # A+, C+E, etc.
        ]
        
        # Initialize all_patterns with chemical patterns
        all_patterns = chemical_patterns.copy()
        
        # Brand-specific product names (customize per brand)
        if 'specialized' in self.brand_domain.lower():
            # Specialized bike models
            brand_patterns = [
                r'\bTarmac\b', r'\bRoubaix\b', r'\bDiverge\b', r'\bAllez\b',
                r'\bStumpjumper\b', r'\bEnduro\b', r'\bEpic\b', r'\bRockhopper\b',
                r'\bTurbo\s+\w+\b',  # Turbo Levo, Turbo Vado, etc.
                r'\bS-Works\b', r'\bFACT\s+\d+\w*\b',  # FACT 10r, FACT 11m
                r'\bRoval\b', r'\bZertz\b', r'\bBrain\b',
            ]
            all_patterns.extend(brand_patterns)
        elif 'sundayriley' in self.brand_domain.lower():
            # Sunday Riley product names
            brand_patterns = [
                r'\bLuna\b', r'\bGood\s+Genes\b', r'\bC\.E\.O\.\b', r'\bA\+\b',
                r'\bJuno\b', r'\bU\.F\.O\.\b', r'\bSaturn\b', r'\bIce\b',
                r'\bAutocorrect\b', r'\bMartian\b', r'\bBlue\s+Moon\b',
                r'\bCeramic\s+Slip\b', r'\bPink\s+Drink\b',
            ]
            all_patterns.extend(brand_patterns)
        
        for pattern in all_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))
    
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
    
    def _is_common_english_term(self, term: str) -> bool:
        """Check if term is common English that STT/TTS handles well."""
        term_lower = term.lower()
        
        # Check against comprehensive common words set (includes wordfreq data)
        if term_lower in self._common_words_set:
            return True
        
        # If wordfreq is available, check frequency score
        if WORDFREQ_AVAILABLE:
            try:
                # Get Zipf frequency (scale of 0-8, where 8 is most common)
                # Words with frequency > 4.0 are quite common
                freq_score = zipf_frequency(term_lower, 'en')
                if freq_score > 4.0:  # Common enough that STT handles well
                    return True
            except:
                pass
        
        # Check individual words in phrases
        words = term_lower.split()
        if len(words) > 1:
            # If all words are common, the phrase is likely common
            if all(self._is_common_english_term(word) for word in words):
                return True
        
        # Common patterns that STT handles well
        common_patterns = [
            r'^(ultra|super|pro|plus|max|mini)\s',  # Common prefixes
            r'\s(series|line|collection|range)$',   # Common suffixes
            r'^(mens?|womens?|kids?)\s',           # Gender/age indicators
            r'\s(small|medium|large|xl|xxl)$',      # Size indicators
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, term_lower):
                return True
        
        return False
    
    def _is_non_standard_term(self, term: str) -> bool:
        """Determine if term is non-standard and needs STT assistance."""
        # Always include model numbers and technical codes
        if self._looks_like_model_or_technical_term(term):
            return True
        
        # Include terms with unusual letter combinations
        unusual_patterns = [
            r'[xz]{2,}',  # Double x or z (e.g., Maxxis, Fizik)
            r'[aeiouy]{3,}',  # Three or more vowels in a row
            r'[bcdfghjklmnpqrstvwxyz]{4,}',  # Four or more consonants
            r'^[a-z]+[A-Z]',  # camelCase or unusual capitalization
        ]
        
        for pattern in unusual_patterns:
            if re.search(pattern, term, re.IGNORECASE):
                return True
        
        # Include brand names and proprietary terms
        # These often have unusual spellings or pronunciations
        if term[0].isupper() and term not in self.COMMON_WORDS:
            # Capitalized terms that aren't common words
            if not self._is_dictionary_word(term.lower()):
                return True
        
        # Include chemical/scientific terms
        chemical_indicators = ['acid', 'peptide', 'oxide', 'glycol', 'amine', 'ol', 'ate']
        if any(term.lower().endswith(indicator) for indicator in chemical_indicators):
            return True
        
        # Include terms with special characters or numbers
        if re.search(r'[\d\-+/]', term):
            return True
        
        return False
    
    def _is_dictionary_word(self, word: str) -> bool:
        """Simple heuristic to check if word is likely a standard dictionary word."""
        # Common English word patterns
        if len(word) < 3 or len(word) > 15:
            return False
        
        # Check for common English suffixes
        common_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ment', 'ness']
        for suffix in common_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return True
        
        # If it's all lowercase and doesn't have unusual letter patterns, probably common
        if word.islower() and not re.search(r'[xzq]{2,}|[aeiouy]{3,}', word):
            # Simple vowel-consonant pattern check
            vowel_count = sum(1 for c in word if c in 'aeiouy')
            consonant_count = len(word) - vowel_count
            # Most English words have a reasonable vowel/consonant ratio
            if 0.25 <= vowel_count / len(word) <= 0.6:
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