"""
STT Vocabulary Extractor V2 - Focused on STT/TTS Challenging Terms

This version takes a whitelist approach - only including terms that are
genuinely challenging for speech recognition and synthesis.
"""

import re
import json
import logging
from typing import List, Set, Dict, Any, Tuple
from collections import Counter
from dataclasses import dataclass

from liddy.models.product import Product
from liddy.storage import get_account_storage_provider

logger = logging.getLogger(__name__)


@dataclass
class TermCandidate:
    """Represents a candidate term with its characteristics"""
    term: str
    category: str  # brand_name, model_number, chemical, technical, foreign
    score: float  # Priority score
    source: str   # Where it came from (name, description, etc.)


class STTVocabularyExtractorV2:
    """
    Extracts vocabulary that is genuinely challenging for STT/TTS engines.
    
    Uses a whitelist approach to only include:
    1. Brand names and proprietary terms
    2. Model numbers and codes
    3. Chemical/scientific terms
    4. Technical jargon with unusual phonetics
    5. Foreign words/names
    """
    
    MAX_VOCAB_SIZE = 2500  # AssemblyAI limit
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.storage = get_account_storage_provider()
        self.candidates: List[TermCandidate] = []
        self.final_vocabulary: Set[str] = set()
        
    def extract_from_catalog(self, products: List[Product]) -> None:
        """Extract challenging vocabulary from product catalog."""
        logger.info(f"📝 Extracting STT vocabulary from {len(products)} products")
        
        # Clear previous data
        self.candidates.clear()
        self.final_vocabulary.clear()
        
        # Extract candidates from each product
        for product in products:
            self._extract_candidates_from_product(product)
        
        # Build final vocabulary
        self._build_final_vocabulary()
        
        logger.info(f"✅ Extracted {len(self.final_vocabulary)} STT-challenging terms")
        
    def _extract_candidates_from_product(self, product: Product) -> None:
        """Extract candidate terms from a single product."""
        
        # 1. Extract brand names (always include the main brand)
        if product.brand and not self._is_generic_brand(product.brand):
            self.candidates.append(TermCandidate(
                term=product.brand,
                category="brand_name",
                score=10.0,
                source="brand"
            ))
        
        # 2. Extract from product name
        if product.name:
            # Look for model numbers and codes
            model_terms = self._extract_model_terms(product.name)
            for term in model_terms:
                self.candidates.append(TermCandidate(
                    term=term,
                    category="model_number",
                    score=9.0,
                    source="product_name"
                ))
            
            # Look for brand-specific product lines
            brand_terms = self._extract_brand_product_names(product.name)
            for term in brand_terms:
                self.candidates.append(TermCandidate(
                    term=term,
                    category="brand_name",
                    score=8.0,
                    source="product_name"
                ))
        
        # 3. Extract from specifications
        if product.specifications:
            tech_terms = self._extract_technical_specifications(product.specifications)
            for term in tech_terms:
                self.candidates.append(TermCandidate(
                    term=term,
                    category="technical",
                    score=6.0,
                    source="specifications"
                ))
        
        # 4. Extract from description and key selling points
        text_sources = []
        if product.description:
            text_sources.append(product.description)
        if product.key_selling_points:
            text_sources.extend(product.key_selling_points)
            
        for text in text_sources:
            # Chemical/scientific terms
            chem_terms = self._extract_chemical_terms(text)
            for term in chem_terms:
                self.candidates.append(TermCandidate(
                    term=term,
                    category="chemical",
                    score=7.0,
                    source="description"
                ))
            
            # Foreign/unusual terms
            foreign_terms = self._extract_foreign_terms(text)
            for term in foreign_terms:
                self.candidates.append(TermCandidate(
                    term=term,
                    category="foreign",
                    score=5.0,
                    source="description"
                ))
    
    def _is_generic_brand(self, brand: str) -> bool:
        """Check if brand is too generic to need STT assistance."""
        generic_brands = {'generic', 'unknown', 'other', 'various', 'multiple'}
        return brand.lower() in generic_brands
    
    def _extract_model_terms(self, text: str) -> Set[str]:
        """Extract model numbers and technical codes."""
        terms = set()
        
        # Patterns for model numbers
        patterns = [
            # Alphanumeric codes with specific structures
            r'\b(?:[A-Z]{1,3}[-\s]?\d{2,4}[A-Z]?)\b',  # SL7, XR-1000
            r'\b(?:\d{2,4}[-\s]?[A-Z]{1,3})\b',        # 1000XR
            r'\b(?:[A-Z]+\d+[A-Z]*)\b',                # ABC123D
            r'\bv\d+(?:\.\d+)?\b',                     # v2.0, v3
            
            # Brand-specific patterns for Specialized
            r'\bS-Works\b',
            r'\bFACT\s+\d+\w*\b',                      # FACT 10r
            r'\bSL\d+\b',                               # SL7
            
            # Technical specifications that need pronunciation
            r'\b\d+(?:mm|nm|cc|ml|oz)\b',              # 100mm, 50ml
            r'\b[A-Z]{2,}(?:/[A-Z]{2,})+\b',           # USB/HDMI
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update(matches)
        
        return terms
    
    def _extract_brand_product_names(self, text: str) -> Set[str]:
        """Extract brand-specific product line names."""
        terms = set()
        
        # Known challenging brand names by domain
        if 'specialized' in self.brand_domain.lower():
            brand_products = [
                'Tarmac', 'Roubaix', 'Diverge', 'Allez', 'Venge',
                'Stumpjumper', 'Enduro', 'Epic', 'Chisel', 'Fuse',
                'Turbo Levo', 'Turbo Vado', 'Turbo Como',
                'Rockhopper', 'Sirrus', 'Roll', 'Como',
                'SWAT', 'Roval', 'Future Shock', 'Brain'
            ]
        elif 'sundayriley' in self.brand_domain.lower():
            brand_products = [
                'Luna', 'Good Genes', 'C.E.O.', 'A+', 'Juno',
                'U.F.O.', 'Saturn', 'Ice', 'Autocorrect',
                'Martian', 'Blue Moon', 'Ceramic Slip', 'Pink Drink',
                'Tidal', 'Flash Fix', '5 Stars'
            ]
        else:
            brand_products = []
        
        # Check if any brand products appear in the text
        text_lower = text.lower()
        for product in brand_products:
            if product.lower() in text_lower:
                terms.add(product)
        
        # Also look for CamelCase names which are often brands
        camelcase_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        matches = re.findall(camelcase_pattern, text)
        terms.update(matches)
        
        return terms
    
    def _extract_technical_specifications(self, specs: Dict[str, Any]) -> Set[str]:
        """Extract technical terms from specifications."""
        terms = set()
        
        # Technical terms that might be mispronounced
        technical_indicators = [
            'alloy', 'carbon', 'composite', 'polymer', 'elastomer',
            'hydraulic', 'pneumatic', 'electronic', 'magnetic',
            'titanium', 'aluminum', 'chromoly', 'kevlar'
        ]
        
        for key, value in specs.items():
            # Skip numeric-only values
            if isinstance(value, (int, float)):
                continue
                
            value_str = str(value)
            
            # Look for compound technical terms
            for indicator in technical_indicators:
                if indicator in value_str.lower():
                    # Extract the full term (e.g., "carbon fiber composite")
                    words = value_str.split()
                    for i, word in enumerate(words):
                        if indicator in word.lower():
                            # Get surrounding context
                            start = max(0, i-1)
                            end = min(len(words), i+2)
                            compound_term = ' '.join(words[start:end])
                            if len(compound_term) > len(indicator):
                                terms.add(compound_term)
        
        return terms
    
    def _extract_chemical_terms(self, text: str) -> Set[str]:
        """Extract chemical and scientific terms."""
        terms = set()
        
        # Chemical term patterns
        patterns = [
            # Specific chemical endings
            r'\b\w+(?:phthalate|peptide|glycol|amine|oxide|hydroxide)\b',
            r'\b\w+(?:ceramide|retinoid|niacinamide)\b',
            r'\b(?:hydroxy|acetyl|methyl|ethyl|propyl)\w+\b',
            
            # Acids
            r'\b\w+\s+acid\b',
            
            # Scientific notation
            r'\b[A-Z]+[+-]\b',  # A+, C+E
            r'\b\w+[-]\d+\b',   # Omega-3
            
            # Complex molecules
            r'\b(?:alpha|beta|gamma|delta)[-\s]\w+\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update(matches)
        
        return terms
    
    def _extract_foreign_terms(self, text: str) -> Set[str]:
        """Extract foreign words or terms with unusual English phonetics."""
        terms = set()
        
        # Patterns for potentially foreign or unusual terms
        patterns = [
            # Double consonants that are unusual in English
            r'\b\w*(?:zz|xx|qq|vv)\w*\b',
            
            # Unusual vowel combinations
            r'\b\w*(?:aa|ii|uu|eo|ae|oe)\w*\b',
            
            # Words ending in unusual patterns for English
            r'\b\w+(?:eux|aux|ois|ese|ini|cci)\b',
            
            # Accented characters (if present)
            r'\b\w*[àáäâèéëêìíïîòóöôùúüûñç]\w*\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Filter out common English words that match patterns
            for match in matches:
                if not self._is_common_english_pattern(match):
                    terms.add(match)
        
        return terms
    
    def _is_common_english_pattern(self, word: str) -> bool:
        """Check if word follows common English patterns."""
        word_lower = word.lower()
        
        # Common English words that might match our patterns
        common_exceptions = {
            'see', 'bee', 'fee', 'free', 'tree', 'speed', 'need',
            'book', 'look', 'good', 'food', 'moon', 'soon',
            'been', 'seen', 'green', 'screen',
        }
        
        return word_lower in common_exceptions
    
    def _build_final_vocabulary(self) -> None:
        """Build final vocabulary from candidates."""
        # Remove duplicates by term
        unique_candidates = {}
        for candidate in self.candidates:
            if candidate.term not in unique_candidates or candidate.score > unique_candidates[candidate.term].score:
                unique_candidates[candidate.term] = candidate
        
        # Sort by score and category priority
        sorted_candidates = sorted(
            unique_candidates.values(),
            key=lambda x: (x.score, x.category == "brand_name", x.category == "model_number"),
            reverse=True
        )
        
        # Build vocabulary within size limit
        current_size = 0
        category_counts = Counter()
        
        for candidate in sorted_candidates:
            term_size = len(candidate.term) + 1  # +1 for space
            if current_size + term_size > self.MAX_VOCAB_SIZE:
                break
            
            self.final_vocabulary.add(candidate.term)
            current_size += term_size
            category_counts[candidate.category] += 1
        
        # Log statistics
        logger.info(f"📊 STT Vocabulary Statistics:")
        logger.info(f"   Total candidates: {len(unique_candidates)}")
        logger.info(f"   Final vocabulary: {len(self.final_vocabulary)} terms")
        logger.info(f"   Character count: {current_size}/{self.MAX_VOCAB_SIZE}")
        logger.info(f"   Categories: {dict(category_counts)}")
        
        # Show sample terms
        sample_terms = list(self.final_vocabulary)[:20]
        logger.info(f"   Sample terms: {', '.join(sample_terms)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics for logging."""
        char_count = sum(len(t) + 1 for t in self.final_vocabulary) - 1 if self.final_vocabulary else 0
        return {
            'term_count': len(self.final_vocabulary),
            'character_count': char_count,
            'coverage_percentage': (char_count / self.MAX_VOCAB_SIZE) * 100 if self.MAX_VOCAB_SIZE > 0 else 0,
            'top_terms': list(self.final_vocabulary)[:20]
        }
    
    async def save_vocabulary(self) -> bool:
        """Save vocabulary to storage."""
        stats = self.get_stats()
        vocab_data = {
            'brand_domain': self.brand_domain,
            'vocabulary': sorted(list(self.final_vocabulary)),
            'vocabulary_string': ' '.join(sorted(self.final_vocabulary)),
            'term_count': stats['term_count'],
            'character_count': stats['character_count'],
            'metadata': {
                'extractor_version': '2.0',
                'approach': 'whitelist',
                'focus': 'stt_challenging_terms'
            }
        }
        
        success = await self.storage.write_file(
            account=self.brand_domain,
            file_path="stt/vocabulary.json",  # Keep same filename for compatibility
            content=json.dumps(vocab_data, indent=2),
            content_type="application/json"
        )
        
        if success:
            logger.info(f"💾 Saved STT vocabulary for {self.brand_domain}")
        else:
            logger.error(f"Failed to save STT vocabulary for {self.brand_domain}")
        
        return success