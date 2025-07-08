"""
STT Vocabulary Extractor V3 - Brand-Agnostic Linguistic Pattern Approach

Focuses on linguistic patterns that are challenging for STT/TTS regardless of brand:
1. Abbreviations and units
2. Alphanumeric codes
3. Non-phonetic spellings
4. Compound technical terms
5. Foreign loanwords
"""

import re
import json
import logging
from typing import List, Set, Dict, Any, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
import unicodedata

from liddy.models.product import Product
from liddy.storage import get_account_storage_provider

logger = logging.getLogger(__name__)


@dataclass
class STTChallenge:
    """Represents why a term is challenging for STT/TTS"""
    term: str
    challenge_type: str  # abbreviation, alphanumeric, non_phonetic, etc.
    confidence: float   # How confident we are this needs STT help
    examples: List[str]  # Example contexts where it appears


class STTVocabularyExtractorV3:
    """
    Brand-agnostic STT vocabulary extraction based on linguistic patterns.
    
    This version focuses on universal linguistic challenges rather than
    brand-specific knowledge.
    """
    
    MAX_VOCAB_SIZE = 2500  # AssemblyAI limit
    
    # Common measurement units that need pronunciation help
    MEASUREMENT_UNITS = {
        # Length/Distance
        'mm', 'cm', 'km', 'in', 'ft', 'yd', 'mi',
        # Weight
        'g', 'kg', 'mg', 'oz', 'lb', 'lbs',
        # Volume
        'ml', 'l', 'cl', 'dl', 'fl oz', 'gal', 'qt', 'pt',
        # Speed/Rate
        'mph', 'kph', 'kmh', 'rpm', 'fps',
        # Power/Energy
        'w', 'kw', 'hp', 'wh', 'kwh', 'mah',
        # Pressure
        'psi', 'bar', 'atm', 'kpa',
        # Temperature
        'c', 'f', 'k',
        # Data
        'gb', 'mb', 'tb', 'mbps', 'gbps',
        # Other
        'v', 'a', 'ah', 'db', 'hz', 'khz'
    }
    
    # Common abbreviations that might be mispronounced
    COMMON_ABBREVIATIONS = {
        'asap', 'etc', 'eg', 'ie', 'vs', 'max', 'min',
        'qty', 'pkg', 'pcs', 'ea', 'doz',
        'mfg', 'mfr', 'oem', 'sku', 'upc',
        'msrp', 'rrp', 'vat', 'gst',
        'usa', 'uk', 'eu', 'intl',
        'xl', 'xxl', 'xxxl', 'xs', 'xxs',
        'jr', 'sr', 'phd', 'md', 'ceo',
        'am', 'pm', 'est', 'pst', 'gmt',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'
    }
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.storage = get_account_storage_provider()
        self.challenges: List[STTChallenge] = []
        self.term_contexts: Dict[str, List[str]] = defaultdict(list)
        self.final_vocabulary: Set[str] = set()
        
    def extract_from_catalog(self, products: List[Product]) -> None:
        """Extract challenging vocabulary from product catalog."""
        logger.info(f"📝 Extracting STT vocabulary from {len(products)} products")
        
        # Clear previous data
        self.challenges.clear()
        self.term_contexts.clear()
        self.final_vocabulary.clear()
        
        # Extract from each product
        for product in products:
            self._analyze_product(product)
        
        # Build final vocabulary
        self._build_final_vocabulary()
        
        logger.info(f"✅ Extracted {len(self.final_vocabulary)} STT-challenging terms")
        
    def _analyze_product(self, product: Product) -> None:
        """Analyze a product for STT-challenging terms."""
        
        # Collect all text from the product
        texts = []
        if product.name:
            texts.append(('name', product.name))
        if product.brand:
            texts.append(('brand', product.brand))
        if product.description:
            texts.append(('description', product.description))
        if product.key_selling_points:
            for ksp in product.key_selling_points:
                texts.append(('key_point', ksp))
        if product.specifications:
            for key, value in product.specifications.items():
                texts.append(('spec', f"{key}: {value}"))
        
        # Analyze each text
        for source, text in texts:
            self._analyze_text(text, source)
    
    def _analyze_text(self, text: str, source: str) -> None:
        """Analyze text for STT challenges."""
        if not text:
            return
            
        # Store context for terms
        self.term_contexts[text[:100]].append(source)
        
        # 1. Extract measurements with units
        measurements = self._extract_measurements(text)
        for term in measurements:
            self._add_challenge(STTChallenge(
                term=term,
                challenge_type="measurement",
                confidence=0.9,  # Measurements are almost always challenging
                examples=[f"Found in {source}: {term}"]
            ))
        
        # 2. Extract abbreviations and acronyms
        abbreviations = self._extract_abbreviations(text)
        for term in abbreviations:
            self._add_challenge(STTChallenge(
                term=term,
                challenge_type="abbreviation",
                confidence=0.8,
                examples=[f"Found in {source}: {term}"]
            ))
        
        # 3. Extract alphanumeric codes
        codes = self._extract_alphanumeric_codes(text)
        for term in codes:
            self._add_challenge(STTChallenge(
                term=term,
                challenge_type="alphanumeric",
                confidence=0.95,  # These are very challenging
                examples=[f"Found in {source}: {term}"]
            ))
        
        # 4. Extract non-phonetic terms
        non_phonetic = self._extract_non_phonetic_terms(text)
        for term in non_phonetic:
            self._add_challenge(STTChallenge(
                term=term,
                challenge_type="non_phonetic",
                confidence=0.7,
                examples=[f"Found in {source}: {term}"]
            ))
        
        # 5. Extract compound technical terms
        compounds = self._extract_compound_terms(text)
        for term in compounds:
            self._add_challenge(STTChallenge(
                term=term,
                challenge_type="compound",
                confidence=0.6,
                examples=[f"Found in {source}: {term}"]
            ))
    
    def _extract_measurements(self, text: str) -> Set[str]:
        """Extract measurements with units."""
        measurements = set()
        
        # Pattern for number + unit (with optional space)
        for unit in self.MEASUREMENT_UNITS:
            # Escape special regex characters in unit
            escaped_unit = re.escape(unit)
            patterns = [
                rf'\b\d+(?:\.\d+)?\s*{escaped_unit}\b',  # 10mm, 3.5 oz
                rf'\b\d+(?:\.\d+)?{escaped_unit}\b',     # 10mm (no space)
                rf'\b\d+(?:\.\d+)?\s*{escaped_unit}s?\b', # plural units
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                measurements.update(matches)
        
        # Also catch compound measurements like "5x10mm" or "3/8 inch"
        compound_patterns = [
            r'\b\d+x\d+(?:\.\d+)?\s*\w{1,4}\b',  # 5x10mm
            r'\b\d+/\d+\s*(?:inch|in|mm|cm)\b',  # 3/8 inch
            r'\b\d+(?:\.\d+)?"\b',               # 5" (inches)
            r"\b\d+(?:\.\d+)?'\b",               # 6' (feet)
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            measurements.update(matches)
        
        return measurements
    
    def _extract_abbreviations(self, text: str) -> Set[str]:
        """Extract abbreviations and acronyms."""
        abbreviations = set()
        
        # All caps words (likely acronyms)
        acronym_pattern = r'\b[A-Z]{2,}\b'
        matches = re.findall(acronym_pattern, text)
        for match in matches:
            # Filter out common words that happen to be capitalized
            if len(match) <= 5 and match.lower() not in ['the', 'and', 'for']:
                abbreviations.add(match)
        
        # Known abbreviations
        text_lower = text.lower()
        for abbr in self.COMMON_ABBREVIATIONS:
            if re.search(rf'\b{abbr}\b', text_lower):
                # Find the actual case used in the text
                pattern = rf'\b{abbr}\b'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    abbreviations.add(match.group())
        
        # Dotted abbreviations (e.g., U.S.A., Ph.D.)
        dotted_pattern = r'\b(?:[A-Za-z]\.)+[A-Za-z]?\b'
        matches = re.findall(dotted_pattern, text)
        abbreviations.update(matches)
        
        return abbreviations
    
    def _extract_alphanumeric_codes(self, text: str) -> Set[str]:
        """Extract alphanumeric model codes and identifiers."""
        codes = set()
        
        patterns = [
            # Classic model numbers
            r'\b[A-Z]{1,3}\d{2,4}[A-Z]?\b',     # A123, ABC1234
            r'\b\d{2,4}[A-Z]{1,3}\b',           # 123ABC
            r'\b[A-Z]\d+[A-Z]\d+\b',            # A1B2
            
            # Version numbers
            r'\b[vV]\d+(?:\.\d+)*\b',           # v1.0, V2.3.1
            
            # Hyphenated codes
            r'\b[A-Z0-9]{2,}-[A-Z0-9]{2,}\b',   # AB-123, X1-Y2
            
            # Special patterns
            r'\b\d+[A-Z]-[A-Z0-9]+\b',          # 10S-PRO
            r'\b[A-Z]+\d+[+-]?\b',              # A1+, B2-
            
            # Size/spec codes
            r'\b\d+[TRX]\b',                    # 11T, 32R
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            codes.update(matches)
        
        return codes
    
    def _extract_non_phonetic_terms(self, text: str) -> Set[str]:
        """Extract terms with non-phonetic or unusual spellings."""
        non_phonetic = set()
        
        # Words with unusual letter combinations
        patterns = [
            # Silent letters or unusual combos
            r'\b\w*[ck]n\w*\b',     # knife, knob
            r'\b\w*ght\b',          # right, light (but common)
            r'\b\w*ph\w*\b',        # phone, graph
            r'\b\w*ps\w*\b',        # psalm, psychology
            
            # Double letters that affect pronunciation
            r'\b\w*[zx]{2}\w*\b',   # jazz, maxx
            r'\b\w*ll[aeiou]\w*\b', # vanilla, guerilla
            
            # Foreign-origin patterns
            r'\b\w*eux?\b',         #ieux, eux (French)
            r'\b\w*sch\w*\b',       # schematic, schnell
            r'\b\w*tsch\w*\b',      # deutsch
            r'\b\w*[aeiou]{3,}\w*\b', # three+ vowels
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Filter out very common words
                if len(match) > 2 and not self._is_common_word(match):
                    non_phonetic.add(match)
        
        # CamelCase words (often brand names or technical terms)
        camelcase_pattern = r'\b[A-Z][a-z]+[A-Z]\w*\b'
        matches = re.findall(camelcase_pattern, text)
        non_phonetic.update(matches)
        
        return non_phonetic
    
    def _extract_compound_terms(self, text: str) -> Set[str]:
        """Extract compound technical terms."""
        compounds = set()
        
        # Hyphenated compounds
        hyphen_pattern = r'\b\w+(?:-\w+)+\b'
        matches = re.findall(hyphen_pattern, text)
        for match in matches:
            # Filter out common hyphenated words
            if not all(self._is_common_word(part) for part in match.split('-')):
                compounds.add(match)
        
        # Slash compounds (and/or, input/output)
        slash_pattern = r'\b\w+/\w+\b'
        matches = re.findall(slash_pattern, text)
        compounds.update(matches)
        
        return compounds
    
    def _is_common_word(self, word: str) -> bool:
        """Simple check for very common English words."""
        # This is a minimal set - in production, could use wordfreq
        common = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are',
            'been', 'has', 'had', 'were', 'said', 'did', 'get', 'may',
            'part', 'made', 'find', 'where', 'much', 'too', 'very',
            'still', 'being', 'going', 'why', 'before', 'never', 'here',
            'more', 'light', 'right', 'night', 'might'
        }
        return word.lower() in common
    
    def _add_challenge(self, challenge: STTChallenge) -> None:
        """Add a challenge if it's not already present."""
        # Check if we already have this term
        for existing in self.challenges:
            if existing.term.lower() == challenge.term.lower():
                # Update confidence if higher
                if challenge.confidence > existing.confidence:
                    existing.confidence = challenge.confidence
                existing.examples.extend(challenge.examples)
                return
        
        self.challenges.append(challenge)
    
    def _build_final_vocabulary(self) -> None:
        """Build final vocabulary from challenges."""
        # Sort by confidence and type priority
        type_priority = {
            'alphanumeric': 1,
            'measurement': 2,
            'abbreviation': 3,
            'non_phonetic': 4,
            'compound': 5
        }
        
        sorted_challenges = sorted(
            self.challenges,
            key=lambda x: (
                -x.confidence,  # Higher confidence first
                type_priority.get(x.challenge_type, 10),  # Type priority
                len(x.term)  # Shorter terms first (more room for others)
            )
        )
        
        # Build vocabulary within size limit
        current_size = 0
        type_counts = Counter()
        
        for challenge in sorted_challenges:
            term_size = len(challenge.term) + 1
            if current_size + term_size > self.MAX_VOCAB_SIZE:
                break
            
            self.final_vocabulary.add(challenge.term)
            current_size += term_size
            type_counts[challenge.challenge_type] += 1
        
        # Log statistics
        logger.info(f"📊 STT Vocabulary Statistics:")
        logger.info(f"   Total challenges identified: {len(self.challenges)}")
        logger.info(f"   Final vocabulary: {len(self.final_vocabulary)} terms")
        logger.info(f"   Character count: {current_size}/{self.MAX_VOCAB_SIZE}")
        logger.info(f"   Challenge types: {dict(type_counts)}")
        
        # Show examples by type
        examples_by_type = defaultdict(list)
        for term in list(self.final_vocabulary)[:50]:
            for challenge in self.challenges:
                if challenge.term == term:
                    examples_by_type[challenge.challenge_type].append(term)
                    break
        
        for challenge_type, examples in examples_by_type.items():
            logger.info(f"   {challenge_type}: {', '.join(examples[:5])}")
    
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
        
        # Group terms by challenge type for the saved data
        terms_by_type = defaultdict(list)
        for challenge in self.challenges:
            if challenge.term in self.final_vocabulary:
                terms_by_type[challenge.challenge_type].append({
                    'term': challenge.term,
                    'confidence': challenge.confidence
                })
        
        vocab_data = {
            'brand_domain': self.brand_domain,
            'vocabulary': sorted(list(self.final_vocabulary)),
            'vocabulary_string': ' '.join(sorted(self.final_vocabulary)),
            'term_count': stats['term_count'],
            'character_count': stats['character_count'],
            'terms_by_type': dict(terms_by_type),
            'metadata': {
                'extractor_version': '3.0',
                'approach': 'linguistic_patterns',
                'focus': 'brand_agnostic_stt_challenges'
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