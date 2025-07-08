#!/usr/bin/env python3
"""
Simple test of STT Vocabulary Extractor V3 with minimal dependencies
"""

import sys
import re
from typing import List, Set, Dict, Any
from collections import Counter, defaultdict
from dataclasses import dataclass

# Add path for imports
sys.path.insert(0, '/Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance')

# Import just the classes we need
from packages.liddy.models.product import Product


# Minimal implementation of STTChallenge for testing
@dataclass
class STTChallenge:
    """Represents why a term is challenging for STT/TTS"""
    term: str
    challenge_type: str
    confidence: float
    examples: List[str]


class SimplifiedSTTExtractorV3:
    """Simplified version of STT V3 for testing"""
    
    MAX_VOCAB_SIZE = 2500
    
    MEASUREMENT_UNITS = {
        'mm', 'cm', 'kg', 'oz', 'ml', 'rpm', 'psi', 'gb', 'v', 'hz',
        'mph', 'kph', 'w', 'kw', 'lbs', 'ft', 'in'
    }
    
    COMMON_ABBREVIATIONS = {
        'asap', 'etc', 'vs', 'max', 'min', 'xl', 'xxl', 'xs',
        'sku', 'msrp', 'oem', 'uk', 'usa', 'eu'
    }
    
    def __init__(self):
        self.challenges: List[STTChallenge] = []
        self.final_vocabulary: Set[str] = set()
    
    def extract_from_products(self, products: List[Product]):
        """Extract challenging terms from products"""
        for product in products:
            texts = []
            if product.name:
                texts.append(product.name)
            if product.description:
                texts.append(product.description)
            if product.specifications:
                for k, v in product.specifications.items():
                    texts.append(f"{k}: {v}")
            
            for text in texts:
                self._analyze_text(text)
        
        self._build_final_vocabulary()
    
    def _analyze_text(self, text: str):
        """Extract challenging terms from text"""
        if not text:
            return
        
        # Measurements
        for unit in self.MEASUREMENT_UNITS:
            pattern = rf'\b\d+(?:\.\d+)?\s*{re.escape(unit)}\b'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                self.challenges.append(STTChallenge(
                    term=match,
                    challenge_type="measurement",
                    confidence=0.9,
                    examples=[text[:50]]
                ))
        
        # Alphanumeric codes
        patterns = [
            r'\b[A-Z]{1,3}\d{2,4}[A-Z]?\b',
            r'\b[vV]\d+(?:\.\d+)*\b',
            r'\b[A-Z0-9]{2,}-[A-Z0-9]{2,}\b'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                self.challenges.append(STTChallenge(
                    term=match,
                    challenge_type="alphanumeric",
                    confidence=0.95,
                    examples=[text[:50]]
                ))
        
        # Abbreviations
        text_lower = text.lower()
        for abbr in self.COMMON_ABBREVIATIONS:
            if re.search(rf'\b{abbr}\b', text_lower):
                self.challenges.append(STTChallenge(
                    term=abbr.upper(),
                    challenge_type="abbreviation", 
                    confidence=0.8,
                    examples=[text[:50]]
                ))
    
    def _build_final_vocabulary(self):
        """Build final vocabulary from challenges"""
        # Sort by confidence
        sorted_challenges = sorted(self.challenges, key=lambda x: -x.confidence)
        
        # Add terms up to size limit
        current_size = 0
        for challenge in sorted_challenges:
            term_size = len(challenge.term) + 1
            if current_size + term_size > self.MAX_VOCAB_SIZE:
                break
            self.final_vocabulary.add(challenge.term)
            current_size += term_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics"""
        type_counts = Counter(c.challenge_type for c in self.challenges)
        return {
            'total_challenges': len(self.challenges),
            'final_terms': len(self.final_vocabulary),
            'by_type': dict(type_counts),
            'sample_terms': list(self.final_vocabulary)[:20]
        }


def test_extractor():
    """Test the STT extractor with sample products"""
    
    # Create test products
    products = [
        Product(
            id="bike1",
            product_id="bike1",
            name="Racing Bike 700c",
            brand="TestBrand",
            description="11-speed groupset, 15.5lbs weight",
            specifications={
                "frame_size": "56cm",
                "weight": "7.03kg",
                "brake_type": "160mm disc"
            }
        ),
        Product(
            id="tv1",
            product_id="tv1", 
            name="UltraHD TV TX-55JZ2000B",
            brand="TechCorp",
            description="4K OLED with 120Hz refresh",
            specifications={
                "model": "TX-55JZ2000B",
                "screen": "55 inch",
                "power": "150W max"
            }
        ),
        Product(
            id="drill1",
            product_id="drill1",
            name="18V Brushless Drill",
            brand="PowerTools",
            description="Max 1800rpm speed",
            specifications={
                "voltage": "18V",
                "torque": "65Nm",
                "speed": "0-600/0-1800rpm"
            }
        )
    ]
    
    # Test extraction
    extractor = SimplifiedSTTExtractorV3()
    extractor.extract_from_products(products)
    
    # Print results
    stats = extractor.get_stats()
    print("\nSTT VOCABULARY EXTRACTOR V3 - TEST RESULTS")
    print("=" * 50)
    print(f"Total challenges found: {stats['total_challenges']}")
    print(f"Final vocabulary size: {stats['final_terms']}")
    print(f"\nBreakdown by type:")
    for challenge_type, count in stats['by_type'].items():
        print(f"  {challenge_type}: {count}")
    print(f"\nSample terms:")
    for term in stats['sample_terms']:
        print(f"  • {term}")
    print("=" * 50)


if __name__ == "__main__":
    test_extractor()