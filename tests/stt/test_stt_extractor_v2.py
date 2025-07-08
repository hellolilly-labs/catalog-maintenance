#!/usr/bin/env python3
"""
Test the new STT Vocabulary Extractor V2
"""

import asyncio
import sys
sys.path.append('/Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance')

from packages.liddy_intelligence.ingestion.core.stt_vocabulary_extractor_v2 import STTVocabularyExtractorV2
from packages.liddy.models.product import Product


async def test_extractor():
    # Create test products with various challenging terms
    test_products = [
        Product(
            product_id="1",
            name="Specialized S-Works Tarmac SL7",
            brand="Specialized",
            description="The S-Works Tarmac SL7 with FACT 12r carbon",
            specifications={"frame": "FACT 12r carbon fiber", "fork": "S-Works FACT"},
            key_selling_points=["Aerodynamic design", "UCI approved"]
        ),
        Product(
            product_id="2", 
            name="Turbo Levo SL Comp Carbon",
            brand="Specialized",
            description="Features the Turbo SL 1.1 motor with 240W of power",
            specifications={"motor": "Specialized SL 1.1", "battery": "320Wh"},
            key_selling_points=["Lightweight e-MTB", "Carbon frame"]
        ),
        Product(
            product_id="3",
            name="Sunday Riley Good Genes All-In-One Lactic Acid Treatment",
            brand="Sunday Riley",
            description="Contains 7% purified lactic acid with licorice and lemongrass",
            specifications={"active": "7% lactic acid", "ingredients": "niacinamide, hyaluronic acid"},
            key_selling_points=["Exfoliating treatment", "Phthalate-free formula"]
        ),
        Product(
            product_id="4",
            name="C.E.O. 15% Vitamin C Brightening Serum", 
            brand="Sunday Riley",
            description="Features THD ascorbate and saccharide isomerate complex",
            specifications={"vitamin_c": "15% THD ascorbate", "texture": "lightweight"},
            key_selling_points=["Advanced vitamin C", "Glycolic acid enhanced"]
        ),
        Product(
            product_id="5",
            name="Generic Mountain Bike Helmet",
            brand="Generic",
            description="A regular mountain bike helmet with standard features",
            specifications={"size": "medium", "color": "black"},
            key_selling_points=["Lightweight", "Comfortable", "Good ventilation"]
        )
    ]
    
    # Test the extractor
    extractor = STTVocabularyExtractorV2("test.com")
    extractor.extract_from_catalog(test_products)
    
    print("\nExtracted vocabulary:")
    print("-" * 50)
    for term in sorted(extractor.final_vocabulary):
        print(f"  {term}")
    print("-" * 50)
    print(f"Total terms: {len(extractor.final_vocabulary)}")
    
    # Check what categories were extracted
    category_examples = {}
    for candidate in extractor.candidates:
        if candidate.term in extractor.final_vocabulary:
            if candidate.category not in category_examples:
                category_examples[candidate.category] = []
            if len(category_examples[candidate.category]) < 3:
                category_examples[candidate.category].append(candidate.term)
    
    print("\nExamples by category:")
    for category, examples in category_examples.items():
        print(f"  {category}: {', '.join(examples)}")


if __name__ == "__main__":
    asyncio.run(test_extractor())