#!/usr/bin/env python3
"""
Test STT Vocabulary Extractor V3 with examples showing brand-agnostic approach
"""

import asyncio
import sys
sys.path.append('/Users/collinbrown/Dropbox/Development/Lily/code/catalog-maintenance')

from packages.liddy_intelligence.ingestion.core.stt_vocabulary_extractor_v3 import STTVocabularyExtractorV3
from packages.liddy.models.product import Product


async def test_with_examples():
    """Test with various product types to show brand-agnostic extraction"""
    
    # Create diverse test products from different domains
    test_products = [
        # Bike product with measurements
        Product(
            product_id="bike1",
            name="Professional Racing Bike 700c",
            brand="GenericBikes",
            description="Features 11-speed Shimano groupset, 15.5lbs total weight",
            specifications={
                "frame_size": "56cm",
                "wheel_size": "700x23c", 
                "weight": "7.03kg",
                "gear_ratio": "53/39T",
                "brake_type": "Hydraulic disc 160mm"
            },
            key_selling_points=[
                "Sub-7kg weight",
                "105mm head tube",
                "BB30 bottom bracket"
            ]
        ),
        
        # Electronics with alphanumeric codes
        Product(
            product_id="elec1",
            name="UltraHD TV Model TX-55JZ2000B",
            brand="TechCorp",
            description="4K OLED display with HDR10+ and 120Hz refresh rate",
            specifications={
                "model": "TX-55JZ2000B",
                "screen": "55 inch / 139.7cm",
                "resolution": "3840x2160",
                "ports": "HDMI 2.1, USB 3.0",
                "power": "150W max"
            },
            key_selling_points=[
                "Dolby Vision IQ",
                "WiFi 6E compatible",
                "A14 processor"
            ]
        ),
        
        # Beauty product with chemical terms
        Product(
            product_id="beauty1",
            name="Anti-Aging Serum pH 5.5",
            brand="SkinScience",
            description="Contains 0.5% retinol and niacinamide complex",
            specifications={
                "volume": "30ml / 1fl.oz",
                "concentration": "0.5% retinol",
                "pH": "5.5",
                "shelf_life": "12M after opening"
            },
            key_selling_points=[
                "Paraben-free formula",
                "Contains vitamin C+E complex",
                "Dermatologist-tested"
            ]
        ),
        
        # Hardware with compound terms
        Product(
            product_id="hw1",
            name="Heavy-Duty Drill/Driver Kit",
            brand="PowerTools",
            description="18V brushless motor with 1,800rpm max speed",
            specifications={
                "voltage": "18V Li-ion",
                "torque": "65Nm",
                "chuck": "13mm / 1/2 inch",
                "speed": "0-600/0-1800rpm",
                "battery": "4.0Ah"
            },
            key_selling_points=[
                "LED worklight",
                "Anti-kickback technology",
                "Quick-change chuck"
            ]
        ),
        
        # Clothing with size abbreviations
        Product(
            product_id="cloth1",
            name="All-Weather Jacket Unisex",
            brand="OutdoorGear",
            description="Gore-Tex membrane with YKK zippers",
            specifications={
                "sizes": "XS, S, M, L, XL, XXL",
                "weight": "350g / 12.3oz",
                "waterproof": "20,000mm",
                "breathability": "15,000g/m²/24hr"
            },
            key_selling_points=[
                "DWR coating",
                "Packable design 15x10cm",
                "RDS-certified down"
            ]
        )
    ]
    
    # Test the extractor
    extractor = STTVocabularyExtractorV3("test.com")
    extractor.extract_from_catalog(test_products)
    
    print("\n" + "="*60)
    print("STT VOCABULARY EXTRACTOR V3 - RESULTS")
    print("="*60)
    
    # Group results by challenge type
    terms_by_type = {}
    for challenge in extractor.challenges:
        if challenge.term in extractor.final_vocabulary:
            if challenge.challenge_type not in terms_by_type:
                terms_by_type[challenge.challenge_type] = []
            terms_by_type[challenge.challenge_type].append(challenge.term)
    
    print(f"\nTotal terms extracted: {len(extractor.final_vocabulary)}")
    print(f"Character count: {sum(len(t) + 1 for t in extractor.final_vocabulary) - 1}")
    
    print("\nTerms by challenge type:")
    print("-" * 40)
    
    for challenge_type, terms in sorted(terms_by_type.items()):
        print(f"\n{challenge_type.upper()} ({len(terms)} terms):")
        # Show up to 10 examples per type
        for term in sorted(terms)[:10]:
            print(f"  • {term}")
        if len(terms) > 10:
            print(f"  ... and {len(terms) - 10} more")
    
    print("\n" + "="*60)
    print("Key observations:")
    print("- Measurements: Captures all units (mm, kg, oz, etc.)")
    print("- Alphanumeric: Model numbers regardless of brand")
    print("- Abbreviations: Technical acronyms and size codes")
    print("- Non-phonetic: Complex spellings and foreign terms")
    print("- Compounds: Hyphenated and technical combinations")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_with_examples())