#!/usr/bin/env python3
"""
Extract STT Vocabulary Runner

Extract STT word boost terms and pronunciation guides for a brand.
Can be run independently or as part of catalog ingestion.

Usage:
    python run/extract_stt_vocabulary.py specialized.com
    python run/extract_stt_vocabulary.py specialized.com --show-terms
    python run/extract_stt_vocabulary.py specialized.com --limit 100
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'packages'))

from liddy_intelligence.research.stt_vocabulary_researcher import STTVocabularyResearcher
from liddy.storage import get_account_storage_provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='Extract STT vocabulary for a brand')
    parser.add_argument('brand_domain', help='Brand domain (e.g., specialized.com)')
    parser.add_argument('--show-terms', action='store_true', help='Show extracted terms')
    parser.add_argument('--show-pronunciations', action='store_true', help='Show pronunciation guides')
    parser.add_argument('--limit', type=int, help='Limit number of terms to show')
    parser.add_argument('--force', action='store_true', help='Force re-extraction even if file exists')
    
    args = parser.parse_args()
    
    # Check if vocabulary already exists
    storage = get_account_storage_provider()
    if not args.force:
        try:
            existing = await storage.read_file(args.brand_domain, "stt_word_boost.json")
            if existing:
                logger.info(f"✅ STT vocabulary already exists for {args.brand_domain}")
                vocabulary_data = json.loads(existing)
                
                # Show stats
                word_boost = vocabulary_data.get('word_boost', [])
                logger.info(f"   Word boost terms: {len(word_boost)}")
                logger.info(f"   Last updated: {vocabulary_data.get('updated', 'unknown')}")
                
                if args.show_terms or args.show_pronunciations:
                    # Load full vocabulary data
                    full_data = await storage.read_file(args.brand_domain, "stt_vocabulary.json")
                    if full_data:
                        vocabulary_data = json.loads(full_data)
                
                if args.show_terms:
                    terms = word_boost[:args.limit] if args.limit else word_boost
                    print("\n📝 Word Boost Terms:")
                    for i, term in enumerate(terms, 1):
                        print(f"   {i}. {term}")
                
                if args.show_pronunciations:
                    pronunciation_guide = vocabulary_data.get('pronunciation_guide', {})
                    items = list(pronunciation_guide.items())[:args.limit] if args.limit else pronunciation_guide.items()
                    print("\n🗣️ Pronunciation Guide:")
                    for term, pronunciation in items:
                        print(f"   {term}: {pronunciation}")
                
                return
        except:
            pass
    
    # Extract vocabulary
    logger.info(f"🎤 Extracting STT vocabulary for {args.brand_domain}...")
    
    researcher = STTVocabularyResearcher(args.brand_domain)
    vocabulary_data = await researcher.extract_vocabulary()
    
    # Show results
    word_boost = vocabulary_data.get('word_boost', [])
    pronunciation_guide = vocabulary_data.get('pronunciation_guide', {})
    stats = vocabulary_data.get('stats', {})
    
    print("\n✅ Extraction Complete!")
    print(f"\n📊 Statistics:")
    print(f"   Total terms: {stats.get('total_terms', len(word_boost))}")
    print(f"   With pronunciation: {stats.get('with_pronunciation', len(pronunciation_guide))}")
    if 'categories' in stats:
        print(f"   Categories:")
        for cat, count in stats['categories'].items():
            print(f"      - {cat}: {count}")
    
    if args.show_terms:
        terms = word_boost[:args.limit] if args.limit else word_boost
        print(f"\n📝 Word Boost Terms ({len(terms)} of {len(word_boost)}):")
        for i, term in enumerate(terms, 1):
            print(f"   {i}. {term}")
    
    if args.show_pronunciations:
        items = list(pronunciation_guide.items())[:args.limit] if args.limit else pronunciation_guide.items()
        print(f"\n🗣️ Pronunciation Guide ({len(items)} of {len(pronunciation_guide)}):")
        for term, pronunciation in items:
            print(f"   {term}: {pronunciation}")
    
    # Show sample from each category if available
    if not args.show_terms and 'vocabulary' in vocabulary_data:
        vocab = vocabulary_data['vocabulary']
        print("\n📋 Sample Terms by Category:")
        
        if 'critical_terms' in vocab:
            critical = vocab['critical_terms'][:5]
            print("\n   Critical (brand/product names):")
            for term in critical:
                if isinstance(term, dict):
                    print(f"      - {term['term']}: {term.get('phonetic', 'N/A')}")
                else:
                    print(f"      - {term}")
        
        if 'important_terms' in vocab:
            important = vocab['important_terms'][:5]
            print("\n   Important (technical terms):")
            for term in important:
                if isinstance(term, dict):
                    print(f"      - {term['term']}: {term.get('phonetic', 'N/A')}")
                else:
                    print(f"      - {term}")
        
        if 'useful_terms' in vocab:
            useful = vocab['useful_terms'][:5]
            print("\n   Useful (categories/features):")
            for term in useful:
                print(f"      - {term}")


if __name__ == "__main__":
    asyncio.run(main())