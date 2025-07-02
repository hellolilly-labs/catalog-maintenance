#!/usr/bin/env python3
"""
Test script for SearchPinecone - Unified Search Interface

This demonstrates how to use the SearchPinecone class for both:
1. catalog-maintenance project
2. voice-assistant project
"""

import asyncio
import argparse
import logging
import json
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_search(brand_domain: str):
    """Test basic search functionality."""
    from src.search.search_pinecone import PineconeRAG
    
    logger.info(f"\n🔍 Testing basic search for {brand_domain}")
    
    # Initialize SearchPinecone directly
    search_pinecone = PineconeRAG(brand_domain=brand_domain)
    await search_pinecone.initialize()
    
    # Test queries
    test_queries = [
        ("mountain bike", "General semantic search"),
        ("specialized stumpjumper", "Specific product search"),
        ("bike under $3000", "Price-based search"),
        ("beginner friendly road bike", "Attribute-based search")
    ]
    
    for query, description in test_queries:
        logger.info(f"\n📝 Query: '{query}' ({description})")
        
        # Perform search
        results = await search_pinecone.search(
            query=query,
            top_k=5,
            search_mode="hybrid",
            rerank=True
        )
        
        # Display results
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results[:3], 1):
            logger.info(f"  {i}. {result.metadata.get('name', 'Unknown')} "
                       f"(Score: {result.score:.3f}, Source: {result.source})")


async def test_search_modes(brand_domain: str):
    """Test different search modes."""
    from src.search.search_pinecone import PineconeRAG
    
    logger.info(f"\n🔬 Testing search modes for {brand_domain}")
    
    search_pinecone = PineconeRAG(brand_domain=brand_domain)
    await search_pinecone.initialize()
    
    query = "lightweight carbon frame"
    
    # Test different modes
    modes = ["dense", "sparse", "hybrid"]
    
    for mode in modes:
        logger.info(f"\n🔍 Testing {mode.upper()} search")
        
        results = await search_pinecone.search(
            query=query,
            top_k=3,
            search_mode=mode,
            rerank=(mode == "hybrid")  # Only rerank hybrid results
        )
        
        logger.info(f"Results from {mode} search:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result.metadata.get('name', 'Unknown')} "
                       f"(Score: {result.score:.3f})")
            if result.debug_info and mode == "hybrid":
                logger.info(f"     Dense: {result.debug_info.get('dense_score', 0):.3f}, "
                           f"Sparse: {result.debug_info.get('sparse_score', 0):.3f}")


async def test_filtered_search(brand_domain: str):
    """Test search with filters."""
    from src.search.search_pinecone import PineconeRAG
    
    logger.info(f"\n🎯 Testing filtered search for {brand_domain}")
    
    search_pinecone = PineconeRAG(brand_domain=brand_domain)
    await search_pinecone.initialize()
    
    # Test with different filters
    test_cases = [
        {
            "query": "bike",
            "filters": {"category": "Mountain"},
            "description": "Category filter"
        },
        {
            "query": "bike",
            "filters": {"price": {"$lte": 3000}},
            "description": "Price filter"
        },
        {
            "query": "bike",
            "filters": {"product_labels.skill_level": {"$in": ["beginner", "intermediate"]}},
            "description": "Product label filter"
        }
    ]
    
    for test in test_cases:
        logger.info(f"\n📋 {test['description']}: {test['filters']}")
        
        results = await search_pinecone.search(
            query=test["query"],
            filters=test["filters"],
            top_k=3,
            search_mode="hybrid"
        )
        
        logger.info(f"Found {len(results)} matching results:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result.metadata.get('name', 'Unknown')} "
                       f"(Price: ${result.metadata.get('price', 0):,.0f})")


async def test_voice_assistant_integration(brand_domain: str):
    """Test integration with voice assistant patterns."""
    from src.search.search_pinecone import get_search_pinecone
    
    logger.info(f"\n🎤 Testing voice assistant integration for {brand_domain}")
    
    # Get singleton instance (as voice assistant would)
    search_pinecone = await get_search_pinecone(brand_domain=brand_domain)
    
    # Simulate voice assistant queries
    voice_queries = [
        "Show me your best mountain bikes",
        "I'm looking for something comfortable for long rides",
        "What do you have for beginners?"
    ]
    
    for query in voice_queries:
        logger.info(f"\n💬 Voice query: '{query}'")
        
        # Use the convenience method that returns legacy format
        results = await search_pinecone.search_products(
            query=query,
            top_k=3
        )
        
        logger.info("Response to user:")
        if results:
            logger.info(f"I found {len(results)} great options for you:")
            for i, result in enumerate(results[:2], 1):
                name = result['metadata'].get('name', 'Product')
                price = result['metadata'].get('price', 0)
                logger.info(f"  {i}. {name} - ${price:,.0f}")
        else:
            logger.info("I couldn't find any products matching your request.")


async def test_knowledge_search(brand_domain: str):
    """Test knowledge base search."""
    from src.search.search_pinecone import PineconeRAG
    
    logger.info(f"\n📚 Testing knowledge search for {brand_domain}")
    
    search_pinecone = PineconeRAG(
        brand_domain=brand_domain,
        namespace="information"  # Knowledge namespace
    )
    await search_pinecone.initialize()
    
    # Test knowledge queries
    queries = [
        "How do I maintain my bike?",
        "What's the warranty policy?",
        "Shipping and returns"
    ]
    
    for query in queries:
        logger.info(f"\n❓ Knowledge query: '{query}'")
        
        results = await search_pinecone.search_knowledge(
            query=query,
            top_k=2
        )
        
        if results:
            logger.info("Found relevant articles:")
            for i, result in enumerate(results, 1):
                title = result['metadata'].get('title', 'Article')
                logger.info(f"  {i}. {title} (Score: {result['score']:.3f})")
        else:
            logger.info("No knowledge articles found.")


async def main():
    parser = argparse.ArgumentParser(
        description='Test SearchPinecone unified search interface'
    )
    parser.add_argument(
        'brand_domain',
        help='Brand domain (e.g., specialized.com)'
    )
    parser.add_argument(
        '--test',
        choices=['all', 'basic', 'modes', 'filters', 'voice', 'knowledge'],
        default='all',
        help='Which test to run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"🚀 Testing SearchPinecone for {args.brand_domain}")
    
    try:
        if args.test == 'all':
            await test_basic_search(args.brand_domain)
            await test_search_modes(args.brand_domain)
            await test_filtered_search(args.brand_domain)
            await test_voice_assistant_integration(args.brand_domain)
            await test_knowledge_search(args.brand_domain)
        elif args.test == 'basic':
            await test_basic_search(args.brand_domain)
        elif args.test == 'modes':
            await test_search_modes(args.brand_domain)
        elif args.test == 'filters':
            await test_filtered_search(args.brand_domain)
        elif args.test == 'voice':
            await test_voice_assistant_integration(args.brand_domain)
        elif args.test == 'knowledge':
            await test_knowledge_search(args.brand_domain)
        
        logger.info("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))