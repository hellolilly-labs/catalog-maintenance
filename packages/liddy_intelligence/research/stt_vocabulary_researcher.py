"""
STT Vocabulary Researcher

A lightweight researcher specifically for extracting industry-specific terms,
pronunciations, and brand vocabulary for STT word_boost and TTS pronunciation guides.

This researcher:
1. Extracts unique product names from the catalog
2. Queries for industry-specific technical terms
3. Gets pronunciation guides for complex terms
4. Outputs a focused list optimized for STT/TTS performance
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Set

from liddy.storage import get_account_storage_provider
from liddy.llm.simple_factory import LLMFactory
from liddy.prompt_manager import PromptManager
from liddy_intelligence.research.data_sources import WebSearchDataSource, DataGatheringContext

logger = logging.getLogger(__name__)


class STTVocabularyResearcher:
    """
    Lightweight researcher for STT/TTS vocabulary extraction.
    
    Unlike full researchers, this is optimized for speed and efficiency,
    extracting only the most important terms for speech recognition/synthesis.
    """
    
    def __init__(self, brand_domain: str, storage_manager=None):
        self.brand_domain = brand_domain
        self.storage_manager = storage_manager or get_account_storage_provider()
        self.prompt_manager = PromptManager()
        
    async def extract_vocabulary(self, products: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract STT/TTS vocabulary from products and industry knowledge.
        
        Args:
            products: Optional list of products (if already loaded)
            
        Returns:
            Dictionary with word_boost terms and pronunciation guides
        """
        start_time = time.time()
        logger.info(f"🎤 Extracting STT/TTS vocabulary for {self.brand_domain}")
        
        # Phase 1: Extract product-based terms
        product_terms = await self._extract_product_terms(products)
        
        # Phase 2: Get industry-specific terms via web search
        industry_terms = await self._get_industry_terms()
        
        # Phase 3: Use LLM to curate and get pronunciations
        vocabulary_data = await self._curate_vocabulary_with_llm(
            product_terms, 
            industry_terms
        )
        
        # Phase 4: Save results
        await self._save_vocabulary(vocabulary_data)
        
        duration = time.time() - start_time
        logger.info(f"✅ Vocabulary extraction completed in {duration:.1f}s")
        
        return vocabulary_data
    
    async def _extract_product_terms(self, products: List[Dict[str, Any]] = None) -> Set[str]:
        """Extract unique terms from product catalog."""
        terms = set()
        
        try:
            # Load products if not provided
            if not products:
                products_json = await self.storage_manager.read_file(
                    account=self.brand_domain,
                    file_path="products.json"
                )
                products = json.loads(products_json) if products_json else []
            
            # Extract unique terms
            for product in products:
                # Product names - try multiple possible field names
                name = product.get('product_name') or product.get('name') or product.get('title')
                if name:
                    # Extract full product name
                    terms.add(name)
                    # Extract individual words that might be technical
                    words = name.split()
                    for word in words:
                        # Keep technical-looking terms (mixed case, numbers, etc.)
                        if (len(word) > 3 and 
                            (any(c.isupper() for c in word[1:]) or 
                             any(c.isdigit() for c in word) or
                             '-' in word)):
                            terms.add(word)
                
                # Brand names - try multiple possible field names
                brand = product.get('brand_name') or product.get('brand') or product.get('manufacturer')
                if brand:
                    terms.add(brand)
                
                # Categories might have technical terms
                if categories := product.get('categories'):
                    for cat in categories:
                        # Extract last part of category (usually most specific)
                        if '>' in cat:
                            specific = cat.split('>')[-1].strip()
                            if len(specific) > 3:
                                terms.add(specific)
                
                # Technical attributes
                for attr in ['material', 'technology', 'series', 'collection']:
                    if value := product.get(attr):
                        if isinstance(value, str) and len(value) > 3:
                            terms.add(value)
            
            logger.info(f"📦 Extracted {len(terms)} unique terms from {len(products)} products")
            return terms
            
        except Exception as e:
            logger.error(f"Error extracting product terms: {e}")
            return terms
    
    async def _get_industry_terms(self) -> Dict[str, Any]:
        """Get industry-specific terms via web search."""
        try:
            web_search = WebSearchDataSource()
            if not web_search.is_available():
                logger.warning("Web search unavailable for industry terms")
                return {}
            
            # Determine industry from domain
            brand_name = self.brand_domain.replace('.com', '').replace('.', ' ').title()
            
            # Search queries for technical terms
            queries = [
                f"{brand_name} technical terminology glossary",
                f"{self.brand_domain} industry jargon terms",
                f"pronunciation guide {brand_name} products",
                f"difficult to pronounce {brand_name} terms",
                f"{brand_name} product technology names"
            ]
            
            context = DataGatheringContext(
                brand_domain=self.brand_domain,
                researcher_name="stt_vocabulary",
                phase_name="industry_terms"
            )
            
            result = await web_search.gather(queries, context)
            
            # Extract terms from search results
            industry_data = {
                "search_results": result.results[:10] if result.results else [],
                "source_count": len(result.sources) if result.sources else 0
            }
            
            logger.info(f"🔍 Found {industry_data['source_count']} sources for industry terms")
            return industry_data
            
        except Exception as e:
            logger.error(f"Error getting industry terms: {e}")
            return {}
    
    async def _curate_vocabulary_with_llm(
        self, 
        product_terms: Set[str], 
        industry_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to curate final vocabulary list with pronunciations."""
        
        # Prepare context for LLM
        product_terms_list = sorted(list(product_terms))[:100]  # Limit for prompt
        
        # Format search results - limit content
        search_context = ""
        for result in industry_data.get("search_results", [])[:3]:
            search_context += f"Title: {result.get('title', '')}\n"
            snippet = result.get('snippet', '')[:200]  # Limit snippet length
            search_context += f"Content: {snippet}...\n\n"
        
        # LLM prompt
        system_prompt = """You are a linguistics expert specializing in speech recognition and synthesis optimization.
Your task is to curate a focused vocabulary list for STT word boost and TTS pronunciation guides."""
        
        user_prompt = f"""
Brand: {self.brand_domain}

Product terms found in catalog:
{json.dumps(product_terms_list, indent=2)}

Industry research context:
{search_context}

Create an optimized vocabulary list with these requirements:

1. Select the TOP 100-150 most important terms for speech recognition that are:
   - Unique to this brand/industry
   - Frequently mispronounced or misrecognized
   - Technical terms not in common English
   - Brand names and product lines
   - DO NOT include common English words

2. For complex terms, provide pronunciation guides using:
   - IPA notation for technical accuracy
   - Simple phonetic spelling for TTS systems
   - Syllable breaks for clarity

3. Categorize terms by importance:
   - Critical (brand names, key products): weight 0.3
   - Important (technical terms, materials): weight 0.2  
   - Useful (categories, features): weight 0.1

Output as JSON:
{{
  "vocabulary": {{
    "critical_terms": [
      {{"term": "example", "ipa": "/ɪɡˈzæmpəl/", "phonetic": "ig-ZAM-pul", "syllables": "ex-am-ple"}}
    ],
    "important_terms": [
      {{"term": "example", "ipa": "...", "phonetic": "..."}}
    ],
    "useful_terms": [
      "term1", "term2"  // Simple list, no pronunciation needed
    ]
  }},
  "word_boost": [
    // Flat list of all terms for STT, ordered by importance
  ],
  "pronunciation_guide": {{
    // Key-value pairs for TTS: term -> phonetic spelling
    "complex_term": "kom-PLEKS term"
  }},
  "stats": {{
    "total_terms": 0,
    "with_pronunciation": 0,
    "categories": {{"critical": 0, "important": 0, "useful": 0}}
  }}
}}
"""
        
        try:
            response = await LLMFactory.chat_completion(
                task="stt_vocabulary_curation",
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            vocabulary_data = json.loads(response.get("content", "{}"))
            
            # Ensure word_boost list exists and is limited
            if "word_boost" not in vocabulary_data:
                # Build from categorized terms
                word_boost = []
                for term_data in vocabulary_data.get("vocabulary", {}).get("critical_terms", []):
                    word_boost.append(term_data.get("term") if isinstance(term_data, dict) else term_data)
                for term_data in vocabulary_data.get("vocabulary", {}).get("important_terms", []):
                    word_boost.append(term_data.get("term") if isinstance(term_data, dict) else term_data)
                word_boost.extend(vocabulary_data.get("vocabulary", {}).get("useful_terms", [])[:50])
                vocabulary_data["word_boost"] = word_boost[:150]  # Limit total
            
            # Add metadata
            vocabulary_data["metadata"] = {
                "brand_domain": self.brand_domain,
                "extraction_time": time.time(),
                "product_terms_analyzed": len(product_terms),
                "industry_sources": industry_data.get("source_count", 0)
            }
            
            return vocabulary_data
            
        except Exception as e:
            logger.error(f"Error curating vocabulary with LLM: {e}")
            # Fallback: just return product terms
            word_boost = list(product_terms)[:150]
            return {
                "word_boost": word_boost,
                "vocabulary": {"all_terms": word_boost},
                "pronunciation_guide": {},
                "metadata": {
                    "brand_domain": self.brand_domain,
                    "extraction_time": time.time(),
                    "error": str(e),
                    "fallback": True
                }
            }
    
    async def _save_vocabulary(self, vocabulary_data: Dict[str, Any]) -> None:
        """Save vocabulary data to storage."""
        try:
            # Save full vocabulary data
            await self.storage_manager.write_file(
                account=self.brand_domain,
                file_path="stt_vocabulary.json",
                content=json.dumps(vocabulary_data, indent=2)
            )
            
            # Also save just the word_boost list for easy access
            word_boost_only = {
                "word_boost": vocabulary_data.get("word_boost", []),
                "boost_weights": {
                    "critical": 0.3,
                    "important": 0.2,
                    "useful": 0.1
                },
                "updated": vocabulary_data.get("metadata", {}).get("extraction_time", time.time())
            }
            
            await self.storage_manager.write_file(
                account=self.brand_domain,
                file_path="stt_word_boost.json",
                content=json.dumps(word_boost_only, indent=2)
            )
            
            logger.info(f"💾 Saved vocabulary data with {len(vocabulary_data.get('word_boost', []))} word boost terms")
            
        except Exception as e:
            logger.error(f"Error saving vocabulary data: {e}")


async def extract_stt_vocabulary(brand_domain: str, products: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to extract STT vocabulary for a brand.
    
    Args:
        brand_domain: The brand domain
        products: Optional pre-loaded products list
        
    Returns:
        Vocabulary data including word_boost and pronunciation guides
    """
    researcher = STTVocabularyResearcher(brand_domain)
    return await researcher.extract_vocabulary(products)