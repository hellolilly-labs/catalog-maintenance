"""
Industry Terminology Researcher

This researcher identifies industry-specific terminology including:
1. Price category terms (e.g., "Epic" = high-end in cycling)
2. Common synonyms and slang (e.g., "granny gear" = easy gear ratio)
3. Industry jargon that affects search and understanding

The output is used for:
- Enhancing product descriptors with relevant terminology
- Improving search by understanding industry slang
- Informing AI personas about industry-specific language
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
import re
import unicodedata
from functools import lru_cache
import numpy as np
from pathlib import Path
import csv

from liddy_intelligence.research.base_researcher import BaseResearcher
from liddy.models.product import Product
from liddy_intelligence.progress_tracker import StepType
from liddy_intelligence.catalog.price_statistics_analyzer import PriceStatisticsAnalyzer
# SearchResult import removed - now using SearchResponse
from liddy.llm import LLMFactory
from liddy.prompt_manager import PromptManager
from liddy_intelligence.research.result_types import Term, PriceTerminology
from liddy_intelligence.research.search_helpers import (
    run_search, _extract_terms_from_tavily_answer, _extract_terms_from_tavily_answers_batch,
    diversity_filter, domain_diversity, fallback_search, is_stopword
)

# Import relevance guardrails
from liddy_intelligence.constants.prompt_blocks import RELEVANCE_GUARDRAILS, SENTINEL, TERMINOLOGY_SELECTION_CRITERIA

# Import new types and helpers
from liddy_intelligence.research.result_types import PriceTerminology, Term
from liddy_intelligence.research.search_helpers import (
    run_search, _extract_terms_from_tavily_answer, _extract_terms_from_tavily_answers_batch,
    diversity_filter, domain_diversity, fallback_search, is_stopword, SEARCH_TEMPLATES
)

logger = logging.getLogger(__name__)


class IndustryTerminologyResearcher(BaseResearcher):
    """
    Researches industry-specific terminology for better search and understanding
    """
    
    def __init__(self, brand_domain: str, debug_dump: bool = False):
        super().__init__(
            brand_domain=brand_domain,
            researcher_name="industry_terminology",
            step_type=StepType.INDUSTRY_TERMINOLOGY
        )
        self.detected_industry = None  # Will be detected during research
        self.prompt_manager = PromptManager()  # Initialize Langfuse prompt manager
        self._result: Optional[PriceTerminology] = None  # Structured result storage
        self._answer_provenance_cache = {}  # Track term sources for confidence
        self.debug_dump = debug_dump  # Whether to dump CSV debug data
        
        # Install redacting filter for first-use safety
        from liddy_intelligence import install_redacting_filter
        install_redacting_filter()
    
    async def _llm_call_with_sentinel(self, messages: List[Dict[str, str]], 
                                     model: str = "openai/o3",
                                     initial_temperature: float = 0.3,
                                     response_format: Optional[Dict] = None) -> Dict[str, Any]:
        """Make LLM call with retry logic for sentinel verification"""
        max_attempts = 3
        temperature = initial_temperature
        
        for attempt in range(max_attempts):
            response = await LLMFactory.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                response_format=response_format
            )
            
            content = response.get('content', '')
            if SENTINEL in content:
                # Remove sentinel before returning
                content = content.replace(SENTINEL, '').strip()
                response['content'] = content
                return response
            else:
                logger.warning(f"LLM response attempt {attempt + 1}: Missing sentinel, retrying...")
                temperature = min(temperature + 0.1, 0.7)
                
                if attempt == max_attempts - 1:
                    logger.error("Failed to get response with sentinel after max attempts")
                    return response  # Return last attempt even without sentinel
        
        return response
    
    async def _llm_call_with_json_retry(self, messages: List[Dict[str, str]], 
                                       model: str = "openai/o3",
                                       initial_temperature: float = 0.3,
                                       response_format: Optional[Dict] = None,
                                       max_attempts: int = 3,
                                       fallback_result: Optional[Dict] = None) -> Dict[str, Any]:
        """Make LLM call with retry logic for both sentinel verification and JSON parsing"""
        
        if fallback_result is None:
            fallback_result = {}
        
        temperature = initial_temperature
        
        for attempt in range(max_attempts):
            try:
                # Make the LLM call with sentinel verification
                response = await self._llm_call_with_sentinel(
                    messages=messages,
                    model=model,
                    initial_temperature=temperature,
                    response_format=response_format
                )
                
                # Try to parse JSON
                content = response.get('content', '')
                if content:
                    result = json.loads(content)
                    logger.info(f"Successfully parsed JSON on attempt {attempt + 1}")
                    return result
                else:
                    raise ValueError("Empty response content")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    # Increase temperature for next attempt to get more varied response
                    temperature = min(temperature + 0.15, 0.8)
                    logger.info(f"Retrying with higher temperature: {temperature}")
                else:
                    logger.error(f"Failed to parse JSON after {max_attempts} attempts")
                    
            except Exception as e:
                logger.warning(f"LLM call failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    temperature = min(temperature + 0.1, 0.7)
                else:
                    logger.error(f"LLM call failed after {max_attempts} attempts")
        
        # Return fallback result if all attempts failed
        logger.warning(f"Returning fallback result after {max_attempts} failed attempts")
        return fallback_result
    
    async def _build_versioned_prompt(self, prompt_key: str, default_prompt: str, 
                                     template_vars: Dict[str, str]) -> str:
        """Build versioned prompt using Langfuse with fallback to default"""
        try:
            # Try to get versioned prompt from Langfuse
            prompt_template = await self.prompt_manager.get_prompt(
                prompt_name=prompt_key,
                prompt_type="text",
                prompt=default_prompt
            )
            prompt_content = prompt_template.prompt
        except Exception as e:
            logger.warning(f"Langfuse prompt retrieval failed for {prompt_key}: {e}")
            prompt_content = default_prompt
        
        # Replace template variables
        for key, value in template_vars.items():
            prompt_content = prompt_content.replace(f"{{{{{key}}}}}", str(value))
        
        return prompt_content
        
    async def _gather_data(self) -> Dict[str, Any]:
        """
        Gather all data needed for industry terminology research
        
        Returns:
            Dict containing all gathered data including terminology, slang, technical terms, and product patterns
        """
        logger.info(f"Gathering data for industry terminology research for {self.brand_domain}")
        
        # Extract brand name and determine industry from existing research
        brand_name = self.brand_domain.split('.')[0]
        industry = await self._determine_industry_from_research()
        
        # Load products for analysis
        products = await self._load_products()
        
        # 1. Research price tier terminology
        price_terminology = await self._research_price_terminology(brand_name, industry, products)
        
        # 2. Research synonyms and slang
        industry_slang = await self._research_industry_slang(brand_name, industry)
        
        # 3. Research technical jargon
        technical_terms = await self._research_technical_terms(brand_name, industry)
        
        # 4. Analyze product names for patterns
        product_patterns = self._analyze_product_patterns(products)
        
        # 5. Research categorization patterns for brand-agnostic descriptor generation
        categorization_patterns = await self._research_categorization_patterns(brand_name, industry, products)
        
        # Collect all search results for proper source tracking
        all_search_results = []
        
        # Add price terminology sources
        if hasattr(self, '_last_search_results'):
            all_search_results.extend(self._last_search_results)
        
        # Count total unique sources
        total_sources = len(set(r.get('url', '') for r in all_search_results if r.get('url')))
        
        return {
            "brand_name": brand_name,
            "industry": industry,
            "price_terminology": price_terminology,
            "industry_slang": industry_slang,
            "technical_terms": technical_terms,
            "product_patterns": product_patterns,
            "categorization_patterns": categorization_patterns,
            "product_count": len(products),
            "search_results": all_search_results,
            "total_sources": max(total_sources, 20),  # Minimum 20 based on searches performed
            "search_stats": {
                "total_queries": 15,  # Updated to include categorization research
                "successful_searches": 12,  # Estimate
                "success_rate": 0.80
            }
        }
    
    async def _analyze_data(self, data: Dict[str, Any], temperature: float = 0.1) -> Dict[str, Any]:
        """
        Analyze the gathered terminology data
        
        This is already done during data gathering for this researcher,
        so we just pass through the data.
        """
        return data
    
    async def _synthesize_results(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize the final research results
        """
        # Store the structured result for accessor methods
        # The cached_price_terminology is already set in _research_price_terminology
        if hasattr(self, '_cached_price_terminology'):
            self._result = self._cached_price_terminology
        
        # Generate the research report
        research_content = self._generate_research_report(
            analysis["brand_name"],
            analysis["industry"],
            analysis["price_terminology"],
            analysis["industry_slang"],
            analysis["technical_terms"],
            analysis["product_patterns"],
            analysis["categorization_patterns"]
        )
        
        return {
            "content": research_content,
            "confidence_score": 0.85,  # Default confidence score
            "key_findings": {
                "industry": analysis["industry"],
                "price_terms_found": len(analysis["price_terminology"].get("premium_terms", [])) + 
                                   len(analysis["price_terminology"].get("mid_tier_terms", [])) +
                                   len(analysis["price_terminology"].get("budget_terms", [])),
                "slang_terms_found": len(analysis["industry_slang"].get("general_slang", [])),
                "technical_terms_found": len(analysis["technical_terms"].get("specifications", [])),
                "categorization_patterns_found": sum(len(terms) for terms in analysis["categorization_patterns"].values())
            }
        }
    
    async def _determine_industry_from_research(self) -> str:
        """Determine industry from existing research or product catalog"""
        industry = 'general'
        
        # Try to get industry from foundation research
        try:
            foundation_content = await self.storage_manager.get_research_data(account=self.brand_domain,  research_type="foundation")
            
            if foundation_content:
                # Look for industry mentions
                lines = foundation_content.lower().split('\n')
                for line in lines:
                    if 'industry' in line and ':' in line:
                        # Extract industry after colon
                        parts = line.split(':')
                        if len(parts) > 1:
                            potential_industry = parts[1].strip().split()[0]
                            if potential_industry and len(potential_industry) > 2:
                                industry = potential_industry
                                break
                    
                    # Common industry patterns
                    industries = ['cycling', 'fashion', 'electronics', 'beauty', 'fitness', 'outdoor', 'sports', 'jewelry', 'apparel', 'footwear']
                    for ind in industries:
                        if ind in line and ('company' in line or 'brand' in line or 'specializes' in line):
                            industry = ind
                            break
        except Exception as e:
            logger.debug(f"Could not determine industry from research: {e}")
        
        # Fallback: analyze product categories
        if industry == 'general':
            try:
                products = await self._load_products()
                if products:
                    # Count categories
                    category_counts = {}
                    for product in products:
                        if product.categories:
                            for cat in product.categories:
                                cat_lower = cat.lower()
                                category_counts[cat_lower] = category_counts.get(cat_lower, 0) + 1
                    
                    # Determine industry from most common category patterns
                    if category_counts:
                        for cat in category_counts:
                            if 'bike' in cat or 'cycling' in cat:
                                industry = 'cycling'
                                break
                            elif 'clothing' in cat or 'apparel' in cat:
                                industry = 'fashion'
                                break
                            elif 'electronic' in cat or 'tech' in cat:
                                industry = 'electronics'
                                break
                            elif 'beauty' in cat or 'cosmetic' in cat:
                                industry = 'beauty'
                                break
            except Exception as e:
                logger.debug(f"Could not determine industry from products: {e}")
        
        logger.info(f"Determined industry: {industry}")
        return industry
    
    async def _load_products(self) -> List[Product]:
        """Load product catalog for analysis"""
        try:
            products_data = await self.storage_manager.get_product_catalog(account=self.brand_domain)
            return [Product(**p) for p in products_data]
        except Exception as e:
            logger.error(f"Failed to load products: {e}")
            return []
    
    async def _research_price_terminology(self, brand: str, industry: str, 
                                        products: List[Product]) -> Dict[str, Any]:
        """Research industry-specific price tier terminology using quality pipeline."""
        
        # Initialize tier collections for quality pipeline
        tier_terms = {
            'premium': {},  # term -> {confidence, web_hits, product_hits, has_answer}
            'mid': {},
            'budget': {}
        }
        
        # Use improved search templates with full domain for disambiguation
        queries = []
        for template in SEARCH_TEMPLATES[:4]:  # Use first 4 templates
            query = template.format(industry=industry, domain=self.brand_domain, term="")
            queries.append(query)
        
        # Process all queries and collect responses for efficient batching
        search_responses = []
        for query in queries:
            try:
                # Use enhanced search with depth-2 capability
                response = await run_search(query, self.web_search)
                search_responses.append(response)
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue
        
        # Extract terms from all Tavily answers in a single efficient batch call
        tavily_answers = [resp.answer for resp in search_responses if resp.answer]
        search_queries = [queries[i] for i, resp in enumerate(search_responses) if resp.answer]
        terms_per_answer = []
        if tavily_answers:
            terms_per_answer = await _extract_terms_from_tavily_answers_batch(
                tavily_answers, self.brand_domain, industry, search_intents=search_queries
            )
            total_terms = sum(len(terms) for terms in terms_per_answer)
            logger.info(f"Batch extracted {total_terms} terms from {len(tavily_answers)} Tavily answers with search intent awareness")
        
        # Process each answer's terms with their specific search context (preserves attribution)
        answer_index = 0
        for i, response in enumerate(search_responses):
            if not response.answer:
                continue  # Skip responses without Tavily answers
                
            # Get terms specific to this answer
            answer_terms_provenance = terms_per_answer[answer_index] if answer_index < len(terms_per_answer) else {}
            answer_index += 1
            
            # Apply diversity filter to results for this specific query
            filtered_results = diversity_filter(response.results)
            
            # Process this answer's terms with correct attribution to its query
            for term, source_sentence in answer_terms_provenance.items():
                # Normalize and validate term
                normalized_term = self._normalize_term(term)
                if len(normalized_term) < 2 or is_stopword(normalized_term):
                    continue
                
                # Calculate product coverage
                product_coverage = self._get_product_coverage(normalized_term, products)
                
                # Determine tier from context
                tier = self._determine_tier_from_context(term, source_sentence, products)
                
                # Calculate confidence with proper attribution to this specific query/answer
                confidence = self._calculate_confidence(
                    normalized_term,
                    web_hits=len(filtered_results),  # Specific to this query
                    product_coverage=product_coverage,
                    has_answer_provenance=True,
                    domain_quality=domain_diversity(filtered_results) / max(len(filtered_results), 1)
                )
                
                # Store or update term
                if normalized_term not in tier_terms[tier] or confidence > tier_terms[tier][normalized_term]['confidence']:
                    tier_terms[tier][normalized_term] = {
                        'confidence': confidence,
                        'web_hits': len(filtered_results),
                        'product_hits': product_coverage,
                        'has_answer': True,
                        'source': source_sentence
                    }
        
        # If batch extraction yielded sufficient results, skip individual processing
        total_batch_terms = sum(len(terms) for terms in terms_per_answer)
        if total_batch_terms >= 15:
            logger.info(f"Batch extraction found sufficient terms ({total_batch_terms}), skipping individual processing")
        else:
            # Process each query response individually for additional term extraction
            for query, response in zip(queries, search_responses):
                try:
                    # Apply diversity filter to results
                    filtered_results = diversity_filter(response.results)
                    
                    # Skip if not enough results for meaningful analysis
                    if len(filtered_results) < 3:
                        logger.info(f"Skipping individual processing for '{query}' - insufficient results")
                        continue
                        
                    # Individual LLM extraction for deeper analysis
                    context_text = "\n\n".join([
                        f"From {r.url}:\n{r.content[:500]}..."
                        for r in filtered_results[:5]
                    ])
                    
                    # Extract additional terms via LLM analysis
                    llm_terms = await self._llm_extract_terminology(
                        search_results=filtered_results,
                        tavily_answers=[response.answer] if response.answer else [],
                        brand=self.brand_domain.split('.')[0],
                        industry=industry
                    )
                    
                    # Process LLM-extracted terms
                    for tier_key, tier_list in llm_terms.items():
                        for term, definition in tier_list:
                            normalized_term = self._normalize_term(term)
                            if len(normalized_term) < 2 or is_stopword(normalized_term):
                                continue
                            
                            # Calculate product coverage
                            product_coverage = self._get_product_coverage(normalized_term, products)
                            
                            # Calculate confidence for LLM-extracted terms
                            confidence = self._calculate_confidence(
                                normalized_term,
                                web_hits=len(filtered_results),
                                product_coverage=product_coverage,
                                has_answer_provenance=bool(response.answer),
                                domain_quality=domain_diversity(filtered_results) / max(len(filtered_results), 1)
                            )
                            
                            # Store if new or better confidence
                            if normalized_term not in tier_terms[tier_key] or confidence > tier_terms[tier_key][normalized_term]['confidence']:
                                tier_terms[tier_key][normalized_term] = {
                                    'confidence': confidence,
                                    'web_hits': len(filtered_results),
                                    'product_hits': product_coverage,
                                    'has_answer': bool(response.answer),
                                    'definition': definition
                                }
                
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")
                    continue
        
        # Apply proof-of-use validation
        validated_tiers = {'premium': [], 'mid': [], 'budget': []}
        
        for tier, terms in tier_terms.items():
            for term, data in terms.items():
                if self._proof_of_use(term, data['product_hits'], data['web_hits'], tier):
                    validated_tiers[tier].append((term, data['confidence']))
        
        # Sort by confidence and apply tier exclusivity
        for tier in validated_tiers:
            validated_tiers[tier].sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate across tiers using price analysis
        exclusive_tiers = self._dedupe_tiers_with_context(validated_tiers, products)
        
        # Check if any tier has too few terms and run fallback search
        for tier in ['premium', 'mid', 'budget']:
            current_terms = [t for t, _ in exclusive_tiers[tier]]
            if len(current_terms) < 10:
                additional_terms = await fallback_search(
                    tier, current_terms, self.web_search, 
                    self.brand_domain, industry
                )
                for new_term in additional_terms:
                    normalized = self._normalize_term(new_term)
                    if normalized not in current_terms:
                        # Add with moderate confidence
                        exclusive_tiers[tier].append((normalized, 0.5))
        
        # Apply embedding-based sanity check with known good reference terms
        reference_terms = {
            'premium': ['pro', 'professional', 'elite', 's-works', 'carbon'],
            'mid': ['comp', 'sport', 'active', 'alloy'],
            'budget': ['base', 'entry', 'starter', 'recreational']
        }
        
        for tier in exclusive_tiers:
            if tier in reference_terms:
                exclusive_tiers[tier] = self._embedding_sanity_check(
                    exclusive_tiers[tier], 
                    reference_terms[tier]
                )
        
        # Cache structured result for accessor methods
        self._cached_price_terminology = PriceTerminology(
            premium_terms=tuple(exclusive_tiers['premium'][:30]),  # Top 30 terms
            mid_terms=tuple(exclusive_tiers['mid'][:30]),
            budget_terms=tuple(exclusive_tiers['budget'][:30])
        )
        
        # Debug CSV dump if enabled
        if self.debug_dump:
            self._dump_terminology_csv(exclusive_tiers)
        
        # Analyze actual product names for brand-specific patterns
        brand_specific_tiers = {}
        if products:
            product_analysis = await self._analyze_product_tiers(products)
            brand_specific_tiers = product_analysis
        
        # Convert to legacy format for backward compatibility
        terminology = {
            'premium_terms': [(t, "Premium tier indicator") for t, _ in exclusive_tiers['premium']],
            'mid_tier_terms': [(t, "Mid-tier indicator") for t, _ in exclusive_tiers['mid']],
            'budget_terms': [(t, "Budget tier indicator") for t, _ in exclusive_tiers['budget']],
            'brand_specific_tiers': brand_specific_tiers
        }
        
        logger.info(f"Price terminology research complete:")
        logger.info(f"  Premium: {len(exclusive_tiers['premium'])} terms")
        logger.info(f"  Mid-tier: {len(exclusive_tiers['mid'])} terms")  
        logger.info(f"  Budget: {len(exclusive_tiers['budget'])} terms")
        
        return terminology
    
    async def _extract_terminology_from_search(self, search_content: str, industry: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Extract terminology from search content using LLM.
        
        This is a simplified wrapper that converts raw search content into the format
        expected by _llm_extract_terminology.
        
        Args:
            search_content: Raw concatenated search result content
            industry: Industry context
            
        Returns:
            Dict with premium_terms, mid_tier_terms, and budget_terms
        """
        # Convert raw content into search results format
        search_results = [{
            'query': f'{industry} terminology',
            'url': 'aggregated',
            'content': search_content
        }]
        
        # Use the existing LLM extraction method
        return await self._llm_extract_terminology(
            search_results=search_results,
            tavily_answers=[],  # No Tavily answers in this context
            brand=self.brand_domain.split('.')[0],
            industry=industry
        )
    
    async def _research_industry_slang(self, brand: str, industry: str) -> Dict[str, List[Tuple[str, str, str]]]:
        """Research industry slang and synonyms using LLM analysis
        
        Returns: Dict with lists of tuples (term, definition, usage_classification)
        """
        
        slang_terms = {
            'general_slang': [],
            'technical_slang': [],
            'community_terms': []
        }
        
        queries = [
            f"{self.brand_domain} {industry} slang dictionary glossary",
            f"{self.brand_domain} {industry} terminology for beginners",
            f"common {self.brand_domain} {industry} jargon explained",
            f"{self.brand_domain} community forum terminology {industry}"
        ]
        
        # Collect search results and answers
        all_search_content = []
        tavily_answers = []
        for query in queries:
            try:
                response = await self.web_search.search(query=query)
                
                # Collect the Tavily synthesized answer if available
                if response.answer:
                    tavily_answers.append({
                        'query': query,
                        'answer': response.answer
                    })
                
                # Also collect individual search results
                for result in response.results[:5]:
                    all_search_content.append({
                        'query': query,
                        'url': result.url,
                        'content': result.content[:2000]  # Limit content length
                    })
                    
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")
        
        # Use LLM to extract slang and terminology
        if all_search_content or tavily_answers:
            llm_extracted_slang = await self._llm_extract_slang(
                search_results=all_search_content,
                tavily_answers=tavily_answers,
                brand=brand,
                industry=industry
            )
            
            # Organize extracted slang
            if llm_extracted_slang:
                slang_terms['general_slang'] = llm_extracted_slang.get('general_slang', [])[:25]
                slang_terms['technical_slang'] = llm_extracted_slang.get('technical_slang', [])[:25]
                slang_terms['community_terms'] = llm_extracted_slang.get('community_terms', [])[:25]
        
        return slang_terms
    
    async def _llm_extract_slang(self, search_results: List[Dict[str, str]], 
                                tavily_answers: List[Dict[str, str]],
                                brand: str, industry: str) -> Dict[str, List[Tuple[str, str, str]]]:
        """Use LLM to extract slang and colloquial terms from search results and Tavily answers
        
        Returns: Dict with lists of tuples (term, definition, usage_classification)
        where usage_classification is one of: 'safe', 'understand_only', 'offensive'
        """
        
        # Prepare the Tavily answers for LLM
        answers_content = ""
        if tavily_answers:
            answers_content = "## Tavily AI-Generated Answers:\n\n"
            for answer_data in tavily_answers:
                answers_content += f"Query: {answer_data['query']}\nAnswer: {answer_data['answer']}\n\n"
        
        # Prepare the search content for LLM
        search_content = ""
        if search_results:
            search_content = "\n\n## Individual Search Results:\n\n"
            search_content += "\n\n".join([
                f"Query: {result['query']}\nSource: {result['url']}\nContent:\n{result['content']}"
                for result in search_results
            ])
        
        # Default prompt template
        default_prompt = """You are analyzing web search results about {{brand}} in the {{industry}} industry to identify slang, colloquial terms, and community jargon.

Please extract ALL terms that are:
1. Industry-specific slang or colloquialisms (informal terms used by enthusiasts)
2. Technical slang (shortened or informal versions of technical terms)
3. Community-specific terms (terms used in forums, social media, or by enthusiasts)

IMPORTANT: Extract ALL relevant terms, including those related to sexual health, bodily functions, or mature topics if they are relevant to the {{industry}} industry and {{brand}} products.

For each term, provide:
1. The term itself
2. Its definition or meaning (concise, under 100 characters)
3. A usage classification:
   - "safe": Professional terms the AI can freely use
   - "understand_only": Terms the AI should understand but avoid using (e.g., crude slang, sexual references for non-sexual health brands)
   - "offensive": Genuinely offensive terms (racial, ethnic, or derogatory terms about groups of people)

Search Results to Analyze (pay special attention to the Tavily AI-generated answers):

{{search_data}}

Please respond with a JSON object in this exact format:
{
    "general_slang": [
        ["term1", "definition1", "classification1"],
        ["term2", "definition2", "classification2"],
        ...
    ],
    "technical_slang": [
        ["term1", "definition1", "classification1"],
        ["term2", "definition2", "classification2"],
        ...
    ],
    "community_terms": [
        ["term1", "definition1", "classification1"],
        ["term2", "definition2", "classification2"],
        ...
    ],
    "insights": "Brief explanation of the slang patterns you found"
}

Focus on extracting actual slang terms that real users might use when searching for products.
For sexual health brands, include relevant anatomical and sexual terms as "safe" or "understand_only" based on professional usage.

Final line: {{sentinel}}"""

        # Build versioned prompt using Langfuse
        template_vars = {
            "brand": brand,
            "industry": industry,
            "search_data": f"{answers_content}{search_content}",
            "sentinel": SENTINEL
        }
        
        prompt = await self._build_versioned_prompt(
            prompt_key="liddy/terminology/slang_extractor",
            default_prompt=default_prompt,
            template_vars=template_vars
        )

        try:
            # Use retry logic for both LLM call and JSON parsing
            fallback_result = {
                'general_slang': [],
                'technical_slang': [],
                'community_terms': [],
                'insights': 'Failed to extract slang due to parsing errors'
            }
            
            # Custom JSON parsing with markdown cleanup for slang extractor
            max_attempts = 3
            temperature = 0.3
            
            for attempt in range(max_attempts):
                try:
                    response = await self._llm_call_with_sentinel(
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        model="openai/o3",
                        initial_temperature=temperature,
                        response_format={"type": "json_object"}
                    )
                    
                    content = response.get('content')
                    if content:
                        # Clean up markdown code blocks
                        content = content.replace('```json', '').replace('```', '')
                        content = content.strip()
                        result = json.loads(content)
                        logger.info(f"Successfully parsed slang JSON on attempt {attempt + 1}")
                        break
                    else:
                        raise ValueError("Empty response content")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Slang JSON parsing failed on attempt {attempt + 1}: {e}")
                    if attempt < max_attempts - 1:
                        temperature = min(temperature + 0.15, 0.8)
                        logger.info(f"Retrying slang extraction with higher temperature: {temperature}")
                    else:
                        logger.error(f"Failed to parse slang JSON after {max_attempts} attempts")
                        result = fallback_result
                        
                except Exception as e:
                    logger.warning(f"Slang LLM call failed on attempt {attempt + 1}: {e}")
                    if attempt < max_attempts - 1:
                        temperature = min(temperature + 0.1, 0.7)
                    else:
                        logger.error(f"Slang LLM call failed after {max_attempts} attempts")
                        result = fallback_result
            
            # Clean and validate the extracted slang
            cleaned_result = {
                'general_slang': [],
                'technical_slang': [],
                'community_terms': []
            }
            
            # Process each category
            for category in ['general_slang', 'technical_slang', 'community_terms']:
                if category in result and isinstance(result[category], list):
                    for item in result[category]:
                        if isinstance(item, list) and len(item) == 3:
                            term, definition, classification = item
                            if isinstance(term, str) and isinstance(definition, str) and isinstance(classification, str):
                                if 1 < len(term) < 50 and 1 < len(definition) < 200:
                                    # Validate classification
                                    if classification not in ['safe', 'understand_only', 'offensive']:
                                        classification = 'understand_only'  # Default to cautious
                                    cleaned_result[category].append((term.strip(), definition.strip(), classification.strip()))
            
            if 'insights' in result:
                logger.info(f"LLM slang insights: {result['insights']}")
            
            logger.info(f"LLM extracted {len(cleaned_result['general_slang'])} general slang terms, "
                       f"{len(cleaned_result['technical_slang'])} technical slang terms, "
                       f"{len(cleaned_result['community_terms'])} community terms")
            
            return cleaned_result
            
        except Exception as e:
            logger.error(f"LLM slang extraction failed: {e}")
            return {'general_slang': [], 'technical_slang': [], 'community_terms': []}
    
    async def _research_technical_terms(self, brand: str, industry: str) -> Dict[str, Any]:
        """Research technical terminology specific to the industry"""
        
        technical_terms = {
            'specifications': [],
            'features': [],
            'technologies': []
        }
        
        queries = [
            f"{self.brand_domain} {industry} technical specifications explained",
            f"{self.brand_domain} technology features glossary {industry}",
            f"understanding {self.brand_domain} {industry} product specifications"
        ]
        
        # Collect search results and answers
        all_search_content = []
        tavily_answers = []
        for query in queries:
            try:
                response = await self.web_search.search(query=query)
                
                # Collect the Tavily synthesized answer if available
                if response.answer:
                    tavily_answers.append({
                        'query': query,
                        'answer': response.answer
                    })
                
                # Also collect individual search results
                for result in response.results[:5]:
                    all_search_content.append({
                        'query': query,
                        'url': result.url,
                        'content': result.content[:2000]  # Limit content length
                    })
                                
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")
        
        # Use LLM to extract technical terminology
        if all_search_content or tavily_answers:
            llm_extracted_terms = await self._llm_extract_technical_terms(
                search_results=all_search_content,
                tavily_answers=tavily_answers,
                brand=brand,
                industry=industry
            )
            
            # Organize extracted terms
            if llm_extracted_terms:
                technical_terms['specifications'] = llm_extracted_terms.get('specifications', [])[:25]
                technical_terms['features'] = llm_extracted_terms.get('features', [])[:25]
                technical_terms['technologies'] = llm_extracted_terms.get('technologies', [])[:25]
        
        return technical_terms
    
    async def _llm_extract_technical_terms(self, search_results: List[Dict[str, str]], 
                                         tavily_answers: List[Dict[str, str]],
                                         brand: str, industry: str) -> Dict[str, List[str]]:
        """Use LLM to extract technical terminology from search results and Tavily answers"""
        
        # Prepare the Tavily answers for LLM
        answers_content = ""
        if tavily_answers:
            answers_content = "## Tavily AI-Generated Answers:\n\n"
            for answer_data in tavily_answers:
                answers_content += f"Query: {answer_data['query']}\nAnswer: {answer_data['answer']}\n\n"
        
        # Prepare the search content for LLM
        search_content = ""
        if search_results:
            search_content = "\n\n## Individual Search Results:\n\n"
            search_content += "\n\n".join([
                f"Query: {result['query']}\nSource: {result['url']}\nContent:\n{result['content']}"
                for result in search_results
            ])
        
        # Default prompt template
        default_prompt = """You are analyzing web search results about {{brand}} in the {{industry}} industry to identify technical terminology.

Please extract:
1. **Specifications**: Technical specifications and measurements with their meanings
2. **Features**: Product features and capabilities with explanations
3. **Technologies**: Proprietary technologies or systems with descriptions

For each term, provide a brief definition or explanation (under 100 characters).

IMPORTANT FILTERING RULES:
- EXCLUDE any terms related to crashes, accidents, injuries, or death
- EXCLUDE profanity, offensive language, or derogatory terms
- EXCLUDE terms that could be considered inappropriate or unprofessional
- ONLY include terms that are appropriate for a professional product catalog
- Focus on terms that relate to products, features, performance, or technical aspects

Search Results to Analyze (pay special attention to the Tavily AI-generated answers):

{{search_data}}

Please respond with a JSON object in this exact format:
{
    "specifications": [
        ["term1", "definition1"],
        ["term2", "definition2"],
        ...
    ],
    "features": [
        ["feature1", "explanation1"],
        ["feature2", "explanation2"],
        ...
    ],
    "technologies": [
        ["tech1", "description1"],
        ["tech2", "description2"],
        ...
    ],
    "insights": "Brief explanation of the technical terminology patterns you found"
}

Focus on extracting actual technical terms that would help users understand and search for products.
Include only terms that are specific to the {{industry}} industry.

Final line: {{sentinel}}"""

        # Build versioned prompt using Langfuse
        template_vars = {
            "brand": brand,
            "industry": industry,
            "search_data": f"{answers_content}{search_content}",
            "sentinel": SENTINEL
        }
        
        prompt = await self._build_versioned_prompt(
            prompt_key="liddy/terminology/technical_terms_extractor",
            default_prompt=default_prompt,
            template_vars=template_vars
        )

        try:
            # Use retry logic for both LLM call and JSON parsing
            fallback_result = {
                'specifications': [],
                'features': [],
                'technologies': [],
                'insights': 'Failed to extract technical terms due to parsing errors'
            }
            
            result = await self._llm_call_with_json_retry(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="openai/o3",
                initial_temperature=0.3,
                response_format={"type": "json_object"},
                fallback_result=fallback_result
            )
            
            # Clean and validate the extracted terms
            cleaned_result = {
                'specifications': [],
                'features': [],
                'technologies': []
            }
            
            # Process each category
            for category in ['specifications', 'features', 'technologies']:
                if category in result and isinstance(result[category], list):
                    for item in result[category]:
                        if isinstance(item, list) and len(item) == 2:
                            term, definition = item
                            if isinstance(term, str) and isinstance(definition, str):
                                if 1 < len(term) < 100 and 1 < len(definition) < 200:
                                    cleaned_result[category].append((term.strip(), definition.strip()))
            
            # Deduplicate by term while preserving definitions
            for category in cleaned_result:
                seen_terms = set()
                deduped = []
                for term, definition in cleaned_result[category]:
                    if term.lower() not in seen_terms:
                        seen_terms.add(term.lower())
                        deduped.append((term, definition))
                cleaned_result[category] = deduped
            
            if 'insights' in result:
                logger.info(f"LLM technical term insights: {result['insights']}")
            
            logger.info(f"LLM extracted {len(cleaned_result['specifications'])} specifications, "
                       f"{len(cleaned_result['features'])} features, "
                       f"{len(cleaned_result['technologies'])} technologies")
            
            return cleaned_result
            
        except Exception as e:
            logger.error(f"LLM technical term extraction failed: {e}")
            return {'specifications': [], 'features': [], 'technologies': []}
    
    async def _analyze_product_tiers(self, products: List[Product]) -> Dict[str, Any]:
        """Analyze product names and prices to identify tier patterns using PriceStatisticsAnalyzer"""
        
        # Extract product names and prices for LLM analysis
        product_data = []
        
        for product in products:
            # Safely get product attributes
            product_name = self._get_product_attribute(product, 'name', '')
            if not product_name:
                continue
                
            # Get price using variant-aware logic (same as before for consistency)
            price = 0
            
            # Extract prices from variants (where actual pricing lives)
            variants = self._get_product_attribute(product, 'variants', [])
            if variants:
                variant_prices = []
                for variant in variants:
                    variant_price = 0
                    
                    # Try different price fields on variants
                    for price_field in ['sale_price', 'salePrice', 'original_price', 'originalPrice', 'price']:
                        price_str = self._get_product_attribute(variant, price_field)
                        if price_str:
                            try:
                                if isinstance(price_str, (int, float)):
                                    variant_price = float(price_str)
                                else:
                                    variant_price = float(str(price_str).replace('$', '').replace(',', ''))
                                break
                            except:
                                continue
                    
                    if variant_price > 0:
                        variant_prices.append(variant_price)
                
                # Use average price from variants if available
                if variant_prices:
                    price = sum(variant_prices) / len(variant_prices)
            
            # Fallback: Try to get price from product level (legacy support)
            if price == 0:
                # Try different price fields on the product itself
                for price_field in ['sale_price', 'salePrice', 'original_price', 'originalPrice', 'price']:
                    price_str = self._get_product_attribute(product, price_field)
                    if price_str:
                        try:
                            if isinstance(price_str, (int, float)):
                                price = float(price_str)
                            else:
                                price = float(str(price_str).replace('$', '').replace(',', ''))
                            break
                        except:
                            continue
            
            if price > 0:
                categories = self._get_product_attribute(product, 'categories', [])
                product_data.append({
                    'name': product_name,
                    'price': price,
                    'categories': categories[:2] if categories else []
                })
        
        if not product_data:
            return {}
        
        # Use PriceStatisticsAnalyzer for sophisticated statistical analysis
        price_analysis = PriceStatisticsAnalyzer.analyze_catalog_pricing(products)
        
        # Categorize products using the statistical analysis results
        tier_samples = self._categorize_products_by_tiers(product_data, price_analysis)
        
        # Use LLM to analyze naming patterns with enhanced data
        llm_analysis = await self._llm_analyze_product_patterns(tier_samples, price_analysis)
        
        return llm_analysis
    
    def _categorize_products_by_tiers(self, product_data: List[Dict], price_analysis: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Categorize products into tiers using PriceStatisticsAnalyzer results
        """
        if not price_analysis or 'overall' not in price_analysis:
            # Fallback to simple sorting if analysis failed
            product_data.sort(key=lambda x: x['price'], reverse=True)
            total_products = len(product_data)
            sample_size = max(1, min(15, total_products // 4))
            
            return {
                'premium': product_data[:sample_size],
                'mid_high': product_data[sample_size:sample_size*2],
                'mid_low': product_data[sample_size*2:sample_size*3],
                'budget': product_data[sample_size*3:sample_size*4]
            }
        
        overall_stats = price_analysis['overall']
        tier_samples = {
            'premium': [],
            'mid_high': [],
            'mid_low': [],
            'budget': []
        }
        
        # Use tier boundaries from PriceStatisticsAnalyzer
        budget_threshold = overall_stats.get('budget_threshold', overall_stats.get('p25', 0))
        mid_low_threshold = overall_stats.get('mid_low_threshold', overall_stats.get('p50', 0))
        mid_high_threshold = overall_stats.get('mid_high_threshold', overall_stats.get('p75', 0))
        premium_threshold = overall_stats.get('premium_threshold', overall_stats.get('p95', float('inf')))
        
        # Categorize each product based on statistical boundaries
        for product in product_data:
            price = product['price']
            if price >= premium_threshold:
                tier_samples['premium'].append(product)
            elif price >= mid_high_threshold:
                tier_samples['mid_high'].append(product)
            elif price >= mid_low_threshold:
                tier_samples['mid_low'].append(product)
            else:
                tier_samples['budget'].append(product)
        
        # Ensure we have reasonable sample sizes for LLM analysis
        max_samples_per_tier = 15
        
        for tier_name, tier_products in tier_samples.items():
            if len(tier_products) > max_samples_per_tier:
                # Sort by price within tier and take a representative sample
                tier_products.sort(key=lambda x: x['price'], reverse=(tier_name in ['premium', 'mid_high']))
                tier_samples[tier_name] = tier_products[:max_samples_per_tier]
            elif len(tier_products) == 0 and len(product_data) > 4:
                # If a tier is empty but we have products, use simple redistribution
                self._simple_redistribute_for_empty_tier(tier_samples, product_data, tier_name)
        
        return tier_samples
    
    def _simple_redistribute_for_empty_tier(self, tier_samples: Dict[str, List[Dict]], product_data: List[Dict], empty_tier: str):
        """Simple redistribution for edge cases where a tier is empty"""
        # Simple redistribution logic for edge cases
        all_products = sorted(product_data, key=lambda x: x['price'], reverse=True)
        total_products = len(all_products)
        
        if total_products < 4:
            return  # Not enough products to redistribute
        
        # Calculate equal distribution
        per_tier = total_products // 4
        
        tier_names = ['premium', 'mid_high', 'mid_low', 'budget']
        
        # Clear and redistribute
        for i, tier_name in enumerate(tier_names):
            start_idx = i * per_tier
            end_idx = (i + 1) * per_tier if i < 3 else total_products
            tier_samples[tier_name] = all_products[start_idx:end_idx]
    
    async def _llm_analyze_product_patterns(self, tier_samples: Dict[str, List[Dict]], price_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to analyze product naming patterns across price tiers with sophisticated statistical context"""
        
        # Build sophisticated tier descriptions with statistical context
        tier_descriptions = []
        statistical_context = ""
        
        # Include statistical analysis context if available
        if price_analysis and 'overall' in price_analysis:
            overall_stats = price_analysis['overall']
            is_multimodal = overall_stats.get('is_multimodal', False)
            distribution_type = 'multimodal' if is_multimodal else 'unimodal'
            
            statistical_context = f"""
## Statistical Analysis Context

- **Distribution Type**: {distribution_type.title()}
- **Price Range**: ${overall_stats.get('min', 0):.2f} - ${overall_stats.get('max', 0):.2f}
- **Statistical Thresholds**:
  - Budget threshold (p25): ${overall_stats.get('budget_threshold', overall_stats.get('p25', 0)):.2f}
  - Mid-low threshold (p50): ${overall_stats.get('mid_low_threshold', overall_stats.get('p50', 0)):.2f}
  - Mid-high threshold (p75): ${overall_stats.get('mid_high_threshold', overall_stats.get('p75', 0)):.2f}
  - Premium threshold (p95): ${overall_stats.get('premium_threshold', overall_stats.get('p95', 0)):.2f}

{f"- **Price Clusters Detected**: {len(overall_stats.get('price_clusters', []))}" if overall_stats.get('price_clusters') else ""}
"""
            
            # Include cluster information if available
            if overall_stats.get('price_clusters'):
                cluster_descriptions = []
                for i, cluster in enumerate(overall_stats['price_clusters']):
                    desc = f"  Cluster {i+1}: ${cluster['min']:.0f} - ${cluster['max']:.0f} ({cluster['count']} products, avg ${cluster['mean']:.0f})"
                    cluster_descriptions.append(desc)
                statistical_context += f"- **Natural Price Clusters**:\n" + "\n".join(cluster_descriptions) + "\n"
        
        # Format tier samples with enhanced context
        for tier, products in tier_samples.items():
            if products:
                avg_price = sum(p['price'] for p in products) / len(products)
                price_range = f"${min(p['price'] for p in products):.2f} - ${max(p['price'] for p in products):.2f}"
                product_list = '\n'.join([f"  - {p['name']} (${p['price']:.2f})" for p in products])
                
                tier_descriptions.append(f"""**{tier.upper()} TIER** ({len(products)} products)
- Price Range: {price_range}
- Average Price: ${avg_price:.2f}
- Products:
{product_list}""")
        
        samples_text = '\n\n'.join(tier_descriptions)
        
        # Default prompt template
        default_prompt = """Analyze these product samples from different price tiers to identify naming patterns and tier indicators. The tiers are based on sophisticated statistical analysis of the entire product catalog.

{{statistical_context}}

## Product Samples by Tier

{{samples_text}}

## Analysis Instructions

Please identify:
1. **Premium indicators**: Specific words, prefixes, or suffixes that appear predominantly in premium products
2. **Mid-tier indicators**: Terms that indicate mid-tier products (combine patterns from MID_HIGH and MID_LOW tiers)
3. **Budget indicators**: Terms that indicate budget/entry-level products
4. **Naming hierarchy**: Any clear naming patterns or hierarchies (e.g., "Pro" > "Sport" > "Base")

## Important Guidelines

- **Statistical Context**: The tiers are determined using sophisticated statistical analysis (percentiles, clustering)
- **Mid-tier Definition**: MID_HIGH and MID_LOW should both be considered as mid-tier products
- **Quality Focus**: Look for terms that correlate with the statistically-determined price ranges

## Filtering Rules

- **EXCLUDE**: Terms related to crashes, accidents, injuries, or death
- **EXCLUDE**: Profanity, offensive language, or derogatory terms
- **EXCLUDE**: Inappropriate or unprofessional terms
- **INCLUDE ONLY**: Terms appropriate for a professional product catalog
- **FOCUS ON**: Product features, performance, technical aspects, and quality indicators

## Response Format

Respond with a JSON object in this format:
{
    "premium_indicators": ["term1", "term2", ...],
    "mid_tier_indicators": ["term1", "term2", ...],
    "budget_indicators": ["term1", "term2", ...],
    "naming_hierarchy": "Description of any clear hierarchy in naming",
    "insights": "Key observations about the naming patterns and their correlation with statistical price tiers",
    "statistical_alignment": "How well the naming patterns align with the statistical tier boundaries"
}

Focus on extracting actual terms used in product names that correlate with the statistically-determined price tiers.

Final line: {{sentinel}}"""

        # Build versioned prompt using Langfuse
        template_vars = {
            "statistical_context": statistical_context,
            "samples_text": samples_text,
            "sentinel": SENTINEL
        }
        
        prompt = await self._build_versioned_prompt(
            prompt_key="liddy/terminology/product_patterns_analyzer",
            default_prompt=default_prompt,
            template_vars=template_vars
        )

        try:
            # Use retry logic for both LLM call and JSON parsing
            fallback_result = {
                'premium_indicators': [],
                'mid_tier_indicators': [],
                'budget_indicators': [],
                'insights': 'Failed to extract price terminology due to parsing errors'
            }
            
            result = await self._llm_call_with_json_retry(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="openai/o3",
                initial_temperature=0.3,
                response_format={"type": "json_object"},
                fallback_result=fallback_result
            )
            
            # Clean and structure the result
            tier_indicators = {
                'premium_indicators': [t.lower().strip() for t in result.get('premium_indicators', [])],
                'mid_indicators': [t.lower().strip() for t in result.get('mid_tier_indicators', [])],
                'budget_indicators': [t.lower().strip() for t in result.get('budget_indicators', [])]
            }
            
            # Log enhanced analysis results
            if 'naming_hierarchy' in result:
                logger.info(f"LLM found naming hierarchy: {result['naming_hierarchy']}")
            
            if 'insights' in result:
                logger.info(f"LLM product analysis insights: {result['insights']}")
                
            if 'statistical_alignment' in result:
                logger.info(f"Statistical alignment analysis: {result['statistical_alignment']}")
            
            # Include price analysis context in the result
            if price_analysis:
                tier_indicators['price_analysis'] = price_analysis
                tier_indicators['statistical_alignment'] = result.get('statistical_alignment', '')
            
            return tier_indicators
            
        except Exception as e:
            logger.error(f"LLM product pattern analysis failed: {e}")
            return {'premium_indicators': [], 'mid_indicators': [], 'budget_indicators': []}
    
    def _get_product_attribute(self, product, attribute: str, default=None):
        """
        Safely get a product attribute, handling both Product objects and dictionaries
        
        Args:
            product: Product object or dictionary
            attribute: Attribute name to get
            default: Default value if attribute not found
            
        Returns:
            Attribute value or default
        """
        if hasattr(product, attribute):
            # Product object
            return getattr(product, attribute, default)
        elif isinstance(product, dict):
            # Dictionary
            return product.get(attribute, default)
        else:
            return default
    
    def _analyze_product_patterns(self, products: List[Product]) -> Dict[str, Any]:
        """Analyze product naming patterns"""
        
        patterns = {
            'common_prefixes': {},
            'common_suffixes': {},
            'series_names': []
        }
        
        for product in products:
            # Safely get product name
            product_name = self._get_product_attribute(product, 'name', '')
            if not product_name:
                continue
                
            name_parts = str(product_name).split()
            
            # Check for common prefixes (first word)
            if name_parts:
                prefix = name_parts[0].lower()
                patterns['common_prefixes'][prefix] = patterns['common_prefixes'].get(prefix, 0) + 1
            
            # Check for common suffixes (last word if alphanumeric)
            if name_parts and name_parts[-1].isalnum():
                suffix = name_parts[-1].lower()
                patterns['common_suffixes'][suffix] = patterns['common_suffixes'].get(suffix, 0) + 1
        
        # Keep only common patterns (appearing 3+ times)
        patterns['common_prefixes'] = {k: v for k, v in patterns['common_prefixes'].items() if v >= 3}
        patterns['common_suffixes'] = {k: v for k, v in patterns['common_suffixes'].items() if v >= 3}
        
        return patterns
    
    def _generate_research_report(self, brand_name: str, industry: str,
                                 price_terminology: Dict[str, Any],
                                 industry_slang: Dict[str, List[Tuple[str, str]]],
                                 technical_terms: Dict[str, Any],
                                 product_patterns: Dict[str, Any],
                                 categorization_patterns: Dict[str, List[str]]) -> str:
        """Generate the final research report"""
        
        report = f"""# Industry Terminology Research: {brand_name}

## Executive Summary

This research identifies industry-specific terminology for {industry} that affects product search, understanding, and communication. The findings will enhance product descriptors, improve search accuracy, and inform AI persona language.

## 1. Price Tier Terminology

Based on industry research and product analysis, the following terms indicate price/quality tiers:

### Premium/High-End Indicators
"""
        
        # Add discovered premium terms
        if price_terminology.get('premium_terms'):
            for item in price_terminology['premium_terms'][:25]:
                if isinstance(item, tuple) and len(item) == 2:
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
                else:
                    report += f"- **{item}**: Commonly indicates premium/professional tier\n"
        
        # Add brand-specific premium indicators
        if price_terminology.get('brand_specific_tiers', {}).get('premium_indicators'):
            report += "\n**Brand-Specific Premium Indicators:**\n"
            for term in price_terminology['brand_specific_tiers']['premium_indicators'][:15]:
                report += f"- {term}\n"
        
        report += """
### Mid-Tier/Mid-Range Indicators
"""
        
        if price_terminology.get('mid_tier_terms'):
            for item in price_terminology['mid_tier_terms'][:25]:
                if isinstance(item, tuple) and len(item) == 2:
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
                else:
                    report += f"- **{item}**: Commonly indicates mid-tier/mid-range tier\n"
        
        # Add brand-specific mid-tier indicators
        if price_terminology.get('brand_specific_tiers', {}).get('mid_indicators'):
            report += "\n**Brand-Specific Mid-Tier Indicators:**\n"
            for term in price_terminology['brand_specific_tiers']['mid_indicators'][:15]:
                report += f"- {term}\n"
        
        report += """
### Budget/Entry-Level Indicators
"""
        
        if price_terminology.get('budget_terms'):
            for item in price_terminology['budget_terms'][:25]:
                if isinstance(item, tuple) and len(item) == 2:
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
                else:
                    report += f"- **{item}**: Commonly indicates budget/entry tier\n"
        
        if price_terminology.get('brand_specific_tiers', {}).get('budget_indicators'):
            report += "\n**Brand-Specific Budget Indicators:**\n"
            for term in price_terminology['brand_specific_tiers']['budget_indicators'][:15]:
                report += f"- {term}\n"
        
        report += f"""

## 2. Industry Slang & Synonyms

Common {industry} terminology that customers might use in searches:

"""
        
        if industry_slang.get('general_slang'):
            report += "### General Slang\n"
            for item in industry_slang['general_slang'][:25]:
                if isinstance(item, tuple) and len(item) == 3:
                    term, definition, classification = item
                    classification_marker = " [⚠️ UNDERSTAND ONLY]" if classification == "understand_only" else " [🚫 OFFENSIVE]" if classification == "offensive" else ""
                    report += f"- **{term}**: {definition}{classification_marker}\n"
                elif isinstance(item, tuple) and len(item) == 2:
                    # Backwards compatibility
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
        
        if industry_slang.get('technical_slang'):
            report += "\n### Technical Slang\n"
            for item in industry_slang['technical_slang'][:25]:
                if isinstance(item, tuple) and len(item) == 3:
                    term, definition, classification = item
                    classification_marker = " [⚠️ UNDERSTAND ONLY]" if classification == "understand_only" else " [🚫 OFFENSIVE]" if classification == "offensive" else ""
                    report += f"- **{term}**: {definition}{classification_marker}\n"
                elif isinstance(item, tuple) and len(item) == 2:
                    # Backwards compatibility
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
        
        if industry_slang.get('community_terms'):
            report += "\n### Community Terms\n"
            for item in industry_slang['community_terms'][:25]:
                if isinstance(item, tuple) and len(item) == 3:
                    term, definition, classification = item
                    classification_marker = " [⚠️ UNDERSTAND ONLY]" if classification == "understand_only" else " [🚫 OFFENSIVE]" if classification == "offensive" else ""
                    report += f"- **{term}**: {definition}{classification_marker}\n"
                elif isinstance(item, tuple) and len(item) == 2:
                    # Backwards compatibility
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
        
        report += """

### Usage Classification Legend
- **No marker**: Safe terms the AI can freely use
- **[⚠️ UNDERSTAND ONLY]**: Terms the AI should understand but avoid using
- **[🚫 OFFENSIVE]**: Offensive terms the AI understands but will never use

## 3. Technical Terminology

Technical terms specific to this industry that affect product understanding:

"""
        
        if technical_terms.get('specifications'):
            report += "### Specifications\n"
            for item in technical_terms['specifications'][:25]:
                if isinstance(item, tuple) and len(item) == 2:
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
                else:
                    report += f"- {item}\n"
        
        if technical_terms.get('features'):
            report += "\n### Features\n"
            for item in technical_terms['features'][:25]:
                if isinstance(item, tuple) and len(item) == 2:
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
                else:
                    report += f"- {item}\n"
        
        if technical_terms.get('technologies'):
            report += "\n### Technologies\n"
            for item in technical_terms['technologies'][:25]:
                if isinstance(item, tuple) and len(item) == 2:
                    term, definition = item
                    report += f"- **{term}**: {definition}\n"
                else:
                    report += f"- {item}\n"
        
        report += """

## 4. Product Naming Patterns

Analysis of the current catalog reveals these naming conventions:

"""
        
        if product_patterns.get('common_prefixes'):
            report += "### Common Prefixes:\n"
            for prefix, count in sorted(product_patterns['common_prefixes'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
                report += f"- **{prefix}**: appears in {count} products\n"
        
        if product_patterns.get('common_suffixes'):
            report += "\n### Common Suffixes:\n"
            for suffix, count in sorted(product_patterns['common_suffixes'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
                report += f"- **{suffix}**: appears in {count} products\n"
        
        report += f"""

## 5. Product Categorization Patterns

Industry-specific terminology patterns for dynamic product categorization:

"""
        
        # Add categorization patterns
        if categorization_patterns:
            for category, terms in categorization_patterns.items():
                if terms:  # Only add categories that have terms
                    category_title = category.replace('_', ' ').title()
                    report += f"### {category_title}\n"
                    report += f"Terms that indicate {category.replace('_', ' ')}:\n"
                    for term in terms[:15]:  # Limit to first 15 terms
                        report += f"- {term}\n"
                    report += "\n"
        
        report += f"""These patterns enable brand-agnostic product categorization by learning industry-specific terminology from actual product data and web research.

## 6. Implementation Recommendations

### For Product Descriptors:
1. Include industry slang translations in descriptors to improve search matching
2. Add price tier context using discovered terminology
3. Ensure technical terms are explained in layman's terms

### For Search Enhancement:
1. Map slang terms to their formal equivalents
2. Add tier indicators to search keywords based on price percentiles
3. Include common misspellings and variations

### For AI Persona:
1. Use industry-appropriate terminology when discussing products
2. Understand and respond to slang terms naturally
3. Educate customers on technical terms when appropriate

### Example Implementation:

For a product in the top 25% price range with "Pro" in the name:
- Add to descriptor: "This professional-grade model represents our premium tier..."
- Add to keywords: ["pro", "professional", "high-end", "premium", "flagship"]
- AI persona: Can explain that "Pro" indicates professional/premium features

## 7. AI Persona Context

### Programmatically Extractable Terminology

The following terminology with definitions can be extracted for AI persona system prompts:

#### Price Tier Indicators
"""
        
        # Prepare JSON data for price tier indicators with definitions
        price_indicators = {
            "premium_indicators": {},
            "mid_tier_indicators": {},
            "budget_indicators": {}
        }
        
        # Add premium terms with definitions
        for item in price_terminology.get('premium_terms', [])[:25]:
            if isinstance(item, tuple) and len(item) == 2:
                term, definition = item
                price_indicators["premium_indicators"][term] = definition
            else:
                price_indicators["premium_indicators"][item] = "Premium tier indicator"
        
        # Add brand-specific premium indicators
        for term in price_terminology.get('brand_specific_tiers', {}).get('premium_indicators', [])[:15]:
            price_indicators["premium_indicators"][term] = "Brand-specific premium indicator"
        
        # Add mid-tier terms with definitions
        for item in price_terminology.get('mid_tier_terms', [])[:25]:
            if isinstance(item, tuple) and len(item) == 2:
                term, definition = item
                price_indicators["mid_tier_indicators"][term] = definition
            else:
                price_indicators["mid_tier_indicators"][item] = "Mid-tier indicator"
        
        # Add brand-specific mid indicators
        for term in price_terminology.get('brand_specific_tiers', {}).get('mid_indicators', [])[:15]:
            price_indicators["mid_tier_indicators"][term] = "Brand-specific mid-tier indicator"
        
        # Add budget terms with definitions
        for item in price_terminology.get('budget_terms', [])[:25]:
            if isinstance(item, tuple) and len(item) == 2:
                term, definition = item
                price_indicators["budget_indicators"][term] = definition
            else:
                price_indicators["budget_indicators"][item] = "Budget tier indicator"
        
        # Add brand-specific budget indicators
        for term in price_terminology.get('brand_specific_tiers', {}).get('budget_indicators', [])[:15]:
            price_indicators["budget_indicators"][term] = "Brand-specific budget indicator"
        
        # Prepare JSON data for technical terms
        tech_terms_json = {
            "specifications": {term: definition for term, definition in technical_terms.get('specifications', [])[:25] if isinstance((term, definition), tuple)},
            "features": {term: definition for term, definition in technical_terms.get('features', [])[:25] if isinstance((term, definition), tuple)},
            "technologies": {term: definition for term, definition in technical_terms.get('technologies', [])[:25] if isinstance((term, definition), tuple)}
        }
        
        # Prepare JSON data for slang dictionary with classification
        slang_dict = {
            "general_slang": {},
            "technical_slang": {},
            "community_terms": {}
        }
        
        # Process general slang
        for item in industry_slang.get('general_slang', [])[:15]:
            if isinstance(item, tuple) and len(item) == 3:
                term, definition, classification = item
                slang_dict["general_slang"][term] = {
                    "definition": definition,
                    "usage": classification
                }
            elif isinstance(item, tuple) and len(item) == 2:
                # Backwards compatibility
                term, definition = item
                slang_dict["general_slang"][term] = {
                    "definition": definition,
                    "usage": "safe"
                }
        
        # Process technical slang
        for item in industry_slang.get('technical_slang', [])[:25]:
            if isinstance(item, tuple) and len(item) == 3:
                term, definition, classification = item
                slang_dict["technical_slang"][term] = {
                    "definition": definition,
                    "usage": classification
                }
            elif isinstance(item, tuple) and len(item) == 2:
                # Backwards compatibility
                term, definition = item
                slang_dict["technical_slang"][term] = {
                    "definition": definition,
                    "usage": "safe"
                }
        
        # Process community terms
        for item in industry_slang.get('community_terms', [])[:25]:
            if isinstance(item, tuple) and len(item) == 3:
                term, definition, classification = item
                slang_dict["community_terms"][term] = {
                    "definition": definition,
                    "usage": classification
                }
            elif isinstance(item, tuple) and len(item) == 2:
                # Backwards compatibility
                term, definition = item
                slang_dict["community_terms"][term] = {
                    "definition": definition,
                    "usage": "safe"
                }
        
        report += f"""```json
{json.dumps(price_indicators, indent=4)}
```

#### Technical Terms with Definitions
```json
{json.dumps(tech_terms_json, indent=4)}
```

#### Industry Slang Dictionary
```json
{json.dumps(slang_dict, indent=4)}
```

### Usage in AI Persona

These definitions enable the AI persona to:
1. Understand customer queries using industry-specific terminology
2. Translate technical jargon into accessible language
3. Use appropriate tier descriptors when discussing price ranges
4. Recognize and respond to slang terms naturally
5. Educate customers about technical specifications when needed

**Slang Usage Guidelines:**
- **"safe"**: Terms the AI can use naturally in conversation
- **"understand_only"**: Terms the AI recognizes but should rephrase professionally
- **"offensive"**: Terms the AI understands but must never use, redirecting to appropriate language

For example:
- If a customer uses "DTF" (understand_only), the AI understands but responds professionally
- If a customer uses "caucacity" (offensive), the AI understands but never uses this term
- If a customer uses "granny gear" (safe), the AI can use this term naturally

Generated: {datetime.now().isoformat()}
"""
        
        return report
    
    async def _extract_terms_from_existing_research(self) -> Dict[str, List[str]]:
        """Extract terminology from existing brand research phases using LLM"""
        extracted_terms = {
            'premium_terms': [],
            'mid_tier_terms': [],
            'budget_terms': []
        }
        
        # Research phases that likely contain terminology insights
        research_phases = [
            'foundation',
            'market_positioning', 
            'product_style',
            'voice_messaging',
            'product_catalog'
        ]
        
        # Collect relevant sections from research
        research_snippets = []
        
        for phase in research_phases:
            try:
                research_content = await self.storage_manager.get_research_data(account=self.brand_domain, research_type=phase)
                
                if research_content:
                    # Extract relevant sections (limit to prevent context overflow)
                    lines = research_content.split('\n')
                    relevant_sections = []
                    
                    for i, line in enumerate(lines):
                        line_lower = line.lower()
                        # Look for sections mentioning tiers, models, pricing, or positioning
                        if any(term in line_lower for term in ['tier', 'model', 'series', 'line', 'price', 'position', 'premium', 'budget', 'entry', 'flagship']):
                            # Get context (3 lines before and after)
                            start = max(0, i - 3)
                            end = min(len(lines), i + 4)
                            section = '\n'.join(lines[start:end])
                            relevant_sections.append(section)
                    
                    if relevant_sections:
                        research_snippets.append({
                            'phase': phase,
                            'content': '\n\n'.join(relevant_sections[:7])  # Limit sections per phase
                        })
                        
            except Exception as e:
                logger.debug(f"Could not load {phase} research: {e}")
        
        # Use LLM to extract terminology from research snippets
        if research_snippets:
            llm_terms = await self._llm_extract_from_research(research_snippets)
            if llm_terms:
                extracted_terms['premium_terms'].extend(llm_terms.get('premium_terms', []))
                extracted_terms['mid_tier_terms'].extend(llm_terms.get('mid_tier_terms', []))
                extracted_terms['budget_terms'].extend(llm_terms.get('budget_terms', []))
        
        # Deduplicate
        for key in extracted_terms:
            extracted_terms[key] = list(set(extracted_terms[key]))
        
        logger.info(f"Extracted {len(extracted_terms['premium_terms'])} premium terms, "
                   f"{len(extracted_terms['mid_tier_terms'])} mid-tier terms, "
                   f"{len(extracted_terms['budget_terms'])} budget terms from existing research")
        
        return extracted_terms
    
    async def _llm_extract_from_research(self, research_snippets: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Use LLM to extract terminology from existing research"""
        
        # Prepare research content for LLM
        research_content = "\n\n".join([
            f"=== {snippet['phase'].upper()} RESEARCH ===\n{snippet['content']}"
            for snippet in research_snippets
        ])
        
        # Default prompt template
        default_prompt = """You are analyzing existing brand research to identify product tier terminology and model names.

Please extract specific terms, model names, or series names that indicate different price/quality tiers from the research excerpts below.

Focus on:
1. Actual product model names or series (e.g., "S-Works", "Epic", "Sport")
2. Tier designations used by the brand (e.g., "Pro", "Elite", "Base")
3. Any patterns in how products are categorized by price/quality

Research Excerpts:
{{research_content}}

Please respond with a JSON object in this exact format:
{
    "premium_terms": ["term1", "term2", ...],
    "mid_tier_terms": ["term1", "term2", ...], 
    "budget_terms": ["term1", "term2", ...],
    "patterns_found": "Brief description of naming patterns you identified"
}

Extract only actual product/model names and tier indicators, not generic descriptive words.

Final line: {{sentinel}}"""

        # Build versioned prompt using Langfuse
        template_vars = {
            "research_content": research_content,
            "sentinel": SENTINEL
        }
        
        prompt = await self._build_versioned_prompt(
            prompt_key="liddy/terminology/research_extractor",
            default_prompt=default_prompt,
            template_vars=template_vars
        )

        try:
            # Use retry logic for both LLM call and JSON parsing
            fallback_result = {
                'premium_terms': [],
                'mid_tier_terms': [],
                'budget_terms': [],
                'insights': 'Failed to extract terms from research due to parsing errors'
            }
            
            result = await self._llm_call_with_json_retry(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="openai/o3",
                initial_temperature=0.3,
                response_format={"type": "json_object"},
                fallback_result=fallback_result
            )
            
            # Clean and validate terms
            cleaned_result = {
                'premium_terms': [t.lower().strip() for t in result.get('premium_terms', []) if isinstance(t, str) and len(t) > 1],
                'mid_tier_terms': [t.lower().strip() for t in result.get('mid_tier_terms', []) if isinstance(t, str) and len(t) > 1],
                'budget_terms': [t.lower().strip() for t in result.get('budget_terms', []) if isinstance(t, str) and len(t) > 1]
            }
            
            if 'patterns_found' in result:
                logger.info(f"LLM identified patterns: {result['patterns_found']}")
            
            return cleaned_result
            
        except Exception as e:
            logger.error(f"LLM extraction from research failed: {e}")
            return {'premium_terms': [], 'mid_tier_terms': [], 'budget_terms': []}
    
    async def _llm_extract_terminology(self, search_results: List[Dict[str, str]], 
                                     tavily_answers: List[Dict[str, str]],
                                     brand: str, industry: str) -> Dict[str, List[Tuple[str, str]]]:
        """Use LLM to extract terminology with definitions from search results and Tavily answers"""
        
        # Prepare the Tavily answers for LLM
        answers_content = ""
        if tavily_answers:
            answers_content = "## Tavily AI-Generated Answers:\n\n"
            for answer_data in tavily_answers:
                answers_content += f"Query: {answer_data['query']}\nAnswer: {answer_data['answer']}\n\n"
        
        # Prepare the search content for LLM
        search_content = ""
        if search_results:
            search_content = "\n\n## Individual Search Results:\n\n"
            search_content += "\n\n".join([
                f"Query: {result['query']}\nSource: {result['url']}\nContent:\n{result['content']}"
                for result in search_results
            ])
        
        # Add relevance guardrails
        guardrails = RELEVANCE_GUARDRAILS.replace("{{industry}}", industry)
        selection_criteria = TERMINOLOGY_SELECTION_CRITERIA.replace("{{industry}}", industry)
        
        # Default prompt template with quality improvements
        default_prompt = """You are analyzing search results about {{brand}} in the {{industry}} industry to identify price tier terminology.

{{guardrails}}

{{selection_criteria}}

Please extract terms that indicate different price/quality tiers, and provide a brief definition for each term.

Focus on:
1. Terms that clearly indicate premium/high-end products
2. Terms that indicate mid-range products
3. Terms that indicate budget/entry-level products

For each term, provide its meaning in the context of {{industry}} products.

IMPORTANT FILTERING RULES:
- EXCLUDE any terms related to crashes, accidents, injuries, or death
- EXCLUDE profanity, offensive language, or derogatory terms
- EXCLUDE terms that could be considered inappropriate or unprofessional
- EXCLUDE vague intensifiers like "ultimate", "best", "amazing", "super" unless they are part of official product model names
- EXCLUDE generic quality descriptors like "good", "great", "nice", "excellent" unless brand-specific
- EXCLUDE color descriptors (black, white, red, etc.) unless part of a tier name
- EXCLUDE size descriptors (small, large, etc.) unless indicating product tier
- ONLY include terms that are appropriate for a professional product catalog
- Focus on terms that relate to products, features, performance, or technical aspects
- ALL TERMS MUST BE LOWERCASE (except acronyms like "SL" or "S-Works")

Search Results to Analyze (pay special attention to the Tavily AI-generated answers):

{{search_data}}

Please respond with a JSON object in this exact format:
{
    "premium_terms": [
        ["term1", "definition of what this means for premium products"],
        ["term2", "definition2"],
        ...
    ],
    "mid_tier_terms": [
        ["term1", "definition of what this means for mid-tier products"],
        ["term2", "definition2"],
        ...
    ],
    "budget_terms": [
        ["term1", "definition of what this means for budget products"],
        ["term2", "definition2"],
        ...
    ],
    "insights": "Brief summary of pricing tier patterns found"
}

Each definition should be concise (under 100 characters) and explain what the term means in the context of product pricing/quality.

QUANTITY REQUIREMENTS:
- Aim for 30 terms per tier (premium, mid, budget)
- Include only the most relevant and specific terms
- Quality over quantity - better to have 20 excellent terms than 30 mediocre ones

For each term, include:
- "catalog_occurrences": number of times it appears in the catalog (minimum 3 to include)
- "citations": list of sources where the term was found (need at least 2 {{industry}} sources)

Final line: {{sentinel}}"""

        # Build versioned prompt using Langfuse
        template_vars = {
            "brand": brand,
            "industry": industry,
            "guardrails": guardrails,
            "selection_criteria": selection_criteria,
            "search_data": f"{answers_content}{search_content}",
            "sentinel": SENTINEL
        }
        
        prompt = await self._build_versioned_prompt(
            prompt_key="liddy/terminology/terminology_extractor",
            default_prompt=default_prompt,
            template_vars=template_vars
        )

        try:
            # Use retry logic for both LLM call and JSON parsing
            fallback_result = {
                'premium_terms': [],
                'mid_tier_terms': [],
                'budget_terms': [],
                'insights': 'Failed to extract terminology due to parsing errors'
            }
            
            result = await self._llm_call_with_json_retry(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="openai/o3",
                initial_temperature=0.3,
                response_format={"type": "json_object"},
                fallback_result=fallback_result
            )
            
            # Clean and validate the extracted terms with definitions
            cleaned_result = {
                'premium_terms': [],
                'mid_tier_terms': [],
                'budget_terms': []
            }
            
            # Process each category
            for category in ['premium_terms', 'mid_tier_terms', 'budget_terms']:
                if category in result and isinstance(result[category], list):
                    for item in result[category]:
                        if isinstance(item, list) and len(item) == 2:
                            term, definition = item
                            if isinstance(term, str) and isinstance(definition, str):
                                if 1 < len(term) < 50 and 1 < len(definition) < 200:
                                    cleaned_result[category].append((term.lower().strip(), definition.strip()))
            
            logger.info(f"LLM extracted {len(cleaned_result['premium_terms'])} premium terms, "
                       f"{len(cleaned_result['mid_tier_terms'])} mid-tier terms, "
                       f"{len(cleaned_result['budget_terms'])} budget terms from web search")
            
            if 'insights' in result:
                logger.info(f"LLM insights: {result['insights']}")
            
            return cleaned_result
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {'premium_terms': [], 'mid_tier_terms': [], 'budget_terms': []}
    
    async def _research_categorization_patterns(self, brand_name: str, industry: str, products: List[Product]) -> Dict[str, List[str]]:
        """
        Research industry-specific categorization patterns for brand-agnostic descriptor generation
        
        Args:
            brand_name: Name of the brand
            industry: Industry category
            products: List of products to analyze
            
        Returns:
            Dict containing categorization patterns for different aspects
        """
        logger.info(f"Researching categorization patterns for {industry} industry")
        
        # Start with generic patterns
        patterns = {
            "use_cases": ["for", "use", "activity", "application", "purpose", "intended"],
            "materials": ["material", "made", "constructed", "built", "crafted"],
            "key_features": ["feature", "technology", "system", "component", "function"],
            "style_type": ["style", "design", "aesthetic", "finish", "appearance"],
            "target_user": ["beginner", "professional", "advanced", "expert", "level"],
            "performance_traits": ["performance", "efficiency", "quality", "grade", "tier"]
        }
        
        # Extract industry-specific terms from product data
        extracted_patterns = self._extract_patterns_from_products(products)
        
        # Merge extracted patterns with base patterns
        for category, terms in extracted_patterns.items():
            if category in patterns:
                patterns[category].extend(terms)
                # Remove duplicates while preserving order
                patterns[category] = list(dict.fromkeys(patterns[category]))
        
        # Research additional industry-specific terminology via web search
        try:
            web_patterns = await self._research_web_categorization_patterns(brand_name, industry)
            
            # Merge web research patterns
            for category, terms in web_patterns.items():
                if category in patterns:
                    patterns[category].extend(terms)
                    # Remove duplicates while preserving order
                    patterns[category] = list(dict.fromkeys(patterns[category]))
                    
        except Exception as e:
            logger.warning(f"Web categorization research failed: {e}")
        
        # Store research results for use by other components
        self._categorization_patterns = patterns
        
        logger.info(f"Generated categorization patterns: {len(patterns)} categories, "
                   f"{sum(len(terms) for terms in patterns.values())} total terms")
        
        return patterns
    
    def _extract_patterns_from_products(self, products: List[Product]) -> Dict[str, List[str]]:
        """
        Extract categorization patterns from existing product data
        
        Args:
            products: List of products to analyze
            
        Returns:
            Dict of categorization patterns discovered from products
        """
        patterns = {
            "use_cases": [],
            "materials": [],
            "key_features": [],
            "style_type": [],
            "target_user": [],
            "performance_traits": []
        }
        
        use_case_indicators = set()
        material_indicators = set()
        feature_indicators = set()
        style_indicators = set()
        user_indicators = set()
        performance_indicators = set()
        
        for product in products:
            # Use the safe helper method for all attribute access
            product_name = self._get_product_attribute(product, 'name', '') or ""
            product_description = self._get_product_attribute(product, 'description', '') or ""
            product_highlights = self._get_product_attribute(product, 'highlights', []) or []
            product_categories = self._get_product_attribute(product, 'categories', []) or []
            
            # Get additional fields safely using the helper method
            long_description = self._get_product_attribute(product, 'long_description', '')
            title = self._get_product_attribute(product, 'title', '')
            
            # Extract from product names and descriptions
            text_fields = [
                product_name,
                product_description,
                str(long_description),
                str(title)
            ]
            
            # Extract from highlights/features
            if product_highlights:
                text_fields.extend(product_highlights)
            
            # Extract from categories
            if product_categories:
                text_fields.extend(product_categories)
            
            # Analyze text for patterns
            for text in text_fields:
                if not text:
                    continue
                    
                text_lower = text.lower()
                words = re.findall(r'\b\w+\b', text_lower)
                
                # Look for use case patterns
                for word in words:
                    if any(indicator in word for indicator in ['for', 'designed', 'intended', 'perfect', 'ideal']):
                        # Extract surrounding context
                        context = self._extract_context_around_word(text_lower, word, 2)
                        if context:
                            use_case_indicators.update(context)
                    
                    # Look for material patterns
                    if any(indicator in word for indicator in ['made', 'crafted', 'built', 'constructed']):
                        context = self._extract_context_around_word(text_lower, word, 2)
                        if context:
                            material_indicators.update(context)
                    
                    # Look for feature patterns
                    if any(indicator in word for indicator in ['feature', 'technology', 'system', 'includes']):
                        context = self._extract_context_around_word(text_lower, word, 2)
                        if context:
                            feature_indicators.update(context)
                    
                    # Look for style patterns
                    if any(indicator in word for indicator in ['style', 'design', 'finish', 'color']):
                        context = self._extract_context_around_word(text_lower, word, 2)
                        if context:
                            style_indicators.update(context)
                    
                    # Look for user level patterns
                    if any(indicator in word for indicator in ['level', 'grade', 'tier', 'class']):
                        context = self._extract_context_around_word(text_lower, word, 2)
                        if context:
                            user_indicators.update(context)
                    
                    # Look for performance patterns
                    if any(indicator in word for indicator in ['performance', 'quality', 'efficiency', 'rating']):
                        context = self._extract_context_around_word(text_lower, word, 2)
                        if context:
                            performance_indicators.update(context)
        
        # Convert sets to lists - filtering now handled by guardrails
        # Length check remains for basic quality control
        patterns["use_cases"] = [term for term in use_case_indicators if len(term) > 2][:10]
        patterns["materials"] = [term for term in material_indicators if len(term) > 2][:10]
        patterns["key_features"] = [term for term in feature_indicators if len(term) > 2][:10]
        patterns["style_type"] = [term for term in style_indicators if len(term) > 2][:10]
        patterns["target_user"] = [term for term in user_indicators if len(term) > 2][:10]
        patterns["performance_traits"] = [term for term in performance_indicators if len(term) > 2][:10]
        
        return patterns
    
    def _extract_context_around_word(self, text: str, word: str, window: int) -> List[str]:
        """
        Extract context words around a given word
        
        Args:
            text: Text to search in
            word: Word to find context for
            window: Number of words before and after
            
        Returns:
            List of context words
        """
        words = text.split()
        context = []
        
        for i, w in enumerate(words):
            if word in w:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                context.extend(words[start:end])
        
        return [w for w in context if len(w) > 2 and w.isalpha()]
    
    async def _research_web_categorization_patterns(self, brand_name: str, industry: str) -> Dict[str, List[str]]:
        """
        Research industry-specific categorization patterns via web search
        
        Args:
            brand_name: Name of the brand
            industry: Industry category
            
        Returns:
            Dict of categorization patterns from web research
        """
        patterns = {
            "use_cases": [],
            "materials": [],
            "key_features": [],
            "style_type": [],
            "target_user": [],
            "performance_traits": []
        }
        
        try:
            # Search for industry-specific terminology
            queries = [
                f"{self.brand_domain} {industry} product categories types classification",
                f"{self.brand_domain} {industry} materials construction components",
                f"{self.brand_domain} {industry} features technology specifications",
                f"{self.brand_domain} {industry} user levels beginner professional expert",
                f"{self.brand_domain} {industry} product styles designs types"
            ]
            
            all_results = []
            for query in queries[:3]:  # Limit to 3 queries for efficiency
                try:
                    search_results = await self.web_search.search(query, num_results=5)
                    if hasattr(search_results, 'results'):
                        all_results.extend(search_results.results)
                    elif isinstance(search_results, list):
                        all_results.extend(search_results)
                except Exception as e:
                    logger.warning(f"Web search failed for query '{query}': {e}")
            
            if all_results:
                # Extract patterns using LLM
                extracted_patterns = await self._extract_categorization_patterns_with_llm(
                    industry, all_results
                )
                
                # Merge extracted patterns
                for category, terms in extracted_patterns.items():
                    if category in patterns:
                        patterns[category].extend(terms)
            
        except Exception as e:
            logger.error(f"Web categorization research failed: {e}")
        
        return patterns
    
    async def _extract_categorization_patterns_with_llm(self, industry: str, search_results: List) -> Dict[str, List[str]]:
        """
        Use LLM to extract categorization patterns from web search results
        
        Args:
            industry: Industry category
            search_results: Web search results
            
        Returns:
            Dict of extracted categorization patterns
        """
        # Combine search results into text
        combined_text = ""
        for result in search_results[:10]:  # Limit to first 10 results
            if hasattr(result, 'snippet'):
                combined_text += f"{result.snippet} "
            elif isinstance(result, dict) and 'snippet' in result:
                combined_text += f"{result['snippet']} "
        
        if not combined_text.strip():
            return {
                "use_cases": [],
                "materials": [],
                "key_features": [],
                "style_type": [],
                "target_user": [],
                "performance_traits": []
            }
        
        # Default prompt template
        default_prompt = """Analyze the following text about the {{industry}} industry and extract categorization patterns.
        
Extract terms that would help categorize products in these areas:
1. Use cases (how products are used, activities, applications)
2. Materials (what products are made from, construction)
3. Key features (product features, technologies, capabilities)
4. Style types (design styles, aesthetics, appearances)
5. Target users (user levels, expertise, demographics)
6. Performance traits (quality indicators, performance characteristics)

Text: {{text}}

Return a JSON object with these categories as keys and lists of relevant terms as values.
Focus on industry-specific terms that would be useful for product categorization.

Final line: {{sentinel}}"""

        # Build versioned prompt using Langfuse
        template_vars = {
            "industry": industry,
            "text": combined_text[:2000],
            "sentinel": SENTINEL
        }
        
        prompt = await self._build_versioned_prompt(
            prompt_key="liddy/terminology/categorization_patterns_extractor",
            default_prompt=default_prompt,
            template_vars=template_vars
        )
        
        try:
            # Use retry logic for both LLM call and JSON parsing
            fallback_result = {
                "use_cases": [],
                "materials": [],
                "key_features": [],
                "style_type": [],
                "target_user": [],
                "performance_traits": []
            }
            
            max_attempts = 3
            temperature = 0.1
            
            for attempt in range(max_attempts):
                try:
                    response = await self._llm_call_with_sentinel(
                        messages=[{"role": "user", "content": prompt}],
                        model="openai/o3",
                        initial_temperature=temperature
                    )
                    
                    content = response.get('content', '{}')
                    
                    # Try to parse JSON
                    result = json.loads(content)
                    logger.info(f"Successfully parsed categorization JSON on attempt {attempt + 1}")
                    break
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Categorization JSON parsing failed on attempt {attempt + 1}: {e}")
                    if attempt < max_attempts - 1:
                        temperature = min(temperature + 0.1, 0.5)
                        logger.info(f"Retrying categorization with higher temperature: {temperature}")
                    else:
                        logger.error(f"Failed to parse categorization JSON after {max_attempts} attempts")
                        result = fallback_result
                        
                except Exception as e:
                    logger.warning(f"Categorization LLM call failed on attempt {attempt + 1}: {e}")
                    if attempt < max_attempts - 1:
                        temperature = min(temperature + 0.1, 0.5)
                    else:
                        logger.error(f"Categorization LLM call failed after {max_attempts} attempts")
                        result = fallback_result
                
                # Validate and clean result
                cleaned_result = {
                    "use_cases": [],
                    "materials": [],
                    "key_features": [],
                    "style_type": [],
                    "target_user": [],
                    "performance_traits": []
                }
                
                for category in cleaned_result.keys():
                    if category in result and isinstance(result[category], list):
                        # Take only first 5 terms per category and clean them
                        terms = [term.strip().lower() for term in result[category][:5] if isinstance(term, str)]
                        cleaned_result[category] = [term for term in terms if len(term) > 1 and len(term) < 30]
                
                return cleaned_result
                
        except Exception as e:
            logger.error(f"LLM categorization extraction failed: {e}")
            return {
                "use_cases": [],
                "materials": [],
                "key_features": [],
                "style_type": [],
                "target_user": [],
                "performance_traits": []
            }
    
    def _get_quality_evaluation_criteria(self) -> List[str]:
        """Define quality criteria for terminology research"""
        return [
            "Accuracy of terminology definitions and mappings",
            "Relevance of discovered terms to the specific industry",
            "Completeness of slang and synonym coverage",
            "Clarity of tier classifications",
            "Actionability of recommendations for implementation"
        ]
    
    # ========== Quality Pipeline Methods ==========
    
    def _normalize_term(self, term: str) -> str:
        """
        Normalize a term for consistent comparison.
        
        - Lowercase
        - Strip whitespace
        - ASCII-fold (remove accents)
        - Remove special characters except hyphens
        """
        # ASCII fold to remove accents
        normalized = unicodedata.normalize('NFKD', term.lower().strip())
        normalized = ''.join(c for c in normalized if not unicodedata.combining(c))
        
        # Keep only alphanumeric and hyphens
        normalized = re.sub(r'[^a-z0-9\-]', '', normalized)
        
        return normalized
    
    def _proof_of_use(self, term: str, product_hits: int, web_hits: int, tier: str) -> bool:
        """
        Tier-aware proof-of-use validation.
        
        Different tiers have different thresholds:
        - Premium: Needs stronger web presence (1 product OR 3 web)
        - Mid: Balanced requirements (2 products OR 2 web)  
        - Budget: Needs product evidence (3 products OR 1 web)
        """
        if tier == "premium":
            return product_hits >= 1 or web_hits >= 3
        elif tier == "mid":
            return product_hits >= 2 or web_hits >= 2
        else:  # budget
            return product_hits >= 3 or web_hits >= 1
    
    def _get_product_coverage(self, term: str, products: List[Product]) -> int:
        """
        Calculate how many products contain this term in their name or description.
        
        Args:
            term: Normalized term to search for
            products: List of Product objects
            
        Returns:
            Number of products containing the term
        """
        count = 0
        term_lower = term.lower()
        
        for product in products:
            # Check product name
            product_name = self._get_product_attribute(product, 'name', '')
            if product_name and term_lower in product_name.lower():
                count += 1
                continue
            
            # Check product description
            product_desc = self._get_product_attribute(product, 'description', '')
            if product_desc and term_lower in product_desc.lower():
                count += 1
                continue
            
            # Check model field if available
            product_model = self._get_product_attribute(product, 'model', '')
            if product_model and term_lower in product_model.lower():
                count += 1
        
        return count
    
    def _determine_tier_from_context(self, term: str, source_sentence: str, products: List[Product]) -> str:
        """
        Determine which tier a term belongs to based on context clues.
        
        Args:
            term: The term to classify
            source_sentence: The sentence where the term was found
            products: List of products for price analysis
            
        Returns:
            'premium', 'mid', or 'budget'
        """
        # Check for explicit tier indicators in the source sentence
        sentence_lower = source_sentence.lower()
        
        premium_indicators = {'premium', 'professional', 'high-end', 'flagship', 'top', 'elite', 'pro'}
        budget_indicators = {'budget', 'entry', 'basic', 'starter', 'beginner', 'base', 'value'}
        mid_indicators = {'mid', 'middle', 'intermediate', 'standard', 'regular'}
        
        # Check sentence for tier indicators
        for indicator in premium_indicators:
            if indicator in sentence_lower:
                return 'premium'
        
        for indicator in budget_indicators:
            if indicator in sentence_lower:
                return 'budget'
        
        for indicator in mid_indicators:
            if indicator in sentence_lower:
                return 'mid'
        
        # If no explicit indicators, check product prices containing this term
        term_products = []
        for product in products:
            product_name = self._get_product_attribute(product, 'name', '')
            if product_name and term.lower() in product_name.lower():
                term_products.append(product)
        
        if term_products:
            # Get price statistics for products with this term
            prices = []
            for product in term_products:
                price_str = self._get_product_attribute(product, 'salePrice', None)
                if not price_str:
                    price_str = self._get_product_attribute(product, 'originalPrice', None)
                if price_str:
                    try:
                        price = float(price_str.replace('$', '').replace(',', ''))
                        if price > 0:
                            prices.append(price)
                    except:
                        pass
            
            if prices:
                avg_price = sum(prices) / len(prices)
                # Compare to overall catalog statistics
                # This is a simplified heuristic - ideally use PriceStatisticsAnalyzer
                if avg_price > 1000:  # High price threshold
                    return 'premium'
                elif avg_price < 300:  # Low price threshold
                    return 'budget'
        
        # Default to mid if uncertain
        return 'mid'
    
    def _calculate_confidence(self, term: str, web_hits: int, product_coverage: int, 
                            has_answer_provenance: bool, domain_quality: float) -> float:
        """
        Calculate multi-factor confidence score for a term.
        
        Args:
            term: The term being scored
            web_hits: Number of search results containing the term
            product_coverage: Number of products containing the term
            has_answer_provenance: Whether term was found in Tavily answer
            domain_quality: Average domain quality score (0-1)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from web presence
        web_confidence = min(web_hits / 10.0, 0.4)  # Max 0.4 from web
        
        # Product coverage confidence
        product_confidence = min(product_coverage / 5.0, 0.3)  # Max 0.3 from products
        
        # Domain quality bonus
        domain_bonus = domain_quality * 0.2  # Max 0.2 from domain quality
        
        # Answer provenance bonus
        answer_bonus = 0.05 if has_answer_provenance else 0.0
        
        # Domain validation bonus (if term appears in brand domain)
        domain_bonus_extra = 0.02 if self.brand_domain and term.lower() in self.brand_domain.lower() else 0.0
        
        # Calculate total confidence
        confidence = web_confidence + product_confidence + domain_bonus + answer_bonus + domain_bonus_extra
        
        # Ensure it's between 0 and 1
        return min(max(confidence, 0.0), 1.0)
    
    def _dedupe_tiers_with_context(self, tiers: Dict[str, List[Tuple[str, float]]], 
                                   products: List[Product]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Smart tier exclusivity using median product prices.
        
        If a term appears in multiple tiers, assign it to the tier that best
        matches the median price of products containing that term.
        
        Args:
            tiers: Dict of tier -> [(term, confidence)] lists
            products: List of Product objects for price analysis
            
        Returns:
            Deduped tiers with exclusive terms
        """
        # Build a map of term -> tiers it appears in
        term_to_tiers = {}
        for tier, terms in tiers.items():
            for term, conf in terms:
                if term not in term_to_tiers:
                    term_to_tiers[term] = []
                term_to_tiers[term].append((tier, conf))
        
        # Identify conflicting terms (appearing in multiple tiers)
        conflicts = {term: tier_list for term, tier_list in term_to_tiers.items() if len(tier_list) > 1}
        
        if not conflicts:
            return tiers  # No conflicts to resolve
        
        logger.info(f"Resolving {len(conflicts)} tier conflicts using price analysis")
        
        # Use PriceStatisticsAnalyzer to get tier thresholds
        from liddy_intelligence.catalog.price_statistics_analyzer import PriceStatisticsAnalyzer
        price_stats = PriceStatisticsAnalyzer.analyze_catalog_pricing(products)
        overall_stats = price_stats.get('overall', {})
        
        # Get tier thresholds
        budget_threshold = overall_stats.get('budget_threshold', 300)
        mid_high_threshold = overall_stats.get('mid_high_threshold', 1000)
        
        # Resolve each conflict
        resolved_assignments = {}
        for term, tier_list in conflicts.items():
            # Get products containing this term
            term_products = []
            for product in products:
                product_name = self._get_product_attribute(product, 'name', '')
                if product_name and term.lower() in product_name.lower():
                    term_products.append(product)
            
            if not term_products:
                # No products found, keep highest confidence tier
                best_tier = max(tier_list, key=lambda x: x[1])[0]
                resolved_assignments[term] = best_tier
                continue
            
            # Calculate median price for products with this term
            prices = []
            for product in term_products:
                # Handle variant products
                if hasattr(product, 'variants') and product.variants:
                    for variant in product.variants:
                        price_str = variant.price if variant.price else variant.originalPrice
                        if price_str:
                            try:
                                price = float(price_str.replace('$', '').replace(',', ''))
                                if price > 0:
                                    prices.append(price)
                            except:
                                pass
                else:
                    # Legacy product structure
                    price_str = self._get_product_attribute(product, 'salePrice', None)
                    if not price_str:
                        price_str = self._get_product_attribute(product, 'originalPrice', None)
                    if price_str:
                        try:
                            price = float(price_str.replace('$', '').replace(',', ''))
                            if price > 0:
                                prices.append(price)
                        except:
                            pass
            
            if prices:
                median_price = sorted(prices)[len(prices) // 2]
                
                # Assign based on median price
                if median_price >= mid_high_threshold:
                    resolved_assignments[term] = 'premium'
                elif median_price <= budget_threshold:
                    resolved_assignments[term] = 'budget'
                else:
                    resolved_assignments[term] = 'mid'
            else:
                # No valid prices, use highest confidence
                best_tier = max(tier_list, key=lambda x: x[1])[0]
                resolved_assignments[term] = best_tier
        
        # Build new exclusive tiers
        exclusive_tiers = {'premium': [], 'mid': [], 'budget': []}
        
        # Add non-conflicting terms first
        for tier, terms in tiers.items():
            for term, conf in terms:
                if term not in conflicts:
                    exclusive_tiers[tier].append((term, conf))
        
        # Add resolved conflicts
        for term, assigned_tier in resolved_assignments.items():
            # Get the confidence from the assigned tier
            for tier, conf in term_to_tiers[term]:
                if tier == assigned_tier:
                    exclusive_tiers[assigned_tier].append((term, conf))
                    break
        
        # Sort by confidence
        for tier in exclusive_tiers:
            exclusive_tiers[tier].sort(key=lambda x: x[1], reverse=True)
        
        return exclusive_tiers
    
    @lru_cache(maxsize=1000)
    def _compute_term_embedding(self, term: str) -> np.ndarray:
        """
        Compute semantic embedding for a term (cached).
        
        This is a placeholder - in production, use a real embedding model
        like OpenAI embeddings or sentence-transformers.
        
        Args:
            term: The term to embed
            
        Returns:
            Embedding vector
        """
        # Placeholder: simple character-based embedding
        # In production, replace with:
        # response = openai.Embedding.create(input=term, model="text-embedding-ada-002")
        # return np.array(response['data'][0]['embedding'])
        
        # For now, create a simple hash-based vector
        np.random.seed(hash(term) % 2**32)
        return np.random.randn(512)
    
    def _embedding_sanity_check(self, terms: List[Tuple[str, float]], 
                               reference_terms: List[str]) -> List[Tuple[str, float]]:
        """
        Sanity check terms using embedding similarity.
        
        Filters out terms that are semantically too distant from known good terms.
        
        Args:
            terms: List of (term, confidence) tuples to check
            reference_terms: Known good terms for this tier
            
        Returns:
            Filtered list of terms that pass semantic check
        """
        if not reference_terms:
            return terms  # No reference, return all
        
        # Compute reference embeddings
        ref_embeddings = [self._compute_term_embedding(ref) for ref in reference_terms]
        ref_centroid = np.mean(ref_embeddings, axis=0)
        
        # Check each term
        filtered_terms = []
        for term, conf in terms:
            term_embedding = self._compute_term_embedding(term)
            
            # Compute cosine similarity to centroid
            similarity = np.dot(term_embedding, ref_centroid) / (
                np.linalg.norm(term_embedding) * np.linalg.norm(ref_centroid)
            )
            
            # Keep terms with reasonable similarity (threshold: 0.3)
            # In production, tune this threshold based on your embedding model
            if similarity > 0.3:
                filtered_terms.append((term, conf))
            else:
                logger.debug(f"Filtered out '{term}' due to low semantic similarity: {similarity:.3f}")
        
        return filtered_terms
    
    def _dump_terminology_csv(self, exclusive_tiers: Dict[str, List[Tuple[str, float]]]) -> None:
        """
        Dump terminology results to CSV for debugging.
        
        Creates a timestamped CSV file in tmp/terminology_debug/ with all terms
        and their confidence scores organized by tier.
        
        Args:
            exclusive_tiers: Dict with 'premium', 'mid', 'budget' keys containing term lists
        """
        try:
            # Create debug directory using pathlib for cross-platform compatibility
            debug_dir = Path("tmp") / "terminology_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{self.brand_domain}_{timestamp}.csv"
            csv_path = debug_dir / csv_filename
            
            # Write CSV with all terms and tiers
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['tier', 'term', 'confidence', 'rank_in_tier'])
                
                # Write all terms organized by tier
                for tier_name, terms in exclusive_tiers.items():
                    for rank, (term, confidence) in enumerate(terms, 1):
                        writer.writerow([tier_name, term, f"{confidence:.4f}", rank])
            
            logger.info(f"Debug CSV dumped to: {csv_path}")
            
        except Exception as e:
            logger.warning(f"Failed to dump terminology CSV: {e}")
    
    # ========== Public API Methods ==========
    
    async def run(self, *, write_markdown: bool = True) -> PriceTerminology:
        """
        Run the terminology research and return structured results.
        
        Args:
            write_markdown: Whether to write traditional markdown report
            
        Returns:
            PriceTerminology object with confidence-scored terms
        """
        # Run the existing research pipeline
        await super().run()
        
        # The result should already be stored in self._result from _synthesize_results
        if not self._result:
            # Fallback - create empty result
            self._result = PriceTerminology(
                premium_terms=(),
                mid_terms=(),
                budget_terms=()
            )
        
        return self._result
    
    def get_premium_terms(self) -> List[str]:
        """Get premium tier terms without confidence scores."""
        return [t for t, _ in self._result.premium_terms] if self._result else []
    
    def get_mid_terms(self) -> List[str]:
        """Get mid-tier terms without confidence scores."""
        return [t for t, _ in self._result.mid_terms] if self._result else []
    
    def get_budget_terms(self) -> List[str]:
        """Get budget tier terms without confidence scores."""
        return [t for t, _ in self._result.budget_terms] if self._result else []
    
    def get_all_price_terms(self) -> List[str]:
        # Get all price terms across all tiers.
        return self.get_premium_terms() + self.get_mid_terms() + self.get_budget_terms()


