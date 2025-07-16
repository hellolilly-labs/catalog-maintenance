"""
Search helper utilities for terminology research

Provides optimized search functions with Tavily answer extraction,
diversity filtering, and depth-2 follow-up queries.
"""

import asyncio
import logging
import random
import re
from typing import Dict, List, Set, Optional
from urllib.parse import urlparse
from functools import lru_cache

logger = logging.getLogger(__name__)


async def llm_call_with_retry(llm_function, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """
    Execute an LLM function call with exponential backoff retry logic.
    
    Args:
        llm_function: Async function that makes the LLM call
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        
    Returns:
        Result from the LLM function call
        
    Raises:
        Exception: Re-raises the last exception if all retries fail
    """
    from liddy.llm.errors import LLMError
    
    last_exception = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            result = await llm_function()
            return result
            
        except LLMError as e:
            last_exception = e
            
            # Check if this is a retryable error
            error_message = str(e).lower()
            non_retryable_errors = [
                'invalid_api_key',
                'authentication_failed', 
                'model not supported',
                'invalid_request_error'
            ]
            
            if any(err in error_message for err in non_retryable_errors):
                logger.error(f"Non-retryable LLM error on attempt {attempt + 1}: {e}")
                raise e
            
            if attempt == max_retries:
                logger.error(f"LLM call failed after {max_retries + 1} attempts: {e}")
                raise e
                
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * 0.1 * random.random()  # Add up to 10% jitter
            total_delay = delay + jitter
            
            logger.warning(f"LLM call failed on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {total_delay:.2f}s...")
            await asyncio.sleep(total_delay)
            
        except Exception as e:
            last_exception = e
            
            # For unexpected errors, still retry but log as unexpected
            if attempt == max_retries:
                logger.error(f"Unexpected error in LLM call after {max_retries + 1} attempts: {e}")
                raise e
                
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * 0.1 * random.random()
            total_delay = delay + jitter
            
            logger.warning(f"Unexpected error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {total_delay:.2f}s...")
            await asyncio.sleep(total_delay)
    
    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    else:
        raise Exception("Unknown error in LLM retry logic")


# Search query templates for terminology discovery
# Uses full domain (e.g. "specialized.com") for better disambiguation
SEARCH_TEMPLATES = [
    "{domain} {industry} premium vs budget terminology model names",
    "{domain} bicycle model hierarchy explained {industry}", 
    "what does {term} mean in {domain} {industry} products",
    "{domain} {industry} price tier names list product lineup",
    "{domain} {industry} product naming conventions",
    "{industry} professional vs entry level terms {domain} products"
]

# Common words to exclude from term extraction
_COMMON_WORDS = {
    'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by',
    'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'that', 'this', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
    'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'should', 'could', 'would', 'may', 'might', 'must',
    'shall', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'as'
}

# Context-aware stop words for filtering vague intensifiers
CONTEXT_STOPWORDS = {
    "generic_quality": {
        "advanced", "premium", "professional", "ultimate", "best", "superior",
        "excellent", "outstanding", "exceptional", "remarkable", "extraordinary",
        "amazing", "fantastic", "great", "good", "better", "finest", "top"
    },
    "size_descriptors": {
        "large", "small", "medium", "big", "mini", "huge", "tiny", "massive",
        "compact", "full-size", "oversized", "undersized"
    },
    "color_descriptors": {
        "black", "white", "red", "blue", "silver", "gold", "gray", "grey",
        "green", "yellow", "orange", "purple", "pink", "brown", "beige"
    },
    "temporal": {
        "new", "latest", "current", "modern", "classic", "vintage", "retro",
        "old", "ancient", "contemporary", "future", "next-gen", "traditional"
    },
    "marketing": {
        "exclusive", "limited", "special", "signature", "unique", "custom",
        "personalized", "bespoke", "luxury", "deluxe", "enhanced", "improved"
    }
}


async def run_search(query: str, web_search) -> 'SearchResponse':
    """
    Run search with depth-2 follow-up if answer is empty (Adjustment A1).
    
    If the initial search returns no answer and fewer than 5 results,
    automatically fires a second query with additional context terms.
    
    Args:
        query: The search query
        web_search: The web search client (Tavily)
        
    Returns:
        SearchResponse with answer and results
    """
    try:
        # First attempt
        response = await web_search.search(query=query)
        
        # Check if we need depth-2 follow-up
        needs_followup = (
            (not response.answer or response.answer.strip() == "") and 
            len(response.results) < 5
        )
        
        if needs_followup:
            # Append context to force better results
            enhanced_query = f'{query} ("explained" OR "model hierarchy")'
            logger.info(f"Depth-2 query triggered for: {query}")
            
            try:
                response = await web_search.search(query=enhanced_query)
                logger.info(f"Depth-2 query improved results: answer={bool(response.answer)}, results={len(response.results)}")
            except Exception as e:
                logger.debug(f"Depth-2 query failed, using original: {e}")
                # Keep original response
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        raise


async def _extract_terms_from_tavily_answers_batch(answers: List[str], brand_domain: str = "", industry: str = "", search_intents: List[str] = None) -> List[Dict[str, str]]:
    """
    Extract product terminology from multiple Tavily answers in a single LLM call.
    
    More efficient than individual calls while preserving per-answer attribution.
    The LLM can see all answers together for better cross-reference and prioritization.
    
    Args:
        answers: List of Tavily synthesized answer texts
        brand_domain: Brand domain for context (e.g., "gucci.com", "specialized.com")
        industry: Industry context (e.g., "luxury fashion", "cycling")
        search_intents: Optional list of search queries that generated each answer for context
        
    Returns:
        List of dicts (one per answer) mapping term -> source_sentence for proper attribution
    """
    if not answers or all(not answer.strip() for answer in answers):
        return {}
    
    # Import here to avoid circular dependencies
    from liddy.llm.simple_factory import LLMFactory
    from liddy.llm.errors import LLMError
    
    try:
        # Prepare the batch of answers for the prompt
        formatted_answers = []
        for i, answer in enumerate(answers, 1):
            if answer and answer.strip():
                # Include search intent if available
                intent_context = ""
                if search_intents and i-1 < len(search_intents) and search_intents[i-1]:
                    intent_context = f"SEARCH QUERY: {search_intents[i-1]}\n"
                
                formatted_answers.append(f"SEARCH RESULT {i}:\n{intent_context}ANSWER: {answer.strip()}\n")
        
        if not formatted_answers:
            return []
        
        answers_text = "\n".join(formatted_answers)
        
        prompt = f"""Extract product terminology from the following search results about {brand_domain} in the {industry} industry.

🚨 CRITICAL: Follow this EXACT 3-STEP PROCESS for each search result:

STEP 1: Identify the search intent (what tier is being searched for)
STEP 2: Find and EXCLUDE contrast phrases that mention OTHER tiers
STEP 3: Extract only terms that belong to the TARGET tier

BRAND CONTEXT: {brand_domain}
INDUSTRY: {industry}

SEARCH RESULTS TO ANALYZE:
{answers_text}

🔍 3-STEP EXTRACTION PROCESS:

STEP 1: READ THE SEARCH QUERY
- Identify if searching for: premium, mid-tier, budget, or specific features
- Remember: ONLY extract terms for the tier being searched for

STEP 2: IDENTIFY AND SKIP CONTRAST PHRASES
Look for these patterns and IGNORE all terms mentioned in them:
- "while [other-tier] models include [term1], [term2]..."
- "unlike [other-tier] [term1], [target-tier] includes..."
- "compared to [other-tier] [term1], [target-tier] features..."
- "[target-tier] differs from [other-tier] [term1] by..."
- "in contrast to [other-tier] [term1], [target-tier]..."

STEP 3: EXTRACT TARGET-TIER TERMS ONLY
- Extract terms that are directly described as belonging to the target tier
- Include model names, series names, technical features
- Exclude generic words and terms from other tiers

⚠️ CONTRAST PHRASE EXAMPLES TO AVOID:
❌ "Premium model is X, while budget models include Y, Z" → If searching premium: extract X only, IGNORE Y, Z
❌ "Unlike premium A, budget options include B, C" → If searching budget: extract B, C only, IGNORE A  
❌ "Compared to entry-level D, mid-tier features E, F" → If searching mid-tier: extract E, F only, IGNORE D

FORMAT YOUR RESPONSE AS JSON ORGANIZED BY SEARCH RESULT:
{{
    "search_result_1": {{
        "search_intent": "premium/mid-tier/budget",
        "contrast_phrases_found": ["phrase that mentions other tiers"],
        "extracted_terms": {{
            "term1": "source sentence containing term1",
            "term2": "source sentence containing term2"
        }}
    }},
    "search_result_2": {{
        "search_intent": "premium/mid-tier/budget", 
        "contrast_phrases_found": ["phrase that mentions other tiers"],
        "extracted_terms": {{
            "term3": "source sentence containing term3"
        }}
    }}
}}

DETAILED EXAMPLE - FOLLOWING THE 3-STEP PROCESS:

Search Query: "{brand_domain} {industry} premium terminology"
Answer: "Brand's premium PlatinumSeries features advanced-tech, while budget models include BasicLine and ValueModel."

STEP 1: Search intent = "premium"
STEP 2: Contrast phrase found = "while budget models include BasicLine and ValueModel"
STEP 3: Extract premium terms only = PlatinumSeries, advanced-tech

CORRECT JSON:
{{
    "search_result_1": {{
        "search_intent": "premium",
        "contrast_phrases_found": ["while budget models include BasicLine and ValueModel"],
        "extracted_terms": {{
            "platinumseries": "Brand's premium PlatinumSeries features advanced-tech",
            "advanced-tech": "PlatinumSeries features advanced-tech"
        }}
    }}
}}

WRONG - DO NOT DO THIS:
{{
    "search_result_1": {{
        "extracted_terms": {{
            "platinumseries": "...",
            "advanced-tech": "...",
            "basicline": "...",  ← WRONG! This is from contrast phrase
            "valuemodel": "..."  ← WRONG! This is from contrast phrase
        }}
    }}
}}

Now process each search result using the 3-step method:"""

        # Use efficient model for this semantic extraction task with retry logic
        retry_count = 0
        async def make_llm_call():
            return await LLMFactory.chat_completion(
                task="terminology_extraction_batch",
                messages=[{"role": "user", "content": prompt}],
                model="openai/o3",
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=4092  # Increased for multiple answers
            )
        
        while retry_count < 3:
            try:
                response = await llm_call_with_retry(make_llm_call, max_retries=3)
                
                # Parse the JSON response
                import json
                content = response.get("content", "").strip()
                
                # Try to extract JSON from the response
                if "{" in content and "}" in content:
                    start_idx = content.find("{")
                    end_idx = content.rfind("}") + 1
                    json_str = content[start_idx:end_idx]
                    
                    try:
                        response_data = json.loads(json_str)
                        
                        # Parse the structured response organized by search result
                        results_per_answer = []
                        
                        for i in range(len(answers)):
                            answer_key = f"search_result_{i+1}"
                            answer_terms = {}
                            
                            if answer_key in response_data and isinstance(response_data[answer_key], dict):
                                # Handle new structured format with search_intent, contrast_phrases_found, extracted_terms
                                search_result = response_data[answer_key]
                                
                                # Extract terms from the extracted_terms field if it exists
                                extracted_terms = search_result.get('extracted_terms', search_result)
                                
                                # Log the filtering process for debugging
                                if 'search_intent' in search_result:
                                    intent = search_result.get('search_intent', 'unknown')
                                    contrast_phrases = search_result.get('contrast_phrases_found', [])
                                    logger.debug(f"LLM identified search intent: {intent}, found {len(contrast_phrases)} contrast phrases")
                                
                                for term, source in extracted_terms.items():
                                    if isinstance(term, str) and isinstance(source, str):
                                        clean_term = term.lower().strip()
                                        if len(clean_term) > 1 and clean_term not in _COMMON_WORDS:
                                            answer_terms[clean_term] = source.strip()
                            
                            results_per_answer.append(answer_terms)
                        
                        total_terms = sum(len(terms) for terms in results_per_answer)
                        logger.info(f"LLM extracted {total_terms} terms from {len(answers)} Tavily answers (preserved per-answer attribution)")
                        return results_per_answer
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse LLM response as JSON: {e}")
                        raise e
                else:
                    logger.warning("LLM response did not contain valid JSON")
                    raise Exception("LLM response did not contain valid JSON")
            except Exception as e:
                logger.warning(f"LLM error during batch term extraction: {e}")
                retry_count += 1
                await asyncio.sleep(0.25)
                
    except LLMError as e:
        logger.warning(f"LLM error during batch term extraction: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error during LLM batch term extraction: {e}")
    
    # Fallback to individual processing if batch fails
    logger.info("Falling back to individual term extraction")
    results_per_answer = []
    for answer in answers:
        if answer and answer.strip():
            terms = await _extract_terms_from_tavily_answer(answer, brand_domain, industry)
            results_per_answer.append(terms)
        else:
            results_per_answer.append({})
    return results_per_answer


async def _extract_terms_from_tavily_answer(answer: str, brand_domain: str = "", industry: str = "") -> Dict[str, str]:
    """
    Extract product terminology from Tavily answer using LLM semantic understanding.
    
    Uses a smaller, efficient LLM to intelligently extract product terms, model names,
    and tier indicators with full brand and industry context awareness.
    
    Args:
        answer: The Tavily synthesized answer text
        brand_domain: Brand domain for context (e.g., "gucci.com", "specialized.com")
        industry: Industry context (e.g., "luxury fashion", "cycling")
        
    Returns:
        Dict mapping term -> source_sentence for confidence boosting
    """
    if not answer or not answer.strip():
        return {}
    
    # Import here to avoid circular dependencies
    from liddy.llm.simple_factory import LLMFactory
    from liddy.llm.errors import LLMError
    
    try:
        prompt = f"""Extract product terminology from the following text about {brand_domain} in the {industry} industry.

TASK: Identify product terms, model names, series names, and tier indicators that customers might use when searching or talking about products.

BRAND CONTEXT: {brand_domain}
INDUSTRY: {industry}

TEXT TO ANALYZE:
{answer}

INSTRUCTIONS:
1. Extract specific product names, model names, series names, and collections
2. Include tier/level indicators (premium, luxury, professional, entry-level, etc.)
3. Include technical terms and features that customers would search for
4. Focus on terms that are specific to this brand/industry, not generic words
5. Include both formal product names and common nicknames/abbreviations
6. Exclude generic words like "the", "and", "product", "brand", etc.

FORMAT YOUR RESPONSE AS JSON:
{{
    "term1": "source sentence containing term1",
    "term2": "source sentence containing term2",
    ...
}}

EXAMPLE FOR SPECIALIZED.COM CYCLING:
{{
    "s-works": "The S-Works line represents our premium performance bikes.",
    "tarmac": "The Tarmac series offers excellent road racing performance.",
    "epic": "Epic mountain bikes are designed for cross-country racing."
}}

Extract the actual terms from the provided text:"""

        # Use efficient model for this semantic extraction task with retry logic
        async def make_llm_call():
            return await LLMFactory.chat_completion(
                task="terminology_extraction",
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-4o",  # Fast and cost-effective for this task
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=800
            )
        
        response = await llm_call_with_retry(make_llm_call, max_retries=3)
        
        # Parse the JSON response
        import json
        content = response.get("content", "").strip()
        
        # Try to extract JSON from the response
        if "{" in content and "}" in content:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            json_str = content[start_idx:end_idx]
            
            try:
                term_provenance = json.loads(json_str)
                # Ensure all keys are lowercase and clean
                cleaned_terms = {}
                for term, source in term_provenance.items():
                    if isinstance(term, str) and isinstance(source, str):
                        clean_term = term.lower().strip()
                        if len(clean_term) > 1 and clean_term not in _COMMON_WORDS:
                            cleaned_terms[clean_term] = source.strip()
                
                logger.info(f"LLM extracted {len(cleaned_terms)} terms from Tavily answer")
                return cleaned_terms
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                
    except LLMError as e:
        logger.warning(f"LLM error during term extraction: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error during LLM term extraction: {e}")
    
    # Fallback to regex-based extraction if LLM fails
    logger.info("Falling back to regex-based term extraction")
    return _regex_extract_terms_fallback(answer)


def _regex_extract_terms_fallback(answer: str) -> Dict[str, str]:
    """
    Fallback regex-based term extraction when LLM is unavailable.
    
    This is the original implementation preserved as a fallback.
    """
    if not answer:
        return {}
    
    term_provenance = {}
    
    # Split into sentences for provenance tracking
    sentences = re.split(r'[.!?]+', answer)
    
    # Tier-related keywords that indicate nearby terms might be relevant
    tier_indicators = {
        'tier', 'level', 'model', 'series', 'line', 'range', 'grade',
        'premium', 'professional', 'high-end', 'flagship', 'top',
        'mid', 'middle', 'intermediate', 'standard',
        'budget', 'entry', 'basic', 'starter', 'beginner', 'base'
    }
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Extract potential terms (alphanumeric with hyphens)
        words = re.findall(r'\b[a-zA-Z0-9\-]+\b', sentence.lower())
        
        # Check if sentence contains tier indicators
        sentence_has_tier_context = any(indicator in words for indicator in tier_indicators)
        
        if sentence_has_tier_context:
            # Extract potential tier terms from this sentence
            for i, word in enumerate(words):
                # Skip common words and very short terms
                if word in _COMMON_WORDS or len(word) <= 2:
                    continue
                
                # Skip the tier indicators themselves
                if word in tier_indicators:
                    continue
                
                # Check if word is near a tier indicator (within 3 words)
                nearby_indicator = False
                for j in range(max(0, i-3), min(len(words), i+4)):
                    if words[j] in tier_indicators:
                        nearby_indicator = True
                        break
                
                if nearby_indicator:
                    # This could be a tier term - store with provenance
                    term_provenance[word] = sentence.strip()
                    
                    # Also check for compound terms (e.g., "s-works")
                    if i < len(words) - 1:
                        compound = f"{word}-{words[i+1]}"
                        if '-' in compound and compound.count('-') == 1:
                            term_provenance[compound] = sentence.strip()
    
    # Special patterns for model names (e.g., "SL7", "Epic 8")
    model_patterns = [
        r'\b[A-Z]{1,3}\d{1,2}\b',  # SL7, GT2
        r'\b\w+\s+\d\b',  # Epic 8, Tarmac 7
        r'\b[A-Z]-Works\b',  # S-Works
    ]
    
    for pattern in model_patterns:
        for sentence in sentences:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                term_provenance[match.lower()] = sentence.strip()
    
    return term_provenance


def diversity_filter(results: List['SearchResult']) -> List['SearchResult']:
    """
    Ensure 1 result per domain for diversity.
    
    Filters search results to include only one result from each domain,
    promoting diverse sources and reducing bias from single sites.
    
    Args:
        results: List of search results
        
    Returns:
        Filtered list with max 1 result per domain
    """
    seen_domains = set()
    filtered = []
    
    for result in results:
        try:
            domain = urlparse(result.url).netloc
            # Remove www. prefix for better deduplication
            domain = domain.replace('www.', '')
            
            if domain not in seen_domains:
                seen_domains.add(domain)
                filtered.append(result)
        except Exception as e:
            logger.debug(f"Failed to parse URL {result.url}: {e}")
            # Keep the result if we can't parse the URL
            filtered.append(result)
    
    return filtered


def domain_diversity(results: List['SearchResult']) -> int:
    """
    Count unique PSL domains for diversity scoring.
    
    Args:
        results: List of search results
        
    Returns:
        Number of unique domains
    """
    domains = set()
    
    for result in results:
        try:
            domain = urlparse(result.url).netloc
            # Remove www. prefix
            domain = domain.replace('www.', '')
            domains.add(domain)
        except:
            pass
    
    return len(domains)


async def fallback_search(tier: str, current_terms: List[str], web_search, 
                         brand_domain: str, industry: str) -> List[str]:
    """
    Additional queries when tier has <10 terms.
    
    Performs forum-specific searches to find community terminology
    that might be missing from general web results.
    
    Args:
        tier: The price tier ("premium", "mid", or "budget")
        current_terms: Terms already found for this tier
        web_search: The web search client
        brand_domain: The brand domain (e.g., "specialized.com")
        industry: The industry context (e.g., "cycling")
        
    Returns:
        Additional terms found through fallback search
    """
    if len(current_terms) >= 10:
        return []
    
    logger.info(f"Fallback search triggered for {tier} tier (current: {len(current_terms)} terms)")
    
    # Try forum and community searches with proper brand/industry context
    fallback_queries = [
        f"{brand_domain} {industry} {tier} tier terminology site:reddit.com",
        f"{brand_domain} {tier} models forum discussion {industry}",
        f'"{brand_domain}" "{tier} level" products explained {industry}'
    ]
    
    additional_terms = []
    
    for query in fallback_queries:
        try:
            response = await run_search(query, web_search)
            
            # Extract terms from answer if available
            if response.answer:
                term_provenance = await _extract_terms_from_tavily_answer(
                    response.answer, brand_domain, industry
                )
                
                # Add new terms not already in current_terms
                for term in term_provenance.keys():
                    if term not in current_terms and term not in additional_terms:
                        additional_terms.append(term)
                
                # Stop if we have enough
                if len(current_terms) + len(additional_terms) >= 10:
                    break
                    
        except Exception as e:
            logger.debug(f"Fallback search failed for '{query}': {e}")
    
    logger.info(f"Fallback search found {len(additional_terms)} additional terms for {tier}")
    return additional_terms


@lru_cache(maxsize=1000)
def is_stopword(term: str, context: str = "all") -> bool:
    """
    Check if a term is a stopword in the given context.
    
    Args:
        term: The term to check
        context: The context category or "all" for any context
        
    Returns:
        True if term is a stopword
    """
    term_lower = term.lower()
    
    if context == "all":
        # Check all contexts
        for words in CONTEXT_STOPWORDS.values():
            if term_lower in words:
                return True
    else:
        # Check specific context
        if context in CONTEXT_STOPWORDS:
            return term_lower in CONTEXT_STOPWORDS[context]
    
    return False