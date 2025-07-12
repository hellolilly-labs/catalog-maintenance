"""
Descriptor Generation System

LLM-powered descriptor and sizing generation for product catalogs.
Uses proven sizing instructions exactly as provided, with multi-provider routing,
comprehensive brand vertical detection (web search + product sampling), and
vertical auto-detection for optimal quality across any brand/industry.

Key Features:
- Multi-source brand vertical detection (web search + product sampling)
- Proven sizing instruction implementation (exact)
- OpenAI/Anthropic/Gemini service integration with configuration management
- Vertical auto-detection (no hardcoded assumptions)
- JSON response formatting with validation
- Comprehensive error handling and logging
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.models.product import Product
from src.llm.simple_factory import LLMFactory
from src.llm.errors import LLMError, TokenLimitError, ModelNotFoundError
from configs.settings import get_settings

logger = logging.getLogger(__name__)


class BrandVerticalDetector:
    """
    Comprehensive brand vertical detection using multiple sources:
    1. Web search of brand domain
    2. Product catalog sampling and analysis
    3. Category distribution analysis
    4. LLM synthesis of all sources
    
    Enhanced with persistent storage - saves results to GCP/local storage
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._cache = {}  # In-memory cache for this session
        self._web_search_engine = None
        self._storage = None  # Lazy-loaded storage provider
    
    def _get_web_search_engine(self):
        """Lazy initialization of web search engine"""
        if self._web_search_engine is None:
            try:
                from src.web_search import get_web_search_engine
                self._web_search_engine = get_web_search_engine()
            except ImportError:
                logger.warning("Web search module not available")
                self._web_search_engine = None
        return self._web_search_engine
    
    def _get_storage(self):
        """Lazy initialization of storage provider"""
        if self._storage is None:
            from src.storage import get_account_storage_provider
            self._storage = get_account_storage_provider()
        return self._storage
    
    async def _load_brand_vertical_from_storage(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """Load cached brand vertical results from storage"""
        try:
            storage = self._get_storage()
            
            # Try to load from storage path: accounts/{brand_domain}/brand_vertical.json
            if hasattr(storage, 'bucket'):
                # GCP storage
                blob = storage.bucket.blob(f"accounts/{brand_domain}/brand_vertical.json")
                if blob.exists():
                    content = blob.download_as_text()
                    result = json.loads(content)
                    logger.info(f"Loaded cached brand vertical for {brand_domain} from GCP storage")
                    return result
            else:
                # Local storage  
                import os
                filepath = os.path.join(storage.base_dir, "accounts", brand_domain, "brand_vertical.json")
                if os.path.exists(filepath):
                    with open(filepath, "r") as f:
                        result = json.load(f)
                        logger.info(f"Loaded cached brand vertical for {brand_domain} from local storage")
                        return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Error loading cached brand vertical for {brand_domain}: {e}")
            return None
    
    async def _save_brand_vertical_to_storage(self, brand_domain: str, detection_results: Dict[str, Any]) -> bool:
        """Save brand vertical detection results to persistent storage"""
        try:
            storage = self._get_storage()
            
            # Add metadata
            enhanced_results = {
                **detection_results,
                "analysis_timestamp": datetime.now().isoformat() + "Z",
                "analysis_duration_seconds": getattr(detection_results, '_duration', None),
                "storage_provider": type(storage).__name__,
                "version": "2.0"  # Enhanced version with persistent storage
            }
            
            # Save to storage
            if hasattr(storage, 'bucket'):
                # GCP storage - accounts/{brand_domain}/brand_vertical.json
                blob = storage.bucket.blob(f"accounts/{brand_domain}/brand_vertical.json")
                blob.upload_from_string(
                    json.dumps(enhanced_results, indent=2),
                    content_type="application/json"
                )
                
                # Also save metadata summary
                metadata = {
                    "brand_domain": brand_domain,
                    "detected_vertical": detection_results.get("detected_vertical"),
                    "confidence": detection_results.get("confidence"),
                    "analysis_methods": detection_results.get("analysis_methods", []),
                    "analysis_timestamp": enhanced_results["analysis_timestamp"],
                    "storage_path": f"accounts/{brand_domain}/brand_vertical.json"
                }
                
                metadata_blob = storage.bucket.blob(f"accounts/{brand_domain}/brand_vertical_metadata.json")
                metadata_blob.upload_from_string(
                    json.dumps(metadata, indent=2),
                    content_type="application/json"
                )
                
                logger.info(f"✅ Saved brand vertical analysis for {brand_domain} to GCP: {storage.bucket.name}")
                return True
                
            else:
                # Local storage
                import os
                account_dir = os.path.join(storage.base_dir, "accounts", brand_domain)
                os.makedirs(account_dir, exist_ok=True)
                
                # Save main results
                filepath = os.path.join(account_dir, "brand_vertical.json")
                with open(filepath, "w") as f:
                    json.dump(enhanced_results, f, indent=2)
                
                # Save metadata
                metadata = {
                    "brand_domain": brand_domain,
                    "detected_vertical": detection_results.get("detected_vertical"),
                    "confidence": detection_results.get("confidence"),
                    "analysis_methods": detection_results.get("analysis_methods", []),
                    "analysis_timestamp": enhanced_results["analysis_timestamp"],
                    "storage_path": filepath
                }
                
                metadata_path = os.path.join(account_dir, "brand_vertical_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✅ Saved brand vertical analysis for {brand_domain} to local storage: {filepath}")
                return True
            
        except Exception as e:
            logger.error(f"Error saving brand vertical results for {brand_domain}: {e}")
            return False
    
    async def detect_brand_vertical(self, brand_domain: str, product_sample: Optional[Product] = None) -> Dict[str, Any]:
        """
        Comprehensive brand vertical detection using multiple sources
        
        Args:
            brand_domain: Brand domain (e.g., "specialized.com")
            product_sample: Optional single product for fallback
            
        Returns:
            Dict with detected vertical, confidence, and analysis method used
        """
        start_time = datetime.now()
        
        # Check in-memory cache first
        if brand_domain in self._cache:
            logger.info(f"Using in-memory cached brand vertical for {brand_domain}")
            return self._cache[brand_domain]
        
        # Check persistent storage
        cached_result = await self._load_brand_vertical_from_storage(brand_domain)
        if cached_result:
            # Load into memory cache and return
            self._cache[brand_domain] = cached_result
            return cached_result
        
        detection_results = {
            "brand_domain": brand_domain,
            "detected_vertical": "general",
            "confidence": 0.0,
            "analysis_methods": [],
            "web_search_data": None,
            "product_analysis": None,
            "category_distribution": None
        }
        
        try:
            # Method 1: Web search analysis (most reliable when available)
            web_search_result = await self._analyze_brand_via_web_search(brand_domain)
            if web_search_result:
                detection_results["web_search_data"] = web_search_result
                detection_results["analysis_methods"].append("web_search")
            
            # Method 2: Product catalog sampling (if available)
            product_analysis = await self._analyze_brand_via_product_sampling(brand_domain)
            if product_analysis:
                detection_results["product_analysis"] = product_analysis
                detection_results["analysis_methods"].append("product_sampling")
            
            # Method 3: Fallback to single product (if provided)
            if product_sample and not detection_results["analysis_methods"]:
                single_product_analysis = await self._analyze_single_product(product_sample)
                if single_product_analysis:
                    detection_results["product_analysis"] = single_product_analysis
                    detection_results["analysis_methods"].append("single_product_fallback")
            
            # Synthesize all available data
            if detection_results["analysis_methods"]:
                synthesis = await self._synthesize_vertical_analysis(detection_results)
                detection_results.update(synthesis)
            
            # Add timing information
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            detection_results["_duration"] = duration
            
            # Save to both memory cache and persistent storage
            self._cache[brand_domain] = detection_results
            await self._save_brand_vertical_to_storage(brand_domain, detection_results)
            
            logger.info(f"Detected brand vertical for {brand_domain}: {detection_results['detected_vertical']} "
                       f"(confidence: {detection_results['confidence']:.2f}, methods: {detection_results['analysis_methods']}, "
                       f"duration: {duration:.1f}s)")
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive brand vertical detection for {brand_domain}: {e}")
            detection_results["detected_vertical"] = "general"
            detection_results["confidence"] = 0.1
            detection_results["analysis_methods"] = ["error_fallback"]
            
            # Still try to save error results for debugging
            await self._save_brand_vertical_to_storage(brand_domain, detection_results)
            
            return detection_results
    
    async def _analyze_brand_via_web_search(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """Use web search to research the brand and determine vertical with direct questions"""
        try:
            web_search = self._get_web_search_engine()
            
            if not web_search or not web_search.is_available():
                # Fallback to domain pattern analysis
                logger.info(f"No web search available, using domain analysis for {brand_domain}")
                domain_vertical = web_search.search_domain_analysis(brand_domain) if web_search else None
                
                if domain_vertical:
                    return {
                        "detected_vertical": domain_vertical,
                        "confidence": 0.6,
                        "reasoning": "Domain pattern analysis fallback when web search unavailable",
                        "evidence": [f"Domain '{brand_domain}' matches {domain_vertical} patterns"],
                        "alternative_verticals": [],
                        "method": "domain_analysis",
                        "search_strategy": "pattern_matching"
                    }
                else:
                    return None
            
            # ENHANCED: More direct vertical detection queries
            search_queries = [
                f"What is the primary business vertical that {brand_domain} operates in",
                f"{brand_domain} company industry sector what do they primarily sell",
                f"{brand_domain} main business focus core products primary market",
                f"site:{brand_domain} about company industry vertical business",
                f"{brand_domain} company profile industry category business type"
            ]
            
            # Perform actual web search with direct vertical questions
            search_results = await web_search.search_brand_info(brand_domain)
            
            if not search_results.get("results"):
                logger.warning(f"No web search results found for {brand_domain}")
                return None
            
            # Compile search data for LLM analysis
            search_context = ""
            for result in search_results["results"][:12]:  # More results for better analysis
                search_context += f"Title: {result.get('title', '')}\n"
                search_context += f"URL: {result.get('url', '')}\n"
                search_context += f"Snippet: {result.get('snippet', '')}\n"
                search_context += f"Query: {result.get('query', '')}\n\n"
            
            # ENHANCED: More sophisticated web research prompt
            web_research_prompt = f"""Analyze web search results to determine this brand's primary business vertical.

Brand: {brand_domain}

SEARCH RESULTS ANALYSIS:
{search_context}

ANALYSIS INSTRUCTIONS:
1. Focus on what the company PRIMARILY does vs what they also sell
2. Look for company descriptions, about pages, industry classifications
3. Distinguish between core business and secondary/accessory products
4. Consider official company statements about their industry/market

Example Analysis:
- "Specialized Bicycle Components designs and manufactures premium bicycles" → Core business: cycling/bicycles
- "Nike athletic footwear and apparel company" → Core business: sports/athletic

Based on these search results, determine the brand's PRIMARY vertical from categories like:
cycling, fashion, footwear, electronics, home, beauty, sports, automotive, health, food, pets, books, tools, jewelry, outdoor, baby, toys, music, art, technology, etc.

Respond with a JSON object containing:
{{
    "detected_vertical": "primary_vertical_name",
    "confidence": 0.85,
    "reasoning": "Detailed explanation based on search evidence focusing on core business",
    "evidence": ["specific quotes or facts from search results"],
    "alternative_verticals": ["secondary_vertical_if_any"],
    "search_strategy": "direct_vertical_questions",
    "source_types": ["company_about_page", "industry_description", "product_focus"]
}}"""

            response = await LLMFactory.chat_completion(
                task="brand_research",
                system="You are a business intelligence analyst. Analyze web search results to determine company's primary business vertical, focusing on core business rather than secondary products.",
                messages=[{
                    "role": "user",
                    "content": web_research_prompt
                }],
                max_tokens=400,
                temperature=0.1
            )
            
            if response and response.get("content"):
                try:
                    # Try to parse JSON response
                    result = json.loads(response["content"])
                    result["method"] = "enhanced_web_search_with_llm"
                    result["search_results_count"] = len(search_results["results"])
                    result["provider_used"] = search_results.get("provider_used", "unknown")
                    result["queries_used"] = search_queries
                    return result
                except json.JSONDecodeError:
                    # Fallback to text analysis
                    content = response["content"].strip().lower()
                    return {
                        "detected_vertical": self._extract_vertical_from_text(content),
                        "confidence": 0.6,
                        "reasoning": "Enhanced web search analysis with text extraction fallback",
                        "evidence": [content[:200] + "..."],
                        "alternative_verticals": [],
                        "method": "enhanced_web_search_text_analysis",
                        "search_results_count": len(search_results["results"]),
                        "search_strategy": "direct_vertical_questions"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced web search analysis for {brand_domain}: {e}")
            return None
    
    async def _analyze_brand_via_product_sampling(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """Analyze brand vertical by intelligent product catalog sampling"""
        try:
            # Get storage provider for this brand
            from src.storage import get_account_storage_provider
            
            storage = get_account_storage_provider()
            
            # Try to get product catalog for analysis
            product_catalog = await storage.get_product_catalog(brand_domain)
            
            if not product_catalog or len(product_catalog) == 0:
                logger.info(f"No product catalog found for {brand_domain}")
                return None
            
            # STEP 1: Get complete category distribution first
            all_categories = []
            product_names = []
            descriptions = []
            
            for product_data in product_catalog:
                if product_data.get('categories'):
                    all_categories.extend(product_data['categories'])
                if product_data.get('name'):
                    product_names.append(product_data['name'])
                if product_data.get('description'):
                    descriptions.append(product_data['description'])
            
            # Count category frequency
            from collections import Counter
            category_counts = Counter(all_categories)
            top_categories = dict(category_counts.most_common(15))  # Get more categories
            
            # STEP 2: Intelligent sampling strategy (not random!)
            # Sample proportionally from top categories to avoid missing core products
            strategic_sample = []
            sample_size = min(15, len(product_catalog))  # Larger sample
            
            # Group products by their primary category
            products_by_category = {}
            for product_data in product_catalog:
                if product_data.get('categories') and len(product_data['categories']) > 0:
                    primary_category = product_data['categories'][0]  # Use first category as primary
                    if primary_category not in products_by_category:
                        products_by_category[primary_category] = []
                    products_by_category[primary_category].append(product_data)
            
            # Sample from each major category proportionally
            for category, count in list(category_counts.most_common(8)):  # Top 8 categories
                category_products = products_by_category.get(category, [])
                if category_products:
                    # Sample 1-3 products from this category based on its prevalence
                    category_sample_size = max(1, min(3, int(sample_size * (count / len(all_categories)))))
                    import random
                    sampled = random.sample(category_products, min(category_sample_size, len(category_products)))
                    strategic_sample.extend(sampled)
            
            # If we haven't hit our target sample size, add some random products
            if len(strategic_sample) < sample_size:
                remaining_products = [p for p in product_catalog if p not in strategic_sample]
                if remaining_products:
                    import random
                    additional = random.sample(remaining_products, min(sample_size - len(strategic_sample), len(remaining_products)))
                    strategic_sample.extend(additional)
            
            # STEP 3: Enhanced LLM analysis with category hierarchy understanding
            catalog_analysis_prompt = f"""Analyze this product catalog to determine the brand's primary business vertical.

Brand: {brand_domain}
Total Products: {len(product_catalog)}
Sample Analyzed: {len(strategic_sample)} products (strategically sampled)

CATEGORY DISTRIBUTION ANALYSIS:
{json.dumps(top_categories, indent=2)}

STRATEGIC PRODUCT SAMPLE:
{json.dumps([{{
    "name": p.get('name', 'Unknown'),
    "categories": p.get('categories', []),
    "description": (p.get('description', '') or '')[:100] + '...' if p.get('description') else 'No description'
}} for p in strategic_sample[:10]], indent=2)}

ANALYSIS INSTRUCTIONS:
1. Identify which categories represent CORE BUSINESS vs ACCESSORIES/SUPPORT
2. Determine what the brand primarily MANUFACTURES vs what they SELL as accessories
3. Consider category hierarchy: Are bikes the core with accessories supporting them?
4. Look for patterns in product naming and descriptions

Example Analysis:
- If categories show "Road Bikes: 156, Accessories: 400" - the core business is likely cycling/bikes, accessories are supportive
- If categories show "Skincare: 200, Makeup: 150, Tools: 50" - core business is beauty, tools are accessories

Based on this analysis, determine the PRIMARY vertical this brand operates in.

Respond with a JSON object:
{{
    "detected_vertical": "primary_vertical_name", 
    "confidence": 0.85,
    "reasoning": "Detailed analysis of core vs accessory products",
    "evidence": ["specific evidence from catalog analysis"],
    "category_hierarchy": {{"core_categories": ["main business"], "support_categories": ["accessories"]}},
    "product_patterns": ["observed patterns in products"],
    "sampling_method": "strategic_category_weighted"
}}"""

            response = await LLMFactory.chat_completion(
                task="brand_research",
                system="You are an expert business analyst. Determine brand verticals by analyzing product catalog hierarchies, distinguishing core business from accessories and support products.",
                messages=[{
                    "role": "user",
                    "content": catalog_analysis_prompt
                }],
                max_tokens=500,
                temperature=0.1
            )
            
            if response and response.get("content"):
                try:
                    result = json.loads(response["content"])
                    result["total_products"] = len(product_catalog)
                    result["sample_size"] = len(strategic_sample)
                    result["method"] = "strategic_product_catalog_analysis"
                    result["category_distribution"] = top_categories
                    return result
                except json.JSONDecodeError:
                    # Fallback analysis
                    content = response["content"].strip().lower()
                    return {
                        "detected_vertical": self._extract_vertical_from_text(content),
                        "confidence": 0.7,
                        "reasoning": "Strategic product catalog analysis with fallback parsing",
                        "evidence": [f"Analyzed {len(strategic_sample)} strategically sampled products"],
                        "category_distribution": top_categories,
                        "total_products": len(product_catalog),
                        "sample_size": len(strategic_sample),
                        "method": "strategic_product_catalog_fallback",
                        "sampling_method": "strategic_category_weighted"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in strategic product sampling analysis for {brand_domain}: {e}")
            return None
    
    async def _analyze_single_product(self, product: Product) -> Optional[Dict[str, Any]]:
        """Fallback analysis using single product (original method)"""
        try:
            analysis_prompt = f"""Analyze this single product to estimate the brand's likely vertical.

Product: {product.name}
Brand: {product.brand}
Categories: {', '.join(product.categories) if product.categories else 'Not specified'}
Description: {product.description or 'Not provided'}
Features: {', '.join(product.highlights) if product.highlights else 'Not provided'}

Based on this single product, estimate the brand's PRIMARY vertical.

Respond with a JSON object:
{{
    "detected_vertical": "estimated_vertical",
    "confidence": 0.4,
    "reasoning": "Single product analysis - limited confidence",
    "evidence": ["evidence from this product"],
    "limitations": ["single product sample", "may not represent full brand"]
}}"""

            response = await LLMFactory.chat_completion(
                task="brand_research",
                system="You are a brand analyst. Estimate brand vertical from single product with appropriate uncertainty.",
                messages=[{
                    "role": "user",
                    "content": analysis_prompt
                }],
                max_tokens=300,
                temperature=0.1
            )
            
            if response and response.get("content"):
                try:
                    result = json.loads(response["content"])
                    result["method"] = "single_product_analysis"
                    return result
                except json.JSONDecodeError:
                    content = response["content"].strip().lower()
                    return {
                        "detected_vertical": self._extract_vertical_from_text(content),
                        "confidence": 0.3,
                        "reasoning": "Single product fallback analysis",
                        "evidence": [f"Product: {product.name}"],
                        "limitations": ["single product sample"],
                        "method": "single_product_text_fallback"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in single product analysis: {e}")
            return None
    
    async def _synthesize_vertical_analysis(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all analysis methods to determine final vertical"""
        try:
            # If only one method succeeded, use its results directly
            if len(detection_results['analysis_methods']) == 1:
                method = detection_results['analysis_methods'][0]
                if method == "web_search" and detection_results.get('web_search_data'):
                    source = detection_results['web_search_data']
                    return {
                        "detected_vertical": source.get('detected_vertical', 'general'),
                        "confidence": source.get('confidence', 0.5),
                        "synthesis_reasoning": f"Single high-quality source: {method}",
                        "method_weights": {method: 1.0},
                        "consensus_level": "high" if source.get('confidence', 0) > 0.8 else "medium"
                    }
                elif method == "product_sampling" and detection_results.get('product_analysis'):
                    source = detection_results['product_analysis']
                    return {
                        "detected_vertical": source.get('detected_vertical', 'general'),
                        "confidence": source.get('confidence', 0.5),
                        "synthesis_reasoning": f"Single high-quality source: {method}",
                        "method_weights": {method: 1.0},
                        "consensus_level": "high" if source.get('confidence', 0) > 0.8 else "medium"
                    }
            
            # Multiple methods - do full synthesis
            synthesis_prompt = f"""Synthesize multiple brand vertical analysis methods to determine the final brand vertical.

Brand: {detection_results['brand_domain']}
Analysis Methods Used: {detection_results['analysis_methods']}

Analysis Results:
{json.dumps({
    'web_search_data': detection_results.get('web_search_data'),
    'product_analysis': detection_results.get('product_analysis')
}, indent=2)}

Synthesize these results to determine:
1. The most reliable vertical detection
2. Overall confidence level
3. Any conflicting signals that need resolution

Respond with a JSON object:
{{
    "detected_vertical": "final_vertical_choice",
    "confidence": 0.85,
    "synthesis_reasoning": "Why this vertical was chosen over others",
    "method_weights": {{"web_search": 0.6, "product_sampling": 0.4}},
    "consensus_level": "high|medium|low"
}}"""

            response = await LLMFactory.chat_completion(
                task="brand_research",
                system="You are a senior brand analyst. Synthesize multiple data sources to make final vertical determinations.",
                messages=[{
                    "role": "user",
                    "content": synthesis_prompt
                }],
                max_tokens=300,
                temperature=0.1
            )
            
            if response and response.get("content"):
                try:
                    result = json.loads(response["content"])
                    return result
                except json.JSONDecodeError:
                    # Use the highest confidence source
                    web_conf = detection_results.get('web_search_data', {}).get('confidence', 0)
                    product_conf = detection_results.get('product_analysis', {}).get('confidence', 0)
                    
                    if web_conf >= product_conf:
                        best_source = detection_results.get('web_search_data', {})
                        method = "web_search"
                    else:
                        best_source = detection_results.get('product_analysis', {})
                        method = "product_analysis"
                    
                    return {
                        "detected_vertical": best_source.get('detected_vertical', 'general'),
                        "confidence": best_source.get('confidence', 0.5),
                        "synthesis_reasoning": f"Used highest confidence source: {method}",
                        "method_weights": {method: 1.0},
                        "consensus_level": "medium"
                    }
            
            # Final fallback
            return {
                "detected_vertical": "general",
                "confidence": 0.3,
                "synthesis_reasoning": "Unable to synthesize analysis results",
                "method_weights": {},
                "consensus_level": "low"
            }
            
        except Exception as e:
            logger.error(f"Error in vertical analysis synthesis: {e}")
            # Emergency fallback - use web search if available
            if detection_results.get('web_search_data'):
                source = detection_results['web_search_data']
                return {
                    "detected_vertical": source.get('detected_vertical', 'general'),
                    "confidence": max(0.2, source.get('confidence', 0.5) * 0.8),  # Reduce confidence due to error
                    "synthesis_reasoning": f"Synthesis error fallback, using web search: {str(e)}",
                    "method_weights": {"web_search": 1.0},
                    "consensus_level": "low"
                }
            return {
                "detected_vertical": "general",
                "confidence": 0.2,
                "synthesis_reasoning": f"Synthesis error: {str(e)}",
                "method_weights": {},
                "consensus_level": "low"
            }
    
    def _extract_vertical_from_text(self, text: str) -> str:
        """Extract likely vertical from text content"""
        vertical_keywords = {
            "cycling": ["bike", "cycling", "bicycle", "wheel", "frame", "specialized", "trek"],
            "fashion": ["clothing", "apparel", "fashion", "style", "wear", "dress", "shirt"],
            "footwear": ["shoes", "boots", "sneakers", "footwear", "running", "athletic"],
            "electronics": ["electronic", "tech", "device", "digital", "computer", "phone"],
            "beauty": ["beauty", "cosmetic", "skincare", "makeup", "fragrance"],
            "sports": ["sports", "athletic", "fitness", "training", "performance"],
            "outdoor": ["outdoor", "hiking", "camping", "adventure", "nature"],
            "automotive": ["car", "auto", "vehicle", "automotive", "racing"],
            "home": ["home", "furniture", "decor", "living", "house"],
            "health": ["health", "wellness", "medical", "fitness", "nutrition"]
        }
        
        text_lower = text.lower()
        vertical_scores = {}
        
        for vertical, keywords in vertical_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                vertical_scores[vertical] = score
        
        if vertical_scores:
            return max(vertical_scores, key=vertical_scores.get)
        
        return "general"


class DescriptorGenerator:
    """LLM-powered product descriptor and sizing generation with enhanced brand intelligence"""
    
    def __init__(self, brand_vertical: str = None):
        """
        Initialize descriptor generator with settings
        
        Args:
            brand_vertical: Optional brand-level vertical (will detect if None)
        """
        self.settings = get_settings()
        self.brand_vertical = brand_vertical
        self.vertical_detector = BrandVerticalDetector()
        self._product_vertical_cache = {}  # Cache for product sub-verticals
        
    async def detect_brand_vertical(self, product: Product) -> str:
        """
        Enhanced brand vertical detection using multiple sources
        
        Args:
            product: Sample product from the brand
            
        Returns:
            str: Detected brand vertical
        """
        if self.brand_vertical:
            return self.brand_vertical
            
        brand_domain = product.brand or "unknown"
        
        # Use comprehensive detection
        detection_result = await self.vertical_detector.detect_brand_vertical(
            brand_domain=brand_domain,
            product_sample=product
        )
        
        return detection_result.get("detected_vertical", "general")
    
    async def detect_product_subvertical(self, product: Product, brand_vertical: str) -> Optional[str]:
        """
        LLM-based product sub-vertical detection within brand vertical
        
        Args:
            product: Product to analyze
            brand_vertical: Already detected brand vertical
            
        Returns:
            Optional[str]: Product sub-vertical or None if same as brand
        """
        try:
            # Build analysis prompt for product sub-vertical
            subvertical_prompt = f"""This product belongs to a {brand_vertical} brand. Determine if this specific product fits into a more specific sub-category within {brand_vertical}.

Product: {product.name}
Categories: {', '.join(product.categories) if product.categories else 'Not specified'}
Description: {product.description or 'Not provided'}
Features: {', '.join(product.highlights) if product.highlights else 'Not provided'}
Specifications: {json.dumps(product.specifications, indent=2) if product.specifications else 'Not provided'}

If this product represents a specific sub-category within {brand_vertical} (like "road bikes" within cycling, or "running shoes" within footwear), respond with the sub-category name.

If this product is just a general {brand_vertical} product without a more specific sub-category, respond with "none".

Respond with ONLY the sub-category name (lowercase) or "none". No explanation needed."""

            response = await LLMFactory.chat_completion(
                task="brand_research", 
                system="You are a product categorization expert. Identify specific sub-categories within broader verticals.",
                messages=[{
                    "role": "user",
                    "content": subvertical_prompt
                }],
                max_tokens=30,
                temperature=0.1
            )
            
            if response and response.get("content"):
                subvertical = response["content"].strip().lower()
                if subvertical == "none" or subvertical == brand_vertical:
                    return None
                logger.info(f"Detected product sub-vertical '{subvertical}' for {product.id}")
                return subvertical
            
            return None
                
        except Exception as e:
            logger.error(f"Error detecting product sub-vertical: {e}")
            return None
    
    async def detect_vertical_context(self, product: Product) -> Dict[str, str]:
        """
        Detect full vertical context for a product using enhanced brand detection
        
        Args:
            product: Product to analyze
            
        Returns:
            Dict with brand_vertical, product_subvertical, and analysis metadata
        """
        # Use enhanced brand vertical detection
        detection_result = await self.vertical_detector.detect_brand_vertical(
            brand_domain=product.brand or "unknown",
            product_sample=product
        )
        
        brand_vertical = detection_result.get("detected_vertical", "general")
        
        # Detect product sub-vertical within brand context
        product_subvertical = await self.detect_product_subvertical(product, brand_vertical)
        
        return {
            "brand_vertical": brand_vertical,
            "product_subvertical": product_subvertical,
            "effective_vertical": product_subvertical or brand_vertical,
            "detection_confidence": detection_result.get("confidence", 0.0),
            "analysis_methods": detection_result.get("analysis_methods", []),
            "brand_domain": product.brand or "unknown"
        }
    
    def build_descriptor_prompt(self, product: Product, vertical: str) -> str:
        """
        Build dynamic descriptor generation prompt based on detected vertical
        
        Args:
            product: Product to generate descriptor for
            vertical: Detected vertical category
            
        Returns:
            str: Customized prompt for the product's vertical
        """
        # Base prompt structure that adapts to any vertical
        base_prompt = f"""You are a professional product copywriter specializing in {vertical} products. 
Create an engaging, informative product descriptor that highlights key features and benefits.

Guidelines:
- Write in a natural, engaging tone appropriate for {vertical} customers
- Highlight unique features and benefits specific to this product type
- Include relevant technical details that matter for purchase decisions
- Keep the descriptor informative but concise (2-4 sentences)
- Focus on what makes this product special and valuable
- Avoid generic marketing language - be specific and authentic

Product Information:
- Name: {product.name}
- Categories: {', '.join(product.categories) if product.categories else 'N/A'}
- Brand: {product.brand}
- Price: {product.salePrice or product.originalPrice or 'N/A'}
- Colors: {', '.join(str(c) for c in product.colors) if product.colors else 'N/A'}
- Sizes: {', '.join(product.sizes) if product.sizes else 'N/A'}
- Highlights: {', '.join(product.highlights) if product.highlights else 'N/A'}
- Description: {product.description or 'N/A'}
- Specifications: {json.dumps(product.specifications, indent=2) if product.specifications else 'N/A'}

Generate a compelling product descriptor that would help customers understand why they should choose this product."""

        return base_prompt
    
    def build_sizing_prompt(self, product: Product, sizing_data: Dict[str, Any]) -> str:
        """
        Build sizing generation prompt using PROVEN SIZING INSTRUCTION exactly as provided
        This is the exact instruction documented as working - DO NOT MODIFY
        
        Args:
            product: Product to generate sizing for
            sizing_data: Available sizing information/chart
            
        Returns:
            str: Proven sizing instruction prompt
        """
        # PROVEN SIZING INSTRUCTION - EXACT IMPLEMENTATION
        # From COPILOT_NOTES.md Decision #3: Use working LLM instructions exactly as provided
        proven_sizing_prompt = """Given these product details and the sizing chart, find the correct sizing and create a 'sizing' field with the appropriate size information in JSON format."""
        
        # Build complete prompt with product context
        full_prompt = f"""{proven_sizing_prompt}

Product Details:
- Name: {product.name}
- Categories: {', '.join(product.categories) if product.categories else 'N/A'}
- Brand: {product.brand}
- Available Sizes: {', '.join(product.sizes) if product.sizes else 'N/A'}
- Description: {product.description or 'N/A'}

Sizing Chart/Information:
{json.dumps(sizing_data, indent=2)}

Please analyze the product and sizing information, then respond with a JSON object containing a 'sizing' field with appropriate size guidance. Include size chart information, fit advice, and any relevant sizing notes that would help customers choose the right size.

Example response format:
{{"sizing": {{"size_chart": {{"S": "Fits X-Y measurements", "M": "Fits A-B measurements"}}, "fit_advice": "This product runs true to size. For best fit, measure [specific area] and refer to size chart."}}}}"""

        return full_prompt
    
    async def generate_descriptor(self, product: Product) -> Optional[str]:
        """
        Generate product descriptor using LLM with vertical-specific optimization
        
        Args:
            product: Product to generate descriptor for
            
        Returns:
            Optional[str]: Generated descriptor or None if generation fails
        """
        try:
            # Detect product vertical for optimal prompt customization
            vertical_context = await self.detect_vertical_context(product)
            effective_vertical = vertical_context["effective_vertical"]
            logger.info(f"Detected vertical context for product {product.id}: {vertical_context}")
            
            # Build vertical-specific prompt
            prompt = self.build_descriptor_prompt(product, effective_vertical)
            
            # Use router to get optimal model for descriptor generation
            # From router configuration: descriptor_generation -> o3
            response = await LLMFactory.chat_completion(
                task="descriptor_generation",
                system="You are a professional product copywriter. Create engaging, informative product descriptors.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=self.settings.openai_max_tokens,
                temperature=0.7  # Creative but controlled
            )
            
            if response and response.get("content"):
                descriptor = response["content"].strip()
                logger.info(f"Generated descriptor for product {product.id}: {len(descriptor)} characters")
                return descriptor
            else:
                logger.warning(f"Empty response from LLM for product {product.id}")
                return None
                
        except TokenLimitError as e:
            logger.error(f"Token limit exceeded generating descriptor for {product.id}: {e}")
            return None
        except ModelNotFoundError as e:
            logger.error(f"Model not available for descriptor generation: {e}")
            return None
        except LLMError as e:
            logger.error(f"LLM error generating descriptor for {product.id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating descriptor for {product.id}: {e}")
            return None
    
    async def generate_sizing(self, product: Product, sizing_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate sizing information using PROVEN SIZING INSTRUCTION exactly as provided
        
        Args:
            product: Product to generate sizing for
            sizing_data: Available sizing chart/information
            
        Returns:
            Optional[Dict]: Generated sizing information in JSON format or None if generation fails
        """
        try:
            # Build prompt using proven sizing instruction
            prompt = self.build_sizing_prompt(product, sizing_data)
            
            # Use router for sizing analysis - optimized for reasoning
            # From router configuration: sizing_analysis -> o3 (superior reasoning)
            response = await LLMFactory.chat_completion(
                task="sizing_analysis",
                system="You are a sizing expert. Analyze product details and sizing charts to provide accurate size guidance in JSON format.",
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=self.settings.openai_max_tokens,
                temperature=0.3  # Lower temperature for accuracy
            )
            
            if response and response.get("content"):
                content = response["content"].strip()
                
                # Parse JSON response
                try:
                    sizing_json = json.loads(content)
                    logger.info(f"Generated sizing for product {product.id}")
                    return sizing_json
                except json.JSONDecodeError:
                    # Try to extract JSON from response if wrapped in text
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            sizing_json = json.loads(json_match.group())
                            logger.info(f"Extracted JSON sizing for product {product.id}")
                            return sizing_json
                        except json.JSONDecodeError:
                            pass
                    
                    logger.warning(f"Could not parse JSON from sizing response for {product.id}: {content[:200]}...")
                    return None
            else:
                logger.warning(f"Empty response from LLM for sizing generation of product {product.id}")
                return None
                
        except TokenLimitError as e:
            logger.error(f"Token limit exceeded generating sizing for {product.id}: {e}")
            return None
        except ModelNotFoundError as e:
            logger.error(f"Model not available for sizing generation: {e}")
            return None
        except LLMError as e:
            logger.error(f"LLM error generating sizing for {product.id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating sizing for {product.id}: {e}")
            return None
    
    async def process_product(self, product: Product, sizing_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a product to generate both descriptor and sizing information
        
        Args:
            product: Product to process
            sizing_data: Optional sizing chart/information
            
        Returns:
            Dict: Results with descriptor, sizing, and metadata
        """
        results = {
            "product_id": product.id,
            "product_name": product.name,
            "detected_vertical": None,
            "descriptor": None,
            "sizing": None,
            "processing_time": None,
            "errors": []
        }
        
        start_time = datetime.now()
        
        try:
            # Detect vertical
            vertical_context = await self.detect_vertical_context(product)
            results["detected_vertical"] = vertical_context["effective_vertical"]
            
            # Generate descriptor
            try:
                descriptor = await self.generate_descriptor(product)
                results["descriptor"] = descriptor
            except Exception as e:
                error_msg = f"Descriptor generation failed: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
            
            # Generate sizing if sizing data provided
            if sizing_data:
                try:
                    sizing = await self.generate_sizing(product, sizing_data)
                    results["sizing"] = sizing
                except Exception as e:
                    error_msg = f"Sizing generation failed: {e}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Calculate processing time
            end_time = datetime.now()
            results["processing_time"] = (end_time - start_time).total_seconds()
            
            logger.info(f"Processed product {product.id} in {results['processing_time']:.2f}s")
            
        except Exception as e:
            error_msg = f"Product processing failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return results


# Factory function for easy integration
def get_descriptor_generator() -> DescriptorGenerator:
    """
    Factory function to create DescriptorGenerator instance
    Uses default LLM router with configuration
    
    Returns:
        DescriptorGenerator: Configured instance ready for use
    """
    return DescriptorGenerator()


# Convenience functions for direct usage
async def generate_product_descriptor(product: Product) -> Optional[str]:
    """
    Generate descriptor for a single product
    
    Args:
        product: Product to generate descriptor for
        
    Returns:
        Optional[str]: Generated descriptor or None
    """
    generator = get_descriptor_generator()
    return await generator.generate_descriptor(product)


async def generate_product_sizing(product: Product, sizing_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate sizing information for a single product
    
    Args:
        product: Product to generate sizing for
        sizing_data: Sizing chart/information
        
    Returns:
        Optional[Dict]: Generated sizing information or None
    """
    generator = get_descriptor_generator()
    return await generator.generate_sizing(product, sizing_data) 