"""
Unified Product Descriptor Generator

Consolidates all descriptor generation approaches into a single, configurable class.
Supports voice-optimized, RAG-optimized, and balanced modes with optional research integration.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
import random
from typing import Dict, Any, List, Optional, Tuple, Literal
from datetime import datetime
from dataclasses import dataclass

from liddy_intelligence.agents.catalog_filter_analyzer import CatalogFilterAnalyzer
from liddy.llm.simple_factory import LLMFactory
from liddy.prompt_manager import PromptManager
from liddy.storage import get_account_storage_provider
from liddy.models.product_manager import get_product_manager
from liddy.models.product import Product, DescriptorMetadata

# Import product catalog researcher (synthesizes all research phases)
from liddy_intelligence.research.product_catalog_research import get_product_catalog_researcher

logger = logging.getLogger(__name__)


@dataclass
class DescriptorConfig:
    """Configuration for descriptor generation"""
    use_research: bool = True
    extract_filters: bool = True
    cache_enabled: bool = True
    quality_threshold: float = 0.8
    descriptor_length: Tuple[int, int] = (100, 200)  # min, max words
    max_search_terms: int = 30
    max_selling_points: int = 5


class UnifiedDescriptorGenerator:
    """
    RAG-optimized descriptor generator with comprehensive content extraction.
    
    Generates dense, searchable product descriptors optimized for vector databases
    while also extracting search terms, key selling points, and voice summaries.
    """
    
    def __init__(self, brand_domain: str, config: Optional[DescriptorConfig] = None):
        self.brand_domain = brand_domain
        self.config = config or DescriptorConfig()
        
        # Initialize prompt manager for Langfuse integration
        self.prompt_manager = PromptManager()
        
        # Initialize filter analyzer
        self.filter_analyzer = CatalogFilterAnalyzer(brand_domain)
        
        # Initialize components
        self.storage = get_account_storage_provider()
        
        # Initialize product catalog researcher (synthesizes all brand research)
        self.product_catalog_researcher = get_product_catalog_researcher(brand_domain)
        
        # Cache for product catalog research
        self.product_catalog_intelligence = ""
        
        logger.info(f"🔧 Initialized RAG-Optimized Descriptor Generator for {brand_domain}")
        logger.info(f"   Research: {'enabled' if self.config.use_research else 'disabled'}")
        logger.info(f"   Filters: {'enabled' if self.config.extract_filters else 'disabled'}")
    
    async def process_catalog(
        self,
        force_regenerate: bool = False,
        limit: Optional[int] = None
    ) -> Tuple[List[Product], Dict[str, Any]]:
        """
        Process product catalog with configured settings.
        
        Returns:
            Tuple of (enhanced_products, filter_labels)
        """
        logger.info(f"📝 Processing descriptors for {self.brand_domain}")
        
        # Load product catalog research if enabled
        if self.config.use_research and not self.product_catalog_intelligence:
            self.product_catalog_intelligence = await self._load_product_catalog_intelligence()
        
        # Load products via ProductManager
        product_manager = await get_product_manager(self.brand_domain)
        products_data: List[Product] = await product_manager.get_products()
        
        if not products_data:
            logger.error(f"❌ No product catalog found")
            return [], {}
        
        products = products_data if isinstance(products_data, list) else products_data.get('products', [])
        logger.info(f"  📁 Loaded {len(products)} products")
        
        # Process each product
        enhanced_products: List[Product] = []
        stats = {'total': len(products), 'cached': 0, 'generated': 0, 'regenerated': 0}

        products_to_process = products
        if limit:
            # pick random limit products from products that have no descriptor
            non_descriptors = [p for p in products if not p.descriptor or not p.descriptor_metadata]
            products_to_process = random.sample(non_descriptors, min(limit, len(non_descriptors)))
        
        save_mod = 13 # save every 13 products
        includes_unsaved_products = False
        
        try:
            for i, product in enumerate(products_to_process):
                logger.info(f"  Processing {i+1}/{len(products_to_process)}: {product.name if product.name else 'Unknown'}")
                
                # Check existing descriptor
                existing_descriptor = product.descriptor if self.config.cache_enabled else None
                existing_product_labels = product.product_labels if self.config.cache_enabled else None
                quality: DescriptorMetadata = await self._assess_quality(existing_descriptor, existing_product_labels, product) if existing_descriptor else DescriptorMetadata(quality_score=0.0, quality_score_reasoning="No existing descriptor")
                
                if existing_descriptor and quality.quality_score >= self.config.quality_threshold and not force_regenerate:
                    logger.info(f"    ✅ Using cached (quality: {quality.quality_score:.2f})")
                    stats['cached'] += 1
                else:
                    if existing_descriptor:
                        logger.info(f"    🔄 Regenerating (quality: {quality.quality_score:.2f})")
                        stats['regenerated'] += 1
                    else:
                        logger.info(f"    🆕 Generating new")
                        stats['generated'] += 1
                    
                    # Generate new descriptor
                    await self._generate_descriptor(product)
                    enhanced_products.append(product)
                    
                    # save our progress
                    if (i+1) % save_mod == 0 and self.config.cache_enabled:
                        success = await product_manager.save_products(products)
                        if not success:
                            logger.warn(f"❌ Failed to save products")
                            includes_unsaved_products = True
                        else:
                            includes_unsaved_products = False
                    else:
                        includes_unsaved_products = True

        except Exception as e:
            logger.error(f"Failed to process catalog: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Save if caching enabled
            if self.config.cache_enabled and includes_unsaved_products:
                logger.info(f"  💾 Saving {len(products)} products to ProductManager")
                success = await product_manager.save_products(products)
                if success:
                    logger.info(f"  💾 Saved to ProductManager")

        # Extract filter labels if enabled
        filter_labels = {}
        if self.config.extract_filters:
            filter_labels = await self.filter_analyzer.analyze_product_catalog(products)
            logger.info(f"  📋 Extracted {len(filter_labels) - 1} filter types")
        
        logger.info(f"✅ Processing complete: {stats}")
        
        return products, filter_labels
    
    async def _generate_descriptor(self, product: Product, model: Optional[str] = None):
        """Generate descriptor based on configured mode"""
        
        # Use Product.to_markdown for better formatted product data
        product_info = Product.to_markdown(product, depth=0, obfuscatePricing=True)
        
        # Build mode-specific prompt using Langfuse versioning
        prompt_key = f"liddy/catalog/descriptor/rag_generation"
        prompt_messages = await self._build_versioned_prompt(product_info, prompt_key)
        
        try:
            # Generate via LLM
            response = await LLMFactory.chat_completion(
                task="rag_descriptor_generation",
                messages=prompt_messages,
                model=model
            )
            
            # Parse response based on mode
            result = self._parse_response(response.get('content', ''), product)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result = self._generate_fallback(product)
        
        # Update product with results
        product.descriptor = result['descriptor']
        product.search_keywords = result.get('search_terms', [])[:self.config.max_search_terms]
        product.key_selling_points = result.get('selling_points', [])[:self.config.max_selling_points]
        product.voice_summary = result.get('voice_summary', '')
        
        # Extract and store product-specific labels for filtering
        product_labels = self._extract_product_labels(product, result)
        product.product_labels = product_labels
        
        # Add metadata
        product.descriptor_metadata = {}
        quality_metadata = await self._assess_quality(result['descriptor'], product_labels, product)
        quality_metadata.generated_at = datetime.now().isoformat()
        quality_metadata.model = model
        quality_metadata.generator_version = '4.0-rag-optimized'
        quality_metadata.mode = 'rag'
        quality_metadata.uses_research = self.config.use_research
        product.descriptor_metadata = quality_metadata
        
        return
    
    async def _build_versioned_prompt(self, product_info: str, prompt_key: str) -> List[Dict[str, str]]:
        """Build versioned prompts using Langfuse prompt management"""
        
        # Try to get versioned prompt from Langfuse first
        system_prompt = self._build_improved_prompt()
        user_prompt = self._build_user_prompt()
        try:
            prompt_template = await self.prompt_manager.get_prompt(
                prompt_name=prompt_key,
                prompt_type="chat",
                prompt=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            prompts = prompt_template.prompt
            
            # Find system and user prompts from the list
            system_prompt = next((p['content'] for p in prompts if p['role'] == 'system'), system_prompt)
            user_prompt = next((p['content'] for p in prompts if p['role'] == 'user'), user_prompt)
        except Exception as e:
            logger.warning(f"Langfuse prompt retrieval failed: {e}")
                
        # Fill in template parameters using ProductCatalogResearcher output
        research_context = ""
        if self.config.use_research and self.product_catalog_intelligence:
            # Use the comprehensive product catalog intelligence that synthesizes all research phases
            research_context = f"\n# PRODUCT CATALOG INTELLIGENCE\n\n{self.product_catalog_intelligence[:8000]}\n"
        
        # Replace template parameters
        system_content = system_prompt.replace("{{mode}}", "rag")
        system_content = system_content.replace("{{focus_instructions}}", self._get_focus_instructions("rag"))
        system_content = system_content.replace("{{research_context}}", research_context)
        system_content = system_content.replace("{{product_info}}", product_info)
        system_content = system_content.replace("{{min_length}}", str(self.config.descriptor_length[0]))
        system_content = system_content.replace("{{max_length}}", str(self.config.descriptor_length[1]))
        
        # Build user prompt with structured output format
        user_content = user_prompt.replace("{{research_context}}", research_context)
        user_content = user_content.replace("{{brand_domain}}", self.brand_domain)
        user_content = user_content.replace("{{product_info}}", product_info)
        user_content = user_content.replace("{{min_length}}", str(self.config.descriptor_length[0]))
        user_content = user_content.replace("{{max_length}}", str(self.config.descriptor_length[1]))
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    def _build_improved_prompt(self) -> str:
        """Build RAG-optimized prompt template"""
        
        return """Generate comprehensive product content optimized for RAG vector databases.

Focus on RAG Optimization:
- MAXIMUM searchable content and keyword density
- All technical specifications and product variations
- Use cases, applications, and synonyms
- NO repetitive brand introductions (waste of vector space)
- NO marketing fluff - pure informational content
- Include all searchable terms users might query

CRITICAL - Avoid Repetitive Patterns:
- DO NOT start with "Meet the [Brand]..." or similar formulaic introductions
- DO NOT use repetitive brand mentions throughout
- Each descriptor should be UNIQUE and specific to the individual product
- Focus on what makes THIS specific product different from others
- Vary sentence structure and opening patterns

{{research_context}}

Product Information:
{{product_info}}

Requirements:
- Factual and comprehensive content
- Include ALL searchable terms and variations
- {{min_length}}-{{max_length}} words for main descriptor
- Natural search query matching
- Extract key selling points and search terms
- Provide concise voice summary"""
    
    def _get_focus_instructions(self, mode: str) -> str:
        """Get RAG-specific focus instructions"""
        return """Focus on RAG optimization:
- MAXIMUM searchable content and keyword density
- All technical specifications and variations
- Use cases, applications, and synonyms
- NO repetitive brand introductions (waste of vector space)
- NO marketing fluff - pure informational content
- Include all searchable terms users might query"""
    
    def _extract_product_data(self, product: Product) -> Dict[str, Any]:
        """Extract comprehensive product data"""
        data = {
            "basic": {},
            "identifiers": {},
            "colors": [],
            "description": "",
            "pricing": {},
            "specifications": {},
            "features": [],
            "benefits": [],
            "use_cases": [],
            "technical": {},
            "visual": {},
            "metadata": {}
        }
        
        # Basic fields
        for field in ["name", "title", "category", "subcategory", "brand", "collection"]:
            if hasattr(product, field) and getattr(product, field):
                data["basic"][field] = getattr(product, field)
        
        # Identifiers
        for field in ["id", "sku", "model_number", "product_code"]:
            if hasattr(product, field) and getattr(product, field):
                data["identifiers"][field] = getattr(product, field)
        
        # Description
        for field in ["description", "long_description", "product_description"]:
            if hasattr(product, field) and getattr(product, field):
                data["description"] = getattr(product, field)
                break
        
        # Pricing
        for field in ["price", "sale_price", "original_price", "msrp"]:
            if hasattr(product, field) and getattr(product, field) is not None:
                data["pricing"][field] = getattr(product, field)
        
        # Colors
        for field in ["colors", "color", "color_options"]:
            if hasattr(product, field) and getattr(product, field):
                data["colors"].extend(getattr(product, field))
        
        # # Sizes
        # for field in ["sizes", "size", "size_options"]:
        #     if field in product and getattr(product, field):
        #         data["sizes"].extend(getattr(product, field))
        
        # Specifications (dynamically detect spec-like fields)
        # Look for fields that appear to be specifications based on patterns
        for key, value in product.to_dict().items():
            key_lower = key.lower()
            # Skip basic fields we already captured
            if key in ["name", "title", "category", "price", "description", "features", "id", "sku"]:
                continue
            
            # Common specification patterns
            if any(pattern in key_lower for pattern in ["_", "-"]) or key_lower != key:
                # Likely a specification field (has underscores, hyphens, or mixed case)
                data["specifications"][key] = value
            elif isinstance(value, (int, float)) and key_lower not in ["price", "salePrice"]:
                # Numeric values are often specifications
                data["specifications"][key] = value
            elif isinstance(value, str) and len(value) < 100 and ":" not in value:
                # Short string values might be specs (color, material, etc)
                data["specifications"][key] = value
        
        # Features (multiple possible fields)
        for field in ["features", "key_features", "highlights"]:
            if hasattr(product, field):
                if isinstance(getattr(product, field), list):
                    data["features"].extend(getattr(product, field))
                elif isinstance(getattr(product, field), str):
                    data["features"].append(getattr(product, field))
        
        # Use cases
        for field in ["use_cases", "intended_use", "applications", "suitable_for"]:
            if hasattr(product, field):
                if isinstance(getattr(product, field), list):
                    data["use_cases"].extend(getattr(product, field))
                elif isinstance(getattr(product, field), str):
                    data["use_cases"].append(getattr(product, field))
        
        # Original description if exists
        for field in ["description", "long_description", "product_description"]:
            if hasattr(product, field) and getattr(product, field):
                data["basic"]["original_description"] = getattr(product, field)
                break
        
        return data
    

    
    def _build_user_prompt(self) -> str:
        """Build structured user prompt for all five components"""
        return """Generate all five components:

DESCRIPTOR:
Write a comprehensive product descriptor optimized for RAG search and retrieval. Focus on maximum searchable content, technical specifications, use cases, and keyword density. Avoid repetitive brand mentions.

SEARCH_TERMS:
List 25-30 search terms and keywords that users might query when looking for this product. Include technical terms, use cases, synonyms, and variations. Separate with commas.

KEY_POINTS:
List 3-5 key selling points that differentiate this product. Focus on unique features, benefits, and competitive advantages. One point per line.

VOICE_SUMMARY:
Write a natural, conversational description (30-50 words) for AI sales agents to use when talking about this product. Use enthusiastic but professional language, focus on benefits and user experience, and make it sound natural when spoken aloud.

PRODUCT_LABELS:
Extract structured labels for precise filtering. Create logical categories based on the product and include relevant labels. Format as JSON.

Example categories (adapt/expand as needed for this specific product):
{
  "use_cases": ["specific application 1", "specific application 2"],
  "materials": ["primary material", "secondary material"],
  "key_features": ["standout feature 1", "standout feature 2"], 
  "style_type": ["aesthetic category", "design style"],
  "target_user": ["primary audience", "secondary audience"],
  "performance_traits": ["performance aspect 1", "performance aspect 2"]
}

IMPORTANT: 
- Create categories that make sense for THIS specific product
- Use category names that are intuitive for filtering
- Include 2-4 relevant labels per category
- Add additional categories if the product warrants them (e.g., "size_options", "compatibility", "skill_level", "occasion", "season", etc.)
- Focus on labels customers would actually search or filter by"""
    
    # def _format_product_info(self, product_data: Dict[str, Any]) -> str:
    #     """Format product data for prompt"""
    #     sections = []
        
    #     # Basic info
    #     if product_data["basic"]:
    #         sections.append("=== PRODUCT ===")
    #         for k, v in product_data["basic"].items():
    #             sections.append(f"{k}: {v}")
        
    #     # Description
    #     if product_data["description"]:
    #         sections.append("=== DESCRIPTION ===")
    #         sections.append(product_data["description"])
        
    #     # Identifiers
    #     if product_data["identifiers"]:
    #         sections.append("\n=== IDENTIFIERS ===")
    #         for k, v in product_data["identifiers"].items():
    #             sections.append(f"{k}: {v}")
        
    #     # Pricing
    #     if product_data["pricing"]:
    #         sections.append("\n=== PRICING ===")
    #         for k, v in product_data["pricing"].items():
    #             sections.append(f"{k}: ${v:,.2f}" if isinstance(v, (int, float)) else f"{k}: {v}")
        
    #     # Specifications
    #     if product_data["specifications"]:
    #         sections.append("\n=== SPECIFICATIONS ===")
    #         for k, v in list(product_data["specifications"].items())[:15]:
    #             sections.append(f"{k}: {v}")
        
    #     # Colors
    #     if product_data["colors"]:
    #         sections.append("\n=== COLORS ===")
    #         for color in product_data["colors"]:
    #             sections.append(f"• {color}")
        
    #     # Features
    #     if product_data["features"]:
    #         sections.append("\n=== FEATURES ===")
    #         for feature in product_data["features"][:10]:
    #             sections.append(f"• {feature}")
        
    #     # Use cases
    #     if product_data["use_cases"]:
    #         sections.append("\n=== USE CASES ===")
    #         for use in product_data["use_cases"][:5]:
    #             sections.append(f"• {use}")
        
    #     return "\n".join(sections)
    
    def _parse_response(self, content: str, product: Product) -> Dict[str, Any]:
        """Parse LLM response"""
        result = {
            "descriptor": "",
            "search_terms": [],
            "selling_points": [],
            "voice_summary": "",
            "product_labels": {}  # Primary LLM-extracted labels
        }
        
        # attempt to use json
        try:
            # trim to first and last "{" "} "
            first_brace = content.find("{")
            last_brace = content.rfind("}")
            content = content[first_brace:last_brace+1]
            content_json = json.loads(content)
            
            result["descriptor"] = content_json["descriptor"]
            result["search_terms"] = content_json["search_terms"]
            result["selling_points"] = content_json["key_points"] if "key_points" in content_json else content_json["selling_points"]
            result["voice_summary"] = content_json["voice_summary"]
            result["product_labels"] = content_json["product_labels"] if "product_labels" in content_json else {}
            
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response as JSON: {e}")
            if "DESCRIPTOR:" in content or "DESCRIPTOR\n":
                descriptor_splitter = "DESCRIPTOR:" if "DESCRIPTOR:" in content else "DESCRIPTOR\n"
                parts = content.split(descriptor_splitter)
                desc_part = parts[1].split("SEARCH_TERMS:")[0] if "SEARCH_TERMS:" in parts[1] else parts[1]
                result["descriptor"] = desc_part.strip()
                
                if "SEARCH_TERMS:" in content or "SEARCH_TERM\n" in content:
                    search_term_splitter = "SEARCH_TERM:" if "SEARCH_TERM:" in content else "SEARCH_TERM\n"
                    terms_part = content.split(search_term_splitter)[1]
                    terms_part = terms_part.split("KEY_POINTS:")[0] if "KEY_POINTS:" in terms_part else terms_part
                    terms_part = terms_part.split("VOICE_SUMMARY:")[0] if "VOICE_SUMMARY:" in terms_part else terms_part
                    terms_part = terms_part.split("PRODUCT_LABELS:")[0] if "PRODUCT_LABELS:" in terms_part else terms_part
                    result["search_terms"] = [t.strip() for t in terms_part.strip().split(',') if t.strip()]
                
                if "KEY_POINTS:" in content or "KEY_POINTS\n" in content:
                    key_points_splitter = "KEY_POINTS:" if "KEY_POINTS:" in content else "KEY_POINTS\n"
                    points_part = content.split(key_points_splitter)[1]
                    points_part = points_part.split("VOICE_SUMMARY:")[0] if "VOICE_SUMMARY:" in points_part else points_part
                    points_part = points_part.split("PRODUCT_LABELS:")[0] if "PRODUCT_LABELS:" in points_part else points_part
                    result["selling_points"] = [p.strip() for p in points_part.strip().split('\n') if p.strip()]
                
                if "VOICE_SUMMARY:" in content or "VOICE_SUMMARY\n" in content:
                    voice_summary_splitter = "VOICE_SUMMARY:" if "VOICE_SUMMARY:" in content else "VOICE_SUMMARY\n"
                    summary_part = content.split(voice_summary_splitter)[1]
                    summary_part = summary_part.split("PRODUCT_LABELS:")[0] if "PRODUCT_LABELS:" in summary_part else summary_part
                    result["voice_summary"] = summary_part.strip()
                
                if "PRODUCT_LABELS:" in content or "PRODUCT_LABELS\n" in content:
                    labels_splitter = "PRODUCT_LABELS:" if "PRODUCT_LABELS:" in content else "PRODUCT_LABELS\n"
                    labels_part = content.split(labels_splitter)[1].strip()
                    try:
                        # Try to parse as JSON
                        result["product_labels"] = json.loads(labels_part)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse PRODUCT_LABELS as JSON: {e}")
                        result["product_labels"] = {}
        
        if not result["descriptor"]:
            raise ValueError("No descriptor found in response")
        
        # Generate missing fields if needed (always ensure all components)
        if not result["search_terms"]:
            result["search_terms"] = self._extract_search_terms(product, result["descriptor"])
        
        if not result["selling_points"]:
            result["selling_points"] = self._extract_selling_points(product, result["descriptor"])
        
        if not result["voice_summary"]:
            result["voice_summary"] = self._generate_voice_summary(product)
        
        return result
    
    async def _assess_quality(self, descriptor: Optional[str], product_labels: Optional[Dict[str, Any]], product: Product) -> DescriptorMetadata:
        """Assess descriptor quality using LLM for accurate evaluation"""
        if not descriptor:
            return DescriptorMetadata(quality_score=0.0, quality_score_reasoning="No descriptor provided")
    
        # if the product labels are not provided, then we need to extract them
        if not product_labels:
            return DescriptorMetadata(quality_score=0.0, quality_score_reasoning="No product labels provided")
        
        # Basic sanity checks first
        word_count = len(descriptor.split())
        min_len, max_len = self.config.descriptor_length
        
        if word_count < min_len * 0.5 or word_count > max_len * 2:
            return DescriptorMetadata(quality_score=0.2, quality_score_reasoning="Descriptor length outside bounds")

        # if the descritpor is current then return the stored quality score
        if product.descriptor_metadata and product.descriptor_metadata.quality_score:
            return product.descriptor_metadata

        # Use LLM for quality assessment with versioned prompt
        try:
            # Build versioned quality assessment prompt
            prompt_key = "liddy/catalog/descriptor/analysis"
            
            system_prompt = "You are a product descriptor quality assessor. Be objective and consistent."
            assessment_prompt = self._build_quality_assessment_template()
            
            # Get the versioned prompt template
            try:
                prompt_template = await self.prompt_manager.get_prompt(
                    prompt_name=prompt_key,
                    prompt_type="chat",
                    prompt=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": assessment_prompt}
                    ],
                )
                
                prompts = prompt_template.prompt
                
                # Find system and user prompts from the list
                system_prompt = next((p['content'] for p in prompts if p['role'] == 'system'), system_prompt)
                assessment_prompt = next((p['content'] for p in prompts if p['role'] == 'user'), assessment_prompt)
                
                # Fill template parameters
                assessment_prompt = assessment_prompt.replace("{{product_name}}", product.name if product.name else 'Unknown')
                assessment_prompt = assessment_prompt.replace("{{product_category}}", ', '.join(product.categories) if product.categories else 'Unknown')
                assessment_prompt = assessment_prompt.replace("{{descriptor}}", descriptor)
                assessment_prompt = assessment_prompt.replace("{{min_length}}", str(min_len))
                assessment_prompt = assessment_prompt.replace("{{max_length}}", str(max_len))
                
            except Exception as e:
                logger.warning(f"Failed to get versioned quality prompt: {e}, using fallback")
                raise

            response = await LLMFactory.chat_completion(
                task="descriptor_quality_assessment",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": assessment_prompt}
                ]
            )
            
            # Parse score from response
            content = response.get('content', '')
            score_match = re.search(r'SCORE:\s*([\d.]+)', content)
            if score_match:
                score = float(score_match.group(1))
                reasoning_match = re.search(r'JUSTIFICATION:\s*(.+)', content)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                else:
                    reasoning = "No justification provided"
                return DescriptorMetadata(
                    quality_score=score,
                    quality_score_reasoning=reasoning
                )
            
        except Exception as e:
            logger.warning(f"LLM quality assessment failed: {e}, using basic assessment")
        
        # Fallback to basic assessment
        return self._basic_quality_assessment(descriptor, product)
    
    def _build_quality_assessment_template(self) -> str:
        """Build quality assessment prompt template for Langfuse"""
        return """Assess the quality of this product descriptor on a scale of 0.0 to 1.0.

Product Name: {{product_name}}
Category: {{product_category}}
Target Length: {{min_length}}-{{max_length}} words

Descriptor:
{{descriptor}}

Assessment Criteria for RAG-Optimized Descriptors:
- Comprehensive product information coverage
- All searchable terms and variations included
- Technical specifications mentioned appropriately
- Use cases and applications covered
- Factual and detailed content
- Natural language that reads well
- Appropriate length for target range

Provide a quality score (0.0-1.0) and brief justification.
Format: SCORE: X.X
JUSTIFICATION: [brief explanation]"""
    
    def _build_fallback_quality_prompt(self, product: Product, descriptor: str, min_len: int, max_len: int) -> str:
        """Build fallback quality assessment prompt when Langfuse fails"""
        return f"""Assess the quality of this product descriptor on a scale of 0.0 to 1.0.

Product Name: {product.name if product.name else 'Unknown'}
Category: {', '.join(product.categories) if product.categories else 'Unknown'}
Target Length: {min_len}-{max_len} words

Descriptor:
{descriptor}

Assessment Criteria for RAG-Optimized Descriptors:
- Comprehensive product information coverage
- All searchable terms and variations included
- Technical specifications mentioned appropriately
- Use cases and applications covered
- Factual and detailed content
- Natural language that reads well
- Appropriate length for target range

Provide a quality score (0.0-1.0) and brief justification.
Format: SCORE: X.X
JUSTIFICATION: [brief explanation]"""
    
    def _basic_quality_assessment(self, descriptor: str, product: Product) -> DescriptorMetadata:
        """Basic quality assessment as fallback"""
        score = 0.0
        word_count = len(descriptor.split())
        
        # Length score
        min_len, max_len = self.config.descriptor_length
        if min_len <= word_count <= max_len:
            score += 0.3
        elif min_len - 20 <= word_count <= max_len + 50:
            score += 0.2
        
        # Product name mentioned
        if product.name.lower() in descriptor.lower():
            score += 0.2
        
        # Has multiple sentences
        if descriptor.count('.') >= 2:
            score += 0.2
        
        # Has some product attributes mentioned
        attribute_count = 0
        for key, value in product.to_dict().items():
            if isinstance(value, str) and len(value) < 50:
                if value.lower() in descriptor.lower():
                    attribute_count += 1
        
        if attribute_count >= 3:
            score += 0.3
        elif attribute_count >= 1:
            score += 0.1
        
        return DescriptorMetadata(
            quality_score=min(1.0, score),
            quality_score_reasoning="Basic assessment"
        )
    

    
    async def _load_product_catalog_intelligence(self) -> str:
        """Load product catalog intelligence from ProductCatalogResearcher"""
        try:
            # Load cached results from ProductCatalogResearcher
            cached_result = await self.product_catalog_researcher._load_cached_results()
            
            if cached_result and cached_result.get("content"):
                logger.info(f"✅ Loaded product catalog intelligence (quality: {cached_result.get('quality_score', 'unknown')})")
                return cached_result["content"]
            else:
                logger.warning(f"⚠️  No product catalog intelligence found - run: python brand_intelligence_pipeline.py --brand {self.brand_domain} --phase product_catalog")
                return ""
                
        except Exception as e:
            logger.warning(f"Could not load product catalog intelligence: {e}")
            return ""
    

    
    def _get_category_emphasis(self, category: str) -> str:
        """Get category-specific emphasis based on research insights"""
        
        # If we have product catalog intelligence, extract key points from it
        if self.product_catalog_intelligence:
            # Extract key differentiators from product catalog intelligence
            lines = self.product_catalog_intelligence.split('\n')[:20]  # First 20 lines likely contain key points
            differentiators = [line for line in lines if any(keyword in line.lower() for keyword in ['differentiator', 'unique', 'advantage', 'key'])]
            if differentiators:
                return f"\n{chr(10).join(differentiators[:3])}\n"
        
        # Otherwise, provide generic guidance based on product type
        category_lower = category.lower()
        
        # Generic patterns that apply broadly
        emphasis = "\nEmphasize key differentiators such as:\n"
        
        # Detect product type and suggest relevant differentiators
        if any(term in category_lower for term in ["apparel", "clothing", "hat", "cap", "shirt"]):
            emphasis += "• Fit type and sizing details\n"
            emphasis += "• Style variations and design elements\n"
            emphasis += "• Target use or occasion\n"
        
        elif any(term in category_lower for term in ["tech", "electronic", "device", "gadget"]):
            emphasis += "• Technical capabilities and compatibility\n"
            emphasis += "• Version or generation differences\n"
            emphasis += "• Key performance metrics\n"
        
        elif any(term in category_lower for term in ["beauty", "cosmetic", "skincare", "makeup"]):
            emphasis += "• Skin type or concern addressed\n"
            emphasis += "• Key ingredients or formulation\n"
            emphasis += "• Application method or coverage\n"
        
        elif any(term in category_lower for term in ["furniture", "home", "decor"]):
            emphasis += "• Dimensions and space requirements\n"
            emphasis += "• Style or aesthetic category\n"
            emphasis += "• Assembly or installation needs\n"
        
        else:
            # Generic differentiators for any product
            emphasis += "• Unique features that set this apart\n"
            emphasis += "• Specific use cases or applications\n"
            emphasis += "• Quality or performance indicators\n"
        
        return emphasis
    
    def _extract_search_terms(self, product: Product, descriptor: str) -> List[str]:
        """Extract search terms"""
        terms = set()
        
        # From product attributes
        product_dict = product.to_dict()
        
        # Add product name and ID
        if product.name:
            terms.add(product.name.lower())
            terms.update(product.name.lower().split())
        
        if product.id:
            terms.add(product.id.lower())
        
        # From categories
        for category in product.categories:
            if isinstance(category, str):
                terms.update(re.findall(r'\b\w{3,}\b', category.lower()))
        
        # From highlights/features
        for highlight in product.highlights[:10]:
            if isinstance(highlight, str):
                terms.update(re.findall(r'\b\w{3,}\b', highlight.lower()))
        
        # From descriptor
        terms.update(re.findall(r'\b\w{4,}\b', descriptor.lower()))
        
        # Remove common words
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been', 'your', 'will', 'can', 'are'}
        terms = terms - stop_words
        
        return sorted(list(terms))[:self.config.max_search_terms]
    
    def _extract_selling_points(self, product: Product, descriptor: str) -> List[str]:
        """Extract selling points"""
        points = []
        
        # Price-based selling points
        try:
            if product.salePrice:
                sale_price = float(product.salePrice.replace('$', '').replace(',', ''))
                if sale_price < 100:
                    points.append("Exceptional value")
                elif sale_price > 1000:
                    points.append("Premium quality")
                elif sale_price > 500:
                    points.append("High-quality")
                elif sale_price > 200:
                    points.append("Good value")
                elif sale_price > 100:
                    points.append("Affordable")
        except (ValueError, AttributeError):
            pass
        
        # Feature-based selling points
        for highlight in product.highlights[:3]:
            if isinstance(highlight, str):
                if "warranty" in highlight.lower():
                    points.append(f"Backed by warranty")
                elif "sustainable" in highlight.lower() or "eco" in highlight.lower():
                    points.append("Eco-friendly design")
        
        # From descriptor content
        if "innovative" in descriptor.lower():
            points.append("Cutting-edge innovation")
        if "quality" in descriptor.lower():
            points.append("Superior quality")
        
        return points[:self.config.max_selling_points]
    
    def _generate_voice_summary(self, product: Product) -> str:
        """Generate conversational voice summary for AI sales agents"""
        import re
        
        parts = []
        
        # Start with product name and category
        if product.name:
            category = product.categories[0] if product.categories else "product"
            parts.append(f"The {product.name} is a {category}")
        
        # Add key highlights/features
        if hasattr(product, 'highlights') and product.highlights:
            key_features = product.highlights[:2]  # Take first 2 highlights
            if key_features:
                parts.append(f"featuring {' and '.join(key_features).lower()}")
        
        # Add benefit or use case from description
        if hasattr(product, 'description') and product.description:
            desc = product.description[:100].lower()
            if 'perfect for' in desc or 'ideal for' in desc:
                benefit_match = re.search(r'(perfect for|ideal for) ([^.]+)', desc)
                if benefit_match:
                    parts.append(f"perfect for {benefit_match.group(2)}")
            elif 'great for' in desc:
                benefit_match = re.search(r'great for ([^.]+)', desc)
                if benefit_match:
                    parts.append(f"great for {benefit_match.group(1)}")
        
        # Add price if available
        if hasattr(product, 'salePrice') and product.salePrice:
            parts.append(f"priced at {product.salePrice}")
        
        # Combine with natural connectors (keep under 50 words)
        if len(parts) >= 2:
            result = f"{parts[0]}, {', '.join(parts[1:-1])}, and {parts[-1]}." if len(parts) > 2 else f"{parts[0]} {parts[1]}."
        elif len(parts) == 1:
            result = f"{parts[0]}."
        else:
            result = f"{product.name} is available for purchase." if product.name else "This product is available for purchase."
        
        # Ensure it's conversational length (30-50 words)
        word_count = len(result.split())
        if word_count > 50:
            # Trim to essential parts
            if product.name and product.categories:
                category = product.categories[0]
                price_part = f" priced at {product.salePrice}" if hasattr(product, 'salePrice') and product.salePrice else ""
                result = f"The {product.name} is a {category}{price_part}."
        
        return result
    

    
    def _generate_fallback(self, product: Product) -> Dict[str, Any]:
        """Generate fallback descriptor"""
        name = product.name or "This product"
        category = product.categories[0] if product.categories else "item"
        
        features = product.highlights[:3] if product.highlights else []
        feature_text = f" featuring {', '.join(features)}" if features else ""
        
        # Try to extract price
        price = 0
        try:
            if product.salePrice:
                price = float(product.salePrice.replace('$', '').replace(',', ''))
        except (ValueError, AttributeError):
            pass
        
        price_text = f" Available at ${price:,.2f}." if price > 0 else ""
        
        descriptor = f"{name} is a quality {category}{feature_text}.{price_text}"
        
        return {
            "descriptor": descriptor,
            "search_terms": self._extract_search_terms(product, descriptor),
            "selling_points": ["Quality construction", "Reliable performance"],
            "voice_summary": self._generate_voice_summary(product)
        }
    
    def _extract_product_labels(self, product: Product, generated_content: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract structured labels/attributes from product data for advanced filtering.
        
        Uses LLM-extracted labels as primary source with rule-based categorization as backup.
        Now supports dynamic categories created by the LLM based on the specific product.
        """
        # Start with empty labels - let LLM define the categories
        labels = {}
        
        # PRIMARY: Use LLM-extracted structured labels if available
        llm_labels = generated_content.get("product_labels", {})
        if llm_labels:
            logger.info(f"Using LLM-extracted product labels with {len(llm_labels)} categories")
            for category, values in llm_labels.items():
                if isinstance(values, list):
                    # Normalize and clean LLM-generated labels
                    cleaned_values = [v.strip().lower() for v in values if v and isinstance(v, str) and v.strip()]
                    if cleaned_values:  # Only add categories that have values
                        labels[category] = cleaned_values
                        logger.debug(f"Added category '{category}' with {len(cleaned_values)} labels from LLM")
        
        # BACKUP: Use rule-based categorization if LLM labels are missing or sparse
        if not llm_labels or len(labels) < 3:  # If we have fewer than 3 categories, supplement
            logger.info("Supplementing with rule-based categorization")
            
            # Define fallback category mapping for rule-based extraction
            fallback_categories = {
                "use_cases": [],
                "materials": [],
                "key_features": [],
                "style_type": [],
                "target_user": [],
                "performance_traits": []
            }
            
            # Extract from generated search keywords
            if generated_content.get('search_terms'):
                search_categories = self._categorize_search_terms_dynamic(generated_content['search_terms'])
                for category, terms in search_categories.items():
                    if category not in labels or len(labels[category]) < 2:
                        if category not in labels:
                            labels[category] = []
                        labels[category].extend(terms)
            
            # Extract from generated selling points  
            if generated_content.get('selling_points'):
                selling_categories = self._categorize_selling_points_dynamic(generated_content['selling_points'])
                for category, points in selling_categories.items():
                    if category not in labels or len(labels[category]) < 2:
                        if category not in labels:
                            labels[category] = []
                        labels[category].extend(points)
            
            # Extract from Product specifications
            if hasattr(product, 'specifications') and product.specifications:
                spec_labels = self._extract_specification_labels_dynamic(product.specifications)
                for category, values in spec_labels.items():
                    if category not in labels or len(labels[category]) < 2:
                        if category not in labels:
                            labels[category] = []
                        labels[category].extend(values)
            
            # Extract from Product highlights/features
            if hasattr(product, 'highlights') and product.highlights:
                feature_labels = self._categorize_highlights_dynamic(product.highlights)
                for category, values in feature_labels.items():
                    if category not in labels or len(labels[category]) < 2:
                        if category not in labels:
                            labels[category] = []
                        labels[category].extend(values)
        
        # Clean and deduplicate all categories
        for category in labels:
            labels[category] = list(set([label.strip().lower() for label in labels[category] if label.strip()]))
        
        # Remove empty categories
        labels = {k: v for k, v in labels.items() if v}
        
        logger.info(f"Final product labels: {len(labels)} categories with {sum(len(v) for v in labels.values())} total labels")
        
        return labels
    
    def _categorize_search_terms_dynamic(self, search_terms: List[str]) -> Dict[str, List[str]]:
        """Categorize search terms into structured labels"""
        categories = {
            "use_cases": [],
            "materials": [],
            "key_features": [],
            "style_type": [],
            "target_user": [],
            "performance_traits": []
        }
        
        # Define flexible patterns for categorization
        patterns = {
            "use_cases": ["for", "use", "activity", "sport", "riding", "racing", "commuting", "touring", "training", "workout", "exercise"],
            "materials": ["carbon", "aluminum", "steel", "titanium", "fabric", "leather", "mesh", "polymer", "cotton", "polyester", "nylon"],
            "key_features": ["feature", "technology", "system", "component", "upgrade", "accessory", "function", "capability"],
            "style_type": ["style", "design", "aesthetic", "color", "finish", "appearance", "look", "classic", "modern", "vintage"],
            "target_user": ["beginner", "professional", "advanced", "men", "women", "youth", "adult", "kids", "senior", "expert"],
            "performance_traits": ["performance", "speed", "efficiency", "comfort", "durability", "lightweight", "strong", "fast", "smooth"]
        }
        
        for term in search_terms:
            term_lower = term.lower()
            categorized = False
            
            for category, keywords in patterns.items():
                if any(keyword in term_lower for keyword in keywords):
                    categories[category].append(term)
                    categorized = True
                    break
            
            # If not categorized, add to key_features as default
            if not categorized:
                categories["key_features"].append(term)
        
        return categories
    
    def _categorize_selling_points_dynamic(self, selling_points: List[str]) -> Dict[str, List[str]]:
        """Categorize selling points into structured labels"""
        categories = {
            "use_cases": [],
            "materials": [],
            "key_features": [],
            "performance_traits": []
        }
        
        for point in selling_points:
            point_lower = point.lower()
            
            # Performance characteristics
            if any(word in point_lower for word in ["performance", "speed", "efficiency", "comfort", "lightweight", "durable"]):
                categories["performance_traits"].append(point)
            # Materials
            elif any(word in point_lower for word in ["carbon", "aluminum", "steel", "material", "construction"]):
                categories["materials"].append(point)
            # Use cases
            elif any(word in point_lower for word in ["for", "ideal", "perfect", "designed for", "suitable"]):
                categories["use_cases"].append(point)
            # Default to features
            else:
                categories["key_features"].append(point)
        
        return categories
    
    def _extract_specification_labels_dynamic(self, specifications: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract labels from Product specifications"""
        labels = {
            "materials": [],
            "technical_specs": [],
            "style_type": []
        }
        
        for key, value in specifications.items():
            key_lower = key.lower()
            value_str = str(value).lower()
            
            # Material specifications
            if any(word in key_lower for word in ["material", "construction", "fabric", "frame"]):
                labels["materials"].append(f"{key}: {value}")
            # Style/appearance specs
            elif any(word in key_lower for word in ["color", "finish", "style", "design"]):
                labels["style_type"].append(f"{key}: {value}")
            # Technical specs
            else:
                labels["technical_specs"].append(f"{key}: {value}")
        
        return labels
    
    def _categorize_highlights_dynamic(self, highlights: List[str]) -> Dict[str, List[str]]:
        """Categorize product highlights into structured labels"""
        categories = {
            "key_features": [],
            "performance_traits": [],
            "use_cases": []
        }
        
        for highlight in highlights:
            highlight_lower = highlight.lower()
            
            if any(word in highlight_lower for word in ["performance", "efficiency", "comfort", "lightweight"]):
                categories["performance_traits"].append(highlight)
            elif any(word in highlight_lower for word in ["for", "ideal", "perfect", "use"]):
                categories["use_cases"].append(highlight)
            else:
                categories["key_features"].append(highlight)
        
        return categories



# Convenience functions
async def generate_descriptors(
    brand_domain: str,
    use_research: bool = True,
    force_regenerate: bool = False,
    quality_threshold: float = 0.8,
    limit: Optional[int] = None
) -> Tuple[List[Product], Dict[str, Any]]:
    """
    Generate RAG-optimized descriptors with comprehensive content extraction.
    
    Extracts five components:
    - DESCRIPTOR: RAG-optimized product description
    - SEARCH_TERMS: Searchable keywords and terms
    - KEY_POINTS: Unique selling points
    - VOICE_SUMMARY: Conversational description for AI sales agents
    - PRODUCT_LABELS: Dynamic structured labels for filtering (LLM-generated categories)
    
    Args:
        brand_domain: Brand domain
        use_research: Whether to use brand research insights
        force_regenerate: Force regeneration of all descriptors
        quality_threshold: Minimum quality score to keep existing descriptors
        
    Returns:
        Tuple of (enhanced_products, filter_labels)
    """
    config = DescriptorConfig(
        use_research=use_research,
        quality_threshold=quality_threshold
    )
    
    generator = UnifiedDescriptorGenerator(brand_domain, config)
    results = await generator.process_catalog(force_regenerate=force_regenerate, limit=limit)
    return results