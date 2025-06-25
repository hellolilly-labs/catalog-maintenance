"""
Descriptor Generation System

LLM-powered descriptor and sizing generation for product catalogs.
Uses proven sizing instructions exactly as provided, with multi-provider routing
and vertical auto-detection for optimal quality across any brand/industry.

Key Features:
- Proven sizing instruction implementation (exact)
- OpenAI service integration with configuration management
- Vertical auto-detection (no hardcoded assumptions)
- JSON response formatting with validation
- Comprehensive error handling and logging
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.models.product import Product
from src.llm.router import LLMRouter
from src.llm.errors import LLMError, TokenLimitError, ModelNotFoundError
from configs.settings import get_settings

logger = logging.getLogger(__name__)


class DescriptorGenerator:
    """LLM-powered product descriptor and sizing generation"""
    
    def __init__(self, llm_router: Optional[LLMRouter] = None, brand_vertical: str = None):
        """
        Initialize descriptor generator with LLM router and settings
        
        Args:
            llm_router: Optional LLMRouter instance (creates new one if None)
            brand_vertical: Optional brand-level vertical (will detect if None)
        """
        self.llm_router = llm_router or LLMRouter()
        self.settings = get_settings()
        self.brand_vertical = brand_vertical
        self._vertical_cache = {}  # Cache for detected verticals
        
    async def detect_brand_vertical(self, product: Product) -> str:
        """
        LLM-based brand vertical detection
        This should be called once per brand and cached
        
        Args:
            product: Sample product from the brand
            
        Returns:
            str: Detected brand vertical
        """
        if self.brand_vertical:
            return self.brand_vertical
            
        brand_key = product.brand or "unknown"
        
        # Check cache first
        if brand_key in self._vertical_cache:
            return self._vertical_cache[brand_key]
        
        try:
            # Build analysis prompt for brand vertical detection
            brand_analysis_prompt = f"""Analyze this brand's primary vertical/industry based on product information.

Brand: {product.brand}
Product Example: {product.name}
Categories: {', '.join(product.categories) if product.categories else 'Not specified'}
Product Description: {product.description or 'Not provided'}
Product Specifications: {json.dumps(product.specifications, indent=2) if product.specifications else 'Not provided'}

Based on this information, determine the PRIMARY vertical/industry this brand operates in. 

Common verticals include: cycling, fashion, footwear, electronics, home, beauty, sports, automotive, health, food, pets, books, tools, jewelry, outdoor, baby, toys, music, art, etc.

Respond with ONLY the primary vertical name (single word or short phrase, lowercase). No explanation needed."""

            response = await self.llm_router.chat_completion(
                task="brand_research",
                system="You are a brand analysis expert. Determine brand verticals accurately based on product information.",
                messages=[{
                    "role": "user",
                    "content": brand_analysis_prompt
                }],
                max_tokens=50,
                temperature=0.1  # Low temperature for consistency
            )
            
            if response and response.get("content"):
                detected_vertical = response["content"].strip().lower()
                # Cache the result
                self._vertical_cache[brand_key] = detected_vertical
                logger.info(f"Detected brand vertical '{detected_vertical}' for {brand_key}")
                return detected_vertical
            else:
                logger.warning(f"Empty response from LLM for brand vertical detection")
                return "general"
                
        except Exception as e:
            logger.error(f"Error detecting brand vertical: {e}")
            return "general"
    
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

            response = await self.llm_router.chat_completion(
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
        Detect full vertical context for a product
        
        Args:
            product: Product to analyze
            
        Returns:
            Dict with brand_vertical and optional product_subvertical
        """
        brand_vertical = await self.detect_brand_vertical(product)
        product_subvertical = await self.detect_product_subvertical(product, brand_vertical)
        
        return {
            "brand_vertical": brand_vertical,
            "product_subvertical": product_subvertical,
            "effective_vertical": product_subvertical or brand_vertical
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
            # From router configuration: descriptor_generation -> gpt-4-turbo
            response = await self.llm_router.chat_completion(
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
            # From router configuration: sizing_analysis -> gpt-4 (superior reasoning)
            response = await self.llm_router.chat_completion(
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