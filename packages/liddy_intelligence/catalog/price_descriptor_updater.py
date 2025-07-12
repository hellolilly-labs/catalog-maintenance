"""
Price Descriptor Updater

This module provides functionality to:
1. Check existing product descriptors for price information
2. Update descriptors with current pricing without full regeneration
3. Handle dynamic pricing changes (sales, price updates)
"""

import re
import json
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime

from liddy.models.product import Product, DescriptorMetadata
from liddy.storage import get_account_storage_provider
from liddy_intelligence.catalog.price_statistics_analyzer import PriceStatisticsAnalyzer

logger = logging.getLogger(__name__)


class PriceDescriptorUpdater:
    """Updates product descriptors with current pricing information"""
    
    # Regex patterns to find price information in descriptors
    PRICE_PATTERNS = [
        r'price[s]?\s*(?:is|are|:)?\s*\$?[\d,]+(?:\.\d{2})?',
        r'\$[\d,]+(?:\.\d{2})?',
        r'costs?\s*\$?[\d,]+(?:\.\d{2})?',
        r'priced at\s*\$?[\d,]+(?:\.\d{2})?',
        r'available for\s*\$?[\d,]+(?:\.\d{2})?',
        r'on sale for\s*\$?[\d,]+(?:\.\d{2})?',
        r'originally\s*\$?[\d,]+(?:\.\d{2})?',
        r'sale price[s]?\s*(?:is|are|:)?\s*\$?[\d,]+(?:\.\d{2})?',
    ]
    
    # Template for price information to add to descriptors
    PRICE_TEMPLATE = """

**Pricing:**
{price_info}
"""
    
    def __init__(self, account: str):
        self.account = account
        self.storage_provider = get_account_storage_provider()
        self.price_stats = None  # Will be calculated on first use
        self.terminology_research = None  # Will be loaded if available
        self.semantic_phrases = None  # Will be generated from stats
    
    def check_descriptor_has_price(self, descriptor: str, product: Product) -> bool:
        """
        Check if a descriptor contains the actual product price
        
        Args:
            descriptor: Product descriptor text
            product: Product object with price information
            
        Returns:
            bool: True if actual price values are found in descriptor
        """
        if not descriptor:
            return False
            
        # Extract actual price values from the product
        prices_to_check = []
        
        if product.originalPrice:
            # Clean the price string and add variations
            original_price = product.originalPrice.strip()
            prices_to_check.append(original_price)
            # Also check without dollar sign
            if original_price.startswith('$'):
                prices_to_check.append(original_price[1:])
        
        if product.salePrice and product.salePrice != product.originalPrice:
            sale_price = product.salePrice.strip()
            prices_to_check.append(sale_price)
            if sale_price.startswith('$'):
                prices_to_check.append(sale_price[1:])
        
        # Check if any of the actual prices appear in the descriptor
        for price in prices_to_check:
            if price in descriptor:
                return True
                
        # Also check for the price range string (e.g., "$1,000 - $2,000")
        price_range = Product.get_product_price_range_string(product)
        if price_range and price_range in descriptor:
            return True
            
        return False
    
    def extract_price_from_product(self, product: Product) -> str:
        """
        Extract formatted price information from a product
        
        Args:
            product: Product object
            
        Returns:
            str: Formatted price information
        """
        price_parts = []
        
        # Handle sale pricing
        if product.salePrice and product.salePrice != product.originalPrice:
            # Parse prices to compare them numerically
            try:
                sale_value = float(product.salePrice.replace('$', '').replace(',', ''))
                orig_value = float(product.originalPrice.replace('$', '').replace(',', ''))
                discount_pct = ((orig_value - sale_value) / orig_value) * 100
                
                price_parts.append(f"On sale for {product.salePrice} (save {discount_pct:.0f}%)")
                price_parts.append(f"Originally {product.originalPrice}")
            except:
                # Fallback if parsing fails
                price_parts.append(f"Sale Price: {product.salePrice}")
                price_parts.append(f"Original Price: {product.originalPrice}")
        else:
            # Regular pricing
            price_parts.append(f"Price: {product.originalPrice}")
        
        # Add price range for context
        price_range = Product.get_product_price_range_string(product)
        if price_range:
            price_parts.append(f"Price range: {price_range}")
        
        return '\n'.join(price_parts)
    
    def update_descriptor_with_price(self, descriptor: str, product: Product, 
                                   price_stats: Optional[Dict[str, float]] = None) -> str:
        """
        Update a descriptor with current price information and semantic context
        
        Args:
            descriptor: Current product descriptor
            product: Product object with current pricing
            price_stats: Optional price statistics for semantic context
            
        Returns:
            str: Updated descriptor with price information and context
        """
        # Get basic price info
        price_info = self.extract_price_from_product(product)
        
        # Add semantic price context based on statistics
        if price_stats:
            semantic_context = self._generate_semantic_price_context(product, price_stats)
            if semantic_context:
                price_info += f"\n\n{semantic_context}"
        
        # Check if descriptor already has a pricing section
        if re.search(r'\*\*Pricing:?\*\*', descriptor):
            # Replace existing pricing section
            pattern = r'\*\*Pricing:?\*\*.*?(?=\n\*\*|\n\n|\Z)'
            replacement = f"**Pricing:**\n{price_info}"
            updated = re.sub(pattern, replacement, descriptor, flags=re.DOTALL)
        else:
            # Add pricing section
            # Try to add it before the search terms or at the end
            if '**Search Terms:**' in descriptor:
                parts = descriptor.split('**Search Terms:**')
                updated = parts[0].rstrip() + self.PRICE_TEMPLATE.format(price_info=price_info) + '\n**Search Terms:**' + parts[1]
            else:
                updated = descriptor.rstrip() + self.PRICE_TEMPLATE.format(price_info=price_info)
        
        return updated
    
    def _generate_semantic_price_context(self, product: Product, price_stats: Dict[str, Any]) -> str:
        """
        Generate semantic context about the product's price positioning
        
        Args:
            product: Product object
            price_stats: Price statistics for the catalog (now includes category stats)
            
        Returns:
            str: Semantic price context
        """
        # Get current price
        price_str = product.salePrice or product.originalPrice
        if not price_str:
            return ""
            
        try:
            price = float(price_str.replace('$', '').replace(',', ''))
        except:
            return ""
        
        context_parts = []
        
        # Check if we have category-specific stats
        category = product.categories[0] if product.categories else None
        category_stats = None
        
        if isinstance(price_stats, dict) and 'by_category' in price_stats:
            # We have the full stats from PriceStatisticsAnalyzer
            overall_stats = price_stats['overall']
            category_stats = price_stats['by_category'].get(category) if category else None
            semantic_phrases = price_stats.get('semantic_phrases', {})
        else:
            # Legacy format - just overall stats
            overall_stats = price_stats
            semantic_phrases = {}
        
        # Use category-specific stats if available
        stats_to_use = category_stats if category_stats else overall_stats
        
        # Determine price tier with better handling of skewed distributions
        if stats_to_use.get('is_multimodal') and 'price_clusters' in stats_to_use:
            # Multi-modal distribution - use clusters
            clusters = stats_to_use['price_clusters']
            for i, cluster in enumerate(clusters):
                if cluster['min'] <= price <= cluster['max']:
                    if i == len(clusters) - 1:  # Top cluster
                        context_parts.append(self._get_premium_context(semantic_phrases.get('premium', [])))
                    elif i == 0:  # Bottom cluster
                        context_parts.append(self._get_budget_context(semantic_phrases.get('budget', [])))
                    else:  # Middle clusters
                        # Determine which mid-tier based on position
                        if i > len(clusters) / 2:
                            context_parts.append(self._get_mid_high_context(semantic_phrases.get('mid_high', [])))
                        else:
                            context_parts.append(self._get_mid_low_context(semantic_phrases.get('mid_low', [])))
                    break
        else:
            # Use adaptive thresholds based on distribution
            thresholds = self._get_adaptive_thresholds(stats_to_use)
            
            if price >= thresholds['premium']:
                context_parts.append(self._get_premium_context(semantic_phrases.get('premium', [])))
            elif price >= thresholds['mid_high']:
                context_parts.append(self._get_mid_high_context(semantic_phrases.get('mid_high', [])))
            elif price >= thresholds['mid_low']:
                context_parts.append(self._get_mid_low_context(semantic_phrases.get('mid_low', [])))
            elif price >= thresholds['budget']:
                context_parts.append(self._get_affordable_context(semantic_phrases.get('budget', [])))
            else:
                context_parts.append(self._get_budget_context(semantic_phrases.get('budget', [])))
        
        # Add sale context if applicable
        if product.salePrice and product.salePrice != product.originalPrice:
            try:
                sale_price = float(product.salePrice.replace('$', '').replace(',', ''))
                orig_price = float(product.originalPrice.replace('$', '').replace(',', ''))
                discount_pct = ((orig_price - sale_price) / orig_price) * 100
                
                if discount_pct >= 30:
                    context_parts.append(
                        f"Currently offered at an exceptional {discount_pct:.0f}% discount, "
                        "this special pricing represents outstanding value."
                    )
                elif discount_pct >= 15:
                    context_parts.append(
                        f"Now available with a {discount_pct:.0f}% discount, making this "
                        "premium product more accessible than ever."
                    )
            except:
                pass
        
        return " ".join(context_parts)
    
    def _get_adaptive_thresholds(self, stats: Dict[str, float]) -> Dict[str, float]:
        """Get adaptive price thresholds based on distribution characteristics"""
        # Check if we have custom thresholds from the analyzer
        if all(key in stats for key in ['budget_threshold', 'mid_low_threshold', 
                                         'mid_high_threshold', 'premium_threshold']):
            return {
                'budget': stats['budget_threshold'],
                'mid_low': stats['mid_low_threshold'],
                'mid_high': stats['mid_high_threshold'],
                'premium': stats['premium_threshold']
            }
        
        # Fallback to percentiles
        return {
            'budget': stats.get('p25', 0),
            'mid_low': stats.get('p50', 0),
            'mid_high': stats.get('p75', 0),
            'premium': stats.get('p95', float('inf'))
        }
    
    def _get_premium_context(self, custom_phrases: List[str] = None) -> str:
        """Get premium tier context with optional custom phrases"""
        
        if custom_phrases and len(custom_phrases) > 0:
            # Use brand-specific terminology
            # Clean and format the phrases
            phrases = [p.strip() for p in custom_phrases[:3] if p.strip()]
            
            if len(phrases) >= 2:
                # Multiple terms - use the most impactful ones
                context = (
                    f"This {phrases[0]}-level product represents the pinnacle of our {phrases[1]} collection. "
                    f"It features premium materials, advanced technology, and exceptional performance "
                    f"that serious enthusiasts and professionals expect from top-tier equipment."
                )
            elif len(phrases) == 1:
                # Single term
                context = (
                    f"As a {phrases[0]} model, this represents our premium offering with "
                    f"the highest quality materials, cutting-edge technology, and exceptional performance. "
                    f"It's designed for discerning customers who demand the very best."
                )
            else:
                # Fallback if no valid phrases
                context = self._get_default_premium_context()
        else:
            context = self._get_default_premium_context()
        
        return context
    
    def _get_default_premium_context(self) -> str:
        """Default premium context when no custom phrases available"""
        return (
            "This represents our premium offering, featuring the highest quality materials, "
            "cutting-edge technology, and exceptional performance. "
            "It's designed for professionals and enthusiasts who demand the very best."
        )
    
    def _get_budget_context(self, custom_phrases: List[str] = None) -> str:
        """Get budget tier context with optional custom phrases"""
        
        if custom_phrases and len(custom_phrases) > 0:
            # Use brand-specific terminology
            phrases = [p.strip() for p in custom_phrases[:3] if p.strip()]
            
            if len(phrases) >= 2:
                # Multiple terms
                context = (
                    f"This {phrases[0]} model offers {phrases[1]}-level features at an accessible price point. "
                    f"It provides reliable performance and essential functionality, making quality "
                    f"available to beginners and budget-conscious buyers."
                )
            elif len(phrases) == 1:
                # Single term
                context = (
                    f"As an {phrases[0]} option, this product makes quality accessible without breaking the bank. "
                    f"It delivers solid performance and essential features at an attractive price point."
                )
            else:
                # Fallback
                context = self._get_default_budget_context()
        else:
            context = self._get_default_budget_context()
        
        return context
    
    def _get_default_budget_context(self) -> str:
        """Default budget context when no custom phrases available"""
        return (
            "This value-focused option makes quality accessible without breaking the bank. "
            "Perfect for beginners or budget-conscious buyers seeking reliable performance."
        )
    
    def _get_mid_high_context(self, custom_phrases: List[str] = None) -> str:
        """Get mid-high tier context"""
        if custom_phrases and len(custom_phrases) > 0:
            phrases = [p.strip() for p in custom_phrases[:2] if p.strip()]
            if phrases:
                return (
                    f"This {phrases[0]}-grade product offers professional-level features and performance "
                    f"at a competitive price. It's an excellent choice for serious users who want "
                    f"high quality without the flagship price tag."
                )
        
        return (
            "This product offers professional-level features and performance at a competitive price. "
            "It's an excellent choice for serious users who want high quality without the flagship price tag."
        )
    
    def _get_mid_low_context(self, custom_phrases: List[str] = None) -> str:
        """Get mid-low tier context"""
        if custom_phrases and len(custom_phrases) > 0:
            phrases = [p.strip() for p in custom_phrases[:2] if p.strip()]
            if phrases:
                return (
                    f"This {phrases[0]} option delivers solid performance and good value. "
                    f"It includes many desirable features while maintaining an accessible price point, "
                    f"perfect for enthusiasts stepping up from entry-level equipment."
                )
        
        return (
            "This well-balanced option delivers solid performance and good value. "
            "It includes many desirable features while maintaining an accessible price point."
        )
    
    def _get_affordable_context(self, custom_phrases: List[str] = None) -> str:
        """Get affordable tier context"""
        if custom_phrases and len(custom_phrases) > 0:
            phrases = [p.strip() for p in custom_phrases[:2] if p.strip()]
            if phrases:
                return (
                    f"This {phrases[0]} option provides reliable performance for everyday use. "
                    f"It's a smart choice for those seeking quality on a budget, offering "
                    f"the essential features you need without unnecessary extras."
                )
        
        return (
            "This affordable option provides reliable performance for everyday use. "
            "It's a smart choice for those seeking quality on a budget."
        )
    
    async def _load_terminology_research(self, auto_run: bool = True, force_refresh: bool = False) -> Optional[Dict]:
        """Load terminology research, optionally running it if missing
        
        Args:
            auto_run: If True, automatically run the researcher if missing
        """
        try:
            research_content = None
            
            # Check if research exists
            if not force_refresh:
                try:
                    research_content = await self.storage_provider.get_research_data(account=self.account, research_type="industry_terminology")
                except Exception as e:
                    logger.debug(f"Terminology research not found: {e}")
                    research_content = None
            
            # If no research content, either run it or warn
            if research_content is None:
                if auto_run:
                    logger.info(f"📚 No terminology research found. Running it now for {self.account}...")
                    
                    # Import and run the researcher
                    from liddy_intelligence.research.industry_terminology_researcher import IndustryTerminologyResearcher
                    
                    researcher = IndustryTerminologyResearcher(
                        brand_domain=self.account
                    )
                    
                    try:
                        result = await researcher.research(force_refresh=True)
                        if result.get("success") or result.get("content"):
                            logger.info(f"✅ Terminology research completed")
                        else:
                            logger.error(f"Failed to run terminology research: {result.get('error', 'Unknown error')}")
                            return None
                        logger.info(f"✅ Terminology research completed")
                        # Now try loading again
                        research_content = result.get("content")
                    except Exception as e:
                        logger.error(f"Failed to run terminology research: {e}")
                        return None
                else:
                    logger.warning(
                        f"⚠️  No terminology research found. Price categorization will use generic terms. "
                        f"Run: python run/research_industry_terminology.py {self.account}"
                    )
                    return None
            
            # Parse the research to extract key terms
            # This is a simplified parser - in production you'd want more robust parsing
            terminology_data = {
                'price_terminology': {
                    'premium_terms': [],
                    'mid_terms': [],
                    'budget_terms': []
                },
                'brand_specific_tiers': {
                    'premium_indicators': [],
                    'mid_indicators': [],
                    'budget_indicators': []
                }
            }
            
            # Extract terms from the research content
            lines = research_content.split('\n')
            current_section = None
            in_brand_specific = False
            
            for line in lines:
                # Check for brand-specific sections FIRST (more specific)
                if 'Brand-Specific Premium Indicators' in line:
                    current_section = 'premium'
                    in_brand_specific = True
                elif 'Brand-Specific Budget Indicators' in line:
                    current_section = 'budget'
                    in_brand_specific = True
                elif 'Brand-Specific Mid' in line:
                    current_section = 'mid'
                    in_brand_specific = True
                # Then check for main sections (less specific)
                elif 'Premium/High-End Indicators' in line:
                    current_section = 'premium'
                    in_brand_specific = False
                elif 'Budget/Entry-Level Indicators' in line:
                    current_section = 'budget'
                    in_brand_specific = False
                elif 'Mid-Range Terms' in line or 'Mid-Tier' in line or 'Mid-Range Indicators' in line:
                    current_section = 'mid'
                    in_brand_specific = False
                
                # Extract terms from bullet points
                elif line.strip().startswith('- ') and current_section:
                    # Handle different bullet formats
                    if '**' in line:
                        # Format: - **term**: description
                        parts = line.split('**')
                        if len(parts) >= 3:
                            term = parts[1].strip().lower()
                        else:
                            continue
                    elif ':' in line:
                        # Format: - term: description
                        term = line.split(':')[0].replace('-', '').strip().lower()
                    else:
                        # Format: - term
                        term = line.replace('-', '').strip().lower()
                    
                    # Add to appropriate list
                    if term and len(term) > 1:
                        if in_brand_specific:
                            if current_section == 'premium':
                                terminology_data['brand_specific_tiers']['premium_indicators'].append(term)
                            elif current_section == 'mid':
                                terminology_data['brand_specific_tiers']['mid_indicators'].append(term)
                            else:
                                terminology_data['brand_specific_tiers']['budget_indicators'].append(term)
                        else:
                            if current_section == 'premium':
                                terminology_data['price_terminology']['premium_terms'].append(term)
                            elif current_section == 'mid':
                                terminology_data['price_terminology']['mid_terms'].append(term)
                            else:
                                terminology_data['price_terminology']['budget_terms'].append(term)
            
            logger.info(f"Loaded terminology research with {len(terminology_data['price_terminology']['premium_terms'])} premium terms")
            return terminology_data
            
        except Exception as e:
            logger.warning(f"Could not load terminology research: {e}")
            return None
    
    def calculate_price_statistics(self, products: List[Product]) -> Dict[str, Any]:
        """
        Calculate price distribution statistics using the enhanced analyzer
        
        Args:
            products: List of products to analyze
            
        Returns:
            Dict with overall stats, category stats, and semantic phrases
        """
        # Use the enhanced analyzer
        stats = PriceStatisticsAnalyzer.analyze_catalog_pricing(products, self.terminology_research)
        
        # Log the analysis results
        overall = stats.get('overall', {})
        logger.info(f"Price statistics for {self.account}: min=${overall.get('min', 0):.2f}, "
                   f"p25=${overall.get('p25', 0):.2f}, p50=${overall.get('p50', 0):.2f}, "
                   f"p75=${overall.get('p75', 0):.2f}, p95=${overall.get('p95', 0):.2f}, "
                   f"max=${overall.get('max', 0):.2f}")
        
        if stats.get('recommendations', {}).get('warnings'):
            for warning in stats['recommendations']['warnings']:
                logger.warning(f"Pricing analysis warning: {warning}")
        
        return stats
    
    def get_price_category_keywords(self, price: float, stats: Dict[str, Any], product: Optional[Product] = None) -> List[str]:
        """
        Get dynamic price category keywords based on catalog statistics
        
        Args:
            price: Product price
            stats: Price statistics for the catalog (can be overall stats or full analyzer output)
            product: Optional product for category-specific keywords
            
        Returns:
            List of relevant price keywords
        """
        keywords = []
        
        # Handle both legacy stats format and new analyzer format
        if isinstance(stats, dict) and 'overall' in stats:
            # New format from PriceStatisticsAnalyzer
            overall_stats = stats['overall']
            semantic_phrases = stats.get('semantic_phrases', {})
            
            # Get category-specific stats if product provided
            category_stats = None
            if product and product.categories and 'by_category' in stats:
                category = product.categories[0]
                category_stats = stats['by_category'].get(category)
            
            # Use category stats if available, otherwise overall
            stats_to_use = category_stats if category_stats else overall_stats
        else:
            # Legacy format - just stats dict
            stats_to_use = stats
            semantic_phrases = {}
        
        if not stats_to_use:
            # Fallback to basic logic if no stats available
            if price < 100:
                keywords.extend(["budget", "affordable"])
            elif price >= 1000:
                keywords.extend(["premium", "high-end"])
            return keywords
        
        # Get adaptive thresholds
        thresholds = self._get_adaptive_thresholds(stats_to_use)
        
        # Dynamic categorization
        if price <= thresholds['budget']:
            keywords.extend([
                "budget", 
                "affordable", 
                "value", 
                "economical",
                "entry-level"
            ])
            # Add semantic phrases if available
            if 'budget' in semantic_phrases:
                keywords.extend(semantic_phrases['budget'][:3])
            # Add price point
            keywords.append(f"under ${int(thresholds['budget'] + 50)}")
            
        elif price <= thresholds['mid_low']:
            keywords.extend([
                "affordable",
                "good value",
                "mid-range"
            ])
            if 'mid_low' in semantic_phrases:
                keywords.extend(semantic_phrases['mid_low'][:2])
            keywords.append(f"under ${int(thresholds['mid_low'] + 50)}")
            
        elif price <= thresholds['mid_high']:
            keywords.extend([
                "mid-range",
                "moderate",
                "quality"
            ])
            if 'mid_high' in semantic_phrases:
                keywords.extend(semantic_phrases['mid_high'][:2])
                
        elif price >= thresholds['premium']:
            keywords.extend([
                "premium",
                "high-end",
                "luxury",
                "top-tier",
                "flagship"
            ])
            # Add semantic phrases if available
            if 'premium' in semantic_phrases:
                keywords.extend(semantic_phrases['premium'][:5])
            else:
                keywords.extend([
                    "best-in-class",
                    "top of the line",
                    "professional",
                    "expert-level"
                ])
        else:  # Between mid_high and premium
            keywords.extend([
                "quality",
                "high-quality",
                "professional",
                "advanced"
            ])
        
        # Add specific price point keywords
        # Round to nearest sensible values
        if price < 100:
            rounded = int(price / 10) * 10 + 10
        elif price < 1000:
            rounded = int(price / 50) * 50 + 50
        else:
            rounded = int(price / 100) * 100 + 100
            
        keywords.append(f"under ${rounded}")
        
        return keywords
    
    def update_search_keywords_with_price(self, keywords: List[str], product: Product, 
                                         price_stats: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Add price-related search keywords using dynamic pricing categories
        
        Args:
            keywords: Current search keywords
            product: Product object
            price_stats: Optional pre-calculated price statistics
            
        Returns:
            List[str]: Updated keywords with price terms
        """
        updated_keywords = keywords.copy() if keywords else []
        
        # Use provided stats or use cached stats
        if price_stats:
            self.price_stats = price_stats
        
        # Extract price values
        price_keywords = []
        
        # Get the current price (sale or original)
        current_price_str = product.salePrice or product.originalPrice
        if current_price_str:
            try:
                current_price = float(current_price_str.replace('$', '').replace(',', ''))
                
                # Get dynamic category keywords
                if self.price_stats:
                    category_keywords = self.get_price_category_keywords(current_price, self.price_stats, product)
                    price_keywords.extend(category_keywords)
                else:
                    # Fallback to simple categorization
                    if current_price < 100:
                        price_keywords.extend(["budget", "affordable", "under $100"])
                    elif current_price < 500:
                        price_keywords.extend(["mid-range", "under $500"])
                    elif current_price < 1000:
                        price_keywords.extend(["under $1000"])
                    else:
                        price_keywords.extend(["premium", "high-end"])
                
                # Add specific price point
                price_keywords.append(f"${int(current_price)}")
                
                # Add sale-specific keywords if on sale
                if product.salePrice and product.salePrice != product.originalPrice:
                    price_keywords.extend([
                        "on sale",
                        "discount",
                        "reduced",
                        "special offer",
                        "sale price"
                    ])
                    
            except Exception as e:
                logger.warning(f"Error processing price for product {product.id}: {e}")
        
        # Add price keywords that aren't already present
        for keyword in price_keywords:
            if keyword and keyword not in updated_keywords:
                updated_keywords.append(keyword)
        
        return updated_keywords[:40]  # Increased limit to accommodate more dynamic keywords
    
    async def check_and_update_products(self, products: Optional[List[Product]] = None, force_refresh: bool = False) -> Dict[str, int]:
        """
        Check all products and update descriptors with price information
        
        Args:
            products: Optional list of products to check. If None, loads all products.
            
        Returns:
            Dict with statistics: updated_count, already_had_price, total_checked
        """
        stats = {
            "updated_count": 0,
            "already_had_price": 0,
            "total_checked": 0,
            "errors": 0
        }
        
        # Load products if not provided
        if products is None:
            try:
                products_data = await self.storage_provider.get_product_catalog(account=self.account)
                products = [Product(**p) for p in products_data]
            except Exception as e:
                logger.error(f"Failed to load products: {e}")
                return stats
        
        # Load terminology research if available
        if not self.terminology_research:
            self.terminology_research = await self._load_terminology_research(force_refresh=force_refresh)
        
        # Calculate price statistics for dynamic categorization using enhanced analyzer
        price_statistics = self.calculate_price_statistics(products)
        self.price_stats = price_statistics  # Cache for later use
        
        for product in products:
            stats["total_checked"] += 1
            
            try:
                # Check if descriptor has price
                if self.check_descriptor_has_price(product.descriptor, product):
                    stats["already_had_price"] += 1
                    
                    # Still update if it's a sale price change
                    if product.salePrice and product.salePrice != product.originalPrice:
                        # Check if the sale price is already in the descriptor
                        if product.salePrice not in product.descriptor:
                            # Update the descriptor with new sale price
                            product.descriptor = self.update_descriptor_with_price(
                                product.descriptor, product, price_statistics
                            )
                            product.search_keywords = self.update_search_keywords_with_price(
                                product.search_keywords, product, price_statistics
                            )
                            if not product.descriptor_metadata:
                                product.descriptor_metadata = DescriptorMetadata()
                            product.descriptor_metadata.price_updated_at = datetime.now().isoformat()
                            stats["updated_count"] += 1
                        else:
                            if not product.descriptor_metadata:
                                product.descriptor_metadata = DescriptorMetadata()
                            if product.descriptor_metadata.price_updated_at is None:
                                product.descriptor_metadata.price_updated_at = datetime.now().isoformat()
                                stats["updated_count"] += 1
                else:
                    # Add price information
                    product.descriptor = self.update_descriptor_with_price(
                        product.descriptor, product, price_statistics
                    )
                    product.search_keywords = self.update_search_keywords_with_price(
                        product.search_keywords, product, price_statistics
                    )
                    if not product.descriptor_metadata:
                        product.descriptor_metadata = DescriptorMetadata()
                    product.descriptor_metadata.price_updated_at = datetime.now().isoformat()
                    stats["updated_count"] += 1
                    
            except Exception as e:
                logger.error(f"Error updating product {product.id}: {e}")
                stats["errors"] += 1
        
        # Save updated products
        if stats["updated_count"] > 0:
            try:
                products_data = [p.to_dict() for p in products]
                await self.storage_provider.save_product_catalog(account=self.account, products=products_data)
                logger.info(f"Saved {stats['updated_count']} updated products")
            except Exception as e:
                logger.error(f"Failed to save updated products: {e}")
        
        return stats
    
    async def update_single_product_price(self, product_id: str, new_sale_price: Optional[str] = None, 
                                        new_original_price: Optional[str] = None) -> bool:
        """
        Update a single product's price in its descriptor
        
        Args:
            product_id: Product ID to update
            new_sale_price: New sale price (optional)
            new_original_price: New original price (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            # Load all products
            products_data = await self.storage_provider.read_json('products.json')
            products = [Product(**p) for p in products_data]
            
            # Find the product
            product = next((p for p in products if p.id == product_id), None)
            if not product:
                logger.error(f"Product {product_id} not found")
                return False
            
            # Update prices
            if new_sale_price is not None:
                product.salePrice = new_sale_price
            if new_original_price is not None:
                product.originalPrice = new_original_price
            
            # Calculate price statistics if not already cached
            if not self.price_stats:
                self.price_stats = self.calculate_price_statistics(products)
            
            # Update descriptor and keywords
            product.descriptor = self.update_descriptor_with_price(
                product.descriptor, product, self.price_stats
            )
            product.search_keywords = self.update_search_keywords_with_price(
                product.search_keywords, product, self.price_stats
            )
            
            # Add update timestamp to metadata
            if not product.descriptor_metadata:
                product.descriptor_metadata = DescriptorMetadata()
            product.descriptor_metadata.price_updated_at = datetime.now().isoformat()
            
            # Save updated products
            products_data = [p.to_dict() for p in products]
            await self.storage_provider.write_json('products.json', products_data)
            
            logger.info(f"Updated price for product {product_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update product {product_id}: {e}")
            return False
    
    async def bulk_update_sale_prices(self, price_updates: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Bulk update sale prices for multiple products
        
        Args:
            price_updates: List of dicts with 'product_id' and 'sale_price' keys
            
        Returns:
            Dict with update statistics
        """
        stats = {
            "updated": 0,
            "failed": 0,
            "not_found": 0
        }
        
        try:
            # Load all products once
            products_data = await self.storage_provider.read_json('products.json')
            products = [Product(**p) for p in products_data]
            products_by_id = {p.id: p for p in products}
            
            # Calculate price statistics for dynamic categorization
            price_statistics = self.calculate_price_statistics(products)
            
            # Apply updates
            for update in price_updates:
                product_id = update.get('product_id')
                sale_price = update.get('sale_price')
                
                if not product_id:
                    stats["failed"] += 1
                    continue
                
                product = products_by_id.get(product_id)
                if not product:
                    stats["not_found"] += 1
                    logger.warning(f"Product {product_id} not found")
                    continue
                
                try:
                    # Update price
                    product.salePrice = sale_price
                    
                    # Update descriptor and keywords
                    product.descriptor = self.update_descriptor_with_price(
                        product.descriptor, product, price_statistics
                    )
                    product.search_keywords = self.update_search_keywords_with_price(
                        product.search_keywords, product, price_statistics
                    )
                    
                    # Add update timestamp
                    if not product.descriptor_metadata:
                        product.descriptor_metadata = DescriptorMetadata()
                    product.descriptor_metadata.price_updated_at = datetime.now().isoformat()
                    
                    stats["updated"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to update product {product_id}: {e}")
                    stats["failed"] += 1
            
            # Save all updates at once
            if stats["updated"] > 0:
                products_data = [p.dict() for p in products]
                await self.storage_provider.write_json('products.json', products_data)
                logger.info(f"Saved {stats['updated']} price updates")
            
        except Exception as e:
            logger.error(f"Failed to load products for bulk update: {e}")
            
        return stats