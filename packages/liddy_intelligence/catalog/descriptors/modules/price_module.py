"""Price enhancement module for descriptors."""

import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from liddy.models.product import Product, DescriptorMetadata
from liddy_intelligence.catalog.price_statistics_analyzer import PriceStatisticsAnalyzer
from ..base import BaseDescriptorModule

logger = logging.getLogger(__name__)


class PriceModule(BaseDescriptorModule):
    """Module for enhancing descriptors with price information and semantic context."""
    
    # Template for price information to add to descriptors
    PRICE_TEMPLATE = """

**Pricing:**
{price_info}
"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional price statistics and terminology."""
        super().__init__(config)
        self.price_stats = config.get('price_stats') if config else None
        self.terminology_research = config.get('terminology_research') if config else None
    
    @property
    def name(self) -> str:
        return "price"
    
    @property
    def priority(self) -> int:
        """Run early to ensure price info is available for other modules."""
        return 10
    
    def is_applicable(self, product: Product) -> bool:
        """Applicable to all products with pricing."""
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        return bool(original_price)
    
    def enhance_descriptor(self, descriptor: str, product: Product, **kwargs) -> str:
        """Add or update price information in descriptor."""
        # Get price statistics from kwargs if not in config
        price_stats = kwargs.get('price_stats') or self.price_stats
        
        # Get basic price info
        price_info = self._extract_price_from_product(product)
        
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
    
    def enhance_search_keywords(self, keywords: List[str], product: Product, **kwargs) -> List[str]:
        """Add price-related search keywords."""
        updated_keywords = keywords.copy() if keywords else []
        
        # Get price stats from kwargs if not in config
        price_stats = kwargs.get('price_stats') or self.price_stats
        
        # Extract price values
        price_keywords = []
        
        # Get the current price (sale or original)
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        current_price_str = sale_price or original_price
        
        if current_price_str:
            try:
                current_price = float(current_price_str.replace('$', '').replace(',', ''))
                
                # Get dynamic category keywords
                if price_stats:
                    category_keywords = self._get_price_category_keywords(current_price, price_stats, product)
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
                if sale_price and sale_price != original_price:
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
        
        return updated_keywords
    
    def get_metadata(self, product: Product, **kwargs) -> Dict[str, Any]:
        """Return price-related metadata."""
        metadata = {
            'price_updated_at': datetime.now().isoformat()
        }
        
        # Add price tier info
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        current_price_str = sale_price or original_price
        
        if current_price_str:
            try:
                price = float(current_price_str.replace('$', '').replace(',', ''))
                price_stats = kwargs.get('price_stats') or self.price_stats
                
                if price_stats:
                    tier = self._get_price_tier(price, price_stats, product)
                    metadata['price_tier'] = tier
                    
                metadata['has_sale'] = bool(sale_price and sale_price != original_price)
                
            except:
                pass
        
        return metadata
    
    def _extract_price_from_product(self, product: Product) -> str:
        """Extract formatted price information from a product."""
        price_parts = []
        
        # Use property access for backward compatibility
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        
        # Handle sale pricing
        if sale_price and sale_price != original_price:
            # Parse prices to compare them numerically
            try:
                sale_value = float(sale_price.replace('$', '').replace(',', ''))
                orig_value = float(original_price.replace('$', '').replace(',', ''))
                discount_pct = ((orig_value - sale_value) / orig_value) * 100
                
                price_parts.append(f"On sale for {sale_price} (save {discount_pct:.0f}%)")
                price_parts.append(f"Originally {original_price}")
            except:
                # Fallback if parsing fails
                price_parts.append(f"Sale Price: {sale_price}")
                price_parts.append(f"Original Price: {original_price}")
        else:
            # Regular pricing
            price_parts.append(f"Price: {original_price}")
        
        # Add price range for context
        price_range = Product.get_product_price_range_string(product)
        if price_range:
            price_parts.append(f"Price range: {price_range}")
        
        return '\n'.join(price_parts)
    
    def _generate_semantic_price_context(self, product: Product, price_stats: Dict[str, Any]) -> str:
        """Generate semantic context about the product's price positioning."""
        # Get current price
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        price_str = sale_price or original_price
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
        if sale_price and sale_price != original_price:
            try:
                sale_val = float(sale_price.replace('$', '').replace(',', ''))
                orig_val = float(original_price.replace('$', '').replace(',', ''))
                discount_pct = ((orig_val - sale_val) / orig_val) * 100
                
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
    
    def _get_price_tier(self, price: float, price_stats: Dict[str, Any], product: Product) -> str:
        """Determine price tier for a product."""
        # Get appropriate stats
        if isinstance(price_stats, dict) and 'by_category' in price_stats:
            category = product.categories[0] if product.categories else None
            category_stats = price_stats['by_category'].get(category) if category else None
            stats_to_use = category_stats if category_stats else price_stats['overall']
        else:
            stats_to_use = price_stats
        
        thresholds = self._get_adaptive_thresholds(stats_to_use)
        
        if price >= thresholds['premium']:
            return 'premium'
        elif price >= thresholds['mid_high']:
            return 'mid_high'
        elif price >= thresholds['mid_low']:
            return 'mid_low'
        elif price >= thresholds['budget']:
            return 'budget'
        else:
            return 'value'
    
    def _get_adaptive_thresholds(self, stats: Dict[str, float]) -> Dict[str, float]:
        """Get adaptive price thresholds based on distribution characteristics."""
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
    
    def _get_price_category_keywords(self, price: float, stats: Dict[str, Any], product: Optional[Product] = None) -> List[str]:
        """Get dynamic price category keywords based on catalog statistics."""
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
    
    # Context generation methods
    def _get_premium_context(self, custom_phrases: List[str] = None) -> str:
        """Get premium tier context with optional custom phrases."""
        if custom_phrases and len(custom_phrases) > 0:
            phrases = [p.strip() for p in custom_phrases[:3] if p.strip()]
            
            if len(phrases) >= 2:
                return (
                    f"This {phrases[0]}-level product represents the pinnacle of our {phrases[1]} collection. "
                    f"It features premium materials, advanced technology, and exceptional performance "
                    f"that serious enthusiasts and professionals expect from top-tier equipment."
                )
            elif len(phrases) == 1:
                return (
                    f"As a {phrases[0]} model, this represents our premium offering with "
                    f"the highest quality materials, cutting-edge technology, and exceptional performance. "
                    f"It's designed for discerning customers who demand the very best."
                )
        
        return (
            "This represents our premium offering, featuring the highest quality materials, "
            "cutting-edge technology, and exceptional performance. "
            "It's designed for professionals and enthusiasts who demand the very best."
        )
    
    def _get_budget_context(self, custom_phrases: List[str] = None) -> str:
        """Get budget tier context with optional custom phrases."""
        if custom_phrases and len(custom_phrases) > 0:
            phrases = [p.strip() for p in custom_phrases[:3] if p.strip()]
            
            if len(phrases) >= 2:
                return (
                    f"This {phrases[0]} model offers {phrases[1]}-level features at an accessible price point. "
                    f"It provides reliable performance and essential functionality, making quality "
                    f"available to beginners and budget-conscious buyers."
                )
            elif len(phrases) == 1:
                return (
                    f"As an {phrases[0]} option, this product makes quality accessible without breaking the bank. "
                    f"It delivers solid performance and essential features at an attractive price point."
                )
        
        return (
            "This value-focused option makes quality accessible without breaking the bank. "
            "Perfect for beginners or budget-conscious buyers seeking reliable performance."
        )
    
    def _get_mid_high_context(self, custom_phrases: List[str] = None) -> str:
        """Get mid-high tier context."""
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
        """Get mid-low tier context."""
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
        """Get affordable tier context."""
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