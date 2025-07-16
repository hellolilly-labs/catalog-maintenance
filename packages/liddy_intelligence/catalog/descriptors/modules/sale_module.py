"""Sale and discount emphasis module for descriptors."""

import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from liddy.models.product import Product
from ..base import BaseDescriptorModule

logger = logging.getLogger(__name__)


class SaleModule(BaseDescriptorModule):
    """Module for emphasizing sales, discounts, and promotional pricing in descriptors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize sale module."""
        super().__init__(config)
        self.min_discount_threshold = config.get('min_discount_threshold', 10) if config else 10
    
    @property
    def name(self) -> str:
        return "sale"
    
    @property
    def priority(self) -> int:
        """Run after price module to enhance sale information."""
        return 20
    
    def is_applicable(self, product: Product) -> bool:
        """Only applicable to products with active sales."""
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        
        if not sale_price or not original_price or sale_price == original_price:
            return False
        
        # Calculate discount percentage
        try:
            sale_val = float(sale_price.replace('$', '').replace(',', ''))
            orig_val = float(original_price.replace('$', '').replace(',', ''))
            discount_pct = ((orig_val - sale_val) / orig_val) * 100
            return discount_pct >= self.min_discount_threshold
        except:
            return False
    
    def enhance_descriptor(self, descriptor: str, product: Product, **kwargs) -> str:
        """Add sale emphasis to descriptor."""
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        
        try:
            sale_val = float(sale_price.replace('$', '').replace(',', ''))
            orig_val = float(original_price.replace('$', '').replace(',', ''))
            discount_pct = ((orig_val - sale_val) / orig_val) * 100
            savings = orig_val - sale_val
        except:
            return descriptor
        
        # Create sale emphasis based on discount level
        sale_emphasis = self._generate_sale_emphasis(discount_pct, savings, sale_price)
        
        # Insert sale emphasis at the beginning of the descriptor (after title if present)
        lines = descriptor.split('\n')
        insert_index = 0
        
        # Skip past any title line (usually first line if it doesn't start with special chars)
        if lines and not lines[0].strip().startswith(('*', '#', '-', '!')):
            insert_index = 1
        
        # Insert the sale emphasis
        lines.insert(insert_index, f"\n{sale_emphasis}\n")
        
        return '\n'.join(lines)
    
    def enhance_search_keywords(self, keywords: List[str], product: Product, **kwargs) -> List[str]:
        """Add sale-specific search keywords."""
        updated_keywords = keywords.copy() if keywords else []
        
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        
        try:
            sale_val = float(sale_price.replace('$', '').replace(',', ''))
            orig_val = float(original_price.replace('$', '').replace(',', ''))
            discount_pct = ((orig_val - sale_val) / orig_val) * 100
            savings = orig_val - sale_val
        except:
            return updated_keywords
        
        # Add discount-specific keywords
        sale_keywords = [
            "clearance",
            "deal",
            "bargain",
            f"{int(discount_pct)}% off",
            f"save ${int(savings)}",
            "limited time"
        ]
        
        if discount_pct >= 50:
            sale_keywords.extend([
                "half off",
                "50% discount",
                "mega sale",
                "deep discount"
            ])
        elif discount_pct >= 30:
            sale_keywords.extend([
                "big sale",
                "major discount",
                "great deal"
            ])
        elif discount_pct >= 20:
            sale_keywords.extend([
                "good deal",
                "nice discount"
            ])
        
        # Add seasonal/event keywords if configured
        if self.config and 'sale_event' in self.config:
            event = self.config['sale_event']
            sale_keywords.extend([
                event.lower(),
                f"{event.lower()} sale",
                f"{event.lower()} deals"
            ])
        
        # Add keywords that aren't already present
        for keyword in sale_keywords:
            if keyword and keyword not in updated_keywords:
                updated_keywords.append(keyword)
        
        return updated_keywords
    
    def get_metadata(self, product: Product, **kwargs) -> Dict[str, Any]:
        """Return sale-related metadata."""
        sale_price = product.sale_price if hasattr(product, 'sale_price') else product.salePrice
        original_price = product.original_price if hasattr(product, 'original_price') else product.originalPrice
        
        try:
            sale_val = float(sale_price.replace('$', '').replace(',', ''))
            orig_val = float(original_price.replace('$', '').replace(',', ''))
            discount_pct = ((orig_val - sale_val) / orig_val) * 100
            savings = orig_val - sale_val
            
            return {
                'discount_percentage': round(discount_pct, 1),
                'savings_amount': round(savings, 2),
                'sale_tier': self._get_sale_tier(discount_pct)
            }
        except:
            return {}
    
    def _generate_sale_emphasis(self, discount_pct: float, savings: float, sale_price: str) -> str:
        """Generate sale emphasis text based on discount level."""
        if discount_pct >= 50:
            return (
                f"🔥 **MEGA SALE - {discount_pct:.0f}% OFF!** 🔥\n"
                f"Incredible savings of ${savings:.0f} on this premium product! "
                f"Now just {sale_price} for a limited time. This exceptional discount "
                f"won't last long - grab this amazing deal while you can!"
            )
        elif discount_pct >= 30:
            return (
                f"⭐ **BIG SALE - Save {discount_pct:.0f}%!** ⭐\n"
                f"Enjoy ${savings:.0f} off the regular price! Now available for just {sale_price}. "
                f"This significant discount makes premium quality more accessible than ever."
            )
        elif discount_pct >= 20:
            return (
                f"💰 **ON SALE - {discount_pct:.0f}% Discount** 💰\n"
                f"Save ${savings:.0f} with our current promotion. Special price of {sale_price} "
                f"for a limited time."
            )
        else:
            return (
                f"✨ **Special Offer - Save ${savings:.0f}** ✨\n"
                f"Currently available at the reduced price of {sale_price}."
            )
    
    def _get_sale_tier(self, discount_pct: float) -> str:
        """Categorize sale by discount percentage."""
        if discount_pct >= 50:
            return "mega_sale"
        elif discount_pct >= 30:
            return "big_sale"
        elif discount_pct >= 20:
            return "moderate_sale"
        elif discount_pct >= 10:
            return "small_sale"
        else:
            return "minimal_discount"