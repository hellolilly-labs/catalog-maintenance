"""
Product Loader with Variant Support and Backward Compatibility

Provides a unified interface for loading products from either:
1. Legacy products.json format (flat product structure)
2. New CSV format with full variant data
3. Enhanced products.json with variants array

Maintains backward compatibility while supporting new variant-aware features.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from liddy.models.product import Product
from liddy.models.product_variant import ProductVariant
from liddy.storage import get_account_storage_provider

logger = logging.getLogger(__name__)


class ProductLoader:
    """
    Unified product loader supporting multiple formats with backward compatibility.
    """
    
    def __init__(self, account: str):
        self.account = account
        self.storage_provider = get_account_storage_provider()
    
    async def load_products(self, source: Optional[str] = None) -> List[Product]:
        """
        Load products from the best available source.
        
        Args:
            source: Optional source preference ('json', 'csv', or None for auto-detect)
            
        Returns:
            List of Product objects with variants if available
        """
        products = []
        
        if source == 'csv':
            products = await self._load_from_csv()
        elif source == 'json':
            products = await self._load_from_json()
        else:
            # Auto-detect: Default to JSON (GCP bucket), only use CSV if explicitly requested
            products = await self._load_from_json()
        
        # Ensure all products have proper variant structure
        products = self._ensure_variant_compatibility(products)
        
        logger.info(f"✅ Loaded {len(products)} products for {self.account}")
        return products
    
    async def _load_from_json(self) -> List[Product]:
        """Load products from products.json in GCP bucket"""
        try:
            logger.info(f"📚 Loading products from GCP bucket for {self.account}")
            products_data = await self.storage_provider.get_product_catalog(account=self.account)
            
            if not products_data:
                logger.warning(f"No products.json found in GCP bucket for {self.account}")
                return []
            
            # Convert to Product objects
            products = []
            for product_dict in products_data:
                try:
                    product = Product.from_dict(product_dict)
                    products.append(product)
                except Exception as e:
                    logger.error(f"Failed to parse product {product_dict.get('id')}: {e}")
            
            logger.info(f"📦 Loaded {len(products)} products from GCP bucket")
            return products
            
        except Exception as e:
            logger.error(f"Failed to load products.json from GCP: {e}")
            return []
    
    async def _load_from_csv(self) -> List[Product]:
        """Load products from CSV with variant data"""
        csv_path = await self._get_csv_path()
        
        if not csv_path:
            raise FileNotFoundError(f"No CSV file found for {self.account}")
        
        # Use appropriate parser based on brand
        if self.account == "specialized.com":
            from liddy_intelligence.ingestion.parsers.specialized_csv_parser import parse_specialized_csv
            return parse_specialized_csv(csv_path, self.account)
        else:
            # Future: Add other brand-specific parsers
            raise NotImplementedError(f"No CSV parser available for {self.account}")
    
    async def _get_csv_path(self) -> Optional[str]:
        """Find CSV file path for the account"""
        # Check local storage
        local_base = Path(f"local/account_storage/accounts/{self.account}")
        
        # Look for CSV files with common patterns
        patterns = [
            "*Hybris*.csv",
            "*product*.csv",
            "*catalog*.csv",
            "*.csv"
        ]
        
        for pattern in patterns:
            csv_files = list(local_base.glob(pattern))
            if csv_files:
                # Return the most recently modified
                csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                logger.info(f"Found CSV files: {[f.name for f in csv_files[:3]]}")
                logger.info(f"Using CSV: {csv_files[0]}")
                return str(csv_files[0])
        
        return None
    
    def _ensure_variant_compatibility(self, products: List[Product]) -> List[Product]:
        """
        Ensure all products have proper variant structure.
        
        For legacy products without variants, create a default variant
        from the product-level data.
        """
        for product in products:
            if not product.variants:
                # Create default variant from product data
                default_variant = self._create_default_variant(product)
                product.variants = [default_variant]
                
                logger.debug(f"Created default variant for legacy product {product.id}")
        
        return products
    
    def _create_default_variant(self, product: Product) -> ProductVariant:
        """Create a default variant from product-level data"""
        # Use product ID as variant ID (with suffix to distinguish)
        variant_id = f"{product.id}-default"
        
        # Determine price - access directly from attributes
        price = product._salePrice or product._originalPrice if hasattr(product, '_salePrice') else product.salePrice or product.originalPrice
        original_price = product._originalPrice if hasattr(product, '_originalPrice') else product.originalPrice
        
        # Collect attributes from product
        attributes = {}
        
        # Add sizes if available
        product_sizes = product._sizes if hasattr(product, '_sizes') else product.sizes
        if product_sizes:
            attributes['size'] = product_sizes[0] if product_sizes else 'One Size'
        
        # Add color if available
        product_colors = product._colors if hasattr(product, '_colors') else product.colors
        if product_colors:
            if isinstance(product_colors[0], dict):
                attributes['color'] = product_colors[0].get('name', 'Default')
            else:
                attributes['color'] = str(product_colors[0])
        
        # Create variant
        variant = ProductVariant(
            id=variant_id,
            productId=product.id,
            price=price,
            originalPrice=original_price,
            inStock=bool(price),  # Assume in stock if has price
            url=product.productUrl,
            image=product.imageUrls[0] if product.imageUrls else None,
            imageUrls=product.imageUrls or [],
            attributes=attributes,
            isDefault=True
        )
        
        return variant
    
    async def save_products(self, products: List[Product]) -> bool:
        """
        Save products back to storage in the enhanced format.
        
        Args:
            products: List of Product objects to save
            
        Returns:
            bool: Success status
        """
        try:
            # Convert to dict format
            products_data = [p.to_dict() for p in products]
            
            # Save to storage
            success = await self.storage_provider.save_product_catalog(
                account=self.account,
                products=products_data
            )
            
            if success:
                logger.info(f"💾 Saved {len(products)} products for {self.account}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save products: {e}")
            return False
    
    def convert_legacy_to_variants(self, legacy_products: List[Dict[str, Any]]) -> List[Product]:
        """
        Convert legacy product format to variant-aware format.
        
        This is useful for migration scripts.
        """
        products = []
        
        for legacy_dict in legacy_products:
            try:
                # Create Product object
                product = Product.from_dict(legacy_dict)
                
                # Ensure it has variants
                if not product.variants:
                    default_variant = self._create_default_variant(product)
                    product.variants = [default_variant]
                
                products.append(product)
                
            except Exception as e:
                logger.error(f"Failed to convert product {legacy_dict.get('id')}: {e}")
        
        return products


# Convenience functions
async def load_products(account: str, source: Optional[str] = None) -> List[Product]:
    """
    Load products for an account.
    
    Args:
        account: Account/brand domain
        source: Optional source preference:
            - 'json': Load from GCP bucket (default)
            - 'csv': Load from local CSV file
            - None: Load from GCP bucket (default behavior)
        
    Returns:
        List of Product objects with variants
    """
    loader = ProductLoader(account)
    return await loader.load_products(source)


async def save_products(account: str, products: List[Product]) -> bool:
    """
    Save products for an account.
    
    Args:
        account: Account/brand domain
        products: Products to save
        
    Returns:
        Success status
    """
    loader = ProductLoader(account)
    return await loader.save_products(products)