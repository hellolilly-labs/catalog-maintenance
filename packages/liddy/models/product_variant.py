# models/product_variant.py
from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class ProductVariant:
    """
    Represents a specific variant of a product (e.g., specific size/color combination).
    Supports both legacy simple variants and full Shopify/Specialized variant data.
    """
    id: str                              # Variant SKU (e.g., "791205")
    price: Optional[str] = None          # Current price (e.g., "$29.99")
    inStock: Optional[bool] = None       # Simple in-stock flag (legacy)
    url: Optional[str] = None            # Variant-specific URL
    image: Optional[str] = None          # Primary image URL (legacy single image)
    
    # New fields for enhanced variant support
    inventoryQuantity: Optional[int] = None    # Actual stock count (e.g., 45)
    imageUrls: List[str] = field(default_factory=list)  # Multiple variant images
    gtin: Optional[str] = None           # Global Trade Item Number (barcode)
    
    # Flexible attributes for variant properties
    attributes: Dict[str, str] = field(default_factory=dict)   # size, color, etc.
    
    # Additional variant metadata
    originalPrice: Optional[str] = None  # Original price if on sale
    isDefault: bool = False              # Is this the default variant?
    productId: Optional[str] = None      # Parent product ID reference
    
    def __post_init__(self):
        """Ensure backward compatibility and data consistency"""
        # If we have a single image but no imageUrls, populate imageUrls
        if self.image and not self.imageUrls:
            self.imageUrls = [self.image]
        # If we have imageUrls but no image, set first as primary
        elif self.imageUrls and not self.image:
            self.image = self.imageUrls[0]
        
        # Convert inventory quantity to inStock flag for backward compatibility
        if self.inventoryQuantity is not None and self.inStock is None:
            self.inStock = self.inventoryQuantity > 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "price": self.price,
            "inStock": self.inStock,
            "url": self.url,
            "image": self.image,
            "inventoryQuantity": self.inventoryQuantity,
            "imageUrls": self.imageUrls,
            "gtin": self.gtin,
            "attributes": self.attributes,
            "originalPrice": self.originalPrice,
            "isDefault": self.isDefault,
            "productId": self.productId
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ProductVariant":
        """Create from dictionary, handling legacy format"""
        # Filter out None values to use defaults
        filtered_data = {k: v for k, v in data.items() if v is not None}
        return cls(**filtered_data)