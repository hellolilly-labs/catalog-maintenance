import json
from typing import List, Optional, Tuple, Dict, Any
import inspect
from datetime import datetime
from .product_variant import ProductVariant

class DescriptorMetadata:
    def __init__(self,
                 generated_at: str | None = None,
                 price_updated_at: str | None = None,
                 model: str | None = None,
                 quality_score: float | None = None,
                 quality_score_reasoning: str | None = None,
                 generator_version: str | None = None,
                 mode: str | None = None,
                 uses_research: bool | None = None):
        self.generated_at = generated_at if generated_at else datetime.now().isoformat()
        self.price_updated_at = price_updated_at if price_updated_at else None
        self.model = model
        self.quality_score = quality_score
        self.quality_score_reasoning = quality_score_reasoning
        self.generator_version = generator_version
        self.mode = mode
        self.uses_research = uses_research
    
    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "price_updated_at": self.price_updated_at,
            "model": self.model,
            "quality_score": self.quality_score,
            "quality_score_reasoning": self.quality_score_reasoning,
            "generator_version": self.generator_version,
            "mode": self.mode,
            "uses_research": self.uses_research
        }

    def to_json(self) -> str:
        """Convert the object to a JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, metadata: dict) -> "DescriptorMetadata":
        # Get the valid field names from the DescriptorMetadata constructor
        sig = inspect.signature(cls.__init__)
        valid_fields = list(sig.parameters.keys())
        
        # Filter the input dict to only include valid fields
        filtered_dict = {}
        for key, value in metadata.items():
            if key in valid_fields:
                filtered_dict[key] = value
        
        # Create the DescriptorMetadata instance with only valid fields
        return cls(**filtered_dict)

class Product:
    def __init__(self, 
                 id: str = '',
                 name: str = '',
                 categories: list[str] | None = None,
                 brand: str = '',
                 salePrice: str | None = None,
                 originalPrice: str = '',
                 productUrl: str = '',
                 imageUrls: list[str] | None = None,
                 videoUrls: list[str] | None = None,
                 colors: list[dict | str] | None = None,
                 sizes: list[str] | None = None,
                 sizing: dict | None = None,
                 sizeSpecifications: dict | None = None,
                 highlights: list[str] | None = None,
                 description: str | None = None,
                 specifications: dict | None = None,
                 updated: str | None = None,
                 created: str | None = None,
                 deleted: str | None = None,
                 imageAnalysis: list[dict] | None = None,
                 descriptor: str | None = None,
                 descriptor_metadata: DescriptorMetadata | None = None,
                 search_keywords: list[str] | None = None,
                 key_selling_points: list[str] | None = None,
                 voice_summary: str | None = None,
                 product_labels: dict | None = None,
                 year: str | None = None,
                 variants: list[ProductVariant] | None = None):
        self.id = id
        self.name = name
        self.categories = categories if categories is not None else []
        self.brand = brand
        self.salePrice = salePrice
        self.originalPrice = originalPrice
        self.productUrl = productUrl
        self.imageUrls = imageUrls if imageUrls is not None else []
        # if each imageUrl is a dict, then convert it to a list of strings
        if isinstance(self.imageUrls, list):
            for i in range(len(self.imageUrls)):
                if isinstance(self.imageUrls[i], str):
                    continue
                elif isinstance(self.imageUrls[i], list):
                    self.imageUrls[i] = ', '.join(self.imageUrls[i])
                elif isinstance(self.imageUrls[i], dict):
                    if "src" in self.imageUrls[i]:
                        self.imageUrls[i] = self.imageUrls[i]["src"]
                    elif "url" in self.imageUrls[i]:
                        self.imageUrls[i] = self.imageUrls[i]["url"]
                else:
                    self.imageUrls[i] = str(self.imageUrls[i])

        self.videoUrls = videoUrls if videoUrls is not None else []
        self.colors = colors if colors is not None else []
        self.sizes = sizes if sizes is not None else []
        self.sizing = sizing if sizing is not None else {}
        self.description = description
        self.specifications = specifications if specifications is not None else {}
        self.sizeSpecifications = sizeSpecifications if sizeSpecifications is not None else {}
        self.highlights = highlights if highlights is not None else []
        self.updated = updated
        self.created = created
        self.deleted = deleted
        self.imageAnalysis = imageAnalysis if imageAnalysis is not None else []
        self.descriptor = descriptor
        if descriptor_metadata is not None:
            if isinstance(descriptor_metadata, DescriptorMetadata):
                self.descriptor_metadata = descriptor_metadata
            else:
                self.descriptor_metadata = DescriptorMetadata(**descriptor_metadata)
        else:
            self.descriptor_metadata = DescriptorMetadata()
        self.search_keywords = search_keywords if search_keywords is not None else []
        self.key_selling_points = key_selling_points if key_selling_points is not None else []
        self.voice_summary = voice_summary if voice_summary is not None else ''
        self.product_labels = product_labels if product_labels is not None else {}
        self.year = year
        self.variants = variants if variants is not None else []
    
    def to_dict(self) -> dict:
        # Convert the object to a dictionary
        return {
            "id": self.id,
            "name": self.name,
            "categories": self.categories,
            "brand": self.brand,
            "salePrice": self.salePrice,
            "originalPrice": self.originalPrice,
            "productUrl": self.productUrl,
            "imageUrls": self.imageUrls,
            "videoUrls": self.videoUrls,
            "colors": self.colors,
            "sizes": self.sizes,
            "sizing": self.sizing,
            "sizeSpecifications": self.sizeSpecifications,
            "highlights": self.highlights,
            "description": self.description,
            "specifications": self.specifications,
            "updated": self.updated,
            "created": self.created,
            "deleted": self.deleted,
            "imageAnalysis": self.imageAnalysis,
            "descriptor": self.descriptor,
            "descriptor_metadata": self.descriptor_metadata.to_dict() if self.descriptor_metadata else None,
            "search_keywords": self.search_keywords,
            "key_selling_points": self.key_selling_points,
            "voice_summary": self.voice_summary,
            "product_labels": self.product_labels,
            "year": self.year,
            "variants": [v.to_dict() for v in self.variants] if self.variants else []
        }
    
    @classmethod
    def from_dict(cls, product: dict) -> "Product":
        # Get the valid field names from the Product constructor
        sig = inspect.signature(cls.__init__)
        valid_fields = list(sig.parameters.keys())
        
        # Filter the input dict to only include valid fields
        filtered_dict = {}
        for key, value in product.items():
            if key in valid_fields:
                if key == 'descriptor_metadata' and value is not None:
                    # Check if it's already a DescriptorMetadata object
                    if isinstance(value, DescriptorMetadata):
                        filtered_dict['descriptor_metadata'] = value
                    else:
                        filtered_dict['descriptor_metadata'] = DescriptorMetadata.from_dict(value)
                elif key == 'variants' and value is not None:
                    # Convert variant dicts to ProductVariant objects
                    if isinstance(value, list):
                        filtered_dict['variants'] = [
                            ProductVariant.from_dict(v) if isinstance(v, dict) else v
                            for v in value
                        ]
                    else:
                        filtered_dict['variants'] = value
                else:
                    filtered_dict[key] = value
        
        # Create the Product instance with only valid fields
        return cls(**filtered_dict)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "Product":
        # Map snake_case to camelCase keys
        mapping = {
            "title": "name",
            "sale_price": "salePrice",
            "original_price": "originalPrice",
            "product_url": "productUrl",
            "image_urls": "imageUrls",
            "images": "imageUrls",
            "size_specifications": "sizeSpecifications",
            "video_urls": "videoUrls",
            "image_analysis": "imageAnalysis",
            "related_products": "relatedProducts",
            "product_reference": "productReference",
            "content_type": "contentType",
        }
        for key, new_key in mapping.items():
            if key in metadata:
                metadata[new_key] = metadata[key]
                del metadata[key]
        
        if not isinstance(metadata.get('id'), str):
            metadata['id'] = f"{metadata.get('id', '')}"
        
        # handle special cases
        # "additional_details"
        if 'additional_details' in metadata:
            metadata['specifications'] = metadata.get('specifications', {})
            if isinstance(metadata['additional_details'], str):
                metadata['additional_details'] = [metadata['additional_details']]

            # if it is a list, then join it
            if isinstance(metadata['additional_details'], list):
                for i in range(len(metadata['additional_details'])):
                    if isinstance(metadata['additional_details'][i], str):
                        # split by the first ":", making the first part the key and the second part the value
                        if ':' in metadata['additional_details'][i]:
                            key, value = metadata['additional_details'][i].split(':', 1)
                            metadata['specifications'][key.strip()] = value.strip()
                        else:
                            # if there is no ":", then just add it to the list
                            metadata['specifications']["Spec"] = metadata['additional_details'][i]
                    elif isinstance(metadata['additional_details'][i], dict):
                        # if it is a dict, then add it to the specifications
                        for key, value in metadata['additional_details'][i].items():
                            metadata['specifications'][key] = value

        if 'variants' in metadata:
            metadata['size'] = metadata.get('size', [])
            if not isinstance(metadata['size'], list):
                metadata['size'] = [metadata['size']]
                
            # if there are variants, then add them to the colors and sizes
            for i in range(len(metadata['variants'])):
                # if the size is a dict, then get the value
                if isinstance(metadata['variants'][i], dict):
                    # merge into a string
                    metadata['size'].append(', '.join([f"{key}: {value}" for key, value in metadata['variants'][i].items()]))
                elif isinstance(metadata['variants'][i], list):
                    # if it is a list, then join it
                    metadata['size'].append(', '.join(metadata['variants'][i]))
                elif isinstance(metadata['variants'][i], str):
                    # if it is a string, then just add it
                    metadata['size'].append(metadata['variants'][i])
            del metadata['variants']

        # remove any keys that are not in the Product class
        keys = list(metadata.keys())
        for key in keys:
            if key not in Product.__init__.__code__.co_varnames:
                del metadata[key]

        # For specifications, if the key is purely a number, then omit it
        if 'specifications' in metadata:
            for key in list(metadata['specifications'].keys()):
                if isinstance(key, str) and key.isdigit():
                    del metadata['specifications'][key]
                elif isinstance(key, int):
                    del metadata['specifications'][key]

        return cls(**metadata)
        

    
    @staticmethod
    def get_product_price_range_string(product):
        # get the price range
        # prices may be in string format as currency, such as "$50.00"
        # so we need to convert them to float
        # remove the dollar sign and commas
        # and convert to float
        salePrice = product.salePrice
        if salePrice:
            salePrice = salePrice.replace('$', '').replace(',', '')
        originalPrice = product.originalPrice
        if originalPrice:
            originalPrice = originalPrice.replace('$', '').replace(',', '')
            
        if not originalPrice:
            if not salePrice:
                return "Price range unknown"
            else:
                originalPrice = salePrice
            
        low = float(salePrice) if salePrice else float(originalPrice)
        high = float(originalPrice) if salePrice else float(originalPrice)
        
        # set the range
        range = ''
        if low == high:
            range = f"${low}"
        elif low == 0 and high == 0:
            range = "Price range unknown"
        elif low < high:
            range = f"${low} - ${high}"
        else:
            range = f"Over ${low}"
        
        return range
    
    @staticmethod
    def to_markdown_short(product, depth=0, additional_info: List[str] | str = None):
        prehash = '#' * depth
        markdown = f"{prehash}# {product.name}\n\n"
        markdown += f"Product ID: {product.id}\n\n"
        # markdown += f"{prehash}## Product URL\n[{product.productUrl}]({product.productUrl})\n\n"
        markdown += f"{prehash}## Category\n- {', '.join(product.categories)}\n\n"
        
        # if there are additional info, then add them
        if additional_info:
            markdown += f"{prehash}## Additional Info\n"
            if isinstance(additional_info, str):
                markdown += f"- {additional_info}\n"
            elif isinstance(additional_info, list):
                for value in additional_info:
                    markdown += f"- {value}\n"
            markdown += "\n"

        if product.highlights and len(product.highlights) > 0:
            markdown += f"{prehash}## Highlights\n"
            for highlight in product.highlights:
                markdown += f"- {highlight}\n"
            markdown += "\n"
        
        if product.descriptor:
            markdown += f"{prehash}## Descriptor\n{product.descriptor}\n\n"
        elif product.description:
            # if the description is markdown, then need to make sure it fits in the markdown at this level
            description = product.description
            while description.startswith('#'):
                # if the description starts with a hash, then it is probably markdown
                # so we need to remove the hash and add a space
                description = description.replace('#', '').strip()
            markdown += f"{prehash}## Description\n{description}\n\n"

        return markdown
    
    @staticmethod
    def to_markdown(product, depth=0, obfuscatePricing=False, additional_info: List[str] | str = None):
        prehash = '#' * depth
        markdown = f"{prehash}# {product.name}\n\n"
        markdown += f"Product ID: {product.id}\n\n"
        markdown += f"{prehash}## Category\n- {', '.join(product.categories)}\n\n"
        
        if product.highlights and len(product.highlights) > 0:
            markdown += f"{prehash}## Highlights\n"
            for highlight in product.highlights:
                markdown += f"- {highlight}\n"
            markdown += "\n"
        
        # if there are additional info, then add them
        if additional_info:
            markdown += f"{prehash}## Additional Info\n"
            if isinstance(additional_info, str):
                markdown += f"- {additional_info}\n"
            elif isinstance(additional_info, list):
                for value in additional_info:
                    markdown += f"- {value}\n"
            markdown += "\n"

        # if obfuscatePricing is true, then show only the approximated price range and include something like "Price range: $50 - $100"
        if obfuscatePricing:
            priceRange = Product.get_product_price_range_string(product)
            markdown += f"{prehash}## Price\n- Price range: {priceRange}\n- See product details page for more information\n\n"
        else:
            if product.salePrice and product.salePrice != product.originalPrice:
                markdown += f"\n{prehash}## Price\n- Sale Price: {product.salePrice}\n"
                markdown += f"- Original Price: {product.originalPrice}\n"
            else:
                markdown += f"\n{prehash}## Price\n- Price: {product.originalPrice}\n"

        if product.colors and len(product.colors) > 0:
            markdown += f"\n{prehash}## Colors\n"
            for color in product.colors:
                if isinstance(color, dict):
                    # if the color is a dict, then get the value
                    color_text = "-"
                    if "name" in color:
                        color_text += f" {color['name']}"
                    if "isDefault" in color and color['isDefault']:
                        color_text += f" **Default Color**"
                    if "id" in color:
                        color_text += f" **ID**: {color['id']}"
                    markdown += color_text + "\n"
                elif isinstance(color, str):
                    markdown += f"- {color}\n"
        
        if product.sizing and 'size_chart' in product.sizing:
            markdown += f"\n{prehash}## Sizing\n"
            markdown += "- Size Chart: \n"
            if isinstance(product.sizing['size_chart'], str):
                markdown += f"  - {product.sizing['size_chart']}\n"
            elif isinstance(product.sizing['size_chart'], dict):
                for key, value in product.sizing['size_chart'].items():
                    markdown += f"  - {key}: {value}\n"
            elif isinstance(product.sizing['size_chart'], list):
                for size_chart in product.sizing['size_chart']:
                    markdown += f"  - {size_chart}\n"
            if 'fit_advice' in product.sizing:
                markdown += f"- Fit Advice: {product.sizing['fit_advice']}\n"
            markdown += "\n"
        elif product.sizes and len(product.sizes) > 0:
            markdown += f"\n{prehash}## Sizes\n- {', '.join(product.sizes)}\n"
        
        #     if product.sizeSpecifications:
        #         markdown += f"\n{prehash}### Size Specifications\n"
        #         for key, value in product.sizeSpecifications.items():
        #             if isinstance(value, str):
        #                 markdown += f"- {key}: {value}\n"
        #             elif isinstance(value, list):
        #                 markdown += f"- {key}: \n"
        #                 for spec in value:
        #                     markdown += f"  - {spec}\n"
        #             else:
        #                 # if it is a dict, then add it to the specifications
        #                 markdown += f"- {key}: \n"
        #                 for specKey, specValue in value.items():
        #                     markdown += f"  - {specKey}: {specValue}\n"
        #         markdown += "\n"

        # elif product.sizeSpecifications:
        #     markdown += f"\n{prehash}## Size Specifications\n"
        #     for key, value in product.sizeSpecifications.items():
        #         if isinstance(value, str):
        #             markdown += f"- {key}: {value}\n"
        #         elif isinstance(value, list):
        #             markdown += f"- {key}: \n"
        #             for spec in value:
        #                 markdown += f"  - {spec}\n"
        #         else:
        #             # if it is a dict, then add it to the specifications
        #             markdown += f"- {key}: \n"
        #             for specKey, specValue in value.items():
        #                 markdown += f"  - {specKey}: {specValue}\n"
        #     markdown += "\n"

        # if the description is markdown, then need to make sure it fits in the markdown at this level
        description = product.description or product.descriptor or ''
        while description.startswith('#'):
            # if the description starts with a hash, then it is probably markdown
            # so we need to remove the hash and add a space
            description = description.replace('#', '').strip()

        # while description start with "Product Details" or "Add Icon", remove it
        while description.startswith('Product Details') or description.startswith('Add Icon'):
            description = description.replace('Product Details', '').strip()
            description = description.replace('Add Icon', '').strip()
        markdown += f"{prehash}## Description\n{description}\n\n"
    
        if product.specifications:
            markdown += f"{prehash}## Technical Specifications\n"
            for key, value in product.specifications.items():
                markdown += f"{prehash}### {key}\n"
                if isinstance(value, str):
                    markdown += f"- {value}\n"
                elif isinstance(value, list):
                    for spec in value.items():
                        if isinstance(spec, str):
                            markdown += f"- {spec}\n"
                        elif isinstance(spec, dict):
                            # if it is a dict, then add it to the specifications
                            for specKey, specValue in spec.items():
                                markdown += f"- {specKey}: {specValue}\n"
                elif isinstance(value, dict):
                    # if it is a dict, then add it to the specifications
                    for specKey, specValue in value.items():
                        if isinstance(specValue, str):
                            markdown += f"- **{specKey}**: {specValue}\n"
                        elif isinstance(specValue, list):
                            markdown += f"- **{specKey}**: \n"
                            for spec in specValue:
                                if isinstance(spec, str):
                                    markdown += f"  - {spec}\n"
                                elif isinstance(spec, dict):
                                    # if it is a dict, then add it to the specifications
                                    for specKey, specValue in spec.items():
                                        markdown += f"  - {specKey}: {specValue}\n"
                        elif isinstance(specValue, dict):
                            markdown += f"- **{specKey}**: \n"
                            for specKey, specValue in specValue.items():
                                markdown += f"  - {specKey}: {specValue}\n"
        # markdown += f"\n{prehash}## Images\n"
        # product.imageUrls.forEach((url) => {
        #   markdown += `![Image](${url})\n`;
        # });
        return markdown
    
    def to_json(self) -> str:
        """Convert the object to a JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    # Variant-aware helper methods
    def price_range(self) -> Tuple[float, float]:
        """
        Get the price range across all variants.
        
        Returns:
            Tuple of (min_price, max_price) in float values
        """
        if not self.variants:
            # Fallback to legacy price fields
            try:
                price_str = self.salePrice or self.originalPrice or "0"
                price = float(price_str.replace('$', '').replace(',', ''))
                return (price, price)
            except:
                return (0.0, 0.0)
        
        prices = []
        for variant in self.variants:
            try:
                # Check sale price first, then regular price
                price_str = variant.price or "0"
                price = float(price_str.replace('$', '').replace(',', ''))
                prices.append(price)
            except:
                continue
        
        if not prices:
            return (0.0, 0.0)
        
        return (min(prices), max(prices))
    
    def get_variant_by_sku(self, sku: str) -> Optional[ProductVariant]:
        """Get a specific variant by its SKU/ID"""
        if not self.variants:
            return None
        
        for variant in self.variants:
            if variant.id == sku:
                return variant
        
        return None
    
    def get_default_variant(self) -> Optional[ProductVariant]:
        """Get the default variant for this product"""
        if not self.variants:
            return None
        
        # First try to find explicitly marked default
        for variant in self.variants:
            if variant.isDefault:
                return variant
        
        # Fallback to first variant
        return self.variants[0] if self.variants else None
    
    def get_variants_by_attribute(self, attr_name: str, attr_value: str) -> List[ProductVariant]:
        """
        Get all variants matching a specific attribute value.
        
        Args:
            attr_name: Attribute name (e.g., 'size', 'color')
            attr_value: Attribute value to match
            
        Returns:
            List of matching variants
        """
        if not self.variants:
            return []
        
        matching = []
        for variant in self.variants:
            if variant.attributes.get(attr_name) == attr_value:
                matching.append(variant)
        
        return matching
    
    def get_available_sizes(self) -> List[str]:
        """Get all unique sizes from variants"""
        if not self.variants:
            return self.sizes or []
        
        sizes = set()
        for variant in self.variants:
            if 'size' in variant.attributes:
                sizes.add(variant.attributes['size'])
        
        return sorted(list(sizes))
    
    def get_available_colors(self) -> List[str]:
        """Get all unique colors from variants"""
        if not self.variants:
            # Return legacy colors field
            return [c['name'] if isinstance(c, dict) else c for c in self.colors] if self.colors else []
        
        colors = set()
        for variant in self.variants:
            if 'color' in variant.attributes:
                colors.add(variant.attributes['color'])
        
        return sorted(list(colors))
    
    def get_total_inventory(self) -> int:
        """Get total inventory across all variants"""
        if not self.variants:
            return 0
        
        total = 0
        for variant in self.variants:
            if variant.inventoryQuantity:
                total += variant.inventoryQuantity
        
        return total
    
    def is_in_stock(self) -> bool:
        """Check if any variant is in stock"""
        if not self.variants:
            # Fallback to checking sale/original price existence
            return bool(self.salePrice or self.originalPrice)
        
        for variant in self.variants:
            if variant.inStock or (variant.inventoryQuantity and variant.inventoryQuantity > 0):
                return True
        
        return False

# if __name__ == "__main__":
#     # Example usage
#     product_ids = [
#         "4278243",
#         "218669",
#     ]
#     for productId in product_ids:
#         product = await Product.find_by_id(account="specialized.com", productId=productId)
#         print(product.to_dict())
#         print(Product.to_markdown(product))