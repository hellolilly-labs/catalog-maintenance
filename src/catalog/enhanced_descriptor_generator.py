"""
Enhanced Product Descriptor Generator with Filter Analysis

Generates RAG-optimized product descriptors while simultaneously analyzing
the catalog to extract filter labels. This ensures consistency and efficiency
by processing the catalog only once.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from src.agents.catalog_filter_analyzer import CatalogFilterAnalyzer

logger = logging.getLogger(__name__)


class EnhancedDescriptorGenerator:
    """
    Generates enhanced product descriptors optimized for RAG while
    simultaneously extracting filter labels from the catalog.
    
    Key features:
    - RAG-optimized product text generation
    - Simultaneous filter label extraction
    - Consistent terminology across descriptors and filters
    - Single-pass catalog processing
    """
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.brand_path = Path(f"accounts/{brand_domain}")
        
        # Initialize filter analyzer for label extraction
        self.filter_analyzer = CatalogFilterAnalyzer(brand_domain)
        
        logger.info(f"🏭 Initialized Enhanced Descriptor Generator for {brand_domain}")
    
    def process_catalog(
        self, 
        catalog_data: List[Dict[str, Any]],
        descriptor_style: str = "voice_optimized"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process catalog to generate descriptors and extract filters in single pass.
        
        Args:
            catalog_data: List of product dictionaries
            descriptor_style: Style of descriptors ("voice_optimized", "detailed", "concise")
            
        Returns:
            Tuple of (enhanced_descriptors, filter_labels)
        """
        
        logger.info(f"🔄 Processing {len(catalog_data)} products for descriptors + filters")
        
        # Step 1: Generate enhanced descriptors
        enhanced_descriptors = self._generate_enhanced_descriptors(
            catalog_data, descriptor_style
        )
        
        # Step 2: Extract filter labels (using same catalog data)
        filter_labels = self.filter_analyzer.analyze_product_catalog(catalog_data)
        
        # Step 3: Enhance descriptors with filter-aware text
        enhanced_descriptors = self._enhance_descriptors_with_filters(
            enhanced_descriptors, filter_labels
        )
        
        logger.info(f"✅ Generated {len(enhanced_descriptors)} enhanced descriptors")
        logger.info(f"✅ Extracted {len(filter_labels) - 1} filter types")  # -1 for metadata
        
        return enhanced_descriptors, filter_labels
    
    def _generate_enhanced_descriptors(
        self, 
        catalog_data: List[Dict[str, Any]], 
        style: str
    ) -> List[Dict[str, Any]]:
        """Generate enhanced product descriptors optimized for RAG and voice"""
        
        enhanced_products = []
        
        for product in catalog_data:
            # Generate different descriptor styles
            if style == "voice_optimized":
                enhanced_desc = self._generate_voice_optimized_descriptor(product)
            elif style == "detailed":
                enhanced_desc = self._generate_detailed_descriptor(product)
            else:  # concise
                enhanced_desc = self._generate_concise_descriptor(product)
            
            # Create enhanced product record
            enhanced_product = {
                **product,  # Keep original data
                "enhanced_description": enhanced_desc,
                "rag_keywords": self._extract_rag_keywords(product),
                "search_terms": self._generate_search_terms(product),
                "voice_summary": self._generate_voice_summary(product)
            }
            
            enhanced_products.append(enhanced_product)
        
        return enhanced_products
    
    def _generate_voice_optimized_descriptor(self, product: Dict[str, Any]) -> str:
        """Generate descriptor optimized for voice AI conversations"""
        
        name = product.get("name", "Product")
        category = product.get("category", "item")
        price = product.get("price", 0)
        
        # Start with natural language intro
        desc_parts = []
        
        # Opening - natural and conversational
        desc_parts.append(f"The {name} is")
        
        # Category and positioning
        if category:
            category_desc = self._get_category_description(category)
            desc_parts.append(category_desc)
        
        # Key features in natural language
        features = product.get("features", [])
        if features:
            feature_text = self._format_features_for_voice(features)
            desc_parts.append(feature_text)
        
        # Use case and benefits
        intended_use = product.get("intended_use", [])
        if intended_use:
            use_text = self._format_use_cases_for_voice(intended_use)
            desc_parts.append(use_text)
        
        # Technical specifications (voice-friendly)
        tech_specs = self._format_specs_for_voice(product)
        if tech_specs:
            desc_parts.append(tech_specs)
        
        # Price positioning
        price_context = self._get_price_context(price, category)
        desc_parts.append(price_context)
        
        # Original description if available
        if product.get("description"):
            desc_parts.append(product["description"])
        
        return " ".join(desc_parts)
    
    def _generate_detailed_descriptor(self, product: Dict[str, Any]) -> str:
        """Generate detailed descriptor with comprehensive information"""
        
        name = product.get("name", "Product")
        
        sections = []
        
        # Product overview
        sections.append(f"**{name}**")
        
        # Specifications
        spec_lines = []
        spec_fields = ["category", "frame_material", "wheel_size", "weight", "gender"]
        for field in spec_fields:
            if product.get(field):
                label = field.replace("_", " ").title()
                value = product[field]
                spec_lines.append(f"- {label}: {value}")
        
        if spec_lines:
            sections.append("**Specifications:**\n" + "\n".join(spec_lines))
        
        # Features
        features = product.get("features", [])
        if features:
            feature_list = "\n".join([f"- {f.replace('_', ' ').title()}" for f in features])
            sections.append(f"**Features:**\n{feature_list}")
        
        # Use cases
        use_cases = product.get("intended_use", [])
        if use_cases:
            use_list = "\n".join([f"- {u.replace('_', ' ').title()}" for u in use_cases])
            sections.append(f"**Ideal For:**\n{use_list}")
        
        # Price
        if product.get("price"):
            sections.append(f"**Price:** ${product['price']:,.2f}")
        
        # Original description
        if product.get("description"):
            sections.append(f"**Description:** {product['description']}")
        
        return "\n\n".join(sections)
    
    def _generate_concise_descriptor(self, product: Dict[str, Any]) -> str:
        """Generate concise descriptor for quick reference"""
        
        name = product.get("name", "Product")
        category = product.get("category", "")
        price = product.get("price", 0)
        
        # Key attributes
        key_attrs = []
        if product.get("frame_material"):
            key_attrs.append(product["frame_material"])
        if category:
            key_attrs.append(category)
        
        # Price
        price_str = f"${price:,.0f}" if price else ""
        
        # Top features
        features = product.get("features", [])[:3]  # Top 3 features
        feature_str = ", ".join([f.replace("_", " ") for f in features])
        
        parts = [name]
        if key_attrs:
            parts.append(f"({', '.join(key_attrs)})")
        if feature_str:
            parts.append(f"- {feature_str}")
        if price_str:
            parts.append(price_str)
        
        return " ".join(parts)
    
    def _get_category_description(self, category: str) -> str:
        """Get natural language description for product category"""
        
        category_descriptions = {
            "road": "a high-performance road bike designed for speed and efficiency on paved surfaces",
            "mountain": "a rugged mountain bike built for off-road trails and challenging terrain",
            "electric": "an electric bike that provides motor assistance for easier riding",
            "hybrid": "a versatile hybrid bike combining comfort and performance for various terrains",
            "gravel": "a gravel bike designed for mixed terrain adventures and exploration"
        }
        
        return category_descriptions.get(category, f"a {category} bike")
    
    def _format_features_for_voice(self, features: List[str]) -> str:
        """Format features in natural language for voice"""
        
        if not features:
            return ""
        
        # Convert technical terms to natural language
        feature_translations = {
            "disc_brakes": "reliable disc brakes",
            "electronic_shifting": "precise electronic shifting",
            "tubeless_ready": "tubeless-ready wheels",
            "suspension": "advanced suspension",
            "electric_motor": "electric motor assistance",
            "comfort_geometry": "comfort-focused geometry"
        }
        
        natural_features = []
        for feature in features[:4]:  # Limit to top 4 for voice
            natural = feature_translations.get(feature, feature.replace("_", " "))
            natural_features.append(natural)
        
        if len(natural_features) == 1:
            return f"featuring {natural_features[0]}"
        elif len(natural_features) == 2:
            return f"featuring {natural_features[0]} and {natural_features[1]}"
        else:
            return f"featuring {', '.join(natural_features[:-1])}, and {natural_features[-1]}"
    
    def _format_use_cases_for_voice(self, use_cases: List[str]) -> str:
        """Format use cases in natural language for voice"""
        
        if not use_cases:
            return ""
        
        use_case_translations = {
            "racing": "competitive racing",
            "commuting": "daily commuting",
            "trail_riding": "trail riding",
            "recreational": "recreational cycling",
            "fitness": "fitness and exercise"
        }
        
        natural_uses = []
        for use_case in use_cases[:3]:  # Limit for voice
            natural = use_case_translations.get(use_case, use_case.replace("_", " "))
            natural_uses.append(natural)
        
        if len(natural_uses) == 1:
            return f"Perfect for {natural_uses[0]}."
        else:
            return f"Ideal for {', '.join(natural_uses[:-1])} and {natural_uses[-1]}."
    
    def _format_specs_for_voice(self, product: Dict[str, Any]) -> str:
        """Format key specifications for voice"""
        
        specs = []
        
        # Weight
        weight = product.get("weight")
        if weight:
            specs.append(f"weighs {weight}kg")
        
        # Frame material
        material = product.get("frame_material")
        if material:
            specs.append(f"built with a {material} frame")
        
        # Wheel size
        wheel_size = product.get("wheel_size")
        if wheel_size and wheel_size != "700c":  # 700c is standard, don't mention
            specs.append(f"equipped with {wheel_size} inch wheels")
        
        if specs:
            return f"It {', '.join(specs)}."
        
        return ""
    
    def _get_price_context(self, price: float, category: str) -> str:
        """Generate price context for voice"""
        
        if not price:
            return ""
        
        # Simple price positioning
        if price < 1000:
            position = "entry-level"
        elif price < 3000:
            position = "mid-range" 
        elif price < 5000:
            position = "performance"
        else:
            position = "premium"
        
        return f"This {position} option is priced at ${price:,.0f}."
    
    def _extract_rag_keywords(self, product: Dict[str, Any]) -> List[str]:
        """Extract keywords for RAG search optimization"""
        
        keywords = []
        
        # Add product name words
        name = product.get("name", "")
        keywords.extend(name.lower().split())
        
        # Add category
        if product.get("category"):
            keywords.append(product["category"])
        
        # Add features
        features = product.get("features", [])
        keywords.extend(features)
        
        # Add use cases
        use_cases = product.get("intended_use", [])
        keywords.extend(use_cases)
        
        # Add material
        if product.get("frame_material"):
            keywords.append(product["frame_material"])
        
        # Add gender
        if product.get("gender"):
            keywords.append(product["gender"])
        
        # Remove duplicates and filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [k for k in set(keywords) if k and k not in stop_words and len(k) > 2]
        
        return sorted(keywords)
    
    def _generate_search_terms(self, product: Dict[str, Any]) -> List[str]:
        """Generate search terms for query matching"""
        
        search_terms = []
        
        # Product name variations
        name = product.get("name", "")
        search_terms.append(name.lower())
        
        # Add name without numbers/model codes
        name_clean = ''.join(c for c in name if not c.isdigit()).strip()
        if name_clean != name:
            search_terms.append(name_clean.lower())
        
        # Category + material combinations
        category = product.get("category", "")
        material = product.get("frame_material", "")
        
        if category and material:
            search_terms.append(f"{material} {category}")
            search_terms.append(f"{category} {material}")
        
        # Feature combinations
        features = product.get("features", [])
        for feature in features[:3]:  # Top 3 features
            if category:
                search_terms.append(f"{category} {feature.replace('_', ' ')}")
        
        return search_terms
    
    def _generate_voice_summary(self, product: Dict[str, Any]) -> str:
        """Generate ultra-concise summary for voice responses"""
        
        name = product.get("name", "Product")
        category = product.get("category", "bike")
        price = product.get("price", 0)
        
        # Key selling point
        features = product.get("features", [])
        key_feature = ""
        if "electronic_shifting" in features:
            key_feature = "with electronic shifting"
        elif "electric_motor" in features:
            key_feature = "electric-powered"
        elif "carbon" in product.get("frame_material", ""):
            key_feature = "lightweight carbon"
        elif "suspension" in features:
            key_feature = "with suspension"
        
        price_str = f"${price:,.0f}" if price else ""
        
        parts = [name]
        if key_feature:
            parts.append(key_feature)
        if category != "bike":
            parts.append(category)
        if price_str:
            parts.append(price_str)
        
        return f"{' '.join(parts[:-1])} - {parts[-1]}" if len(parts) > 1 else parts[0]
    
    def _enhance_descriptors_with_filters(
        self, 
        descriptors: List[Dict[str, Any]], 
        filter_labels: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Enhance descriptors by ensuring filter terms are included"""
        
        # Extract all filter values that should be mentioned
        filter_terms = set()
        
        for filter_config in filter_labels.values():
            if isinstance(filter_config, dict):
                values = filter_config.get("values", [])
                if isinstance(values, list):
                    filter_terms.update(values)
                
                aliases = filter_config.get("aliases", {})
                for alias_list in aliases.values():
                    if isinstance(alias_list, list):
                        filter_terms.update(alias_list)
        
        # Enhance each descriptor
        for descriptor in descriptors:
            enhanced_desc = descriptor.get("enhanced_description", "")
            
            # Ensure key filter terms are mentioned
            product = descriptor
            missing_terms = []
            
            # Check if category is mentioned
            category = product.get("category", "")
            if category and category not in enhanced_desc.lower():
                missing_terms.append(category)
            
            # Check if material is mentioned
            material = product.get("frame_material", "")
            if material and material not in enhanced_desc.lower():
                missing_terms.append(material)
            
            # Add missing terms naturally
            if missing_terms:
                addition = f" This {' '.join(missing_terms)} model"
                enhanced_desc = enhanced_desc.replace(".", f"{addition}.", 1)
                descriptor["enhanced_description"] = enhanced_desc
        
        return descriptors
    
    def save_enhanced_catalog(
        self, 
        enhanced_descriptors: List[Dict[str, Any]], 
        filter_labels: Dict[str, Any]
    ) -> None:
        """Save enhanced descriptors and filter labels"""
        
        # Save enhanced product catalog
        catalog_path = self.brand_path / "enhanced_product_catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(catalog_path, 'w') as f:
            json.dump(enhanced_descriptors, f, indent=2)
        
        logger.info(f"💾 Saved enhanced catalog to {catalog_path}")
        
        # Save filter labels
        self.filter_analyzer.save_filters_to_file(filter_labels, "catalog_filters.json")
        
        logger.info(f"✅ Enhanced descriptor generation complete")


# Factory function for easy usage
def generate_enhanced_catalog(
    brand_domain: str, 
    catalog_data: List[Dict[str, Any]],
    descriptor_style: str = "voice_optimized"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate enhanced product catalog with descriptors and filters"""
    
    generator = EnhancedDescriptorGenerator(brand_domain)
    
    enhanced_descriptors, filter_labels = generator.process_catalog(
        catalog_data, descriptor_style
    )
    
    generator.save_enhanced_catalog(enhanced_descriptors, filter_labels)
    
    return enhanced_descriptors, filter_labels