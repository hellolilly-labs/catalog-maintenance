"""
Enhanced Customer & Cultural Intelligence Research Phase

Implements Phase 4 of the Brand Research Pipeline with catalog-enhanced analysis.
Leverages product catalog data for customer segmentation and cultural insights.

Focus: Target audience and cultural intelligence with quantitative analysis
Cache Duration: 3-4 months (moderate stability)
Research Time: 2-3 minutes
Quality Threshold: 8.5 (enhanced)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.llm.simple_factory import LLMFactory
from src.llm.prompt_manager import PromptManager
from src.storage import get_account_storage_provider
from src.progress_tracker import ProgressTracker, StepType, create_console_listener
from src.web_search import TavilySearchProvider

logger = logging.getLogger(__name__)


class CustomerCulturalResearcher:
    """Enhanced Customer & Cultural Intelligence Research Phase Implementation"""
    
    def __init__(self, storage_manager=None):
        self.storage_manager = storage_manager or get_account_storage_provider()
        self.llm_service = LLMFactory.get_service("openai/o3")
        self.prompt_manager = PromptManager.get_prompt_manager()
        
        self.quality_threshold = 8.5  # Enhanced threshold
        self.cache_duration_days = 105  # 3.5 months default
        
        self.progress_tracker = ProgressTracker(storage_manager=self.storage_manager, enable_checkpoints=True)
        console_listener = create_console_listener()
        self.progress_tracker.add_progress_listener(console_listener)
        
    async def research_customer_cultural(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Enhanced Customer & Cultural Intelligence Research"""
        start_time = time.time()
        
        logger.info(f"👥 Starting Enhanced Customer & Cultural Intelligence Research for {brand_domain}")
        
        step_id = self.progress_tracker.create_step(
            step_type=StepType.CUSTOMER_INTELLIGENCE,
            brand=brand_domain,
            phase_name="Enhanced Customer & Cultural Intelligence Research",
            total_operations=8
        )
        
        try:
            self.progress_tracker.start_step(step_id, "Checking cache and initializing...")
            
            if not force_refresh:
                cached_result = await self._load_cached_customer_cultural(brand_domain)
                if cached_result:
                    self.progress_tracker.complete_step(step_id, cache_hit=True)
                    logger.info(f"✅ Using cached customer cultural research for {brand_domain}")
                    return cached_result
            
            # Enhanced analysis with product catalog integration
            self.progress_tracker.update_progress(step_id, 1, "📦 Loading product catalog for customer insights...")
            product_catalog = await self.storage_manager.get_product_catalog(brand_domain)
            
            self.progress_tracker.update_progress(step_id, 2, "🔍 Analyzing customer segments from product data...")
            
            if product_catalog:
                logger.info(f"🎉 Analyzing {len(product_catalog)} products for customer insights")
                analysis_result = await self._analyze_with_product_catalog(
                    brand_domain=brand_domain,
                    product_catalog=product_catalog,
                    step_id=step_id
                )
            else:
                logger.info("📋 No product catalog - using web-only cultural analysis")
                analysis_result = await self._analyze_without_product_catalog(
                    brand_domain=brand_domain,
                    step_id=step_id
                )
            
            self.progress_tracker.update_progress(step_id, 6, "💾 Saving enhanced customer research...")
            saved_files = await self._save_customer_cultural_research(brand_domain, analysis_result)
            
            duration = time.time() - start_time
            logger.info(f"✅ Enhanced Customer & Cultural Intelligence Research completed in {duration:.1f}s")
            
            self.progress_tracker.complete_step(
                step_id,
                output_files=saved_files,
                quality_score=analysis_result.get("confidence", 0.8),
                cache_hit=False
            )
            
            return {
                "brand": brand_domain,
                "customer_cultural_content": analysis_result.get("content", ""),
                "quality_score": analysis_result.get("confidence", 0.8),
                "files": saved_files,
                "data_sources": analysis_result.get("source_count", 0),
                "research_method": analysis_result.get("analysis_type", "enhanced_cultural_analysis"),
                "product_count": len(product_catalog) if product_catalog else 0
            }
            
        except Exception as e:
            self.progress_tracker.fail_step(step_id, str(e))
            logger.error(f"❌ Error in enhanced customer cultural research: {e}")
            raise

    async def _analyze_with_product_catalog(
        self,
        brand_domain: str,
        product_catalog: List[Dict[str, Any]],
        step_id: str
    ) -> Dict[str, Any]:
        """Enhanced customer analysis using product catalog data"""
        
        self.progress_tracker.update_progress(step_id, 3, "🧮 Analyzing customer segmentation from products...")
        
        # Analyze customer segments from product data
        customer_segments = await self._analyze_customer_segments(product_catalog)
        
        self.progress_tracker.update_progress(step_id, 4, "🎯 Analyzing cultural patterns from product design...")
        
        # Analyze cultural patterns
        cultural_patterns = await self._analyze_cultural_patterns(product_catalog)
        
        self.progress_tracker.update_progress(step_id, 5, "🧠 Generating enhanced cultural intelligence...")
        
        # Get enhanced prompt for catalog-based customer analysis
        prompt_template = await self._get_customer_catalog_analysis_prompt()
        
        # Prepare comprehensive analysis data
        analysis_data = {
            "brand_domain": brand_domain,
            "product_catalog": {
                "total_products": len(product_catalog),
                "customer_segments": customer_segments,
                "cultural_patterns": cultural_patterns,
                "sample_products": product_catalog[:10]  # First 10 for analysis
            }
        }
        
        # Generate enhanced analysis using LLM
        analysis_content = await self._generate_enhanced_customer_analysis(
            prompt_template, 
            analysis_data
        )
        
        # Calculate quality score based on data richness
        quality_score = self._calculate_enhanced_quality_score(product_catalog, analysis_content)
        
        return {
            "content": analysis_content,
            "confidence": quality_score,
            "source_count": len(product_catalog),
            "analysis_type": "catalog_enhanced_customer_intelligence",
            "customer_segments": customer_segments,
            "cultural_patterns": cultural_patterns
        }

    async def _analyze_without_product_catalog(
        self,
        brand_domain: str,
        step_id: str
    ) -> Dict[str, Any]:
        """Fallback customer analysis using web research only"""
        
        self.progress_tracker.update_progress(step_id, 3, "🌐 Conducting web-based cultural research...")
        
        # Simplified web-only analysis
        analysis_content = f"""# Customer & Cultural Intelligence: {brand_domain.replace('.com', '').title()}

**Analysis Type**: WEB-ONLY (Limited Data)
**Note**: Enhanced analysis requires product catalog integration

## 1. Customer Demographics & Psychographics
Limited to web research - specific customer data not available without product catalog.

## 2. Cultural Patterns & Trends
General cultural analysis based on available web sources.

## 3. Target Audience Segmentation  
Unable to provide specific segmentation without product data.

## 4. Customer Journey & Touchpoints
Web-based analysis of customer interaction patterns.

## 5. Cultural Influences & Values
General cultural context analysis.

## 6. Customer Behavior Intelligence
Limited insights without product purchase data.

**Recommendation**: Enable product catalog integration for comprehensive customer intelligence.
"""
        
        return {
            "content": analysis_content,
            "confidence": 0.6,  # Lower confidence for web-only
            "source_count": 0,
            "analysis_type": "web_only_customer_analysis"
        }

    async def _analyze_customer_segments(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze customer segments from product catalog"""
        
        # Price-based segmentation
        prices = []
        for product in products:
            if 'originalPrice' in product and product['originalPrice']:
                try:
                    price_str = str(product['originalPrice']).replace('$', '').replace(',', '')
                    if price_str.replace('.', '').isdigit():
                        prices.append(float(price_str))
                except:
                    pass
        
        # Category-based segmentation  
        categories = {}
        for product in products:
            if 'categories' in product and product['categories']:
                for cat in product['categories']:
                    categories[cat] = categories.get(cat, 0) + 1
        
        # Gender-based segmentation
        gender_segments = {}
        for product in products:
            product_name = product.get('name', '').lower()
            if 'women' in product_name or 'wmn' in product_name:
                gender_segments['women'] = gender_segments.get('women', 0) + 1
            elif 'men' in product_name or 'mens' in product_name:
                gender_segments['men'] = gender_segments.get('men', 0) + 1
            else:
                gender_segments['unisex'] = gender_segments.get('unisex', 0) + 1
        
        return {
            "price_segments": {
                "budget": len([p for p in prices if p < 100]),
                "mid_range": len([p for p in prices if 100 <= p < 500]),
                "premium": len([p for p in prices if p >= 500]),
                "price_range": f"${min(prices):.2f} - ${max(prices):.2f}" if prices else "No pricing data"
            },
            "category_preferences": dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]),
            "gender_segmentation": gender_segments,
            "total_analyzed": len(products)
        }

    async def _analyze_cultural_patterns(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cultural patterns from product design and positioning"""
        
        # Performance vs lifestyle orientation
        performance_indicators = 0
        lifestyle_indicators = 0
        
        for product in products:
            name = product.get('name', '').lower()
            categories = product.get('categories', [])
            
            # Performance indicators
            if any(term in name for term in ['pro', 'performance', 'race', 'speed', 'aero']):
                performance_indicators += 1
            
            # Lifestyle indicators  
            if any(term in name for term in ['casual', 'comfort', 'everyday', 'urban']):
                lifestyle_indicators += 1
        
        # Color culture analysis
        colors = []
        for product in products:
            if 'colors' in product and product['colors']:
                for color in product['colors']:
                    if 'name' in color:
                        colors.append(color['name'].lower())
        
        color_culture = {}
        for color in colors:
            color_culture[color] = color_culture.get(color, 0) + 1
        
        # Technology adoption patterns
        tech_features = 0
        for product in products:
            name = product.get('name', '').lower()
            if any(term in name for term in ['™', '®', 'tech', 'smart', 'connect']):
                tech_features += 1
        
        return {
            "performance_vs_lifestyle": {
                "performance_products": performance_indicators,
                "lifestyle_products": lifestyle_indicators,
                "orientation": "performance" if performance_indicators > lifestyle_indicators else "lifestyle"
            },
            "color_culture": dict(sorted(color_culture.items(), key=lambda x: x[1], reverse=True)[:15]),
            "technology_adoption": {
                "tech_enhanced_products": tech_features,
                "adoption_rate": f"{tech_features/len(products)*100:.1f}%" if products else "0%"
            },
            "cultural_themes": self._identify_cultural_themes(products)
        }

    def _identify_cultural_themes(self, products: List[Dict[str, Any]]) -> List[str]:
        """Identify cultural themes from product patterns"""
        themes = []
        
        # Analyze product names for cultural themes
        all_names = [p.get('name', '').lower() for p in products]
        name_text = ' '.join(all_names)
        
        if 'therminal' in name_text:
            themes.append("Climate-Adaptive Performance")
        if 'deflect' in name_text:
            themes.append("Protection-Focused Design")
        if any(term in name_text for term in ['women', 'wmn']):
            themes.append("Gender-Inclusive Design")
        if any(term in name_text for term in ['alpha', 'pro', 'expert']):
            themes.append("Performance Hierarchy Culture")
        
        return themes

    async def _get_customer_catalog_analysis_prompt(self) -> str:
        """Get enhanced prompt for customer catalog analysis"""
        
        default_prompt = """
You are conducting comprehensive Customer & Cultural Intelligence research for {{brand_domain}}.

You have access to a complete product catalog with {{product_count}} products for customer segmentation analysis.

Your task is to analyze customer demographics, cultural patterns, and target audience intelligence.

**PRODUCT CATALOG DATA:**
Total Products: {{product_count}}
Customer Segments: {{customer_segments}}
Cultural Patterns: {{cultural_patterns}}
Sample Products: {{sample_products}}

**ANALYSIS REQUIREMENTS:**

Generate comprehensive customer intelligence covering these 6 sections:

## 1. Customer Demographics & Psychographics
Analyze target customer profiles based on product data:
- Age demographics indicated by product types
- Income levels reflected in pricing tiers
- Lifestyle preferences from product categories
- Performance vs. recreational customer splits
*Use specific product examples and pricing data*

## 2. Cultural Patterns & Trends
Document cultural influences on brand positioning:
- Performance culture vs. lifestyle culture
- Gender inclusivity and representation
- Technology adoption patterns
- Color and design cultural preferences
*Reference actual product color and design data*

## 3. Target Audience Segmentation
Map customer segments with quantitative analysis:
- Price-based customer tiers (budget/mid/premium)
- Category-based interest segments
- Gender-based product preferences
- Performance level segmentation
*Provide specific numbers and percentages*

## 4. Customer Journey & Touchpoints
Analyze customer interaction patterns:
- Entry-level to advanced product progression
- Cross-category purchase patterns
- Seasonal engagement cycles
- Brand loyalty indicators
*Use product hierarchy and categorization data*

## 5. Cultural Influences & Values
Assess brand cultural alignment:
- Performance excellence culture
- Innovation and technology values
- Sustainability and responsibility
- Community and inclusivity themes
*Reference cultural themes from product analysis*

## 6. Customer Behavior Intelligence
Document behavioral patterns from product data:
- Purchase tier preferences
- Category loyalty patterns
- Feature preference trends
- Brand engagement indicators
*Use quantitative insights from catalog analysis*

**SOURCING REQUIREMENTS:**
- Use numbered citations [1], [2], [3] for all insights
- Include specific product examples and data points
- Provide quantitative analysis with percentages
- Reference actual pricing and categorization data

**OUTPUT FORMAT:**
Write professional customer intelligence suitable for strategic planning.
Focus on actionable insights backed by product data.
Include confidence levels based on data quality.
"""

        prompt = await self.prompt_manager.get_prompt(
            "customer_cultural_catalog_analysis",
            default_prompt
        )
        
        return prompt.prompt if prompt else default_prompt

    async def _generate_enhanced_customer_analysis(
        self, 
        prompt_template: str, 
        analysis_data: Dict[str, Any]
    ) -> str:
        """Generate enhanced customer analysis using product catalog data"""
        
        # Prepare template variables
        template_vars = {
            "brand_domain": analysis_data["brand_domain"],
            "product_count": analysis_data["product_catalog"]["total_products"],
            "customer_segments": str(analysis_data["product_catalog"]["customer_segments"]),
            "cultural_patterns": str(analysis_data["product_catalog"]["cultural_patterns"]),
            "sample_products": str(analysis_data["product_catalog"]["sample_products"][:3])
        }
        
        # Replace template variables
        final_prompt = prompt_template
        for var, value in template_vars.items():
            final_prompt = final_prompt.replace(f"{{{{{var}}}}}", str(value))
        
        # Prepare context for LLM
        context = f"""
BRAND: {analysis_data['brand_domain']}

PRODUCT CATALOG ANALYSIS:
- Total Products: {analysis_data['product_catalog']['total_products']}
- Customer Segments: {analysis_data['product_catalog']['customer_segments']}
- Cultural Patterns: {analysis_data['product_catalog']['cultural_patterns']}

SAMPLE PRODUCTS FOR REFERENCE:
{json.dumps(analysis_data['product_catalog']['sample_products'][:5], indent=2)}
"""
        
        response = await LLMFactory.chat_completion(
            task="customer_research",
            system="You are an expert customer intelligence analyst specializing in target audience and cultural analysis. Generate comprehensive, data-driven customer insights based on product catalog analysis.",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        return response.get("content", "Customer analysis generation failed")

    def _calculate_enhanced_quality_score(
        self, 
        product_catalog: List[Dict[str, Any]], 
        analysis_content: str
    ) -> float:
        """Calculate quality score for catalog-enhanced customer analysis"""
        
        base_score = 0.7  # Higher base for catalog data
        
        # Product catalog quality
        product_count = len(product_catalog)
        catalog_bonus = min(0.15, product_count * 0.0001)  # Up to 0.15 for product richness
        
        # Analysis content quality
        content_length = len(analysis_content)
        content_bonus = min(0.08, content_length / 12000)  # Up to 0.08 for comprehensive analysis
        
        # Check for citations in content
        citation_count = analysis_content.count("[") + analysis_content.count("]")
        citation_bonus = min(0.05, citation_count * 0.002)  # Up to 0.05 for good citations
        
        # Check for quantitative data
        quantitative_indicators = analysis_content.count("%") + analysis_content.count("$")
        quantitative_bonus = min(0.02, quantitative_indicators * 0.005)  # Bonus for data-driven insights
        
        final_score = base_score + catalog_bonus + content_bonus + citation_bonus + quantitative_bonus
        return min(0.95, final_score)  # Cap at 0.95

    async def _save_customer_cultural_research(self, brand_domain: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Save customer cultural research in three-file format"""
        
        saved_files = []
        
        try:
            research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
            os.makedirs(research_dir, exist_ok=True)
            
            # Save content
            content_path = os.path.join(research_dir, "customer_cultural_research.md")
            with open(content_path, "w") as f:
                f.write(analysis_result["content"])
            saved_files.append(content_path)
            
            # Save metadata
            metadata = {
                "phase": "customer_cultural",
                "confidence_score": analysis_result.get("confidence", 0.8),
                "analysis_type": analysis_result.get("analysis_type", "enhanced_customer_analysis"),
                "product_count": analysis_result.get("source_count", 0),
                "customer_segments": analysis_result.get("customer_segments", {}),
                "cultural_patterns": analysis_result.get("cultural_patterns", {}),
                "research_metadata": {
                    "phase": "customer_cultural",
                    "research_duration_seconds": time.time(),
                    "timestamp": datetime.now().isoformat() + "Z",
                    "quality_threshold": self.quality_threshold,
                    "version": "2.0_enhanced"
                }
            }
            
            metadata_path = os.path.join(research_dir, "customer_cultural_research_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            saved_files.append(metadata_path)
            
            # Save sources/data
            sources_data = {
                "analysis_sources": ["product_catalog_analysis", "customer_segmentation", "cultural_pattern_analysis"],
                "total_products_analyzed": analysis_result.get("source_count", 0),
                "collection_timestamp": datetime.now().isoformat() + "Z"
            }
            
            sources_path = os.path.join(research_dir, "customer_cultural_research_sources.json")
            with open(sources_path, "w") as f:
                json.dump(sources_data, f, indent=2)
            saved_files.append(sources_path)
            
            logger.info(f"✅ Saved enhanced customer cultural research for {brand_domain}")
            
        except Exception as e:
            logger.error(f"❌ Error saving customer cultural research: {e}")
            raise
        
        return saved_files

    async def _load_cached_customer_cultural(self, brand_domain: str) -> Optional[Dict[str, Any]]:
        """Load cached customer cultural research"""
        try:
            research_dir = os.path.join(self.storage_manager.base_dir, "accounts", brand_domain, "research_phases")
            
            content_path = os.path.join(research_dir, "customer_cultural_research.md")
            metadata_path = os.path.join(research_dir, "customer_cultural_research_metadata.json")
            
            if all(os.path.exists(p) for p in [content_path, metadata_path]):
                # Check cache expiry
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                research_metadata = metadata.get("research_metadata", {})
                cache_expires = research_metadata.get("cache_expires")
                
                if cache_expires and datetime.now() < datetime.fromisoformat(cache_expires.replace("Z", "")):
                    # Load cached data
                    with open(content_path, "r") as f:
                        content = f.read()
                    
                    return {
                        "brand": brand_domain,
                        "customer_cultural_content": content,
                        "quality_score": metadata.get("confidence_score", 0.8),
                        "files": [content_path, metadata_path],
                        "data_sources": metadata.get("product_count", 0),
                        "research_method": metadata.get("analysis_type", "cached_analysis")
                    }
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None


def get_customer_cultural_researcher() -> CustomerCulturalResearcher:
    """Get enhanced customer cultural researcher instance"""
    return CustomerCulturalResearcher()
