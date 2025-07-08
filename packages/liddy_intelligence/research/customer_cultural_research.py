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

from liddy.llm.simple_factory import LLMFactory
from liddy.prompt_manager import PromptManager
from liddy.storage import get_account_storage_provider
from liddy_intelligence.progress_tracker import ProgressTracker, StepType, create_console_listener
from liddy_intelligence.web_search import TavilySearchProvider
from liddy_intelligence.research.base_researcher import BaseResearcher

logger = logging.getLogger(__name__)


class CustomerCulturalResearcher(BaseResearcher):
    """Enhanced Customer & Cultural Intelligence Research Phase Implementation"""
    
    def __init__(self, brand_domain: str, storage_manager=None):
        super().__init__(
            brand_domain=brand_domain,
            researcher_name="customer_cultural",
            step_type=StepType.CUSTOMER_CULTURAL,
            quality_threshold=8.5,
            cache_duration_days=105,
            storage_manager=storage_manager
        )
        
        self.llm_service = LLMFactory.get_service("openai/o3")
        
    # async def research_customer_cultural(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
    #     """Enhanced Customer & Cultural Intelligence Research"""
        
    #     logger.info(f"👥 Starting Enhanced Customer & Cultural Intelligence Research for {brand_domain}")
        
    #     # Use the base class research method
    #     result = await self.research(force_refresh=force_refresh)
        
    #     return {
    #         "brand": brand_domain,
    #         "customer_cultural_content": result.get("content", ""),
    #         "quality_score": result.get("quality_score", 0.8),
    #         "files": result.get("files", []),
    #         "data_sources": result.get("data_sources", 0),
    #         "research_method": result.get("research_method", "enhanced_cultural_analysis")
    #     }

    async def _gather_data(self) -> Dict[str, Any]:
        """Gather customer and cultural data - implements BaseResearcher abstract method"""
        
        brand_name = self.brand_domain.replace('.com', '').replace('.', ' ').title()
        research_queries = [
            f"{brand_name} target audience customer demographics",
            f"{brand_name} customer personas buyer profiles",
            f"{brand_name} brand community culture values",
            f"{brand_name} customer reviews testimonials feedback",
            f"{brand_name} social media followers audience engagement",
            f"{brand_name} customer lifestyle preferences behavior",
            f"{brand_name} market research consumer insights",
            f"{brand_name} brand loyalty customer retention",
            f"{brand_name} customer journey touchpoints experience",
            f"{brand_name} cultural influences brand positioning",
            f"{brand_name} customer feedback product reviews",
            f"{brand_name} brand advocates community members"
        ]
        
        try:
            from liddy_intelligence.web_search import get_web_search_engine
            web_search = get_web_search_engine()
            
            if web_search and web_search.is_available():
                all_results = []
                detailed_sources = []
                successful_searches = 0
                failed_searches = 0
                ssl_errors = 0
                
                for query_idx, query in enumerate(research_queries):
                    try:
                        results = await web_search.search(query)
                        if results.get("results"):
                            successful_searches += 1
                            for result_idx, result in enumerate(results["results"][:3]):
                                result_dict = {
                                    "title": result.title,
                                    "url": result.url,
                                    "content": result.content,
                                    "snippet": result.content,
                                    "score": result.score,
                                    "published_date": result.published_date,
                                    "source_query": query,
                                    "source_type": "customer_cultural_research",
                                    "query_index": query_idx,
                                    "result_index": result_idx
                                }
                                all_results.append(result_dict)
                                
                                source_record = {
                                    "source_id": f"query_{query_idx}_result_{result_idx}",
                                    "title": result_dict.get("title", ""),
                                    "url": result_dict.get("url", ""),
                                    "snippet": result_dict.get("snippet", ""),
                                    "search_query": query,
                                    "search_score": result_dict.get("score", 0.0),
                                    "collected_at": datetime.now().isoformat() + "Z",
                                    "source_type": "web_search",
                                    "provider": results.get("provider_used", "unknown")
                                }
                                detailed_sources.append(source_record)
                        else:
                            failed_searches += 1
                            
                        await asyncio.sleep(0.25)
                        
                    except Exception as e:
                        failed_searches += 1
                        error_msg = str(e).lower()
                        if 'ssl' in error_msg and 'certificate' in error_msg:
                            ssl_errors += 1
                        logger.warning(f"Search failed for query '{query}': {e}")
                
                total_searches = len(research_queries)
                success_rate = successful_searches / total_searches if total_searches > 0 else 0
                
                if ssl_errors >= 3:
                    error_msg = f"ABORTING: SSL certificate verification failed for {ssl_errors} searches."
                    logger.error(f"🚨 {error_msg}")
                    raise RuntimeError(error_msg)
                
                if success_rate < 0.3:
                    error_msg = f"ABORTING: Only {successful_searches}/{total_searches} searches succeeded ({success_rate:.1%})."
                    logger.error(f"🚨 {error_msg}")
                    raise RuntimeError(error_msg)
                
                if success_rate < 0.7:
                    logger.warning(f"⚠️ Reduced data quality: Only {successful_searches}/{total_searches} searches succeeded ({success_rate:.1%})")
                
                logger.info(f"✅ Web search completed: {successful_searches}/{total_searches} successful searches, {len(all_results)} total sources")
                
                return {
                    "brand_domain": self.brand_domain,
                    "brand_name": brand_name,
                    "search_results": all_results,
                    "detailed_sources": detailed_sources,
                    "research_queries": research_queries,
                    "total_sources": len(all_results),
                    "search_stats": {
                        "successful_searches": successful_searches,
                        "failed_searches": failed_searches,
                        "success_rate": success_rate,
                        "ssl_errors": ssl_errors
                    }
                }
            else:
                error_msg = "ABORTING: Web search service not available."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
                
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"ABORTING: Critical error in data gathering: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)

    def _get_default_user_prompt(self) -> str:
        """Get the default user prompt for customer cultural analysis - implements BaseResearcher abstract method"""
        
        default_prompt = """Analyze this customer & cultural research data to extract comprehensive audience intelligence.

**Brand:** {{brand_name}}
**Domain:** {{brand_domain}}

## Research Data Quality Notice:
- **Total Sources**: {{total_sources}} sources analyzed
- **Search Success Rate**: {{success_rate}}
- **Data Quality**: {{data_quality}}

## Research Data Sources:

{{search_context}}

## Source Reference Guide:
{{source_reference_guide}}

## Customer & Cultural Intelligence Analysis Requirements:

Please create a comprehensive customer & cultural intelligence report in **markdown format**. When referencing information, cite your sources using the numbers provided (e.g., [1], [2], [3]).

Structure your analysis as follows:

# Customer & Cultural Intelligence: {{brand_name}}

## 1. Customer Demographics & Psychographics
- **Target Age Groups:** [Primary age demographics and generations] [cite sources]
- **Income & Lifestyle:** [Economic segments and lifestyle preferences] [cite sources]
- **Geographic Distribution:** [Regional and global customer presence] [cite sources]
- **Psychographic Profiles:** [Values, interests, and motivations] [cite sources]

## 2. Cultural Patterns & Trends
- **Cultural Values:** [Core cultural values driving purchase decisions] [cite sources]
- **Lifestyle Influences:** [Cultural lifestyle factors and trends] [cite sources]
- **Social Identity:** [How brand fits into customer identity] [cite sources]
- **Community Engagement:** [Cultural community and social connection] [cite sources]

## 3. Target Audience Segmentation
- **Primary Segments:** [Main customer groups and characteristics] [cite sources]
- **Secondary Segments:** [Additional customer segments] [cite sources]
- **Segment Behaviors:** [Different behaviors across segments] [cite sources]
- **Segment Preferences:** [Varying preferences and needs] [cite sources]

## 4. Customer Journey & Touchpoints
- **Discovery Phase:** [How customers discover the brand] [cite sources]
- **Consideration:** [Research and evaluation process] [cite sources]
- **Purchase Decision:** [Key factors in buying decisions] [cite sources]
- **Post-Purchase:** [Customer retention and loyalty patterns] [cite sources]

## 5. Cultural Influences & Values
- **Performance Culture:** [Emphasis on achievement and excellence] [cite sources]
- **Innovation Values:** [Appreciation for cutting-edge solutions] [cite sources]
- **Sustainability Focus:** [Environmental and social responsibility] [cite sources]
- **Community Values:** [Importance of belonging and connection] [cite sources]

## 6. Customer Behavior Intelligence
- **Purchase Patterns:** [Buying behavior and frequency] [cite sources]
- **Brand Loyalty:** [Customer retention and advocacy] [cite sources]
- **Engagement Preferences:** [Preferred communication channels] [cite sources]
- **Feedback Patterns:** [Customer review and feedback behavior] [cite sources]

## Analysis Quality & Confidence

**Data Sources:** {{total_sources}} search results analyzed
**Search Success Rate:** {{success_rate}}
**Information Quality:** {{information_quality}}
**Confidence Level:** {{confidence_level}} confidence in findings
**Key Gaps:** [Note any information that was missing or unclear due to limited data availability]

## Summary

[Provide a 2-3 sentence executive summary of the customer base and cultural positioning]

## Sources

{{source_reference_guide}}

---

**Important Instructions:**
- **ALWAYS cite sources** using the provided reference numbers [1], [2], [3], etc.
- Focus on factual, verifiable information only
- Clearly distinguish between direct customer feedback and external analysis
- Note confidence levels for different claims based on data availability
- If information is missing, clearly state "Not available in research data"
- Given the {{data_quality_text}} data quality, be appropriately cautious in claims
- Use markdown formatting for structure and readability
- Include the complete sources list at the end"""

        return default_prompt

    def _get_default_instruction_prompt(self) -> str:
        """Get the default instruction prompt for customer cultural analysis - implements BaseResearcher abstract method"""
        
        return "You are an expert customer intelligence analyst specializing in target audience and cultural analysis. Generate comprehensive, data-driven customer insights based on research data. Always cite your sources using the provided reference numbers. Adjust confidence levels based on data quality."

    async def _analyze_with_product_catalog(
        self,
        brand_domain: str,
        product_catalog: List[Dict[str, Any]],
        step_id: str
    ) -> Dict[str, Any]:
        """Enhanced customer analysis using product catalog data"""
        
        await self.progress_tracker.update_progress(step_id, 3, "🧮 Analyzing customer segmentation from products...")
        
        # Analyze customer segments from product data
        customer_segments = await self._analyze_customer_segments(product_catalog)
        
        await self.progress_tracker.update_progress(step_id, 4, "🎯 Analyzing cultural patterns from product design...")
        
        # Analyze cultural patterns
        cultural_patterns = await self._analyze_cultural_patterns(product_catalog)
        
        await self.progress_tracker.update_progress(step_id, 5, "🧠 Generating enhanced cultural intelligence...")
        
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
            "analysis_type": "catalog_enhanced_customer_cultural",
            "customer_segments": customer_segments,
            "cultural_patterns": cultural_patterns
        }

    async def _analyze_without_product_catalog(
        self,
        brand_domain: str,
        step_id: str
    ) -> Dict[str, Any]:
        """Fallback customer analysis using web research only"""
        
        await self.progress_tracker.update_progress(step_id, 3, "🌐 Conducting web-based cultural research...")
        
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
            # max_tokens=4000
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


def get_customer_cultural_researcher(brand_domain: str) -> CustomerCulturalResearcher:
    """Get enhanced customer cultural researcher instance"""
    return CustomerCulturalResearcher(brand_domain)
