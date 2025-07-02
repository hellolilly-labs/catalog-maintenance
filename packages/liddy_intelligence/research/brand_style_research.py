"""
Brand Style Attribute Research Phase

Extracts concrete, programmatically-accessible brand style attributes:
- Primary and secondary brand colors (hex codes)
- Typography/font information
- Logo specifications
- Visual identity elements

Outputs structured data for account.json and generates style.css files.

Focus: Concrete style attributes for programmatic use
Cache Duration: 6 months (styles change infrequently)
Research Time: 2-3 minutes
Quality Threshold: 8.0
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from liddy_intelligence.llm.simple_factory import LLMFactory
from liddy.storage import get_account_storage_provider
from configs.settings import get_settings
from liddy_intelligence.progress_tracker import (
    get_progress_tracker, 
    StepType, 
    create_console_listener,
    ProgressTracker
)
from liddy_intelligence.research.base_researcher import BaseResearcher
from liddy_intelligence.research.data_sources import WebSearchDataSource, DataGatheringContext

logger = logging.getLogger(__name__)


class BrandStyleResearcher(BaseResearcher):
    """
    Brand Style Attribute Research Phase Implementation
    
    Extracts concrete, programmatically-accessible brand style attributes:
    - Primary and secondary brand colors (hex codes)
    - Typography/font information  
    - Logo specifications
    - Visual identity elements
    
    Outputs structured data for account.json and generates style.css files.
    """
    
    def __init__(self, brand_domain: str, storage_manager=None):
        super().__init__(
            brand_domain=brand_domain,
            researcher_name="brand_style",
            step_type=StepType.BRAND_STYLE,
            quality_threshold=8.0,
            cache_duration_days=180,  # 6 months - styles change infrequently
            storage_manager=storage_manager
        )

    async def _get_brand_name(self) -> str:
        """Get the brand name from account config or fallback to domain-derived name"""
        try:
            config = await self.storage_manager.get_account_config(self.brand_domain)
            if config and config.get("brand_name"):
                return config["brand_name"]
            else:
                # Fallback to domain-derived name
                return self.brand_domain.replace('.com', '').replace('.', ' ').title()
        except Exception as e:
            logger.warning(f"⚠️ Error getting brand name from config: {e}")
            return self.brand_domain.replace('.com', '').replace('.', ' ').title()

    async def _analyze_website_css(self) -> List[Dict[str, Any]]:
        """Directly fetch and analyze the website's CSS for color and font extraction"""
        
        try:
            import aiohttp
            import re
            
            css_data = []
            
            async with aiohttp.ClientSession() as session:
                # Try to fetch the main website
                try:
                    async with session.get(f"https://{self.brand_domain}", timeout=10) as response:
                        if response.status == 200:
                            html_content = await response.text()
                            
                            # Extract CSS links from HTML
                            css_links = re.findall(r'<link[^>]*rel=["\']stylesheet["\'][^>]*href=["\']([^"\']+)["\']', html_content)
                            
                            # Also look for inline styles
                            inline_styles = re.findall(r'<style[^>]*>(.*?)</style>', html_content, re.DOTALL)
                            
                            # Process inline styles
                            if inline_styles:
                                combined_inline = '\n'.join(inline_styles)
                                colors_found = self._extract_colors_from_css(combined_inline)
                                fonts_found = self._extract_fonts_from_css(combined_inline)
                                
                                if colors_found or fonts_found:
                                    css_data.append({
                                        'title': f'{self.brand_domain} - Inline CSS Analysis',
                                        'url': f'https://{self.brand_domain}',
                                        'snippet': f'Extracted {len(colors_found)} colors and {len(fonts_found)} fonts from inline CSS: Colors: {colors_found[:5]} Fonts: {fonts_found[:3]}',
                                        'source_query': 'direct_css_analysis'
                                    })
                            
                            # Process external CSS files (first few only)
                            for css_link in css_links[:3]:  # Limit to first 3 CSS files
                                if not css_link.startswith('http'):
                                    css_link = f"https://{self.brand_domain}{css_link}"
                                
                                try:
                                    async with session.get(css_link, timeout=10) as css_response:
                                        if css_response.status == 200:
                                            css_content = await css_response.text()
                                            colors_found = self._extract_colors_from_css(css_content)
                                            fonts_found = self._extract_fonts_from_css(css_content)
                                            
                                            if colors_found or fonts_found:
                                                css_data.append({
                                                    'title': f'{self.brand_domain} - CSS File Analysis',
                                                    'url': css_link,
                                                    'snippet': f'Extracted {len(colors_found)} colors and {len(fonts_found)} fonts from CSS: Colors: {colors_found[:10]} Fonts: {fonts_found[:5]}',
                                                    'source_query': 'direct_css_file_analysis'
                                                })
                                except Exception as e:
                                    logger.debug(f"Failed to fetch CSS file {css_link}: {e}")
                                    
                except Exception as e:
                    logger.warning(f"Failed to fetch website HTML: {e}")
            
            if css_data:
                logger.info(f"✅ Direct CSS analysis found {len(css_data)} sources with style data")
            else:
                logger.info("ℹ️ No CSS data extracted from direct website analysis")
                
            return css_data
            
        except ImportError:
            logger.warning("⚠️ aiohttp not available for direct CSS analysis")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Direct CSS analysis failed: {e}")
            return []

    async def _capture_visual_analysis(self) -> List[Dict[str, Any]]:
        """Capture and analyze website screenshots for visual design patterns"""
        
        try:
            visual_data = []
            
            # Try to capture screenshots of key pages
            pages_to_analyze = [
                {"url": f"https://{self.brand_domain}", "name": "homepage"},
                {"url": f"https://{self.brand_domain}/products", "name": "products"},
                {"url": f"https://{self.brand_domain}/about", "name": "about"},
            ]
            
            for page in pages_to_analyze:
                try:
                    screenshot_data = await self._capture_screenshot(page["url"])
                    if screenshot_data:
                        visual_analysis = await self._analyze_screenshot_design(screenshot_data, page["name"])
                        if visual_analysis:
                            visual_data.append({
                                'title': f'{self.brand_domain} - {page["name"].title()} Visual Analysis',
                                'url': page["url"],
                                'snippet': visual_analysis,
                                'source_query': f'visual_screenshot_analysis_{page["name"]}'
                            })
                except Exception as e:
                    logger.debug(f"Failed to analyze {page['name']} page: {e}")
            
            if visual_data:
                logger.info(f"📸 Visual analysis captured {len(visual_data)} page designs")
            else:
                logger.info("ℹ️ No visual analysis data captured")
                
            return visual_data
            
        except Exception as e:
            logger.warning(f"⚠️ Visual analysis failed: {e}")
            return []

    async def _capture_screenshot(self, url: str) -> Optional[str]:
        """Capture screenshot of webpage using playwright or similar"""
        
        try:
            # Try to use playwright for screenshot capture
            try:
                from playwright.async_api import async_playwright
                
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    context = await browser.new_context(
                        viewport={'width': 1200, 'height': 800},
                        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    )
                    page = await context.new_page()
                    
                    # Navigate and wait for load with longer timeout
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    
                    # Take screenshot (PNG doesn't support quality parameter)
                    screenshot_bytes = await page.screenshot(
                        type='png',
                        full_page=False  # Just above the fold
                    )
                    
                    await browser.close()
                    
                    # Convert to base64 for LLM analysis
                    import base64
                    return base64.b64encode(screenshot_bytes).decode('utf-8')
                    
            except ImportError:
                logger.warning("⚠️ Playwright not available for screenshot capture")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Screenshot capture failed for {url}: {e}")
            return None

    async def _analyze_screenshot_design(self, screenshot_base64: str, page_name: str) -> Optional[str]:
        """Analyze screenshot using LLM vision to extract design patterns"""
        
        try:
            from liddy_intelligence.llm.simple_factory import LLMFactory
            
            visual_analysis_prompt = f"""Analyze this {page_name} page screenshot from {self.brand_domain} and extract detailed design characteristics:

VISUAL DESIGN ANALYSIS REQUIREMENTS:

1. **Typography Hierarchy:**
   - Primary headings: font weight, size relationships, letter spacing
   - Body text: font families, line heights, text colors
   - UI elements: button text, navigation, labels

2. **Color Usage Patterns:**
   - Primary brand colors and where they're used
   - Secondary colors for backgrounds, borders, accents
   - Text color hierarchy (headings, body, muted text)
   - Interactive element colors (hover, active states)

3. **Layout & Spacing:**
   - Grid systems and content width patterns
   - Spacing patterns between elements
   - Card/component styling and shadows
   - Border radius patterns

4. **Component Styles:**
   - Button designs (primary, secondary, outline styles)
   - Navigation patterns and styling
   - Card/product tile designs
   - Form element styling

5. **Brand Personality Indicators:**
   - Modern vs traditional feel
   - Minimal vs rich design
   - Technical vs lifestyle oriented
   - Professional vs casual tone

6. **Industry-Specific Patterns:**
   - What design patterns indicate the industry/category?
   - How does the visual style position the brand?

Provide specific, actionable insights about colors, fonts, spacing, and component styling that would help recreate this brand's authentic design language."""

            # Use vision-capable LLM for analysis
            response = await LLMFactory.chat_completion(
                task="visual_design_analysis",
                system="You are an expert UI/UX designer and brand analyst. Analyze website screenshots to extract detailed design patterns, typography, colors, and brand styling characteristics.",
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": visual_analysis_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                    ]
                }],
                temperature=0.1
            )
            
            if response and response.get("content"):
                return response["content"]
            else:
                logger.warning(f"⚠️ No visual analysis response for {page_name}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Visual analysis failed for {page_name}: {e}")
            return None

    def _extract_colors_from_css(self, css_content: str) -> List[str]:
        """Extract hex colors from CSS content"""
        # Find hex colors
        hex_colors = re.findall(r'#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})', css_content)
        # Normalize to uppercase and add # prefix
        normalized_colors = [f"#{color.upper()}" for color in hex_colors]
        # Remove duplicates while preserving order
        unique_colors = list(dict.fromkeys(normalized_colors))
        return unique_colors

    def _extract_fonts_from_css(self, css_content: str) -> List[str]:
        """Extract font families from CSS content"""
        # Find font-family declarations
        font_families = re.findall(r'font-family\s*:\s*([^;}]+)', css_content, re.IGNORECASE)
        fonts = []
        for family in font_families:
            # Clean up font family names
            family = family.strip().strip(',').strip('"').strip("'")
            if family and family not in ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']:
                fonts.append(family)
        # Remove duplicates while preserving order
        unique_fonts = list(dict.fromkeys(fonts))
        return unique_fonts

    async def _gather_data(self) -> Dict[str, Any]:
        """Gather brand style and visual identity data"""
        
        # Get discovered brand name from account config
        brand_name = await self._get_brand_name()
        
        # Brand style and visual identity research queries
        research_queries = [
            # Official brand color documentation - PRIORITY SEARCHES
            f'"{brand_name}" brand colors hex code primary secondary official',
            f'"{brand_name}" color palette hex values RGB brand guidelines',
            f'"{brand_name}" official brand colors style guide hex codes',
            f'"{brand_name}" logo colors hex code primary brand color',
            f'"{brand_name}" brand kit color palette download PDF',
            f'"{brand_name}" corporate colors hex RGB official guidelines',
            f'"{brand_name}" primary brand color hex code official',
            f'"{brand_name}" visual identity colors hex values brand manual',
            
            # Site-specific searches for brand guidelines
            f'site:{self.brand_domain} brand colors hex code style guide',
            f'site:{self.brand_domain} color palette brand guidelines colors',
            f'site:{self.brand_domain} brand kit press kit colors',
            
            # Typography searches
            f'"{brand_name}" official fonts typography brand guidelines',
            f'"{brand_name}" corporate font family typeface brand identity',
            f'site:{self.brand_domain} fonts typography brand guidelines',
            
            # Logo and visual identity
            f'"{brand_name}" logo guidelines brand standards visual identity',
            f'"{brand_name}" brand manual style guide download'
        ]
        
        # Use WebSearchDataSource for data gathering
        try:
            web_search_source = WebSearchDataSource()
            
            if not web_search_source.is_available():
                error_msg = "ABORTING: Web search service not available. Cannot proceed with brand style research without external data sources."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            # Create data gathering context
            context = DataGatheringContext(
                brand_domain=self.brand_domain,
                researcher_name=self.researcher_name,
                phase_name="brand_style"
            )
            
            # Gather data using the data source strategy
            search_result = await web_search_source.gather(research_queries, context)
            
            # Additionally, try to directly fetch and analyze the website's CSS
            website_css_data = await self._analyze_website_css()
            if website_css_data:
                # Add CSS analysis results to search results
                search_result.results.extend(website_css_data)
            
            # 🆕 VISUAL ANALYSIS: Capture and analyze website screenshots
            visual_analysis_data = await self._capture_visual_analysis()
            if visual_analysis_data:
                # Add visual analysis results to search results
                search_result.results.extend(visual_analysis_data)
                logger.info(f"📸 Added {len(visual_analysis_data)} visual design analysis sources")
            
            # Quality checks
            total_searches = len(research_queries)
            success_rate = search_result.successful_searches / total_searches if total_searches > 0 else 0
            
            if search_result.ssl_errors >= 3:
                error_msg = f"ABORTING: SSL certificate verification failed for {search_result.ssl_errors} searches. Cannot proceed with research without reliable web access."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            if success_rate < 0.3:
                error_msg = f"ABORTING: Only {search_result.successful_searches}/{total_searches} searches succeeded ({success_rate:.1%}). Insufficient data for quality brand style research."
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
            
            if success_rate < 0.7:
                logger.warning(f"⚠️ Reduced data quality: Only {search_result.successful_searches}/{total_searches} searches succeeded ({success_rate:.1%})")
            
            logger.info(f"✅ Brand style search completed: {search_result.successful_searches}/{total_searches} successful searches, {len(search_result.results)} total sources")
            
            # Return data in the expected format
            return {
                "brand_domain": self.brand_domain,
                "brand_name": brand_name,
                "search_results": search_result.results,
                "detailed_sources": search_result.sources,
                "research_queries": research_queries,
                "total_sources": len(search_result.results),
                "search_stats": {
                    "successful_searches": search_result.successful_searches,
                    "failed_searches": search_result.failed_searches,
                    "success_rate": success_rate,
                    "ssl_errors": search_result.ssl_errors
                },
                "sources_by_type": {
                    "web_search": len(search_result.sources)
                }
            }
                
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"ABORTING: Critical error in brand style data gathering: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)

    async def _analyze_data(self, data: Dict[str, Any], temperature: float = 0.1) -> Dict[str, Any]:
        """Analyze data and extract structured brand style attributes"""
        
        # Check if we have sufficient data for analysis
        total_sources = data.get("total_sources", 0)
        search_stats = data.get("search_stats", {})
        success_rate = search_stats.get("success_rate", 0)
        
        if total_sources == 0:
            error_msg = "ANALYSIS ABORTED: No search results available for brand style analysis."
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)
        
        if total_sources < 5:
            logger.warning(f"⚠️ Limited data available: Only {total_sources} sources for brand style analysis.")
        
        # Compile search context for LLM analysis
        search_context = ""
        for result in data["search_results"]:
            search_context += f"**Title:** {result.get('title', '')}\n"
            search_context += f"**URL:** {result.get('url', '')}\n"
            search_context += f"**Content:** {result.get('snippet', '')}\n\n---\n\n"

        brand_style_extraction_prompt = f"""Extract COMPREHENSIVE brand style attributes from this research data for {data.get('brand_name')} ({data.get('brand_domain')}).

RESEARCH DATA:
{search_context}

TASK: Extract AUTHENTIC brand design language from ALL sources including official guidelines, CSS analysis, and visual screenshot analysis.

EXTRACTION GUIDELINES:

1. **VISUAL DESIGN ANALYSIS (PRIORITY)**: 
   - Use screenshot analysis to understand actual implementation
   - Look for typography hierarchy used in practice
   - Identify authentic color usage patterns 
   - Understand spacing and layout patterns
   - Extract component styling approaches

2. **OFFICIAL BRAND GUIDELINES**:
   - Look for official brand colors with hex codes
   - Corporate typography specifications
   - Official design principles and standards

3. **CSS & IMPLEMENTATION ANALYSIS**:
   - Extract colors and fonts from actual website CSS
   - Understand technical implementation patterns
   - Identify design tokens and spacing systems

4. **BRAND PERSONALITY & POSITIONING**:
   - Technical vs lifestyle oriented
   - Modern vs traditional approach
   - Minimal vs rich design language
   - Industry-specific patterns

REQUIRED OUTPUT FORMAT (JSON):
{{
  "brand_colors": {{
    "primary": "#RRGGBB",
    "secondary": "#RRGGBB", 
    "additional": ["#RRGGBB", "#RRGGBB"],
    "text_primary": "#RRGGBB",
    "text_secondary": "#RRGGBB",
    "background_primary": "#RRGGBB",
    "background_secondary": "#RRGGBB"
  }},
  "typography": {{
    "primary_font": "Font Family Name",
    "secondary_font": "Font Family Name", 
    "display_font": "Display Font Name",
    "font_weights": ["300", "400", "600", "700"],
    "font_fallbacks": ["Arial", "sans-serif"],
    "heading_styles": {{
      "h1_size": "32px",
      "h1_weight": "700",
      "h1_line_height": "1.2",
      "h1_letter_spacing": "-0.5px"
    }},
    "body_styles": {{
      "size": "16px",
      "line_height": "1.6",
      "letter_spacing": "0px"
    }}
  }},
  "design_system": {{
    "spacing_unit": "8px",
    "border_radius_sm": "4px",
    "border_radius_md": "8px", 
    "border_radius_lg": "12px",
    "shadow_sm": "0 1px 2px rgba(0,0,0,0.05)",
    "shadow_md": "0 4px 6px rgba(0,0,0,0.1)", 
    "shadow_lg": "0 10px 15px rgba(0,0,0,0.1)",
    "transition_fast": "150ms ease",
    "transition_normal": "250ms ease"
  }},
  "component_patterns": {{
    "button_primary": {{
      "background": "#RRGGBB",
      "color": "#RRGGBB",
      "padding": "12px 24px",
      "border_radius": "4px",
      "font_weight": "600",
      "hover_transform": "translateY(-1px)"
    }},
    "button_secondary": {{
      "background": "transparent",
      "color": "#RRGGBB", 
      "border": "2px solid #RRGGBB",
      "padding": "10px 22px"
    }},
    "card_style": {{
      "background": "#FFFFFF",
      "border_radius": "8px",
      "shadow": "0 4px 6px rgba(0,0,0,0.1)",
      "padding": "24px"
    }}
  }},
  "brand_personality": {{
    "style_category": "modern|classic|minimalist|bold|playful|technical|luxury",
    "industry_alignment": "sports|tech|luxury|lifestyle|B2B|consumer",
    "design_principles": ["principle1", "principle2", "principle3"],
    "target_demographic": "professionals|enthusiasts|general_consumers|luxury_buyers"
  }},
  "logo": {{
    "primary_logo_url": "https://...",
    "logo_variations": ["primary", "white", "black", "horizontal"],
    "minimum_size": "24px",
    "clear_space": "2x logo height"
  }},
  "extraction_confidence": {{
    "colors": "high|medium|low",
    "typography": "high|medium|low",
    "design_system": "high|medium|low", 
    "visual_analysis": "high|medium|low",
    "overall": "high|medium|low"
  }},
  "sources_used": [
    "url1", 
    "url2"
  ]
}}

PRIORITIZATION:
1. Visual screenshot analysis provides AUTHENTIC implementation patterns
2. Official brand guidelines provide AUTHORITATIVE specifications  
3. CSS analysis provides TECHNICAL implementation details
4. Combine all sources for COMPREHENSIVE brand design language

Extract the AUTHENTIC brand design language now:"""

        try:
            logger.info(f"🎨 Extracting brand style attributes from {total_sources} sources")
            
            response = await LLMFactory.chat_completion(
                task="brand_style_extraction",
                system="You are an expert brand designer and CSS developer. Extract concrete, programmatically-usable brand style attributes from research data. Always respond with valid JSON only.",
                messages=[{
                    "role": "user",
                    "content": brand_style_extraction_prompt
                }],
                temperature=temperature
            )
            
            if response and response.get("content"):
                # Parse the JSON response
                extracted_data = self._parse_style_extraction_response(response["content"])
                
                # Calculate confidence based on data quality
                base_confidence = 0.8
                if success_rate < 0.5:
                    base_confidence = 0.5
                elif success_rate < 0.7:
                    base_confidence = 0.65
                
                if total_sources < 10:
                    base_confidence *= 0.9
                
                return {
                    "brand_style_attributes": extracted_data,
                    "analysis_method": "structured_style_extraction",
                    "confidence": base_confidence,
                    "data_sources": total_sources,
                    "search_success_rate": success_rate,
                    "detailed_sources": data.get('detailed_sources', []),
                    "search_stats": search_stats
                }
            else:
                error_msg = "ANALYSIS FAILED: No response from LLM brand style analysis"
                logger.error(f"🚨 {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise e
            error_msg = f"ANALYSIS FAILED: Error in LLM brand style analysis: {str(e)}"
            logger.error(f"🚨 {error_msg}")
            raise RuntimeError(error_msg)

    def _parse_style_extraction_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response and extract brand style attributes"""
        try:
            # Clean up response
            content = response_text.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            extracted_data = json.loads(content)
            
            # Validate and clean the extracted data
            validated_data = {
                "brand_colors": extracted_data.get("brand_colors", {}),
                "typography": extracted_data.get("typography", {}),
                "logo": extracted_data.get("logo", {}),
                "visual_identity": extracted_data.get("visual_identity", {}),
                "extraction_confidence": extracted_data.get("extraction_confidence", {}),
                "sources_used": extracted_data.get("sources_used", [])
            }
            
            # Validate hex colors
            if "brand_colors" in validated_data:
                colors = validated_data["brand_colors"]
                for color_key in ["primary", "secondary"]:
                    if color_key in colors:
                        validated_data["brand_colors"][color_key] = self._validate_hex_color(colors[color_key])
                
                if "additional" in colors and isinstance(colors["additional"], list):
                    validated_data["brand_colors"]["additional"] = [
                        self._validate_hex_color(color) for color in colors["additional"]
                    ]
            
            return validated_data
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse brand style JSON: {e}")
            return {
                "brand_colors": {"primary": None, "secondary": None},
                "typography": {"primary_font": None},
                "logo": {},
                "visual_identity": {},
                "extraction_confidence": {"overall": "low"},
                "sources_used": [],
                "parsing_error": str(e)
            }

    def _validate_hex_color(self, color_value: str) -> Optional[str]:
        """Validate and normalize hex color codes"""
        if not color_value:
            return None
        
        # Remove any whitespace
        color_value = str(color_value).strip()
        
        # Add # if missing
        if not color_value.startswith('#'):
            color_value = '#' + color_value
        
        # Validate hex format
        hex_pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
        if re.match(hex_pattern, color_value):
            return color_value.upper()
        else:
            logger.warning(f"Invalid hex color format: {color_value}")
            return None

    async def _synthesize_results(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize brand style results and prepare for storage"""
        
        style_attributes = analysis.get("brand_style_attributes", {})
        
        # Structure the final brand style intelligence
        intelligence = {
            "brand_domain": self.brand_domain,
            "brand_style_attributes": style_attributes,
            "confidence_score": analysis.get("confidence", 0.8),
            "data_quality": "high" if analysis.get("analysis_method") == "structured_style_extraction" else "medium",
            "data_sources_count": analysis.get("data_sources", 0),
            "search_success_rate": analysis.get("search_success_rate", 0.0),
            "detailed_sources": analysis.get("detailed_sources", []),
            "search_stats": analysis.get("search_stats", {}),
            "extraction_timestamp": datetime.now().isoformat() + "Z"
        }
        
        return intelligence

    async def _save_results(self, research_results: Dict[str, Any]) -> List[str]:
        """Save brand style results to account.json and generate style.css"""
        saved_files = []
        
        try:
            style_attributes = research_results.get("brand_style_attributes", {})
            
            # 1. Update account.json with brand style attributes
            await self._update_account_config_with_style(style_attributes)
            
            # 2. Generate and save brand-specific style.css
            css_file = await self._generate_brand_css(style_attributes)
            if css_file:
                saved_files.append(css_file)
            
            # 3. Save detailed research results (using base class method)
            base_files = await super()._save_results(research_results)
            saved_files.extend(base_files)
            
            logger.info(f"✅ Saved brand style attributes and generated CSS for {self.brand_domain}")
            
        except Exception as e:
            logger.error(f"❌ Error saving brand style results: {e}")
            raise
        
        return saved_files

    async def _update_account_config_with_style(self, style_attributes: Dict[str, Any]) -> None:
        """Update account.json with extracted brand style attributes"""
        
        try:
            # Get existing config or create new one
            config = await self.storage_manager.get_account_config(self.brand_domain) or {}
            
            # Update with brand style attributes
            config.update({
                "brand_style": style_attributes,
                "brand_style_updated": datetime.now().isoformat() + "Z"
            })
            
            # Save config
            success = await self.storage_manager.save_account_config(self.brand_domain, config)
            if success:
                logger.info(f"💾 Updated account config with brand style attributes")
            else:
                logger.warning(f"⚠️ Failed to update account config with brand style")
                
        except Exception as e:
            logger.warning(f"⚠️ Error updating account config with brand style: {e}")

    async def _generate_brand_css(self, style_attributes: Dict[str, Any]) -> Optional[str]:
        """Generate brand-specific style.css file with comprehensive design system"""
        
        try:
            colors = style_attributes.get("brand_colors", {})
            typography = style_attributes.get("typography", {})
            design_system = style_attributes.get("design_system", {})
            component_patterns = style_attributes.get("component_patterns", {})
            brand_personality = style_attributes.get("brand_personality", {})
            visual_identity = style_attributes.get("visual_identity", {})
            
            # Brand personality context
            style_category = brand_personality.get("style_category", "modern")
            industry_alignment = brand_personality.get("industry_alignment", "general")
            
            # Generate CSS content with comprehensive design system
            css_content = f"""/* {self.brand_domain} Brand Design System */
/* Generated from comprehensive brand research including visual analysis */
/* Style Category: {style_category} | Industry: {industry_alignment} */
/* Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} */

:root {{
  /* ========================================
     BRAND COLOR SYSTEM
     ======================================== */
  
  /* Primary Brand Colors */
  --brand-primary: {colors.get('primary', '#2563eb')};
  --brand-secondary: {colors.get('secondary', '#1e293b')};
  
  /* Extended Color Palette */"""
            
            # Add comprehensive color system
            for color_key, default_value in [
                ('text_primary', '#1e293b'),
                ('text_secondary', '#64748b'),
                ('background_primary', '#ffffff'),
                ('background_secondary', '#f8fafc')
            ]:
                color_value = colors.get(color_key, default_value)
                css_content += f"\n  --{color_key.replace('_', '-')}: {color_value};"
            
            # Add additional colors if available
            additional_colors = colors.get("additional", [])
            for i, color in enumerate(additional_colors, 1):
                if color:
                    css_content += f"\n  --brand-accent-{i}: {color};"
            
            # Typography System
            primary_font = typography.get("primary_font", "Inter")
            secondary_font = typography.get("secondary_font", "Arial")
            display_font = typography.get("display_font", primary_font)
            font_fallbacks = typography.get("font_fallbacks", ["system-ui", "sans-serif"])
            fallback_str = ", ".join(font_fallbacks)
            
            css_content += f"""

  /* ========================================
     TYPOGRAPHY SYSTEM
     ======================================== */
  
  /* Font Families */
  --font-primary: '{primary_font}', {fallback_str};
  --font-secondary: '{secondary_font}', {fallback_str};
  --font-display: '{display_font}', {fallback_str};
  
  /* Font Weights */"""
            
            font_weights = typography.get("font_weights", ["400", "600", "700"])
            for weight in font_weights:
                css_content += f"\n  --font-weight-{weight}: {weight};"
            
            # Typography scales from research
            heading_styles = typography.get("heading_styles", {})
            body_styles = typography.get("body_styles", {})
            
            css_content += f"""
  
  /* Typography Scales */
  --text-h1-size: {heading_styles.get('h1_size', '32px')};
  --text-h1-weight: {heading_styles.get('h1_weight', '700')};
  --text-h1-line-height: {heading_styles.get('h1_line_height', '1.2')};
  --text-h1-letter-spacing: {heading_styles.get('h1_letter_spacing', '-0.5px')};
  
  --text-body-size: {body_styles.get('size', '16px')};
  --text-body-line-height: {body_styles.get('line_height', '1.6')};
  --text-body-letter-spacing: {body_styles.get('letter_spacing', '0px')};

  /* ========================================
     DESIGN SYSTEM TOKENS
     ======================================== */
  
  /* Spacing Scale */
  --spacing-xs: {design_system.get('spacing_unit', '8px').replace('8px', '4px')};
  --spacing-sm: {design_system.get('spacing_unit', '8px')};
  --spacing-md: {design_system.get('spacing_unit', '8px').replace('8px', '16px')};
  --spacing-lg: {design_system.get('spacing_unit', '8px').replace('8px', '24px')};
  --spacing-xl: {design_system.get('spacing_unit', '8px').replace('8px', '32px')};
  --spacing-2xl: {design_system.get('spacing_unit', '8px').replace('8px', '48px')};
  
  /* Border Radius Scale */
  --radius-sm: {design_system.get('border_radius_sm', '4px')};
  --radius-md: {design_system.get('border_radius_md', '8px')};
  --radius-lg: {design_system.get('border_radius_lg', '12px')};
  
  /* Shadow System */
  --shadow-sm: {design_system.get('shadow_sm', '0 1px 2px rgba(0,0,0,0.05)')};
  --shadow-md: {design_system.get('shadow_md', '0 4px 6px rgba(0,0,0,0.1)')};
  --shadow-lg: {design_system.get('shadow_lg', '0 10px 15px rgba(0,0,0,0.1)')};
  
  /* Animation System */
  --transition-fast: {design_system.get('transition_fast', '150ms ease')};
  --transition-normal: {design_system.get('transition_normal', '250ms ease')};
  --transition-slow: 350ms ease;
}}

/* ========================================
   UTILITY CLASSES
   ======================================== */

/* Color Utilities */
.brand-primary {{ color: var(--brand-primary); }}
.brand-secondary {{ color: var(--brand-secondary); }}
.text-primary {{ color: var(--text-primary); }}
.text-secondary {{ color: var(--text-secondary); }}

.bg-brand-primary {{ background-color: var(--brand-primary); }}
.bg-brand-secondary {{ background-color: var(--brand-secondary); }}
.bg-primary {{ background-color: var(--background-primary); }}
.bg-secondary {{ background-color: var(--background-secondary); }}

/* Typography Utilities */
.font-primary {{ font-family: var(--font-primary); }}
.font-secondary {{ font-family: var(--font-secondary); }}
.font-display {{ font-family: var(--font-display); }}

/* Component Base Styles */
.btn {{
  font-family: var(--font-primary);
  font-weight: var(--font-weight-600);
  border-radius: var(--radius-md);
  transition: all var(--transition-normal);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
}}"""
            
            # Add component patterns from research
            button_primary = component_patterns.get("button_primary", {})
            button_secondary = component_patterns.get("button_secondary", {})
            card_style = component_patterns.get("card_style", {})
            
            css_content += f"""

/* Primary Button (from brand analysis) */
.btn-primary {{
  background: {button_primary.get('background', 'var(--brand-primary)')};
  color: {button_primary.get('color', 'white')};
  border: none;
  padding: {button_primary.get('padding', 'var(--spacing-md) var(--spacing-lg)')};
  border-radius: {button_primary.get('border_radius', 'var(--radius-md)')};
  font-weight: {button_primary.get('font_weight', 'var(--font-weight-600)')};
}}

.btn-primary:hover {{
  transform: {button_primary.get('hover_transform', 'translateY(-1px)')};
  box-shadow: var(--shadow-lg);
}}

/* Secondary Button (from brand analysis) */
.btn-secondary {{
  background: {button_secondary.get('background', 'transparent')};
  color: {button_secondary.get('color', 'var(--brand-primary)')};
  border: {button_secondary.get('border', '2px solid var(--brand-primary)')};
  padding: {button_secondary.get('padding', 'calc(var(--spacing-md) - 2px) calc(var(--spacing-lg) - 2px)')};
}}

.btn-secondary:hover {{
  background: var(--brand-primary);
  color: white;
}}

/* Card Component (from brand analysis) */
.card {{
  background: {card_style.get('background', 'var(--background-primary)')};
  border-radius: {card_style.get('border_radius', 'var(--radius-md)')};
  box-shadow: {card_style.get('shadow', 'var(--shadow-md)')};
  padding: {card_style.get('padding', 'var(--spacing-lg)')};
  transition: all var(--transition-normal);
}}

.card:hover {{
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}}"""
            
            # Add enhanced product modal CSS using the comprehensive design system
            css_content += f"""

/* ========================================
   ENHANCED PRODUCT MODAL - BRAND AUTHENTIC
   ======================================== */
   
/* Generated using visual analysis and brand research */
/* Adapts to {style_category} {industry_alignment} brand characteristics */

/* Modal Overlay with Backdrop Blur */
.product-modal-overlay {{
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(30, 41, 59, 0.8);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transition: all var(--transition-slow);
}}

.product-modal-overlay.active {{
  opacity: 1;
  visibility: visible;
}}

/* Enhanced Modal Container */
.product-modal {{
  background: var(--background-primary);
  border-radius: var(--radius-lg);
  max-width: 1000px;
  width: 90%;
  max-height: 90vh;
  overflow: hidden;
  position: relative;
  box-shadow: var(--shadow-lg);
  transform: scale(0.95) translateY(20px);
  transition: transform var(--transition-slow);
  border: 1px solid var(--text-secondary, #e2e8f0);
}}

.product-modal-overlay.active .product-modal {{
  transform: scale(1) translateY(0);
}}

/* Enhanced Close Button */
.product-modal-close {{
  position: absolute;
  top: var(--spacing-md);
  right: var(--spacing-md);
  width: 40px;
  height: 40px;
  border: none;
  background: var(--background-secondary);
  color: var(--text-primary);
  border-radius: var(--radius-md);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  z-index: 10;
  transition: all var(--transition-fast);
  box-shadow: var(--shadow-sm);
}}

.product-modal-close:hover {{
  background: var(--brand-primary);
  color: white;
  transform: scale(1.1);
}}

/* Enhanced Modal Content Layout */
.product-modal-content {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  min-height: 500px;
}}

/* Enhanced Product Images */
.product-images {{
  position: relative;
  background: var(--background-secondary);
  padding: var(--spacing-xl);
}}

.product-image-carousel {{
  position: relative;
  aspect-ratio: 1;
  overflow: hidden;
  border-radius: var(--radius-md);
  background: var(--background-primary);
  box-shadow: var(--shadow-md);
}}

.product-image {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: none;
  transition: opacity var(--transition-normal);
}}

.product-image.active {{
  display: block;
}}

/* Enhanced Image Navigation */
.image-nav {{
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  background: var(--background-primary);
  color: var(--text-primary);
  border: none;
  width: 44px;
  height: 44px;
  border-radius: var(--radius-md);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition-fast);
  box-shadow: var(--shadow-md);
  font-size: 18px;
}}

.image-nav:hover {{
  background: var(--brand-primary);
  color: white;
  transform: translateY(-50%) scale(1.1);
}}

.image-nav.prev {{
  left: var(--spacing-md);
}}

.image-nav.next {{
  right: var(--spacing-md);
}}

/* Enhanced Thumbnails */
.product-thumbnails {{
  display: flex;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-md);
  overflow-x: auto;
  padding: var(--spacing-xs);
}}

.thumbnail {{
  width: 60px;
  height: 60px;
  object-fit: cover;
  border-radius: var(--radius-sm);
  cursor: pointer;
  opacity: 0.6;
  transition: all var(--transition-fast);
  border: 2px solid transparent;
  box-shadow: var(--shadow-sm);
}}

.thumbnail.active,
.thumbnail:hover {{
  opacity: 1;
  border-color: var(--brand-primary);
  transform: scale(1.05);
}}

/* Enhanced Product Details */
.product-details {{
  padding: var(--spacing-xl);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  background: var(--background-primary);
}}

.product-name {{
  font-family: var(--font-display);
  font-size: var(--text-h1-size);
  font-weight: var(--text-h1-weight);
  color: var(--text-primary);
  margin: 0;
  line-height: var(--text-h1-line-height);
  letter-spacing: var(--text-h1-letter-spacing);
}}

.product-price {{
  font-family: var(--font-primary);
  font-size: 28px;
  font-weight: var(--font-weight-600);
  color: var(--brand-primary);
  margin: 0;
}}

.product-description {{
  font-family: var(--font-secondary);
  font-size: var(--text-body-size);
  line-height: var(--text-body-line-height);
  color: var(--text-secondary);
  margin: 0;
}}

/* Enhanced Product Variants */
.product-variants {{
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}}

.variant-group {{
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}}

.variant-label {{
  font-family: var(--font-primary);
  font-size: 14px;
  font-weight: var(--font-weight-600);
  color: var(--text-primary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

.variant-options {{
  display: flex;
  gap: var(--spacing-sm);
  flex-wrap: wrap;
}}

.variant-option {{
  padding: var(--spacing-sm) var(--spacing-md);
  border: 2px solid var(--text-secondary, #e2e8f0);
  background: var(--background-primary);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-family: var(--font-primary);
  font-size: 14px;
  font-weight: 500;
  transition: all var(--transition-fast);
}}

.variant-option:hover,
.variant-option.selected {{
  border-color: var(--brand-primary);
  background: var(--brand-primary);
  color: white;
}}

/* Enhanced Quantity Selector */
.quantity-selector {{
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}}

.quantity-label {{
  font-family: var(--font-primary);
  font-size: 14px;
  font-weight: var(--font-weight-600);
  color: var(--text-primary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

.quantity-controls {{
  display: flex;
  align-items: center;
  border: 2px solid var(--text-secondary, #e2e8f0);
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--background-primary);
}}

.quantity-btn {{
  width: 44px;
  height: 44px;
  border: none;
  background: var(--background-secondary);
  color: var(--text-primary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  transition: all var(--transition-fast);
  font-weight: var(--font-weight-600);
}}

.quantity-btn:hover {{
  background: var(--brand-primary);
  color: white;
}}

.quantity-input {{
  width: 60px;
  height: 44px;
  border: none;
  text-align: center;
  font-family: var(--font-primary);
  font-size: 16px;
  font-weight: var(--font-weight-600);
  background: var(--background-primary);
}}

/* Enhanced Add to Cart Button */
.add-to-cart-btn {{
  background: var(--brand-primary);
  color: white;
  border: none;
  padding: var(--spacing-md) var(--spacing-xl);
  border-radius: var(--radius-md);
  font-family: var(--font-primary);
  font-size: 16px;
  font-weight: var(--font-weight-600);
  cursor: pointer;
  transition: all var(--transition-normal);
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-sm);
}}

.add-to-cart-btn:hover {{
  background: var(--brand-secondary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-lg);
}}

.add-to-cart-btn:active {{
  transform: translateY(0);
}}

.add-to-cart-btn.loading {{
  pointer-events: none;
  opacity: 0.8;
}}

/* Enhanced Product Features */
.product-features {{
  border-top: 1px solid var(--text-secondary, #e2e8f0);
  padding-top: var(--spacing-lg);
}}

.features-title {{
  font-family: var(--font-primary);
  font-size: 16px;
  font-weight: var(--font-weight-600);
  color: var(--text-primary);
  margin: 0 0 var(--spacing-md) 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

.features-list {{
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}}

.feature-item {{
  font-family: var(--font-secondary);
  font-size: 14px;
  color: var(--text-secondary);
  position: relative;
  padding-left: var(--spacing-lg);
}}

.feature-item::before {{
  content: "✓";
  position: absolute;
  left: 0;
  color: var(--brand-primary);
  font-weight: bold;
}}

/* Enhanced Responsive Design */
@media (max-width: 768px) {{
  .product-modal-content {{
    grid-template-columns: 1fr;
  }}
  
  .product-images {{
    order: 1;
  }}
  
  .product-details {{
    order: 2;
  }}
  
  .product-modal {{
    width: 95%;
    margin: 20px;
  }}
  
  .product-name {{
    font-size: 24px;
  }}
  
  .product-price {{
    font-size: 20px;
  }}
  
  .add-to-cart-btn {{
    padding: 14px 24px;
    font-size: 16px;
  }}
}}
"""
            
            # Save CSS file in style directory at same level as account.json
            css_file_path = "style/style.css"
            success = await self.storage_manager.write_file(
                self.brand_domain, 
                css_file_path, 
                css_content, 
                content_type="text/css"
            )
            
            if success:
                logger.info(f"📄 Generated brand CSS file: {css_file_path}")
                return css_file_path
            else:
                logger.warning(f"⚠️ Failed to save brand CSS file")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error generating brand CSS: {e}")
            return None


# Factory function
def get_brand_style_researcher(brand_domain: str) -> BrandStyleResearcher:
    """Get configured brand style researcher"""
    return BrandStyleResearcher(brand_domain=brand_domain)
