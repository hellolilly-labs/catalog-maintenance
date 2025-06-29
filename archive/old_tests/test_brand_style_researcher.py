#!/usr/bin/env python3
"""
Test Brand Style Researcher standalone
"""

import asyncio
import logging
from src.research.brand_style_research import get_brand_style_researcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_brand_style_researcher():
    """Test the brand style researcher with specialized.com"""
    
    brand_domain = "specialized.com"
    
    try:
        logger.info(f"🎨 Testing Brand Style Researcher for {brand_domain}")
        
        # Get the researcher
        researcher = get_brand_style_researcher(brand_domain)
        
        # Run the research
        result = await researcher.research(force_refresh=True)
        
        logger.info(f"✅ Brand Style Research completed!")
        logger.info(f"Quality Score: {result.get('quality_score', 'Unknown')}")
        logger.info(f"Files: {result.get('files', [])}")
        
        # Print some style attributes if available
        if 'brand_style_attributes' in result:
            style_attrs = result['brand_style_attributes']
            colors = style_attrs.get('brand_colors', {})
            if colors:
                logger.info(f"🎨 Brand Colors:")
                logger.info(f"  Primary: {colors.get('primary', 'Not found')}")
                logger.info(f"  Secondary: {colors.get('secondary', 'Not found')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Brand Style Research failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_brand_style_researcher())