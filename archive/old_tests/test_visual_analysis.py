#!/usr/bin/env python3
"""
Test script for visual analysis functionality
"""

import asyncio
from src.research.brand_style_research import BrandStyleResearcher

async def test_visual_analysis():
    print("🎯 Testing Visual Analysis for Specialized.com")
    print("=" * 50)
    
    researcher = BrandStyleResearcher("specialized.com")
    
    try:
        # Test visual analysis capability
        visual_data = await researcher._capture_visual_analysis()
        
        if visual_data:
            print(f"✅ Visual analysis captured {len(visual_data)} screenshots")
            for i, data in enumerate(visual_data):
                page_name = data.get('page_name', data.get('source_query', 'Unknown'))
                print(f"   📷 Screenshot {i+1}: {page_name}")
                if data.get('design_analysis'):
                    analysis = data['design_analysis']
                    print(f"   🎨 Design analysis: {analysis[:150]}...")
                    print()
        else:
            print("❌ No visual data captured")
    
    except Exception as e:
        print(f"❌ Error in visual analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_visual_analysis())