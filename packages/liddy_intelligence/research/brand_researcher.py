"""
Brand Researcher CLI

Main entry point for the Brand Research Pipeline per ROADMAP Section 4.1.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any

from liddy_intelligence.research.foundation_research import get_foundation_researcher
from liddy_intelligence.research.market_positioning_research import get_market_positioning_researcher
from liddy_intelligence.workflow.workflow_state_manager import get_workflow_manager, WorkflowState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrandResearcher:
    """Main Brand Research Pipeline Controller"""
    
    def __init__(self):
        self.foundation_researcher = get_foundation_researcher()
        self.market_positioning_researcher = get_market_positioning_researcher()
        self.workflow_manager = get_workflow_manager()
        
    async def run_foundation_research(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Run foundation research phase only"""
        
        logger.info(f"🏗️ Starting Foundation Research for {brand_domain}")
        
        # Update workflow state to indicate research in progress
        self.workflow_manager.update_brand_state(
            brand_domain, 
            WorkflowState.RESEARCH_IN_PROGRESS, 
            phase_name="foundation"
        )
        
        start_time = time.time()
        
        try:
            result = await self.foundation_researcher.research(
                brand_domain=brand_domain,
                force_refresh=force_refresh
            )
            
            duration = time.time() - start_time
            quality_score = result.get("quality_score", result.get("confidence_score", 0.8))
            
            # Mark foundation research phase as complete
            self.workflow_manager.mark_phase_complete(
                brand_domain, 
                "foundation", 
                quality_score=quality_score, 
                duration=duration / 60.0
            )
            
            # Check if all research phases are complete
            research_summary = self.workflow_manager.get_research_progress_summary(brand_domain)
            all_research_complete = research_summary["all_complete"]
            
            if all_research_complete:
                # All 8 research phases complete - ready for catalog
                self.workflow_manager.update_brand_state(
                    brand_domain,
                    WorkflowState.RESEARCH_COMPLETE,
                    phase_name="all_research_phases"
                )
                logger.info(f"🎉 ALL RESEARCH COMPLETE: {brand_domain} → RESEARCH_COMPLETE")
            else:
                # Still more research phases to complete
                self.workflow_manager.update_brand_state(
                    brand_domain,
                    WorkflowState.RESEARCH_IN_PROGRESS,
                    phase_name="foundation"
                )
                next_phase = research_summary["next_phase"]
                logger.info(f"📋 Research phase complete: {brand_domain} → RESEARCH_IN_PROGRESS")
                logger.info(f"🔄 Progress: {research_summary['completed_count']}/{research_summary['total_phases']} phases ({research_summary['completion_percentage']:.1f}%)")
                logger.info(f"🔄 Next: {next_phase}")
            
            logger.info(f"✅ Foundation Research completed in {duration:.1f}s")
            
            return result
            
        except Exception as e:
            # Update workflow state to indicate failure
            self.workflow_manager.update_brand_state(
                brand_domain,
                WorkflowState.FAILED,
                phase_name="foundation",
                error=str(e)
            )
            
            logger.error(f"❌ Foundation Research failed: {e}")
            logger.error(f"📊 Workflow state updated: {brand_domain} → FAILED")
            raise

    async def run_market_positioning_research(self, brand_domain: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Run market positioning research phase only"""
        
        logger.info(f"🏁 Starting Market Positioning Research for {brand_domain}")
        
        # Update workflow state to indicate research in progress
        self.workflow_manager.update_brand_state(
            brand_domain, 
            WorkflowState.RESEARCH_IN_PROGRESS, 
            phase_name="market_positioning"
        )
        
        start_time = time.time()
        
        try:
            result = await self.market_positioning_researcher.research(
                brand_domain=brand_domain,
                force_refresh=force_refresh
            )
            
            duration = time.time() - start_time
            quality_score = result.get("quality_score", result.get("confidence_score", 0.75))
            
            # Mark market positioning research phase as complete
            self.workflow_manager.mark_phase_complete(
                brand_domain, 
                "market_positioning", 
                quality_score=quality_score, 
                duration=duration / 60.0
            )
            
            # Check if all research phases are complete
            research_summary = self.workflow_manager.get_research_progress_summary(brand_domain)
            all_research_complete = research_summary["all_complete"]
            
            if all_research_complete:
                # All 8 research phases complete - ready for catalog
                self.workflow_manager.update_brand_state(
                    brand_domain,
                    WorkflowState.RESEARCH_COMPLETE,
                    phase_name="all_research_phases"
                )
                logger.info(f"🎉 ALL RESEARCH COMPLETE: {brand_domain} → RESEARCH_COMPLETE")
            else:
                # Still more research phases to complete
                self.workflow_manager.update_brand_state(
                    brand_domain,
                    WorkflowState.RESEARCH_IN_PROGRESS,
                    phase_name="market_positioning"
                )
                next_phase = research_summary["next_phase"]
                logger.info(f"📋 Research phase complete: {brand_domain} → RESEARCH_IN_PROGRESS")
                logger.info(f"🔄 Progress: {research_summary['completed_count']}/{research_summary['total_phases']} phases ({research_summary['completion_percentage']:.1f}%)")
                logger.info(f"🔄 Next: {next_phase}")
            
            logger.info(f"✅ Market Positioning Research completed in {duration:.1f}s")
            
            return result
            
        except Exception as e:
            # Update workflow state to indicate failure
            self.workflow_manager.update_brand_state(
                brand_domain,
                WorkflowState.FAILED,
                phase_name="market_positioning",
                error=str(e)
            )
            
            logger.error(f"❌ Market Positioning Research failed: {e}")
            logger.error(f"📊 Workflow state updated: {brand_domain} → FAILED")
            raise

    def generate_brand_details_md(self, research_result: Dict[str, Any]) -> str:
        """Generate brand_details.md content from research results"""
        
        brand_domain = research_result.get("brand_domain", "unknown")
        foundation_content = research_result.get("foundation_content", "")
        market_positioning_content = research_result.get("market_positioning_content", "")
        
        # If we have foundation research content, use it directly
        if foundation_content:
            return foundation_content
        
        # If we have market positioning content, use it directly
        if market_positioning_content:
            return market_positioning_content
        
        # Fallback for old format
        foundation = research_result.get("research_phases", {}).get("foundation", {})
        foundation_intelligence = foundation.get("foundation_intelligence", {})
        
        # Extract key information for markdown
        brand_name = brand_domain.replace('.com', '').replace('.', ' ').title()
        
        md_content = f"""# Brand Intelligence: {brand_name}

## Foundation Research

**Research Date**: {foundation.get("research_metadata", {}).get("timestamp", "Unknown")}
**Cache Expires**: {foundation.get("research_metadata", {}).get("cache_expires", "Unknown")}
**Confidence Score**: {foundation.get("confidence_score", "Unknown")}
**Data Quality**: {foundation.get("data_quality", "Unknown")}

---

## Analysis Summary

{foundation_intelligence.get("foundation_analysis", "No foundation analysis available")}

---

## Research Metadata

- **Domain**: {brand_domain}
- **Research Duration**: {foundation.get("research_metadata", {}).get("research_duration_seconds", 0):.1f} seconds
- **Quality Threshold**: {foundation.get("research_metadata", {}).get("quality_threshold", "Unknown")}
- **Analysis Method**: {foundation_intelligence.get("analysis_method", "Unknown")}

---

*Generated by Brand Research Pipeline v{foundation.get("research_metadata", {}).get("version", "1.0")}*
"""
        
        return md_content


async def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(description="Brand Research Pipeline")
    
    parser.add_argument("--brand", required=True, help="Brand domain (e.g., specialized.com)")
    parser.add_argument("--foundation", action="store_true", help="Run foundation research only")
    parser.add_argument("--market-positioning", action="store_true", help="Run market positioning research only")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh of cached research")
    parser.add_argument("--output-format", choices=["json", "markdown"], default="json", help="Output format")
    parser.add_argument("--status", action="store_true", help="Show workflow status for brand")
    
    args = parser.parse_args()
    
    # Initialize workflow manager for status check
    workflow_manager = get_workflow_manager()
    
    # If status flag is set, show status and exit
    if args.status:
        workflow_info = workflow_manager.get_brand_info(args.brand)
        next_step = workflow_manager.get_next_step(args.brand)
        
        print(f"\n🔍 WORKFLOW STATUS: {args.brand}")
        print("=" * 50)
        print(f"Current State: {workflow_info.current_state.value}")
        print(f"Last Updated: {workflow_info.last_updated}")
        print(f"Completed Phases: {workflow_info.completed_phases}")
        if workflow_info.failed_phases:
            print(f"Failed Phases: {workflow_info.failed_phases}")
        if workflow_info.total_research_time > 0:
            print(f"Total Research Time: {workflow_info.total_research_time:.1f} minutes")
        print(f"\nNext Step: {next_step}")
        return
    
    # Check that exactly one research phase is selected
    if not (args.foundation or args.market_positioning):
        parser.error("Must specify one research phase: --foundation or --market-positioning")
    
    if args.foundation and args.market_positioning:
        parser.error("Cannot specify multiple research phases at once")
    
    # Initialize researcher
    researcher = BrandResearcher()
    
    try:
        if args.foundation:
            result = await researcher.run_foundation_research(
                brand_domain=args.brand,
                force_refresh=args.force_refresh
            )
        elif args.market_positioning:
            result = await researcher.run_market_positioning_research(
                brand_domain=args.brand,
                force_refresh=args.force_refresh
            )
        
        # Output results based on format
        if args.output_format == "json":
            # For JSON output, show metadata only (not the full markdown content)
            output_result = {
                "brand_domain": result.get("brand_domain", result.get("brand")),
                "confidence_score": result.get("confidence_score", result.get("quality_score")),
                "data_quality": result.get("data_quality"),
                "data_sources_count": result.get("data_sources_count", result.get("data_sources")),
                "research_metadata": result.get("research_metadata"),
                "content_preview": (
                    result.get("foundation_content", result.get("market_positioning_content", ""))[:200] + "..." 
                    if result.get("foundation_content") or result.get("market_positioning_content") 
                    else "No content"
                )
            }
            print(json.dumps(output_result, indent=2))
        elif args.output_format == "markdown":
            content = result.get("foundation_content") or result.get("market_positioning_content", "")
            if content:
                print(content)
            else:
                print("No markdown content available")
        
        # Show next step recommendation
        next_step = workflow_manager.get_next_step(args.brand)
        print(f"\n📋 Next Step: {next_step}")
        
        logger.info("✅ Brand research completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Brand research failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
