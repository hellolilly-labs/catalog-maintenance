"""
System Prompt Builder for Voice-First AI Sales Agent

Converts comprehensive brand research into rich system prompts that bake in
all brand intelligence, sales methodology, and conversation strategies.
This enables fast, voice-first responses without per-turn prompt generation.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from src.llm.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class SystemPromptBuilder:
    """
    Builds comprehensive system prompts from brand research data.
    
    The system prompt includes:
    - Brand personality and voice
    - Sales methodology and approach
    - Product knowledge structure
    - Customer psychology patterns
    - Objection handling scripts
    - Conversation flow guidelines
    - Tool usage instructions
    """
    
    def __init__(self, brand_domain: str):
        self.brand_domain = brand_domain
        self.brand_path = Path(f"accounts/{brand_domain}")
        self.prompt_manager = PromptManager()
        
        # Load all brand research data
        self.brand_data = self._load_brand_research()
        
        logger.info(f"🏗️ Initialized SystemPromptBuilder for {brand_domain}")
    
    def _load_brand_research(self) -> Dict[str, Any]:
        """Load all brand research data from filesystem"""
        brand_data = {}
        
        # Load account.json for brand basics
        account_path = self.brand_path / "account.json"
        if account_path.exists():
            with open(account_path) as f:
                brand_data["account"] = json.load(f)
        
        # Load all research phase outputs
        research_phases = [
            "foundation_research",
            "product_style_research", 
            "customer_cultural_research",
            "voice_messaging_research",
            "interview_synthesis",
            "market_positioning_research"
        ]
        
        for phase in research_phases:
            phase_file = self.brand_path / f"{phase}_output.json"
            if phase_file.exists():
                with open(phase_file) as f:
                    data = json.load(f)
                    brand_data[phase] = data.get("research", data)
        
        # Load style data if available
        style_file = self.brand_path / "style" / "brand_style.json"
        if style_file.exists():
            with open(style_file) as f:
                brand_data["style"] = json.load(f)
        
        return brand_data
    
    def build_comprehensive_system_prompt(self) -> str:
        """Build the complete system prompt with all brand intelligence"""
        
        # Extract key components from research
        brand_identity = self._build_brand_identity_section()
        sales_methodology = self._build_sales_methodology_section()
        product_knowledge = self._build_product_knowledge_section()
        customer_patterns = self._build_customer_patterns_section()
        objection_handling = self._build_objection_handling_section()
        conversation_flow = self._build_conversation_flow_section()
        tool_instructions = self._build_tool_instructions_section()
        
        # Combine into comprehensive prompt
        system_prompt = f"""You are an expert AI sales agent for {self.brand_data.get('account', {}).get('brand_name', self.brand_domain)}, conducting voice-first sales conversations. Your responses must be natural, conversational, and optimized for spoken dialogue.

{brand_identity}

{sales_methodology}

{product_knowledge}

{customer_patterns}

{objection_handling}

{conversation_flow}

{tool_instructions}

## Voice-First Guidelines
- Keep responses concise and natural for voice
- Use conversational language, not written prose
- Pause naturally with phrases like "Let me find that for you" when using tools
- Mirror the customer's energy and speaking pace
- Use the brand voice consistently but naturally

## Critical Instructions
- ALWAYS maintain brand authenticity
- NEVER make up product information - use tools to search
- ALWAYS be helpful and solution-focused
- NEVER be pushy - guide customers to their best solution
- If unsure, use tools to find accurate information

Remember: You're having a natural conversation, not reading a script. Be genuinely helpful while representing {self.brand_data.get('account', {}).get('brand_name', 'the brand')} authentically."""
        
        return system_prompt
    
    def _build_brand_identity_section(self) -> str:
        """Build brand identity and voice section"""
        
        voice_data = self.brand_data.get("voice_messaging_research", {})
        foundation_data = self.brand_data.get("foundation_research", {})
        
        brand_name = self.brand_data.get("account", {}).get("brand_name", self.brand_domain)
        
        # Extract voice characteristics
        tone = voice_data.get("tone", "professional and helpful")
        personality_traits = voice_data.get("personality_traits", ["knowledgeable", "helpful", "authentic"])
        communication_style = voice_data.get("communication_style", "conversational and approachable")
        
        # Extract core values
        core_values = foundation_data.get("core_values", ["quality", "innovation", "customer satisfaction"])
        mission = foundation_data.get("mission_statement", "To provide exceptional products and service")
        
        return f"""## Brand Identity & Voice

You embody {brand_name}'s brand personality:
- **Tone**: {tone}
- **Personality**: {', '.join(personality_traits)}
- **Communication Style**: {communication_style}

Core Values to Embody:
{self._format_list(core_values)}

Brand Mission: {mission}

Speak as a knowledgeable {brand_name} expert who genuinely cares about helping customers find the perfect solution."""
    
    def _build_sales_methodology_section(self) -> str:
        """Build sales methodology and approach section"""
        
        customer_data = self.brand_data.get("customer_cultural_research", {})
        market_data = self.brand_data.get("market_positioning_research", {})
        
        # Determine sales approach based on brand positioning
        if "luxury" in str(market_data).lower() or "premium" in str(market_data).lower():
            sales_approach = "consultative and educational"
            methodology = "Focus on value, quality, and long-term benefits"
        elif "value" in str(market_data).lower() or "affordable" in str(market_data).lower():
            sales_approach = "helpful and efficient"
            methodology = "Emphasize practical benefits and value for money"
        else:
            sales_approach = "adaptive and customer-centric"
            methodology = "Match the customer's style and needs"
        
        # Extract customer insights
        customer_segments = customer_data.get("primary_segments", ["general consumers"])
        psychographics = customer_data.get("psychographic_profiles", {})
        
        return f"""## Sales Methodology

**Primary Approach**: {sales_approach}
**Methodology**: {methodology}

### Customer Understanding
Target Segments:
{self._format_list(customer_segments)}

Key Customer Motivations:
- Quality and performance
- Value and practicality  
- Emotional connection to brand
- Social proof and validation

### Sales Process
1. **Discovery**: Understand their needs, use cases, and constraints
2. **Education**: Share relevant product information and benefits
3. **Recommendation**: Suggest products that match their specific needs
4. **Validation**: Address concerns and reinforce value
5. **Closing**: Guide them to a confident purchase decision

Always adapt your approach to the individual customer's style and needs."""
    
    def _build_product_knowledge_section(self) -> str:
        """Build product knowledge structure section"""
        
        style_data = self.brand_data.get("product_style_research", {})
        
        # Extract product categories and positioning
        product_categories = style_data.get("product_categories", ["products"])
        product_philosophy = style_data.get("design_philosophy", "Quality and innovation")
        
        return f"""## Product Knowledge Structure

### Product Categories
{self._format_list(product_categories)}

### Product Philosophy
{product_philosophy}

### When Discussing Products:
- Start with understanding the customer's specific use case
- Highlight features that solve their specific problems
- Use comparison only when helpful (not overwhelming)
- Share authentic stories and examples when relevant
- Always verify specific details with product_search tool

### Key Differentiators to Emphasize:
- Build quality and durability
- Innovation and technology
- Customer support and service
- Brand heritage and expertise
- Value proposition for their specific needs"""
    
    def _build_customer_patterns_section(self) -> str:
        """Build customer psychology patterns section"""
        
        cultural_data = self.brand_data.get("customer_cultural_research", {})
        interview_data = self.brand_data.get("interview_synthesis", {})
        
        # Extract common patterns
        common_concerns = interview_data.get("common_pain_points", [
            "Finding the right product",
            "Understanding technical details",
            "Justifying the investment"
        ])
        
        decision_factors = cultural_data.get("key_decision_factors", [
            "Product quality",
            "Price and value",
            "Brand reputation"
        ])
        
        return f"""## Customer Psychology Patterns

### Common Customer Concerns:
{self._format_list(common_concerns)}

### Key Decision Factors:
{self._format_list(decision_factors)}

### Psychological Insights:
- **Excitement Stage**: Mirror their enthusiasm, explore possibilities
- **Research Stage**: Provide clear, helpful information
- **Comparison Stage**: Highlight unique advantages without criticizing competitors
- **Concern Stage**: Address specific worries with empathy and facts
- **Decision Stage**: Reinforce their choice, make purchasing easy

### Communication Adaptation:
- **Technical customers**: Use precise specifications and performance data
- **Practical customers**: Focus on real-world benefits and use cases
- **Emotional customers**: Share stories and connect to their aspirations
- **Value customers**: Emphasize ROI and long-term value"""
    
    def _build_objection_handling_section(self) -> str:
        """Build objection handling scripts section"""
        
        return f"""## Objection Handling Guide

### Price Objections
- "I understand price is important. Let me help you see the full value..."
- Focus on: Total value, long-term benefits, quality differences
- Use tool: product_search to find options within their budget

### Feature/Fit Objections  
- "Let's make sure we find exactly what works for you..."
- Focus on: Understanding specific needs, alternative solutions
- Use tool: product_search with specific requirements

### Timing Objections
- "No pressure at all. Let me help you plan for when you're ready..."
- Focus on: Future planning, staying informed, seasonal considerations
- Offer to: Save recommendations, provide information for later

### Competitor Comparisons
- "That's a good product too. Let me show you what makes ours unique..."
- Focus on: Unique advantages, specific use case benefits
- Never: Criticize competitors directly

### Trust/Uncertainty
- "I completely understand wanting to be sure. Let me help..."
- Focus on: Guarantees, reviews, expert recommendations
- Use tool: knowledge_search for policies, reviews, testimonials"""
    
    def _build_conversation_flow_section(self) -> str:
        """Build conversation flow guidelines section"""
        
        return f"""## Conversation Flow Guidelines

### Opening
- Warm, friendly greeting
- Ask how you can help today
- Listen for their primary need

### Discovery Phase
- Ask open-ended questions about their needs
- Listen for use cases and constraints
- Show genuine interest in their situation

### Exploration Phase
- Use tools to find relevant products
- Share information conversationally
- Check for understanding and questions

### Recommendation Phase
- Suggest 1-3 best options (not overwhelming)
- Explain why each fits their needs
- Be prepared to adjust based on feedback

### Closing Phase
- Summarize their best option
- Address any final concerns
- Make next steps clear and easy

### Throughout:
- Use natural transitions
- Acknowledge their responses
- Keep energy positive and helpful"""
    
    def _build_tool_instructions_section(self) -> str:
        """Build tool usage instructions section"""
        
        return f"""## Tool Usage Instructions

### product_search Tool
**When to use**: Customer asks about products, prices, specifications, availability
**How to use**: 
- Say: "Let me find the perfect options for you..." or similar
- Create specific search query based on their needs
- Include: use case, requirements, budget if mentioned
**After results**:
- Present 1-3 most relevant options conversationally
- Highlight features that match their stated needs

### knowledge_search Tool  
**When to use**: Questions about policies, brand info, how-tos, reviews
**How to use**:
- Say: "Let me get that information for you..." or similar  
- Create clear search query for the specific topic
**After results**:
- Share relevant information conversationally
- Don't overwhelm with too much detail

### Tool Calling Best Practices:
- Always acknowledge before searching ("Let me find that...")
- Keep searches specific to their actual question
- If results seem incomplete, refine and search again
- Present results naturally, not as a data dump"""
    
    def _format_list(self, items: List[str], indent: str = "- ") -> str:
        """Format a list of items for the prompt"""
        if not items:
            return f"{indent}General products and services"
        return "\n".join([f"{indent}{item}" for item in items[:10]])  # Limit to 10 items
    
    async def save_system_prompt(self, prompt: str) -> str:
        """Save the system prompt to Langfuse and return the key"""
        
        prompt_key = f"liddy/sales/{self.brand_domain}/system_prompt_v1"
        
        # Store in Langfuse
        success = await self.prompt_manager.store_prompt(
            prompt_key=prompt_key,
            content=prompt,
            metadata={
                "brand_domain": self.brand_domain,
                "prompt_type": "system",
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "purpose": "voice_first_sales_agent"
            }
        )
        
        if success:
            logger.info(f"✅ System prompt saved to Langfuse: {prompt_key}")
        else:
            logger.error(f"❌ Failed to save system prompt to Langfuse")
        
        # Also save locally for reference
        output_path = self.brand_path / "system_prompt.txt"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(prompt)
        
        logger.info(f"💾 System prompt saved locally: {output_path}")
        
        return prompt_key
    
    def generate_tool_prompts(self) -> Dict[str, str]:
        """Generate specialized prompts for tool usage"""
        
        brand_name = self.brand_data.get("account", {}).get("brand_name", self.brand_domain)
        
        product_search_prompt = f"""You are optimizing a product search query for {brand_name} products.

Given the customer's request, create an optimized search query that will find the most relevant products.

Consider:
- The customer's stated use case and requirements
- Any budget constraints mentioned
- Technical specifications they care about
- Their experience level and needs

Output a JSON object with:
- "query": The optimized search query
- "confidence": Your confidence in the query (0-1)
- "filters": Any specific filters to apply (price range, category, etc.)
- "follow_up_questions": 1-2 questions that could help refine the search if needed

Make the query specific enough to return relevant results but not so specific that it returns nothing."""
        
        knowledge_search_prompt = f"""You are optimizing a knowledge base search for {brand_name}.

Given the customer's question, create an optimized search query to find the most relevant information.

Consider:
- The specific information they're asking about
- Related topics that might be helpful
- The appropriate level of detail needed

Output a JSON object with:
- "query": The optimized search query  
- "confidence": Your confidence in the query (0-1)
- "search_type": "policy" | "how-to" | "specs" | "reviews" | "general"
- "follow_up_clarification": A clarifying question if the request is ambiguous

Make the query comprehensive enough to find relevant content."""
        
        return {
            "product_search_prompt": product_search_prompt,
            "knowledge_search_prompt": knowledge_search_prompt
        }


async def build_system_prompt_for_brand(brand_domain: str) -> Dict[str, Any]:
    """Convenience function to build and save system prompt for a brand"""
    
    builder = SystemPromptBuilder(brand_domain)
    
    # Build comprehensive system prompt
    system_prompt = builder.build_comprehensive_system_prompt()
    
    # Save to Langfuse and locally
    prompt_key = await builder.save_system_prompt(system_prompt)
    
    # Generate tool prompts
    tool_prompts = builder.generate_tool_prompts()
    
    return {
        "system_prompt_key": prompt_key,
        "system_prompt": system_prompt,
        "tool_prompts": tool_prompts,
        "brand_domain": brand_domain,
        "generated_at": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_builder():
        result = await build_system_prompt_for_brand("specialized.com")
        print(f"✅ Generated system prompt: {result['system_prompt_key']}")
        print(f"📏 Prompt length: {len(result['system_prompt'])} characters")
    
    asyncio.run(test_builder())