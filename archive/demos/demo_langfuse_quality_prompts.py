#!/usr/bin/env python3
"""
Demo: Langfuse Integration for Quality Evaluation Prompts
Shows how to set up and use Langfuse chat templates for quality evaluation
"""

import asyncio
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockPromptManager:
    """Mock prompt manager to demonstrate Langfuse integration pattern"""
    
    def __init__(self):
        # Simulate Langfuse prompts for different research phases
        self.langfuse_prompts = {
            "liddy/catalog/quality/foundation_evaluator": {
                "system": """You are an expert quality evaluator for FOUNDATION brand research.

Foundation research focuses on core brand identity that rarely changes:
- Company founding story, history, timeline
- Mission, vision, core values statements  
- Leadership team and organizational culture
- Legal structure, ownership, headquarters
- Patents, foundational innovations

Evaluate the research quality on a scale of 0.0 to 10.0 based on these criteria:

EVALUATION CRITERIA:
- Accuracy: Information is factual and well-sourced (0-2 points)
- Completeness: All required foundation elements present (0-2 points)  
- Consistency: No contradictions or conflicts (0-2 points)
- Authenticity: Captures genuine brand voice and history (0-2 points)
- Actionability: Provides implementable insights for sales agents (0-2 points)

QUALITY STANDARDS FOR FOUNDATION RESEARCH:
- 9.0-10.0: Exceptional quality, comprehensive foundation coverage
- 8.0-8.9: High quality, minor gaps in historical details
- 7.0-7.9: Good quality, some foundation elements missing
- 6.0-6.9: Acceptable quality, significant historical gaps
- Below 6.0: Poor quality, insufficient foundation research

Respond in JSON format with your evaluation.""",
                
                "user": """Evaluate this FOUNDATION research for {brand_domain}:

RESEARCH CONTENT:
{research_content}

CONTEXT:
- Current confidence score: {current_confidence_score}
- Quality threshold: {quality_threshold}/10.0

Focus your evaluation on foundation-specific requirements:
✅ Company founding story and historical timeline
✅ Mission, vision, and core values clearly articulated
✅ Leadership team and organizational culture
✅ Legal structure and business fundamentals
✅ Foundational innovations and patents

Provide evaluation in this JSON format:
{{
    "quality_score": 8.2,
    "criteria_met": {{
        "accuracy": true,
        "completeness": true, 
        "consistency": true,
        "authenticity": true,
        "actionability": true
    }},
    "improvement_feedback": [
        "Add more specific founding date and initial funding details",
        "Include more leadership background information"
    ],
    "confidence_level": "high"
}}"""
            },
            
            "liddy/catalog/quality/product_style_evaluator": {
                "system": """You are an expert quality evaluator for PRODUCT STYLE brand research.

Product Style research focuses on design philosophy and aesthetic elements:
- Visual design language and aesthetic principles
- Product styling consistency across portfolio
- Design innovation and differentiation
- Materials, colors, and form factors
- User experience and design philosophy

Evaluate the research quality on a scale of 0.0 to 10.0 based on these criteria:

EVALUATION CRITERIA:
- Accuracy: Design information is factual and well-sourced (0-2 points)
- Completeness: All required product style elements present (0-2 points)  
- Consistency: No contradictions in design philosophy (0-2 points)
- Authenticity: Captures genuine brand design voice (0-2 points)
- Actionability: Provides implementable design insights (0-2 points)

QUALITY STANDARDS FOR PRODUCT STYLE RESEARCH:
- 9.0-10.0: Exceptional quality, comprehensive design analysis
- 8.0-8.9: High quality, minor gaps in design details
- 7.0-7.9: Good quality, some style elements missing
- 6.0-6.9: Acceptable quality, significant design gaps
- Below 6.0: Poor quality, insufficient product style research

Respond in JSON format with your evaluation.""",
                
                "user": """Evaluate this PRODUCT STYLE research for {brand_domain}:

RESEARCH CONTENT:
{research_content}

CONTEXT:
- Current confidence score: {current_confidence_score}
- Quality threshold: {quality_threshold}/10.0

Focus your evaluation on product style-specific requirements:
✅ Visual design language and aesthetic principles
✅ Product styling consistency across product lines
✅ Design innovation and competitive differentiation
✅ Materials, colors, form factors, and finishes
✅ User experience philosophy and design thinking

Provide evaluation in this JSON format:
{{
    "quality_score": 8.2,
    "criteria_met": {{
        "accuracy": true,
        "completeness": true, 
        "consistency": true,
        "authenticity": true,
        "actionability": true
    }},
    "improvement_feedback": [
        "Include more specific color palette and material details",
        "Add more user experience design philosophy"
    ],
    "confidence_level": "high"
}}"""
            }
        }
    
    async def get_prompt(self, prompt_name: str, prompt_type: str = "chat", prompt: list = None):
        """Mock Langfuse get_prompt method"""
        
        if prompt_name in self.langfuse_prompts:
            # Simulate successful Langfuse retrieval
            langfuse_prompt = self.langfuse_prompts[prompt_name]
            
            class MockPromptTemplate:
                def __init__(self, prompt_data):
                    self.prompt = [
                        {"role": "system", "content": prompt_data["system"]},
                        {"role": "user", "content": prompt_data["user"]}
                    ]
                    self.version = "v1.2"
                    self.name = prompt_name
            
            logger.info(f"✅ Langfuse prompt found: {prompt_name}")
            return MockPromptTemplate(langfuse_prompt)
        else:
            # Simulate fallback to provided prompts
            if prompt:
                class MockPromptTemplate:
                    def __init__(self, fallback_prompt):
                        self.prompt = fallback_prompt
                        self.version = "fallback_1.0"
                        self.name = prompt_name + "_fallback"
                
                logger.warning(f"⚠️ Langfuse prompt not found, using fallback: {prompt_name}")
                return MockPromptTemplate(prompt)
            else:
                logger.error(f"❌ No Langfuse prompt or fallback found: {prompt_name}")
                return None


class MockQualityEvaluator:
    """Mock quality evaluator using the Langfuse integration pattern"""
    
    def __init__(self, researcher_name: str):
        self.researcher_name = researcher_name
        self.prompt_manager = MockPromptManager()
    
    async def _get_quality_evaluator_prompt(self) -> Dict[str, str]:
        """Get phase-specific quality evaluator prompt using Langfuse chat templates"""
        
        # Get fallback prompts first
        fallback_prompts = self._get_fallback_quality_evaluator_prompt()
        
        try:
            # Try to get phase-specific prompt from Langfuse using chat template pattern
            prompt_key = f"liddy/catalog/quality/{self.researcher_name}_evaluator"
            
            prompt_template = await self.prompt_manager.get_prompt(
                prompt_name=prompt_key,
                prompt_type="chat", 
                prompt=[
                    {"role": "system", "content": fallback_prompts["system"]},
                    {"role": "user", "content": fallback_prompts["user_template"]}
                ]
            )
            
            if prompt_template and prompt_template.prompt:
                prompts = prompt_template.prompt
                
                system_prompt = next((msg["content"] for msg in prompts if msg["role"] == "system"), None)
                user_prompt = next((msg["content"] for msg in prompts if msg["role"] == "user"), None)
                
                if system_prompt and user_prompt:
                    logger.info(f"✅ Using Langfuse quality evaluator prompt for {self.researcher_name}")
                    return {
                        "system": system_prompt,
                        "user_template": user_prompt
                    }
            
            logger.warning(f"Using fallback quality evaluator prompt for {self.researcher_name}")
            return fallback_prompts
            
        except Exception as e:
            logger.warning(f"Langfuse prompt failed for {self.researcher_name}_evaluator, using fallback: {e}")
            return fallback_prompts
    
    def _get_fallback_quality_evaluator_prompt(self) -> Dict[str, str]:
        """Fallback quality evaluator prompt when Langfuse is unavailable"""
        
        system_prompt = f"""You are an expert quality evaluator for {self.researcher_name} brand research.

Evaluate the research quality on a scale of 0.0 to 10.0 based on these criteria:

EVALUATION CRITERIA:
- Accuracy: Information is factual and well-sourced (0-2 points)
- Completeness: All required elements present (0-2 points)  
- Consistency: No contradictions or conflicts (0-2 points)
- Authenticity: Captures genuine brand voice (0-2 points)
- Actionability: Provides implementable insights (0-2 points)

QUALITY STANDARDS:
- 9.0-10.0: Exceptional quality, production ready
- 8.0-8.9: High quality, minor improvements possible
- 7.0-7.9: Good quality, some improvements needed
- 6.0-6.9: Acceptable quality, significant improvements needed
- Below 6.0: Poor quality, major rework required

Respond in JSON format with your evaluation."""

        user_template = """Evaluate this {phase_name} research for {brand_domain}:

RESEARCH CONTENT:
{research_content}

CONTEXT:
- Current confidence score: {current_confidence_score}
- Quality threshold: {quality_threshold}/10.0

Provide evaluation in this JSON format:
{{
    "quality_score": 8.2,
    "criteria_met": {{
        "accuracy": true,
        "completeness": true, 
        "consistency": true,
        "authenticity": true,
        "actionability": true
    }},
    "improvement_feedback": [
        "Specific suggestion 1",
        "Specific suggestion 2"
    ],
    "confidence_level": "high"
}}"""

        return {
            "system": system_prompt,
            "user_template": user_template
        }


async def demo_langfuse_integration():
    """Demonstrate Langfuse integration for quality evaluation prompts"""
    
    print("🔌 Langfuse Quality Evaluation Prompts Demo")
    print("=" * 60)
    
    # Test different research phases with Langfuse prompts
    test_phases = [
        ("foundation", "Has Langfuse prompt"),
        ("product_style", "Has Langfuse prompt"), 
        ("market_positioning", "No Langfuse prompt - will use fallback"),
        ("customer_cultural", "No Langfuse prompt - will use fallback")
    ]
    
    for phase_name, description in test_phases:
        print(f"\n📋 Testing: {phase_name} ({description})")
        print("-" * 50)
        
        evaluator = MockQualityEvaluator(phase_name)
        prompt_data = await evaluator._get_quality_evaluator_prompt()
        
        # Show prompt details
        system_preview = prompt_data["system"][:100] + "..." if len(prompt_data["system"]) > 100 else prompt_data["system"]
        user_preview = prompt_data["user_template"][:100] + "..." if len(prompt_data["user_template"]) > 100 else prompt_data["user_template"]
        
        print(f"✅ System Prompt: {system_preview}")
        print(f"✅ User Template: {user_preview}")


async def demo_prompt_customization():
    """Show how to customize quality evaluation prompts per research phase"""
    
    print(f"\n🎯 Quality Prompt Customization Guide")
    print("=" * 60)
    
    customization_examples = {
        "foundation": {
            "focus": "Company history, mission, values, leadership",
            "key_criteria": [
                "Historical accuracy and timeline completeness",
                "Mission/vision statement clarity and authenticity", 
                "Leadership background and organizational culture",
                "Legal structure and business fundamentals"
            ],
            "quality_threshold": 8.0
        },
        "product_style": {
            "focus": "Design language, aesthetics, user experience",
            "key_criteria": [
                "Visual design consistency across product portfolio",
                "Material, color, and form factor analysis",
                "Design innovation and competitive differentiation",
                "User experience philosophy documentation"
            ],
            "quality_threshold": 8.0
        },
        "customer_cultural": {
            "focus": "Customer psychology, cultural insights, behavior",
            "key_criteria": [
                "Cultural context and demographic analysis",
                "Customer psychology and behavioral patterns",
                "Community engagement and brand loyalty",
                "Cultural sensitivity and inclusivity"
            ],
            "quality_threshold": 8.5
        },
        "interview_synthesis": {
            "focus": "Human insights synthesis, qualitative analysis",
            "key_criteria": [
                "Interview data synthesis accuracy",
                "Qualitative insight extraction quality",
                "Human perspective authenticity",
                "Actionable recommendation generation"
            ],
            "quality_threshold": 9.0
        }
    }
    
    print(f"🎯 **Phase-Specific Quality Evaluation Customization:**\n")
    
    for phase, config in customization_examples.items():
        print(f"**{phase.upper()} Research Evaluation:**")
        print(f"   Focus: {config['focus']}")
        print(f"   Quality Threshold: {config['quality_threshold']}/10.0")
        print(f"   Key Criteria:")
        for criterion in config['key_criteria']:
            print(f"     • {criterion}")
        print()


async def demo_setup_instructions():
    """Show setup instructions for Langfuse quality evaluation prompts"""
    
    print(f"\n⚙️ Langfuse Setup Instructions")
    print("=" * 60)
    
    setup_steps = [
        "1. 🔐 **Configure Langfuse Credentials**",
        "   Set LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST in environment",
        "",
        "2. 📝 **Create Quality Evaluator Prompts in Langfuse UI**",
        "   Navigate to Langfuse → Prompts → Create New Prompt",
        "",
        "3. 🏷️ **Use Standardized Naming Convention**",
        "   Format: `liddy/catalog/quality/{phase_name}_evaluator`",
        "   Examples:",
        "     • liddy/catalog/quality/foundation_evaluator",
        "     • liddy/catalog/quality/product_style_evaluator", 
        "     • liddy/catalog/quality/customer_cultural_evaluator",
        "",
        "4. 💬 **Set Prompt Type to 'Chat'**",
        "   Use chat template format with system and user messages",
        "",
        "5. 🎯 **Customize System Message**",
        "   Include phase-specific evaluation criteria and quality standards",
        "",
        "6. 📋 **Customize User Message Template**",
        "   Include context variables: {brand_domain}, {research_content}, etc.",
        "",
        "7. 🔄 **Test and Iterate**",
        "   Run quality evaluations and refine prompts based on results",
        "",
        "8. ✅ **Fallback Protection**",
        "   System automatically falls back to default prompts if Langfuse unavailable"
    ]
    
    for step in setup_steps:
        print(step)
    
    print(f"\n📋 **Example Langfuse Prompt Structure:**")
    print(f"```")
    print(f"Prompt Name: liddy/catalog/quality/foundation_evaluator")
    print(f"Prompt Type: chat")
    print(f"")
    print(f"System Message:")
    print(f"  You are an expert quality evaluator for FOUNDATION brand research...")
    print(f"")
    print(f"User Message:")
    print(f"  Evaluate this FOUNDATION research for {{brand_domain}}:")
    print(f"  RESEARCH CONTENT: {{research_content}}")
    print(f"  ...")
    print(f"```")


async def main():
    """Run the Langfuse quality evaluation demo"""
    
    try:
        await demo_langfuse_integration()
        await demo_prompt_customization()
        await demo_setup_instructions()
        
        print(f"\n🎉 Langfuse Integration Benefits:")
        print(f"   ✅ Phase-specific quality evaluation criteria")
        print(f"   ✅ Centralized prompt management in Langfuse UI")
        print(f"   ✅ Easy prompt iteration and version control")
        print(f"   ✅ Automatic fallback to default prompts")
        print(f"   ✅ Consistent chat template pattern across all researchers")
        print(f"   ✅ Real-time prompt updates without code deployment")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Configure Langfuse credentials in environment")
        print(f"   2. Create quality evaluator prompts for each research phase")
        print(f"   3. Test quality evaluation with real research content")
        print(f"   4. Iterate and refine prompts based on evaluation results")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 