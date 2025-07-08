#!/usr/bin/env python3
"""
Pre-generate Product Descriptors with Brand Research

This script pre-generates and caches product descriptors in products.json
using brand research intelligence for enhanced quality. Uses ProductManager for all catalog access.

Also includes model evaluation mode to compare descriptor generation quality between:
- OpenAI o3
- OpenAI o3-mini  
- Anthropic Claude 3.5 Sonnet

Usage:
    python pre_generate_descriptors.py specialized.com
    python pre_generate_descriptors.py --force specialized.com
    python pre_generate_descriptors.py --quality-threshold 0.9 brand.com
    python pre_generate_descriptors.py --evaluate-models specialized.com
    python pre_generate_descriptors.py --evaluate-models --sample-size 10 brand.com
"""

import argparse
import logging
import sys
import os
import asyncio
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liddy_intelligence.catalog.unified_descriptor_generator import generate_descriptors, UnifiedDescriptorGenerator, DescriptorConfig
from liddy.models.product_manager import get_product_manager
from liddy.models.product import Product
from liddy.llm.simple_factory import LLMFactory

# Import researcher classes
from liddy_intelligence.research.foundation_research import get_foundation_researcher
from liddy_intelligence.research.voice_messaging_research import get_voice_messaging_researcher
from liddy_intelligence.research.product_style_research import get_product_style_researcher
from liddy_intelligence.research.market_positioning_research import get_market_positioning_researcher
from liddy_intelligence.research.customer_cultural_research import get_customer_cultural_researcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def evaluate_models(brand_domain: str, sample_size: int = 6) -> Dict[str, Any]:
    """
    Compare descriptor generation between o3 and o3-mini models.
    
    Args:
        brand_domain: The brand to evaluate
        sample_size: Number of products to sample for evaluation
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"\n🔬 Starting model evaluation for {brand_domain}")
    logger.info(f"   Comparing: openai/o3 vs openai/o3-mini vs anthropic/claude-3-5-sonnet")
    logger.info(f"   Sample size: {sample_size} products")
    
    # Save results
    output_dir = Path(f"evaluation_results/{brand_domain}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load products
    product_manager = await get_product_manager(brand_domain)
    all_products = await product_manager.get_products()
    
    if not all_products:
        raise ValueError(f"No products found for {brand_domain}")
    
    # Sample random products
    sample_products = random.sample(all_products, min(sample_size, len(all_products)))
    logger.info(f"   Sampled {len(sample_products)} products for evaluation")
    
    models_to_compare = ["anthropic/claude-sonnet-4-20250514", "openai/o3", "openai/o3-mini"]
    
    # Results storage
    evaluation_results = {
        "brand": brand_domain,
        "evaluation_date": datetime.now().isoformat(),
        "models_compared": models_to_compare,
        "sample_size": len(sample_products),
        "product_evaluations": []
    }
    
    # Test both models on each product
    for idx, product in enumerate(sample_products, 1):
        logger.info(f"\n📦 Evaluating product {idx}/{len(sample_products)}: {product.name}")
        
        product_result = {
            "product_id": product.id,
            "product_name": product.name,
            "category": product.categories[0] if product.categories else "Unknown",
            "model_results": {}
        }
        
        # Generate descriptors with each model
        for model in models_to_compare:
            logger.info(f"   🤖 Generating with {model}...")
            
            # Create generator with specific model
            config = DescriptorConfig(use_research=True)
            generator = UnifiedDescriptorGenerator(brand_domain, config)
            
            # Load research if available
            generator.product_catalog_intelligence = await generator._load_product_catalog_intelligence()
            
            # Generate descriptor
            start_time = datetime.now()
            
            try:
                # Create a copy of the product to avoid modifying the original
                test_product = Product.from_dict(product.to_dict())
                
                # Generate descriptor with specific model
                await generator._generate_descriptor(test_product, model=model)
                
                generation_time = (datetime.now() - start_time).total_seconds()
                
                # Store results
                product_result["model_results"][model] = {
                    "descriptor": test_product.descriptor,
                    "search_keywords": test_product.search_keywords,
                    "key_selling_points": test_product.key_selling_points,
                    "voice_summary": test_product.voice_summary,
                    "quality_score": test_product.descriptor_metadata.quality_score if test_product.descriptor_metadata else 0,
                    "quality_reasoning": test_product.descriptor_metadata.quality_score_reasoning if test_product.descriptor_metadata else "",
                    "generation_time_seconds": generation_time,
                    "descriptor_length": len(test_product.descriptor.split()) if test_product.descriptor else 0
                }
                
            except Exception as e:
                logger.error(f"   ❌ Error with {model}: {e}")
                product_result["model_results"][model] = {
                    "error": str(e),
                    "generation_time_seconds": (datetime.now() - start_time).total_seconds()
                }
            # finally:
            #     # Restore original model
            #     LLMFactory.default_model = original_model
        
        # Compare the results
        if all("error" not in product_result["model_results"][m] for m in models_to_compare):
            comparison = await compare_descriptors_multi(
                product,
                product_result["model_results"],
                models_to_compare
            )
            product_result["comparison"] = comparison
        
        evaluation_results["product_evaluations"].append(product_result)
    
    # Calculate aggregate statistics
    evaluation_results["aggregate_stats"] = calculate_aggregate_stats(evaluation_results["product_evaluations"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"model_evaluation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"\n📊 Evaluation complete! Results saved to: {output_file}")
    
    # Print summary
    print_evaluation_summary(evaluation_results)
    
    return evaluation_results


async def compare_descriptors_multi(product: Product, model_results: Dict[str, Dict], models: List[str]) -> Dict[str, Any]:
    """Use LLM to compare the quality of multiple descriptors."""
    
    # Build comparison prompt
    comparison_prompt = f"""Compare these product descriptors for the same product:

PRODUCT: {product.name}
CATEGORY: {', '.join(product.categories) if product.categories else 'Unknown'}

"""
    
    # Add each model's descriptor
    for model in models:
        model_display = model.split('/')[-1].upper()  # Extract model name
        comparison_prompt += f"\nDESCRIPTOR ({model_display}):\n{model_results[model]['descriptor']}\n"
    
    comparison_prompt += """
Please evaluate and compare all descriptors:
1. Information completeness and accuracy
2. Searchability and keyword coverage
3. Natural language quality
4. Technical detail appropriateness
5. Overall effectiveness for RAG/vector search

Provide:
- A score for each descriptor (0-10)
- Which one is best overall and why
- Key differences between them
- Rank them from best to worst
"""
    
    response = await LLMFactory.chat_completion(
        task="descriptor_comparison",
        messages=[
            {"role": "system", "content": "You are an expert at evaluating product descriptors for e-commerce search systems."},
            {"role": "user", "content": comparison_prompt}
        ],
        model="openai/gpt-4.1"  # Use a fast model for evaluation
    )
    
    # Build comparison results
    result = {
        "comparison": response.get('content', ''),
    }
    
    # Add quality scores and generation times for each model
    for model in models:
        model_key = model.split('/')[-1].replace('-', '_').replace('.', '_')
        result[f"{model_key}_quality_score"] = model_results[model].get('quality_score', 0)
        result[f"{model_key}_generation_time"] = model_results[model].get('generation_time_seconds', 0)
    
    return result


def calculate_aggregate_stats(evaluations: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregate statistics from evaluations."""
    
    stats = {
        "o3": {
            "avg_quality_score": 0,
            "avg_generation_time": 0,
            "avg_descriptor_length": 0,
            "error_count": 0
        },
        "o3-mini": {
            "avg_quality_score": 0,
            "avg_generation_time": 0,
            "avg_descriptor_length": 0,
            "error_count": 0
        },
        "claude-3.5-sonnet": {
            "avg_quality_score": 0,
            "avg_generation_time": 0,
            "avg_descriptor_length": 0,
            "error_count": 0
        }
    }
    
    for model_key, display_name in [
        ("openai/o3", "o3"), 
        ("openai/o3-mini", "o3-mini"),
        ("anthropic/claude-3-5-sonnet-20241022", "claude-3.5-sonnet")
    ]:
        valid_results = []
        
        for eval_item in evaluations:
            if model_key in eval_item["model_results"]:
                result = eval_item["model_results"][model_key]
                if "error" not in result:
                    valid_results.append(result)
                else:
                    stats[display_name]["error_count"] += 1
        
        if valid_results:
            stats[display_name]["avg_quality_score"] = sum(r.get("quality_score", 0) for r in valid_results) / len(valid_results)
            stats[display_name]["avg_generation_time"] = sum(r.get("generation_time_seconds", 0) for r in valid_results) / len(valid_results)
            stats[display_name]["avg_descriptor_length"] = sum(r.get("descriptor_length", 0) for r in valid_results) / len(valid_results)
    
    return stats


def print_evaluation_summary(results: Dict[str, Any]):
    """Print a summary of the evaluation results."""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    stats = results["aggregate_stats"]
    
    print(f"\nBrand: {results['brand']}")
    print(f"Sample Size: {results['sample_size']} products")
    
    print("\n📊 AGGREGATE RESULTS:")
    print("-"*40)
    
    for model in ["o3", "o3-mini", "claude-3.5-sonnet"]:
        print(f"\n{model.upper()}:")
        print(f"  Average Quality Score: {stats[model]['avg_quality_score']:.2f}")
        print(f"  Average Generation Time: {stats[model]['avg_generation_time']:.2f}s")
        print(f"  Average Descriptor Length: {stats[model]['avg_descriptor_length']:.0f} words")
        print(f"  Errors: {stats[model]['error_count']}")
    
    # Determine winner
    scores = {
        "o3": stats["o3"]["avg_quality_score"],
        "o3-mini": stats["o3-mini"]["avg_quality_score"],
        "claude-3.5-sonnet": stats["claude-3.5-sonnet"]["avg_quality_score"]
    }
    
    # Sort models by score
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_model = sorted_models[0][0]
    best_score = sorted_models[0][1]
    
    print("\n🏆 VERDICT:")
    print(f"  Best Model: {best_model.upper()} (quality: {best_score:.2f})")
    
    # Check if models are close in performance
    score_differences = [abs(sorted_models[i][1] - sorted_models[i+1][1]) for i in range(len(sorted_models)-1)]
    if all(diff < 0.1 for diff in score_differences):
        print(f"  All models perform similarly (within 0.1 quality score)")
        # Find fastest model
        times = {model: stats[model]["avg_generation_time"] for model in ["o3", "o3-mini", "claude-3.5-sonnet"]}
        fastest_model = min(times.items(), key=lambda x: x[1])[0]
        print(f"  {fastest_model.upper()} is recommended due to faster generation time")
    else:
        print(f"\n  Rankings:")
        for i, (model, score) in enumerate(sorted_models, 1):
            print(f"  {i}. {model.upper()}: {score:.2f}")
    
    # Speed comparison
    times = {
        "o3": stats["o3"]["avg_generation_time"],
        "o3-mini": stats["o3-mini"]["avg_generation_time"],
        "claude-3.5-sonnet": stats["claude-3.5-sonnet"]["avg_generation_time"]
    }
    
    slowest_time = max(times.values())
    print(f"\n⚡ Speed Comparison:")
    for model, time in sorted(times.items(), key=lambda x: x[1]):
        if time > 0:
            speed_factor = slowest_time / time
            print(f"  {model.upper()}: {time:.1f}s ({speed_factor:.1f}x speed)")
    
    print("\n" + "="*60)


async def main():
    parser = argparse.ArgumentParser(
        description='Pre-generate product descriptors using brand research (uses ProductManager for all catalog access)'
    )
    parser.add_argument(
        'brand_domain',
        help='Brand domain (e.g., specialized.com)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of all descriptors'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.8,
        help='Minimum quality score for cached descriptors (0.0-1.0, default: 0.8)'
    )
    parser.add_argument(
        '--evaluate-models',
        action='store_true',
        help='Compare o3 vs o3-mini model performance on descriptor generation'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of products to sample for model evaluation (default: 6)'
    )
    
    args = parser.parse_args()
    
    # Handle model evaluation mode
    if args.evaluate_models:
        try:
            args.sample_size = args.sample_size or 6
            await evaluate_models(args.brand_domain, args.sample_size)
            return 0
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Regular descriptor generation mode
    # Check if brand research exists using Researcher classes
    researchers = {
        "foundation": get_foundation_researcher(args.brand_domain),
        "voice": get_voice_messaging_researcher(args.brand_domain),
        "style": get_product_style_researcher(args.brand_domain),
        "positioning": get_market_positioning_researcher(args.brand_domain),
        "customer": get_customer_cultural_researcher(args.brand_domain)
    }
    
    has_research = False
    research_summary = []
    
    for research_type, researcher in researchers.items():
        try:
            cached_results = await researcher._load_cached_results()
            if cached_results:
                quality_score = cached_results.get('quality_score', 0.0)
                data_sources = cached_results.get('data_sources', 0)
                research_summary.append(f"{research_type.capitalize()}: {quality_score:.1f} quality ({data_sources} sources)")
                has_research = True
            else:
                research_summary.append(f"{research_type.capitalize()}: Not found")
        except Exception as e:
            research_summary.append(f"{research_type.capitalize()}: Error ({str(e)[:50]}...)")
    
    if has_research:
        logger.info(f"✅ Found brand research for {args.brand_domain}")
        logger.info("   Research Summary:")
        for summary in research_summary:
            logger.info(f"     • {summary}")
        logger.info("   Will use research-enhanced descriptor generation")
    else:
        logger.warning(f"⚠️  No brand research found for {args.brand_domain}")
        logger.warning("   Research Summary:")
        for summary in research_summary:
            logger.warning(f"     • {summary}")
        logger.warning("   Descriptors will be generated without brand intelligence")
        logger.warning("   Run brand_intelligence_pipeline.py first for better results")
    
    # Generate descriptors
    try:
        logger.info(f"\n🚀 Starting descriptor pre-generation")
        logger.info(f"   Brand: {args.brand_domain}")
        logger.info(f"   Using ProductManager for products")
        logger.info(f"   Force regenerate: {args.force}")
        logger.info(f"   Quality threshold: {args.quality_threshold}")
        
        results = await generate_descriptors(
            brand_domain=args.brand_domain,
            force_regenerate=args.force,
            quality_threshold=args.quality_threshold,
            limit=args.sample_size
        )
        
        logger.info("\n✅ Descriptor pre-generation complete!")
        logger.info(f"   Descriptors saved to ProductManager")
        logger.info(f"   Results: {len(results[0])} products")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate descriptors: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))