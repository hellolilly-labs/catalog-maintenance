"""
Verify Descriptor Generation Quality

This script analyzes our descriptor generation to ensure it follows
best practices for RAG and voice AI applications.
"""

import json
import logging
import statistics
from typing import Dict, List, Any, Tuple
from pathlib import Path
import re

from src.ingestion.universal_product_processor import UniversalProductProcessor
from src.catalog.enhanced_descriptor_generator import EnhancedDescriptorGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DescriptorQualityAnalyzer:
    """Analyze descriptor quality for RAG best practices."""
    
    def __init__(self):
        self.quality_checks = {
            'length': self._check_length,
            'completeness': self._check_completeness,
            'readability': self._check_readability,
            'keyword_density': self._check_keyword_density,
            'voice_optimization': self._check_voice_optimization,
            'search_optimization': self._check_search_optimization
        }
    
    def analyze_descriptors(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze descriptor quality across all products."""
        
        results = {
            'total_products': len(products),
            'quality_scores': {},
            'issues': [],
            'recommendations': []
        }
        
        # Run quality checks
        for check_name, check_func in self.quality_checks.items():
            scores = []
            issues = []
            
            for product in products:
                score, product_issues = check_func(product)
                scores.append(score)
                if product_issues:
                    issues.extend([(product['id'], issue) for issue in product_issues])
            
            results['quality_scores'][check_name] = {
                'average': statistics.mean(scores),
                'min': min(scores),
                'max': max(scores),
                'issues': issues[:10]  # Top 10 issues
            }
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results['quality_scores'])
        
        return results
    
    def _check_length(self, product: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check if descriptor length is optimal for RAG."""
        issues = []
        score = 1.0
        
        descriptor = product.get('enhanced_descriptor', '')
        word_count = len(descriptor.split())
        
        # Optimal range: 50-200 words for voice AI
        if word_count < 30:
            score = 0.5
            issues.append(f"Descriptor too short ({word_count} words)")
        elif word_count > 300:
            score = 0.7
            issues.append(f"Descriptor too long ({word_count} words)")
        elif word_count < 50:
            score = 0.8
        
        return score, issues
    
    def _check_completeness(self, product: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check if descriptor includes all key information."""
        issues = []
        score = 1.0
        
        descriptor = product.get('enhanced_descriptor', '').lower()
        universal = product.get('universal_fields', {})
        
        # Check for essential elements
        essential_elements = {
            'name': universal.get('name', '').lower(),
            'category': universal.get('category', [''])[0].lower() if universal.get('category') else '',
            'price': str(universal.get('price', ''))
        }
        
        missing = []
        for element, value in essential_elements.items():
            if value and value not in descriptor:
                missing.append(element)
                score -= 0.2
        
        if missing:
            issues.append(f"Missing elements: {', '.join(missing)}")
        
        # Check for features
        if not any(word in descriptor for word in ['feature', 'includes', 'with', 'has']):
            score -= 0.1
            issues.append("No features mentioned")
        
        return max(0, score), issues
    
    def _check_readability(self, product: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check readability for voice AI."""
        issues = []
        score = 1.0
        
        descriptor = product.get('enhanced_descriptor', '')
        
        # Check sentence length
        sentences = re.split(r'[.!?]+', descriptor)
        avg_sentence_length = statistics.mean([len(s.split()) for s in sentences if s.strip()])
        
        if avg_sentence_length > 25:
            score -= 0.2
            issues.append(f"Long sentences (avg {avg_sentence_length:.1f} words)")
        
        # Check for complex punctuation
        complex_punct_count = len(re.findall(r'[;:\(\)\[\]]', descriptor))
        if complex_punct_count > 3:
            score -= 0.1
            issues.append(f"Complex punctuation ({complex_punct_count} instances)")
        
        # Check for natural language flow
        if not any(phrase in descriptor.lower() for phrase in ['is', 'are', 'features', 'includes']):
            score -= 0.2
            issues.append("Lacks natural language flow")
        
        return max(0, score), issues
    
    def _check_keyword_density(self, product: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check keyword optimization for search."""
        issues = []
        score = 1.0
        
        descriptor = product.get('enhanced_descriptor', '').lower()
        keywords = product.get('search_keywords', [])
        
        if not keywords:
            score = 0.5
            issues.append("No search keywords defined")
        else:
            # Check keyword presence
            missing_keywords = []
            for keyword in keywords[:10]:  # Check top 10 keywords
                if keyword.lower() not in descriptor:
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                score -= 0.1 * len(missing_keywords) / 10
                issues.append(f"Missing keywords: {', '.join(missing_keywords[:5])}")
        
        return max(0, score), issues
    
    def _check_voice_optimization(self, product: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check optimization for voice AI delivery."""
        issues = []
        score = 1.0
        
        descriptor = product.get('enhanced_descriptor', '')
        voice_summary = product.get('voice_summary', '')
        
        # Check for conversational tone
        conversational_indicators = ['perfect for', 'ideal for', 'great for', 'designed for']
        if not any(phrase in descriptor.lower() for phrase in conversational_indicators):
            score -= 0.2
            issues.append("Lacks conversational tone")
        
        # Check voice summary
        if not voice_summary:
            score -= 0.3
            issues.append("No voice summary")
        elif len(voice_summary.split()) > 50:
            score -= 0.1
            issues.append("Voice summary too long")
        
        # Check for technical jargon
        jargon_pattern = r'\b[A-Z]{3,}\b'  # 3+ letter acronyms
        jargon_count = len(re.findall(jargon_pattern, descriptor))
        if jargon_count > 3:
            score -= 0.1
            issues.append(f"Too much technical jargon ({jargon_count} acronyms)")
        
        return max(0, score), issues
    
    def _check_search_optimization(self, product: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check optimization for search retrieval."""
        issues = []
        score = 1.0
        
        descriptor = product.get('enhanced_descriptor', '')
        universal = product.get('universal_fields', {})
        
        # Check for variety in terminology
        name = universal.get('name', '')
        if name:
            name_words = set(name.lower().split())
            descriptor_words = set(descriptor.lower().split())
            
            # Should include name but also synonyms/variations
            if not name_words.intersection(descriptor_words):
                score -= 0.3
                issues.append("Product name not in descriptor")
            
            # Check for synonym usage
            if len(descriptor_words) < 20:
                score -= 0.2
                issues.append("Limited vocabulary in descriptor")
        
        # Check for category variations
        category = universal.get('category', [''])[0]
        if category and category.lower() not in descriptor.lower():
            score -= 0.1
            issues.append("Category not mentioned")
        
        return max(0, score), issues
    
    def _generate_recommendations(self, quality_scores: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []
        
        for check_name, results in quality_scores.items():
            avg_score = results['average']
            
            if avg_score < 0.7:
                if check_name == 'length':
                    recommendations.append(
                        "Optimize descriptor length: Aim for 50-200 words for optimal voice delivery and search performance."
                    )
                elif check_name == 'completeness':
                    recommendations.append(
                        "Ensure descriptors include: product name, category, key features, and price information."
                    )
                elif check_name == 'readability':
                    recommendations.append(
                        "Improve readability: Use shorter sentences (15-20 words) and natural language flow."
                    )
                elif check_name == 'keyword_density':
                    recommendations.append(
                        "Enhance keyword integration: Include search keywords naturally within descriptors."
                    )
                elif check_name == 'voice_optimization':
                    recommendations.append(
                        "Optimize for voice: Add conversational phrases and ensure voice summaries are concise (30-50 words)."
                    )
                elif check_name == 'search_optimization':
                    recommendations.append(
                        "Improve search optimization: Include product name, category, and synonyms in descriptors."
                    )
        
        return recommendations


def main():
    """Analyze descriptor quality for a sample catalog."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Verify descriptor generation quality")
    parser.add_argument("brand_domain", help="Brand domain (e.g., specialized.com)")
    parser.add_argument("catalog_path", help="Path to product catalog")
    parser.add_argument("--output", help="Output file for results", default="descriptor_quality_report.json")
    
    args = parser.parse_args()
    
    # Load catalog
    logger.info(f"Loading catalog from {args.catalog_path}")
    with open(args.catalog_path, 'r') as f:
        catalog_data = json.load(f)
    
    if isinstance(catalog_data, dict):
        catalog_data = catalog_data.get('products', [catalog_data])
    
    # Process products
    logger.info("Processing products...")
    processor = UniversalProductProcessor(args.brand_domain)
    processed_products = processor.process_catalog(catalog_data)
    
    # Generate enhanced descriptors
    logger.info("Generating enhanced descriptors...")
    generator = EnhancedDescriptorGenerator(args.brand_domain)
    enhanced_products, _ = generator.process_catalog(processed_products)
    
    # Analyze quality
    logger.info("Analyzing descriptor quality...")
    analyzer = DescriptorQualityAnalyzer()
    results = analyzer.analyze_descriptors(enhanced_products)
    
    # Display results
    print("\n" + "="*60)
    print("DESCRIPTOR QUALITY ANALYSIS")
    print("="*60)
    print(f"\nTotal Products Analyzed: {results['total_products']}")
    
    print("\nQuality Scores:")
    for check_name, scores in results['quality_scores'].items():
        print(f"\n{check_name.replace('_', ' ').title()}:")
        print(f"  Average: {scores['average']:.2f}")
        print(f"  Range: {scores['min']:.2f} - {scores['max']:.2f}")
        if scores['issues']:
            print(f"  Sample Issues:")
            for product_id, issue in scores['issues'][:3]:
                print(f"    - {product_id}: {issue}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Save detailed results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()