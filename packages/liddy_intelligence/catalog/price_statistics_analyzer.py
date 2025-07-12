"""
Price Statistics Analyzer

Analyzes product pricing to create meaningful price tiers, handling:
1. Multi-modal distributions (e.g., $20 accessories + $5000 bikes)
2. Category-specific pricing tiers
3. Standard deviation-based clustering
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from liddy.models.product import Product

logger = logging.getLogger(__name__)


class PriceStatisticsAnalyzer:
    """Analyzes product pricing for meaningful categorization"""
    
    @staticmethod
    def analyze_catalog_pricing(products: List[Product], terminology_research: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze pricing with category awareness and distribution detection
        
        Returns:
            Dict with overall and category-specific statistics
        """
        # Extract prices by category
        prices_by_category = defaultdict(list)
        all_prices = []
        
        for product in products:
            price_str = product.salePrice or product.originalPrice
            if price_str:
                try:
                    price = float(price_str.replace('$', '').replace(',', ''))
                    if price > 0:  # Skip free items
                        all_prices.append(price)
                        
                        # Get primary category
                        if product.categories:
                            primary_category = product.categories[0]
                            prices_by_category[primary_category].append(price)
                        else:
                            prices_by_category['uncategorized'].append(price)
                except:
                    pass
        
        if not all_prices:
            return {}
        
        # Analyze overall distribution
        overall_stats = PriceStatisticsAnalyzer._calculate_statistics(all_prices)
        
        # Detect if we have a multi-modal distribution
        multi_modal = PriceStatisticsAnalyzer._detect_multimodal_distribution(all_prices)
        
        # Calculate category-specific statistics
        category_stats = {}
        for category, prices in prices_by_category.items():
            if len(prices) >= 5:  # Need enough data points
                category_stats[category] = PriceStatisticsAnalyzer._calculate_statistics(prices)
        
        # Determine pricing strategy
        if multi_modal:
            # Use clustering to find natural price groups
            price_clusters = PriceStatisticsAnalyzer._cluster_prices(all_prices)
            overall_stats['price_clusters'] = price_clusters
            overall_stats['is_multimodal'] = True
            
            # Define tiers based on clusters
            if len(price_clusters) >= 3:
                overall_stats['budget_threshold'] = price_clusters[0]['max']
                overall_stats['mid_low_threshold'] = price_clusters[1]['max'] if len(price_clusters) > 1 else price_clusters[0]['max'] * 2
                overall_stats['mid_high_threshold'] = price_clusters[2]['max'] if len(price_clusters) > 2 else price_clusters[1]['max'] * 2
                overall_stats['premium_threshold'] = price_clusters[-1]['min']
        else:
            # Use standard percentile approach but check if it makes sense
            overall_stats['is_multimodal'] = False
            
            # Check if the distribution is too skewed
            if overall_stats['std'] > overall_stats['mean']:
                # High variance - use log scale percentiles
                log_prices = np.log10(all_prices)
                log_percentiles = {
                    'p25': np.percentile(log_prices, 25),
                    'p50': np.percentile(log_prices, 50),
                    'p75': np.percentile(log_prices, 75),
                    'p95': np.percentile(log_prices, 95)
                }
                
                overall_stats['budget_threshold'] = 10 ** log_percentiles['p25']
                overall_stats['mid_low_threshold'] = 10 ** log_percentiles['p50']
                overall_stats['mid_high_threshold'] = 10 ** log_percentiles['p75']
                overall_stats['premium_threshold'] = 10 ** log_percentiles['p95']
            else:
                # Normal distribution - use regular percentiles
                overall_stats['budget_threshold'] = overall_stats['p25']
                overall_stats['mid_low_threshold'] = overall_stats['p50']
                overall_stats['mid_high_threshold'] = overall_stats['p75']
                overall_stats['premium_threshold'] = overall_stats['p95']
        
        # Find the most relevant category for a given price
        overall_stats['get_relevant_category'] = lambda price: PriceStatisticsAnalyzer._find_relevant_category(
            price, category_stats
        )
        
        # Generate semantic phrases based on actual distribution
        semantic_phrases = PriceStatisticsAnalyzer._generate_semantic_phrases(
            overall_stats, category_stats, terminology_research
        )
        
        return {
            'overall': overall_stats,
            'by_category': category_stats,
            'semantic_phrases': semantic_phrases,
            'recommendations': PriceStatisticsAnalyzer._generate_recommendations(
                overall_stats, category_stats
            )
        }
    
    @staticmethod
    def _calculate_statistics(prices: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a price list"""
        prices_array = np.array(prices)
        
        return {
            'min': float(np.min(prices_array)),
            'max': float(np.max(prices_array)),
            'mean': float(np.mean(prices_array)),
            'std': float(np.std(prices_array)),
            'p5': float(np.percentile(prices_array, 5)),
            'p25': float(np.percentile(prices_array, 25)),
            'p50': float(np.percentile(prices_array, 50)),
            'p75': float(np.percentile(prices_array, 75)),
            'p95': float(np.percentile(prices_array, 95)),
            'count': len(prices)
        }
    
    @staticmethod
    def _detect_multimodal_distribution(prices: List[float]) -> bool:
        """
        Detect if the price distribution has multiple distinct modes
        (e.g., accessories at $20-100 and bikes at $1000-5000)
        """
        if len(prices) < 20:
            return False
        
        # Use log scale to detect gaps
        log_prices = np.log10([p for p in prices if p > 0])
        
        # Sort and find gaps
        sorted_log_prices = np.sort(log_prices)
        gaps = np.diff(sorted_log_prices)
        
        # If we have gaps > 0.5 in log scale (>3x price difference), it's likely multimodal
        large_gaps = gaps > 0.5
        
        # Need at least one significant gap with data on both sides
        if np.any(large_gaps):
            gap_positions = np.where(large_gaps)[0]
            for gap_pos in gap_positions:
                # Check if we have meaningful data on both sides of the gap
                if gap_pos > len(prices) * 0.1 and gap_pos < len(prices) * 0.9:
                    return True
        
        return False
    
    @staticmethod
    def _cluster_prices(prices: List[float], max_clusters: int = 4) -> List[Dict[str, float]]:
        """
        Cluster prices into natural groups using gap detection
        """
        if len(prices) < 10:
            return [{'min': min(prices), 'max': max(prices), 'mean': np.mean(prices)}]
        
        # Sort prices
        sorted_prices = np.sort(prices)
        
        # Find natural breaks using log scale gaps
        log_prices = np.log10(sorted_prices)
        gaps = np.diff(log_prices)
        
        # Find significant gaps (> 0.3 in log scale = 2x price difference)
        significant_gaps = np.where(gaps > 0.3)[0]
        
        # Create clusters
        clusters = []
        start_idx = 0
        
        for gap_idx in significant_gaps[:max_clusters-1]:
            end_idx = gap_idx + 1
            cluster_prices = sorted_prices[start_idx:end_idx]
            
            if len(cluster_prices) > 0:
                clusters.append({
                    'min': float(np.min(cluster_prices)),
                    'max': float(np.max(cluster_prices)),
                    'mean': float(np.mean(cluster_prices)),
                    'count': len(cluster_prices)
                })
            
            start_idx = end_idx
        
        # Add final cluster
        final_cluster = sorted_prices[start_idx:]
        if len(final_cluster) > 0:
            clusters.append({
                'min': float(np.min(final_cluster)),
                'max': float(np.max(final_cluster)),
                'mean': float(np.mean(final_cluster)),
                'count': len(final_cluster)
            })
        
        return clusters
    
    @staticmethod
    def _find_relevant_category(price: float, category_stats: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Find which category this price most likely belongs to"""
        best_category = None
        best_score = float('inf')
        
        for category, stats in category_stats.items():
            # Score based on how many standard deviations away from the mean
            if stats['std'] > 0:
                z_score = abs(price - stats['mean']) / stats['std']
                
                # Penalize if outside the range
                if price < stats['min'] or price > stats['max']:
                    z_score += 2.0
                
                if z_score < best_score:
                    best_score = z_score
                    best_category = category
        
        return best_category
    
    @staticmethod
    def _generate_recommendations(overall_stats: Dict, category_stats: Dict) -> Dict[str, str]:
        """Generate recommendations for handling the pricing structure"""
        
        recommendations = {
            'strategy': '',
            'warnings': [],
            'suggestions': []
        }
        
        if overall_stats.get('is_multimodal'):
            recommendations['strategy'] = 'multi_modal'
            recommendations['suggestions'].append(
                "Use category-specific price tiers for more accurate classification"
            )
            
            if overall_stats.get('price_clusters'):
                cluster_descriptions = []
                for i, cluster in enumerate(overall_stats['price_clusters']):
                    desc = f"Cluster {i+1}: ${cluster['min']:.2f} - ${cluster['max']:.2f} ({cluster['count']} products)"
                    cluster_descriptions.append(desc)
                recommendations['suggestions'].append(
                    f"Detected {len(overall_stats['price_clusters'])} distinct price groups: " + 
                    ", ".join(cluster_descriptions)
                )
        else:
            recommendations['strategy'] = 'continuous'
            
            # Check for high variance
            if overall_stats['std'] > overall_stats['mean']:
                recommendations['warnings'].append(
                    "High price variance detected - consider category-specific analysis"
                )
        
        # Check for extreme percentile compression
        if overall_stats['p75'] < overall_stats['mean'] * 0.5:
            recommendations['warnings'].append(
                "Most products are in the lower price range with few expensive outliers"
            )
        
        # Add category-specific recommendations
        if len(category_stats) > 1:
            recommendations['suggestions'].append(
                "Use category-specific price tiers for more accurate classification"
            )
            
            # Check if categories have significantly different price ranges
            category_means = [stats['mean'] for stats in category_stats.values()]
            if max(category_means) > min(category_means) * 5:
                recommendations['warnings'].append(
                    "Categories have vastly different price ranges - consider separate tier systems"
                )
        
        return recommendations
    
    @staticmethod
    def _generate_semantic_phrases(overall_stats: Dict, category_stats: Dict, 
                                  terminology_research: Optional[Dict] = None) -> Dict[str, List[str]]:
        """
        Generate semantic phrases for price tiers based on actual distribution
        and industry terminology research
        """
        semantic_phrases = {
            'premium': [],
            'mid_high': [],
            'mid_low': [],
            'budget': []
        }
        
        # Extract industry-specific terms from terminology research
        if terminology_research:
            # Look for price tier indicators in the research
            if 'price_terminology' in terminology_research:
                price_terms = terminology_research['price_terminology']
                if 'premium_terms' in price_terms:
                    semantic_phrases['premium'].extend(price_terms['premium_terms'])
                if 'budget_terms' in price_terms:
                    semantic_phrases['budget'].extend(price_terms['budget_terms'])
                # Check for mid-tier terms
                if 'mid_terms' in price_terms:
                    semantic_phrases['mid_high'].extend(price_terms['mid_terms'])
                    semantic_phrases['mid_low'].extend(price_terms['mid_terms'])
            
            # Look for brand-specific tier indicators
            if 'brand_specific_tiers' in terminology_research:
                brand_tiers = terminology_research['brand_specific_tiers']
                if 'premium_indicators' in brand_tiers:
                    semantic_phrases['premium'].extend(brand_tiers['premium_indicators'])
                if 'budget_indicators' in brand_tiers:
                    semantic_phrases['budget'].extend(brand_tiers['budget_indicators'])
                # Check for mid-tier indicators
                if 'mid_indicators' in brand_tiers:
                    semantic_phrases['mid_high'].extend(brand_tiers['mid_indicators'])
                    semantic_phrases['mid_low'].extend(brand_tiers['mid_indicators'])
        
        # Add distribution-aware phrases
        if overall_stats.get('is_multimodal'):
            # For multi-modal distributions, use cluster-based phrases
            if 'price_clusters' in overall_stats:
                clusters = overall_stats['price_clusters']
                if len(clusters) >= 2:
                    # Top cluster
                    semantic_phrases['premium'].extend([
                        f"premium tier (${clusters[-1]['min']:.0f}+)",
                        f"flagship models",
                        f"professional grade"
                    ])
                    # Bottom cluster
                    semantic_phrases['budget'].extend([
                        f"value tier (under ${clusters[0]['max']:.0f})",
                        f"entry level",
                        f"starter models"
                    ])
        else:
            # For continuous distributions, use percentile-based phrases
            # Adjust phrases based on price skew
            if overall_stats['std'] > overall_stats['mean']:
                # High variance - use log-scale friendly phrases
                semantic_phrases['premium'].extend([
                    f"premium collection",
                    f"investment pieces",
                    f"top {100 - 95}% of range"
                ])
                semantic_phrases['budget'].extend([
                    f"accessible options",
                    f"value collection",
                    f"smart buys"
                ])
            else:
                # Normal distribution - use standard phrases
                semantic_phrases['premium'].extend([
                    f"premium tier",
                    f"high-end models",
                    f"top of the line"
                ])
                semantic_phrases['budget'].extend([
                    f"budget friendly",
                    f"affordable range",
                    f"entry tier"
                ])
        
        # Add category-aware phrases if categories have distinct pricing
        if len(category_stats) > 1:
            for category, stats in category_stats.items():
                if stats['mean'] > overall_stats['mean'] * 1.5:
                    semantic_phrases['premium'].append(f"premium {category.lower()}")
                elif stats['mean'] < overall_stats['mean'] * 0.5:
                    semantic_phrases['budget'].append(f"value {category.lower()}")
        
        # Remove duplicates and clean up
        for tier in semantic_phrases:
            semantic_phrases[tier] = list(set(phrase.lower() for phrase in semantic_phrases[tier] if phrase))
        
        return semantic_phrases