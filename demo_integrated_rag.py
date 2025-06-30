#!/usr/bin/env python3
"""
Demo: Integrated RAG System

Demonstrates the complete RAG system with all features:
- Universal product processing
- Hybrid search
- Automatic synchronization  
- Langfuse integration
- Caching
- Monitoring
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Mock the system for demo purposes
class MockIntegratedRAGSystem:
    """Mock RAG system for demonstration."""
    
    def __init__(self, brand_domain, catalog_path, index_name, **kwargs):
        self.brand_domain = brand_domain
        self.catalog_path = catalog_path
        self.index_name = index_name
        self.monitoring_enabled = kwargs.get('enable_monitoring', True)
        self.cache_enabled = kwargs.get('enable_caching', True)
        self.auto_sync = kwargs.get('auto_sync', False)
        
        # Mock components
        self.search_count = 0
        self.cache_hits = 0
        self.sync_count = 0
        
        print(f"🚀 Initialized Integrated RAG System")
        print(f"   Brand: {brand_domain}")
        print(f"   Index: {index_name}")
        print(f"   Monitoring: {'✓' if self.monitoring_enabled else '✗'}")
        print(f"   Caching: {'✓' if self.cache_enabled else '✗'}")
        print(f"   Auto-sync: {'✓' if self.auto_sync else '✗'}")
    
    def search(self, query, top_k=10, filters=None, user_context=None, use_cache=None):
        """Mock search with realistic behavior."""
        self.search_count += 1
        
        # Simulate cache behavior
        cache_hit = False
        if use_cache is not False and self.cache_enabled and self.search_count % 3 != 1:
            cache_hit = True
            self.cache_hits += 1
            latency = 15  # Cache hit is fast
        else:
            latency = 120 + (len(query) * 2)  # Simulate processing time
        
        time.sleep(latency / 1000)  # Simulate latency
        
        # Generate mock results
        results = []
        for i in range(min(top_k, 5)):
            results.append({
                'id': f'PROD-{i+1:04d}',
                'score': 0.95 - (i * 0.05),
                'name': f'Product {i+1} for "{query}"',
                'brand': self.brand_domain.split('.')[0].title(),
                'price': 1000 + (i * 500),
                'description': f'Great product matching your search for {query}',
                'key_selling_points': [
                    f'Feature {j+1}' for j in range(3)
                ]
            })
        
        print(f"\n🔍 Search: '{query}'")
        print(f"   Cache: {'HIT' if cache_hit else 'MISS'}")
        print(f"   Latency: {latency}ms")
        print(f"   Results: {len(results)}")
        
        return results
    
    def get_system_status(self):
        """Mock system status."""
        cache_hit_rate = self.cache_hits / self.search_count if self.search_count > 0 else 0
        
        return {
            'brand': self.brand_domain,
            'index': self.index_name,
            'components': {
                'search': 'ready',
                'sync': 'running' if self.auto_sync else 'stopped',
                'cache': 'enabled' if self.cache_enabled else 'disabled',
                'monitoring': 'enabled' if self.monitoring_enabled else 'disabled'
            },
            'search_stats': {
                'total_searches': self.search_count,
                'cache_hit_rate': cache_hit_rate,
                'latency': {
                    'avg_ms': 85,
                    'p95_ms': 142
                }
            },
            'sync_stats': {
                'last_sync': datetime.now().isoformat(),
                'products_synced': 1250,
                'pending_changes': 3
            }
        }
    
    def sync_changes(self):
        """Mock sync operation."""
        self.sync_count += 1
        print(f"\n🔄 Syncing catalog changes...")
        time.sleep(1)
        print(f"✅ Sync complete: 3 products updated")
        return True


def demo_search_scenarios(rag):
    """Demonstrate various search scenarios."""
    print("\n" + "="*60)
    print("SEARCH DEMONSTRATIONS")
    print("="*60)
    
    # Scenario 1: Basic search
    print("\n1️⃣ Basic Product Search")
    results = rag.search("carbon road bike")
    display_results(results[:3])
    
    # Scenario 2: Filtered search
    print("\n2️⃣ Filtered Search")
    results = rag.search(
        "bike",
        filters={'category': 'road', 'price': {'max': 3000}}
    )
    display_results(results[:3])
    
    # Scenario 3: Personalized search
    print("\n3️⃣ Personalized Search with Context")
    results = rag.search(
        "lightweight bike",
        user_context={
            'preferences': {
                'preferred_brands': ['Specialized', 'Trek'],
                'max_price': 5000
            },
            'messages': [
                'I need something for racing',
                'Carbon frame is important'
            ]
        }
    )
    display_results(results[:3])
    
    # Scenario 4: Cache demonstration
    print("\n4️⃣ Cache Performance")
    query = "mountain bike with suspension"
    
    # First search - cache miss
    print("\nFirst search (cache miss expected):")
    results = rag.search(query)
    
    # Second search - cache hit
    print("\nSecond search (cache hit expected):")
    results = rag.search(query)
    
    # Third search - still cached
    print("\nThird search (still cached):")
    results = rag.search(query)


def demo_monitoring(rag):
    """Demonstrate monitoring capabilities."""
    print("\n" + "="*60)
    print("MONITORING & OBSERVABILITY")
    print("="*60)
    
    # Get system status
    status = rag.get_system_status()
    
    print("\n📊 System Status:")
    print(f"   Brand: {status['brand']}")
    print(f"   Index: {status['index']}")
    
    print("\n🔧 Component Status:")
    for component, state in status['components'].items():
        icon = '✅' if state in ['ready', 'enabled', 'running'] else '⚠️'
        print(f"   {icon} {component}: {state}")
    
    print("\n📈 Performance Metrics:")
    search_stats = status['search_stats']
    print(f"   Total searches: {search_stats['total_searches']}")
    print(f"   Cache hit rate: {search_stats['cache_hit_rate']:.1%}")
    print(f"   Avg latency: {search_stats['latency']['avg_ms']}ms")
    print(f"   P95 latency: {search_stats['latency']['p95_ms']}ms")
    
    print("\n🔄 Sync Status:")
    sync_stats = status['sync_stats']
    print(f"   Last sync: {sync_stats['last_sync']}")
    print(f"   Products synced: {sync_stats['products_synced']}")
    print(f"   Pending changes: {sync_stats['pending_changes']}")


def demo_langfuse_integration():
    """Demonstrate Langfuse integration."""
    print("\n" + "="*60)
    print("LANGFUSE PROMPT MANAGEMENT")
    print("="*60)
    
    print("\n📝 Managed Prompts:")
    prompts = [
        "filter_dictionary - Product filters and values",
        "query_optimizer - Query enhancement rules",
        "search_enhancer - Search improvement",
        "product_presenter - Result formatting",
        "filter_extractor - NLP filter extraction"
    ]
    
    for prompt in prompts:
        print(f"   • {prompt}")
    
    print("\n🔧 Filter Dictionary Example:")
    filter_dict = {
        'categories': ['road', 'mountain', 'hybrid', 'electric'],
        'attributes': {
            'material': ['carbon', 'aluminum', 'steel', 'titanium'],
            'brand': ['Specialized', 'Trek', 'Giant', 'Cannondale'],
            'size': ['XS', 'S', 'M', 'L', 'XL']
        },
        'price_ranges': {
            'budget': {'min': 0, 'max': 1000},
            'mid-range': {'min': 1000, 'max': 3000},
            'premium': {'min': 3000, 'max': 6000},
            'luxury': {'min': 6000, 'max': 20000}
        }
    }
    
    print(json.dumps(filter_dict, indent=2)[:500] + "...")
    
    print("\n✅ Prompts automatically updated on catalog sync")


def demo_production_patterns():
    """Demonstrate production deployment patterns."""
    print("\n" + "="*60)
    print("PRODUCTION DEPLOYMENT PATTERNS")
    print("="*60)
    
    print("\n1️⃣ API Server Pattern:")
    print("""
from fastapi import FastAPI
from src.rag_system import create_rag_system

app = FastAPI()
rag = create_rag_system(...)

@app.post("/search")
async def search(query: str, filters: dict = {}):
    results = rag.search(query, filters=filters)
    return {"results": results}
    """)
    
    print("\n2️⃣ Background Sync Pattern:")
    print("""
# Scheduled sync every 30 minutes
import schedule

schedule.every(30).minutes.do(rag.sync_changes)

# Or webhook-triggered
@app.post("/catalog-updated")
async def catalog_updated():
    rag.sync_changes()
    return {"status": "sync_triggered"}
    """)
    
    print("\n3️⃣ Monitoring Integration:")
    print("""
# Export metrics to Prometheus
from prometheus_client import Counter, Histogram

search_counter = Counter('rag_searches', 'Total searches')
search_latency = Histogram('rag_latency', 'Search latency')

# Alert on high latency
rag.monitor.add_alert_handler(
    lambda type, msg, sev: send_pagerduty_alert(msg)
)
    """)


def display_results(results):
    """Display search results nicely."""
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. {result['name']}")
        print(f"      Score: {result['score']:.3f}")
        print(f"      Price: ${result['price']:,}")
        print(f"      Points: {', '.join(result['key_selling_points'][:2])}")


def main():
    """Run the integrated RAG demo."""
    print("🎯 Integrated RAG System Demo")
    print("Demonstrating all features of the production-ready RAG system")
    print("="*60)
    
    # Initialize system
    print("\n🚀 Initializing RAG System...")
    rag = MockIntegratedRAGSystem(
        brand_domain="specialized.com",
        catalog_path="data/products.json",
        index_name="specialized-hybrid-v2",
        enable_monitoring=True,
        enable_caching=True,
        auto_sync=True
    )
    
    # Run demonstrations
    demo_search_scenarios(rag)
    demo_monitoring(rag)
    demo_langfuse_integration()
    demo_production_patterns()
    
    # Final summary
    print("\n" + "="*60)
    print("SYSTEM CAPABILITIES SUMMARY")
    print("="*60)
    
    capabilities = [
        "✅ Universal product processing for any brand/category",
        "✅ Hybrid search combining dense + sparse embeddings",
        "✅ Automatic catalog synchronization with change detection",
        "✅ Langfuse integration for prompt management",
        "✅ Multi-layer caching for <200ms response times",
        "✅ Comprehensive monitoring with alerts",
        "✅ Production-ready with API patterns",
        "✅ Voice AI optimized for natural conversation"
    ]
    
    for capability in capabilities:
        print(f"   {capability}")
    
    print("\n🎉 RAG system ready for production deployment!")
    print("\n💡 Next Steps:")
    print("   1. Configure environment variables")
    print("   2. Ingest your product catalog")
    print("   3. Test with real queries")
    print("   4. Deploy with monitoring")
    print("   5. Integrate with voice assistant")


if __name__ == "__main__":
    main()