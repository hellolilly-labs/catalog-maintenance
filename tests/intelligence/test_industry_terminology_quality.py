"""
Test suite for IndustryTerminologyResearcher quality improvements.

Tests the enhanced functionality including:
- Quality pipeline (normalization, stop-words, proof-of-use, deduplication)
- Tavily answer extraction with provenance
- Depth-2 search capability
- Confidence scoring
- Tier exclusivity with price analysis
- Structured API access
"""

import pytest
import asyncio
import re
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from hypothesis import given, settings, strategies as st

from liddy_intelligence.research.industry_terminology_researcher import IndustryTerminologyResearcher
from liddy_intelligence.research.result_types import PriceTerminology, Term
from liddy_intelligence.research.search_helpers import (
    _extract_terms_from_tavily_answer,
    diversity_filter,
    domain_diversity,
    is_stopword
)
from liddy.models.product import Product


class MockSearchResult:
    """Mock search result for testing."""
    def __init__(self, url: str, content: str):
        self.url = url
        self.content = content


class MockSearchResponse:
    """Mock search response for testing."""
    def __init__(self, answer: str = "", results: List[MockSearchResult] = None):
        self.answer = answer
        self.results = results or []


@pytest.fixture
def mock_products():
    """Create mock products for testing."""
    products = []
    
    # Premium products
    for i in range(5):
        product = Mock(spec=Product)
        product.name = f"S-Works Elite Model {i}"
        product.salePrice = "$5000"
        product.description = "Professional grade carbon fiber"
        products.append(product)
    
    # Mid-tier products
    for i in range(5):
        product = Mock(spec=Product)
        product.name = f"Comp Sport Model {i}"
        product.salePrice = "$1500"
        product.description = "Enthusiast level performance"
        products.append(product)
    
    # Budget products
    for i in range(5):
        product = Mock(spec=Product)
        product.name = f"Base Entry Model {i}"
        product.salePrice = "$500"
        product.description = "Great for beginners"
        products.append(product)
    
    return products


@pytest.fixture
def mock_researcher():
    """Create a mock IndustryTerminologyResearcher."""
    researcher = IndustryTerminologyResearcher("specialized.com")
    researcher.web_search = AsyncMock()
    researcher.storage_manager = AsyncMock()
    researcher._llm_call_with_json_retry = AsyncMock()
    return researcher


class TestSearchHelpers:
    """Test search helper functions."""
    
    async def test_extract_terms_from_tavily_answer(self):
        """Test term extraction from Tavily answers."""
        answer = """
        In the cycling industry, premium models often use terms like S-Works and SL7 
        to indicate top-tier products. Mid-range bikes use Comp or Sport designations,
        while entry-level models are marked as Base or Entry.
        """
        
        # Test with brand and industry context
        terms = await _extract_terms_from_tavily_answer(
            answer, "specialized.com", "cycling"
        )
        
        assert "s-works" in terms
        assert "sl7" in terms
        assert "comp" in terms
        assert "sport" in terms
        assert "base" in terms
        assert "entry" in terms
        
        # Check provenance is maintained
        assert "top-tier" in terms["s-works"]
        assert "entry-level" in terms["entry"]
    
    def test_diversity_filter(self):
        """Test domain diversity filtering."""
        results = [
            MockSearchResult("https://example.com/1", "content1"),
            MockSearchResult("https://example.com/2", "content2"),
            MockSearchResult("https://other.com/1", "content3"),
            MockSearchResult("https://third.com/1", "content4"),
        ]
        
        filtered = diversity_filter(results)
        
        # Should only have one result per domain
        assert len(filtered) == 3
        domains = [r.url.split('/')[2] for r in filtered]
        assert len(set(domains)) == len(domains)
    
    def test_is_stopword(self):
        """Test stopword detection."""
        # Generic quality terms should be stopwords
        assert is_stopword("ultimate", "generic_quality")
        assert is_stopword("best", "generic_quality")
        assert is_stopword("premium", "generic_quality")
        
        # Size descriptors
        assert is_stopword("large", "size_descriptors")
        assert is_stopword("small", "size_descriptors")
        
        # Colors
        assert is_stopword("black", "color_descriptors")
        assert is_stopword("white", "color_descriptors")
        
        # Non-stopwords
        assert not is_stopword("s-works")
        assert not is_stopword("comp")
        assert not is_stopword("pro")


class TestQualityPipeline:
    """Test quality pipeline methods."""
    
    @pytest.mark.asyncio
    async def test_normalize_term(self, mock_researcher):
        """Test term normalization."""
        # Test basic normalization
        assert mock_researcher._normalize_term("S-Works") == "s-works"
        assert mock_researcher._normalize_term("COMP") == "comp"
        assert mock_researcher._normalize_term("Pro Series") == "proseries"
        
        # Test special character removal
        assert mock_researcher._normalize_term("Elite™") == "elite"
        assert mock_researcher._normalize_term("Sport+") == "sport"
        
        # Test hyphen preservation
        assert mock_researcher._normalize_term("entry-level") == "entry-level"
    
    @pytest.mark.asyncio
    async def test_proof_of_use(self, mock_researcher):
        """Test tier-aware proof-of-use validation."""
        # Premium tier - needs 1 product OR 3 web hits
        assert mock_researcher._proof_of_use("s-works", 1, 0, "premium")
        assert mock_researcher._proof_of_use("s-works", 0, 3, "premium")
        assert not mock_researcher._proof_of_use("s-works", 0, 2, "premium")
        
        # Mid tier - needs 2 products OR 2 web hits
        assert mock_researcher._proof_of_use("comp", 2, 0, "mid")
        assert mock_researcher._proof_of_use("comp", 0, 2, "mid")
        assert not mock_researcher._proof_of_use("comp", 1, 1, "mid")
        
        # Budget tier - needs 3 products OR 1 web hit
        assert mock_researcher._proof_of_use("base", 3, 0, "budget")
        assert mock_researcher._proof_of_use("base", 0, 1, "budget")
        assert not mock_researcher._proof_of_use("base", 2, 0, "budget")
    
    @pytest.mark.asyncio
    async def test_get_product_coverage(self, mock_researcher, mock_products):
        """Test product coverage calculation."""
        # S-Works appears in 5 products
        coverage = mock_researcher._get_product_coverage("s-works", mock_products)
        assert coverage == 5
        
        # Comp appears in 5 products
        coverage = mock_researcher._get_product_coverage("comp", mock_products)
        assert coverage == 5
        
        # Non-existent term
        coverage = mock_researcher._get_product_coverage("nonexistent", mock_products)
        assert coverage == 0
    
    @given(st.lists(st.text(min_size=1), unique=True, min_size=1, max_size=20))
    @settings(deadline=None)
    def test_normalize_terms_property(self, terms):
        """Property test: term normalization should be idempotent and consistent."""
        researcher = IndustryTerminologyResearcher("test.com")
        
        for term in terms:
            # Normalization should be idempotent
            normalized_once = researcher._normalize_term(term)
            normalized_twice = researcher._normalize_term(normalized_once)
            assert normalized_once == normalized_twice
            
            # Result should be lowercase
            assert normalized_once == normalized_once.lower()
            
            # Should not be empty unless input was empty/whitespace or only special chars
            if term.strip() and re.search(r'[a-zA-Z0-9]', term):
                assert len(normalized_once) > 0
    
    @given(st.lists(st.tuples(st.text(min_size=1), st.floats(min_value=0.0, max_value=1.0)), unique=True, min_size=0, max_size=10))
    @settings(deadline=None)
    def test_price_terminology_immutability_property(self, term_tuples):
        """Property test: PriceTerminology should be immutable."""
        terminology = PriceTerminology(
            premium_terms=tuple(term_tuples[:3]),
            mid_terms=tuple(term_tuples[3:6]),
            budget_terms=tuple(term_tuples[6:])
        )
        
        # Should not be able to modify the tuples
        original_premium = terminology.premium_terms
        original_mid = terminology.mid_terms  
        original_budget = terminology.budget_terms
        
        # These should be the same objects (immutable)
        assert terminology.premium_terms is original_premium
        assert terminology.mid_terms is original_mid
        assert terminology.budget_terms is original_budget
        
        # Legacy list properties should return new lists each time
        if terminology.premium_terms:
            list1 = terminology.premium_terms_list
            list2 = terminology.premium_terms_list
            assert list1 == list2  # Same content
            assert list1 is not list2  # Different objects
    
    @pytest.mark.asyncio
    async def test_calculate_confidence(self, mock_researcher):
        """Test multi-factor confidence scoring."""
        # High confidence - good web presence, product coverage, and answer provenance
        confidence = mock_researcher._calculate_confidence(
            "s-works",
            web_hits=8,
            product_coverage=4,
            has_answer_provenance=True,
            domain_quality=0.8
        )
        assert confidence > 0.7
        
        # Low confidence - minimal presence
        confidence = mock_researcher._calculate_confidence(
            "obscure",
            web_hits=1,
            product_coverage=0,
            has_answer_provenance=False,
            domain_quality=0.2
        )
        assert confidence < 0.3
        
        # Answer provenance boost
        conf_with_answer = mock_researcher._calculate_confidence(
            "term", 5, 2, True, 0.5
        )
        conf_without_answer = mock_researcher._calculate_confidence(
            "term", 5, 2, False, 0.5
        )
        assert conf_with_answer > conf_without_answer


class TestTierExclusivity:
    """Test tier exclusivity enforcement."""
    
    @pytest.mark.asyncio
    async def test_dedupe_tiers_with_context(self, mock_researcher, mock_products):
        """Test smart tier deduplication using price analysis."""
        # Create overlapping tiers
        tiers = {
            'premium': [('s-works', 0.9), ('pro', 0.8), ('comp', 0.7)],
            'mid': [('comp', 0.85), ('sport', 0.7), ('pro', 0.6)],
            'budget': [('base', 0.8), ('entry', 0.7), ('sport', 0.5)]
        }
        
        # Mock PriceStatisticsAnalyzer
        with patch('liddy_intelligence.research.industry_terminology_researcher.PriceStatisticsAnalyzer') as mock_analyzer:
            mock_analyzer.analyze_catalog_pricing.return_value = {
                'overall': {
                    'budget_threshold': 800,
                    'mid_high_threshold': 3000
                }
            }
            
            exclusive = mock_researcher._dedupe_tiers_with_context(tiers, mock_products)
            
            # Check that overlapping terms are resolved
            all_terms = set()
            for tier_terms in exclusive.values():
                tier_names = {t for t, _ in tier_terms}
                # No term should appear in multiple tiers
                assert len(all_terms & tier_names) == 0
                all_terms.update(tier_names)


class TestTavilyIntegration:
    """Test Tavily-specific optimizations."""
    
    @pytest.mark.asyncio
    async def test_depth_2_search(self, mock_researcher):
        """Test depth-2 follow-up queries."""
        from liddy_intelligence.research.search_helpers import run_search
        
        mock_web_search = AsyncMock()
        
        # First search returns empty answer
        mock_web_search.search.side_effect = [
            MockSearchResponse(answer="", results=[MockSearchResult("url1", "content1")]),
            MockSearchResponse(answer="Enhanced answer with terms", results=[
                MockSearchResult("url2", "content2"),
                MockSearchResult("url3", "content3")
            ])
        ]
        
        response = await run_search("test query", mock_web_search)
        
        # Should have called search twice
        assert mock_web_search.search.call_count == 2
        
        # Second call should have enhanced query
        second_call_query = mock_web_search.search.call_args_list[1][1]['query']
        assert "explained" in second_call_query or "model hierarchy" in second_call_query
    
    @pytest.mark.asyncio
    async def test_quick_scan_optimization(self, mock_researcher, mock_products):
        """Test quick-scan skips LLM when Tavily answer is sufficient."""
        # Mock search response with rich answer
        mock_response = MockSearchResponse(
            answer="""Premium bikes include S-Works, SL7, and Epic models.
                     Mid-range options are Comp, Sport, and Active series.
                     Entry-level bikes include Base, Entry, and Starter models.""",
            results=[MockSearchResult(f"url{i}", f"content{i}") for i in range(5)]
        )
        
        mock_researcher.web_search.search.return_value = mock_response
        
        # Mock LLM should not be called due to quick-scan
        mock_researcher._extract_terminology_from_search = AsyncMock()
        
        # Run research
        result = await mock_researcher._research_price_terminology("specialized", "cycling", mock_products)
        
        # Verify terms were extracted from answer
        assert any('s-works' in str(term).lower() for term in result['premium_terms'])
        assert any('comp' in str(term).lower() for term in result['mid_tier_terms'])
        assert any('base' in str(term).lower() for term in result['budget_terms'])


class TestStructuredAPI:
    """Test the structured API methods."""
    
    @pytest.mark.asyncio
    async def test_run_returns_price_terminology(self, mock_researcher):
        """Test that run() returns PriceTerminology object."""
        # Mock the parent run method
        with patch.object(mock_researcher, '_gather_data', new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = {
                'brand_name': 'specialized',
                'industry': 'cycling',
                'price_terminology': {
                    'premium_terms': [('s-works', 'Premium indicator')],
                    'mid_tier_terms': [('comp', 'Mid-tier indicator')],
                    'budget_terms': [('base', 'Budget indicator')]
                },
                'industry_slang': {},
                'technical_terms': {},
                'product_patterns': {},
                'categorization_patterns': {}
            }
            
            # Set cached terminology
            mock_researcher._cached_price_terminology = PriceTerminology(
                premium_terms=[('s-works', 0.9)],
                mid_terms=[('comp', 0.8)],
                budget_terms=[('base', 0.7)]
            )
            
            result = await mock_researcher.run(write_markdown=False)
            
            assert isinstance(result, PriceTerminology)
            assert len(result.premium_terms) > 0
            assert len(result.mid_terms) > 0
            assert len(result.budget_terms) > 0
    
    def test_accessor_methods(self, mock_researcher):
        """Test accessor methods for clean API."""
        # Set up cached terminology
        mock_researcher._result = PriceTerminology(
            premium_terms=(('s-works', 0.9), ('sl7', 0.85)),
            mid_terms=(('comp', 0.8), ('sport', 0.75)),
            budget_terms=(('base', 0.7), ('entry', 0.65))
        )
        
        # Test individual tier accessors
        premium = mock_researcher.get_premium_terms()
        assert premium == ['s-works', 'sl7']
        
        mid = mock_researcher.get_mid_terms()
        assert mid == ['comp', 'sport']
        
        budget = mock_researcher.get_budget_terms()
        assert budget == ['base', 'entry']
        
        # Test all terms accessor
        all_terms = mock_researcher.get_all_price_terms()
        assert len(all_terms) == 6
        assert 's-works' in all_terms
        assert 'comp' in all_terms
        assert 'base' in all_terms


class TestPromptEnhancements:
    """Test LLM prompt improvements."""
    
    @pytest.mark.asyncio
    async def test_prompt_excludes_vague_intensifiers(self, mock_researcher):
        """Test that prompts exclude vague intensifiers."""
        # The prompt should be built with exclusion rules
        template_vars = {
            'brand': 'test',
            'industry': 'cycling',
            'search_data': 'test data'
        }
        
        # Check the default prompt contains exclusion rules
        assert "EXCLUDE vague intensifiers" in mock_researcher._llm_extract_terminology.__code__.co_consts
        assert "ultimate" in str(mock_researcher._llm_extract_terminology.__code__.co_consts)
        assert "best" in str(mock_researcher._llm_extract_terminology.__code__.co_consts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])