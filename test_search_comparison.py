#!/usr/bin/env python3
"""
Voice Assistant Search Comparison Test Framework

This script tests and compares:
1. Enhanced search (separate dense/sparse indexes with reranking)
2. Baseline search (single dense index with Product.to_markdown())

It simulates voice assistant conversations and evaluates search quality.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Import our components
from src.models.product import Product
from src.models.product_manager import ProductManager
from src.storage import get_account_storage_provider
from src.llm.simple_factory import LLMFactory
from src.llm.prompt_manager import PromptManager
from src.search.search_pinecone import PineconeRAG
from src.search.search_service import SearchService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UserState:
    """Simple user state for testing."""
    user_id: str = "test_user"
    session_id: str = "test_session"
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    speaker: str  # 'user' or 'assistant'
    message: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SearchTestScenario:
    """Defines a test scenario for search comparison."""
    scenario_id: str
    description: str
    conversation: List[ConversationTurn]
    expected_product_types: List[str]
    evaluation_criteria: Dict[str, Any]


@dataclass
class SearchResult:
    """Standardized search result for comparison."""
    product_id: str
    score: float
    name: str
    price: float
    category: str
    relevance_explanation: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ComparisonMetrics:
    """Metrics for comparing search results."""
    query: str
    enhanced_results: List[SearchResult]
    baseline_results: List[SearchResult]
    enhanced_time: float
    baseline_time: float
    overlap_ratio: float
    ranking_correlation: float
    llm_evaluation: Dict[str, Any]


class BaselineSearchIndex:
    """
    Creates and manages a baseline single dense index using Product.to_markdown().
    """
    
    def __init__(self, brand_domain: str, index_name: str = None):
        self.brand_domain = brand_domain
        self.index_name = index_name or f"{brand_domain.replace('.', '-')}-baseline"
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = None
        self.namespace = "products"
        
    async def create_index(self):
        """Create the baseline index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating baseline index: {self.index_name}")
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud="gcp",
                region="us-central1",
                # Configure server-side embeddings
                embed={
                    "model": "llama-text-embed-v2",
                    "field_map": {"text": "text"},
                    "dimension": 2048
                }
            )
            logger.info(f"✅ Created baseline index: {self.index_name}")
        
        self.index = self.pc.Index(self.index_name)
    
    async def ingest_products(self, products: List[Product], batch_size: int = 100):
        """Ingest products using to_markdown() for text generation."""
        logger.info(f"Ingesting {len(products)} products into baseline index")
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            vectors = []
            
            for product in batch:
                # Generate text using to_markdown
                text = Product.to_markdown(product, obfuscatePricing=False)
                
                # Prepare record for server-side embeddings
                # Metadata must be stored as JSON string for search_records
                metadata = {
                    "product_id": str(product.id),
                    "name": product.name,
                    "price": self._extract_price(product),
                    "category": product.categories[0] if product.categories else "",
                    "brand": product.brand,
                    "text_snippet": text[:1000],  # Store truncated text for reranking
                    "content_type": "product",
                    "last_updated": datetime.now().isoformat()
                }
                
                vector = {
                    "_id": str(product.id),
                    "text": text,
                    "metadata": json.dumps(metadata)  # Store as JSON string
                }
                
                vectors.append(vector)
            
            # Upsert batch with server-side embeddings
            self.index.upsert_records(
                namespace=self.namespace,
                records=vectors
            )
            
            logger.info(f"Ingested batch {i//batch_size + 1}/{(len(products) + batch_size - 1)//batch_size}")
    
    async def search(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search the baseline index."""
        # Import required class for search_records
        from pinecone import SearchQuery
        
        # Build search query for server-side embeddings
        search_query = SearchQuery(
            inputs={"text": query},
            top_k=top_k * 2 if filters else top_k  # Get more results if filtering locally
        )
        
        # Use search_records which supports server-side embeddings
        # Note: This method doesn't support filters, so we filter locally
        search_result = self.index.search_records(
            namespace=self.namespace,
            query=search_query,
            fields=["metadata"]
        )
        
        results = []
        for hit in search_result.result.hits:
            # Parse metadata from JSON string
            metadata = hit.fields.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            # Apply local filtering if filters were provided
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            results.append(SearchResult(
                product_id=metadata.get('product_id', hit._id),
                score=hit._score,
                name=metadata.get('name', 'Unknown'),
                price=metadata.get('price', 0),
                category=metadata.get('category', ''),
                metadata=metadata
            ))
        
        # Limit to top_k after filtering
        return results[:top_k]
    
    def _extract_price(self, product: Product) -> float:
        """Extract numeric price from product."""
        price_str = product.salePrice or product.originalPrice or "0"
        try:
            return float(price_str.replace('$', '').replace(',', ''))
        except:
            return 0.0
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, filter_value in filters.items():
            metadata_value = metadata.get(key)
            
            # Handle missing metadata
            if metadata_value is None:
                return False
            
            # Handle different filter operators
            if isinstance(filter_value, dict):
                if '$in' in filter_value:
                    # Check if value is in the list
                    if metadata_value not in filter_value['$in']:
                        return False
                elif '$gte' in filter_value:
                    # Greater than or equal
                    if not (isinstance(metadata_value, (int, float)) and metadata_value >= filter_value['$gte']):
                        return False
                elif '$lte' in filter_value:
                    # Less than or equal
                    if not (isinstance(metadata_value, (int, float)) and metadata_value <= filter_value['$lte']):
                        return False
                elif '$gt' in filter_value:
                    # Greater than
                    if not (isinstance(metadata_value, (int, float)) and metadata_value > filter_value['$gt']):
                        return False
                elif '$lt' in filter_value:
                    # Less than
                    if not (isinstance(metadata_value, (int, float)) and metadata_value < filter_value['$lt']):
                        return False
                elif '$regex' in filter_value:
                    # Regex matching
                    import re
                    pattern = filter_value['$regex']
                    flags = re.IGNORECASE if filter_value.get('$options') == 'i' else 0
                    if not re.search(pattern, str(metadata_value), flags):
                        return False
            else:
                # Simple equality check
                if metadata_value != filter_value:
                    return False
        
        return True


class AssistantLLM:
    """LLM to simulate the voice assistant's responses and tool calls."""
    
    def __init__(self, account: str, primary_model: str = "gpt-4.1"):
        self.account = account
        self.primary_model = primary_model
        self.tools = self._define_tools()
        
    def _define_tools(self):
        """Define the tools available to the assistant in OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "product_search",
                    "description": "Search for products that the user may be interested in",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Be as descriptive and detailed as possible based on the conversation"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "display_product",
                    "description": "Display a specific product to the user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "The ID of the product"
                            },
                            "variant": {
                                "type": "object",
                                "description": "Optional variant details"
                            },
                            "resumption_message": {
                                "type": "string",
                                "description": "Message to say after displaying the product"
                            }
                        },
                        "required": ["product_id", "resumption_message"]
                    }
                }
            }
        ]
    
    async def generate_response(self, chat_context: List[Dict[str, str]], system_prompt: str = None):
        """Generate assistant response using actual LLM with tool calls."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt + "\n\nIMPORTANT: Focus on helping the customer find products. Use the product_search tool proactively."
            })
        
        messages.extend(chat_context)
        
        try:
            # Call the actual LLM
            response = await LLMFactory.chat_completion(
                task="assistant_response",
                messages=messages,
                tools=self.tools,
                model=self.primary_model,
                temperature=0.7
            )
            
            # Parse response - handle different response formats
            tool_calls = []
            message = ""
            
            if isinstance(response, dict):
                if "content" in response:
                    message = response["content"]
                elif "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    if "message" in choice:
                        message = choice["message"].get("content", "")
                        
                        # Extract tool calls
                        if "tool_calls" in choice["message"]:
                            for tc in choice["message"]["tool_calls"]:
                                tool_calls.append({
                                    "name": tc["function"]["name"],
                                    "parameters": json.loads(tc["function"]["arguments"])
                                })
            
            return {
                "message": message or "Let me help you find the perfect product.",
                "tool_calls": tool_calls
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "message": "I'd be happy to help you find products. What are you looking for?",
                "tool_calls": []
            }


class ConversationSimulator:
    """
    Simulates voice assistant conversations for testing search functionality.
    """
    
    def __init__(self, account: str, primary_model: str = "openai/gpt-4.1"):
        self.account = account
        self.conversation_history = []
        self.user_state = UserState()
        self.system_prompt = None
        self.prompt_manager = PromptManager()
        self.assistant_llm = AssistantLLM(account, primary_model)
        self.tool_call_history = []
        
    async def load_system_prompt(self):
        """Load the system prompt for the brand from Langfuse."""
        try:
            # Try to load brand-specific system prompt
            prompt_name = f"{self.account}/full_instructions"
            logger.info(f"Loading system prompt: {prompt_name}")
            
            prompt_template = await self.prompt_manager.get_prompt(
                prompt_name=prompt_name,
                prompt_type="text"
            )
            
            self.system_prompt = prompt_template.prompt
            logger.info(f"✅ Loaded system prompt for {self.account}")
            
        except Exception as e:
            logger.warning(f"Could not load system prompt from Langfuse: {e}")
            # Fallback system prompt
            self.system_prompt = f"""You are a knowledgeable and helpful sales assistant for {self.account}.
            
Help customers find the perfect products by understanding their needs and providing personalized recommendations.
Be conversational, friendly, and focus on solving their problems."""
        
    def add_turn(self, speaker: str, message: str):
        """Add a conversation turn."""
        turn = ConversationTurn(speaker=speaker, message=message)
        self.conversation_history.append(turn)
        
    def get_chat_context(self) -> List[Dict[str, str]]:
        """Convert conversation history to chat context format including system prompt."""
        chat_ctx = []
        
        # Add system prompt if available
        if self.system_prompt:
            chat_ctx.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        for turn in self.conversation_history:
            role = "user" if turn.speaker == "user" else "assistant"
            chat_ctx.append({"role": role, "content": turn.message})
        return chat_ctx
    
    async def simulate_conversation(self, scenario: SearchTestScenario) -> Dict[str, Any]:
        """Simulate a conversation scenario and capture search queries."""
        self.conversation_history = []
        search_queries = []
        
        for turn in scenario.conversation:
            self.add_turn(turn.speaker, turn.message)
            
            # If this is a user turn that might trigger a search
            if turn.speaker == "user" and self._is_search_query(turn.message):
                search_queries.append({
                    'query': turn.message,
                    'context': self.get_chat_context(),
                    'turn_number': len(self.conversation_history)
                })
        
        return {
            'scenario_id': scenario.scenario_id,
            'search_queries': search_queries,
            'final_context': self.get_chat_context()
        }
    
    def _is_search_query(self, message: str) -> bool:
        """Determine if a message is likely a product search query."""
        search_indicators = [
            'looking for', 'need', 'want', 'show me', 'find',
            'search', 'recommend', 'suggest', 'what about',
            'do you have', 'bike', 'product', 'under', 'over',
            'similar', 'like', 'category', 'type'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in search_indicators)


class SearchComparator:
    """
    Compares search results between enhanced and baseline approaches.
    """
    
    def __init__(self, account: str):
        self.account = account
        self.enhanced_search = None
        self.baseline_search = None
        self.comparison_results = []
        
    async def initialize(self):
        """Initialize both search systems."""
        # Initialize enhanced search
        self.enhanced_search = PineconeRAG(
            brand_domain=self.account,
            dense_index_name=f"{self.account.split('.')[0]}-dense",
            sparse_index_name=f"{self.account.split('.')[0]}-sparse"
        )
        await self.enhanced_search.initialize()
        
        # Initialize baseline search
        self.baseline_search = BaselineSearchIndex(self.account)
        await self.baseline_search.create_index()
        
        # Initialize SearchService
        await SearchService.preload_catalog_labels(self.account)
        
    async def compare_searches(
        self,
        query: str,
        chat_context: List[Dict[str, str]],
        user_state: UserState
    ) -> ComparisonMetrics:
        """Compare search results for a single query."""
        
        # Enhanced search
        start_time = time.time()
        enhanced_results, enhanced_metrics = await SearchService.unified_product_search(
            query=query,
            user_state=user_state,
            chat_ctx=chat_context,
            account=self.account,
            use_separate_indexes=True,
            enable_research_enhancement=True,
            enable_filter_extraction=True,
            enable_reranking=True,
            top_k=20,
            top_n=10
        )
        enhanced_time = time.time() - start_time
        
        # Convert to standardized format
        enhanced_standardized = []
        for result in enhanced_results:
            enhanced_standardized.append(SearchResult(
                product_id=result['metadata'].get('product_id', result['id']),
                score=result['score'],
                name=result['metadata'].get('name', 'Unknown'),
                price=result['metadata'].get('price', 0),
                category=result['metadata'].get('category', ''),
                relevance_explanation=result.get('relevance_explanation'),
                metadata=result['metadata']
            ))
        
        # Baseline search
        start_time = time.time()
        baseline_results = await self.baseline_search.search(query, top_k=10)
        baseline_time = time.time() - start_time
        
        # Calculate comparison metrics
        overlap_ratio = self._calculate_overlap(enhanced_standardized, baseline_results)
        ranking_correlation = self._calculate_ranking_correlation(enhanced_standardized, baseline_results)
        
        # LLM evaluation
        llm_evaluation = await self._evaluate_with_llm(
            query, 
            enhanced_standardized, 
            baseline_results,
            chat_context
        )
        
        return ComparisonMetrics(
            query=query,
            enhanced_results=enhanced_standardized,
            baseline_results=baseline_results,
            enhanced_time=enhanced_time,
            baseline_time=baseline_time,
            overlap_ratio=overlap_ratio,
            ranking_correlation=ranking_correlation,
            llm_evaluation=llm_evaluation
        )
    
    def _calculate_overlap(self, results1: List[SearchResult], results2: List[SearchResult]) -> float:
        """Calculate the overlap ratio between two result sets."""
        ids1 = {r.product_id for r in results1}
        ids2 = {r.product_id for r in results2}
        
        if not ids1 or not ids2:
            return 0.0
        
        intersection = ids1.intersection(ids2)
        union = ids1.union(ids2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_ranking_correlation(self, results1: List[SearchResult], results2: List[SearchResult]) -> float:
        """Calculate Spearman rank correlation for common products."""
        # Get common product IDs
        ids1 = {r.product_id for r in results1}
        ids2 = {r.product_id for r in results2}
        common_ids = ids1.intersection(ids2)
        
        if len(common_ids) < 2:
            return 0.0
        
        # Create rank mappings
        rank1 = {r.product_id: i for i, r in enumerate(results1)}
        rank2 = {r.product_id: i for i, r in enumerate(results2)}
        
        # Calculate Spearman correlation
        ranks1 = [rank1[id] for id in common_ids]
        ranks2 = [rank2[id] for id in common_ids]
        
        return np.corrcoef(ranks1, ranks2)[0, 1]
    
    async def _evaluate_with_llm(
        self,
        query: str,
        enhanced_results: List[SearchResult],
        baseline_results: List[SearchResult],
        chat_context: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Use LLM to evaluate and compare search results."""
        
        # Prepare evaluation prompt
        system_prompt = """You are an expert at evaluating search result quality for voice commerce applications.
        
        Evaluate the two sets of search results based on:
        1. Relevance to the query and conversation context
        2. Result diversity and coverage
        3. Ranking quality (best matches first)
        4. Suitability for voice interaction
        
        Provide scores (0-10) and detailed reasoning for each criterion."""
        
        # Format results for evaluation
        enhanced_formatted = self._format_results_for_llm(enhanced_results[:5])
        baseline_formatted = self._format_results_for_llm(baseline_results[:5])
        
        # Build conversation context summary
        context_summary = "Recent conversation:\n"
        for msg in chat_context[-5:]:  # Last 5 messages
            context_summary += f"{msg['role'].upper()}: {msg['content']}\n"
        
        user_prompt = f"""Query: "{query}"

{context_summary}

ENHANCED SEARCH RESULTS (with separate indexes, reranking, etc.):
{enhanced_formatted}

BASELINE SEARCH RESULTS (single dense index with to_markdown):
{baseline_formatted}

Please evaluate both result sets and provide:
1. Relevance scores (0-10) for each approach
2. Which approach better understands the user's intent
3. Which results would work better for voice interaction
4. Overall recommendation with reasoning
"""
        
        response = await LLMFactory.chat_completion(
            task="search_evaluation",
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3,
            model="claude-3-5-sonnet-latest"
        )
        
        return {
            'evaluation': response.get('content', ''),
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format search results for LLM evaluation."""
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.name}\n"
            formatted += f"   - Score: {result.score:.3f}\n"
            formatted += f"   - Price: ${result.price}\n"
            formatted += f"   - Category: {result.category}\n"
            if result.relevance_explanation:
                formatted += f"   - Relevance: {result.relevance_explanation}\n"
            formatted += "\n"
        return formatted


class SearchTestRunner:
    """
    Main test runner that orchestrates the comparison tests.
    """
    
    def __init__(self, account: str):
        self.account = account
        self.simulator = ConversationSimulator(account)
        self.comparator = SearchComparator(account)
        self.test_scenarios = []
        self.results = []
        
    async def setup(self):
        """Set up the test environment."""
        await self.comparator.initialize()
        await self.simulator.load_system_prompt()
        self._create_test_scenarios()
        
    def _create_test_scenarios(self):
        """Create test scenarios for different search types."""
        self.test_scenarios = [
            SearchTestScenario(
                scenario_id="basic_category",
                description="Basic category search",
                conversation=[
                    ConversationTurn("assistant", "Hello! How can I help you find the perfect bike today?"),
                    ConversationTurn("user", "I'm looking for a mountain bike")
                ],
                expected_product_types=["mountain bikes"],
                evaluation_criteria={"focus": "category matching"}
            ),
            
            SearchTestScenario(
                scenario_id="price_constraint",
                description="Search with price constraint",
                conversation=[
                    ConversationTurn("assistant", "Welcome! What kind of bike are you interested in?"),
                    ConversationTurn("user", "I need a road bike under $2000")
                ],
                expected_product_types=["road bikes"],
                evaluation_criteria={"focus": "price filtering", "max_price": 2000}
            ),
            
            SearchTestScenario(
                scenario_id="feature_specific",
                description="Search for specific features",
                conversation=[
                    ConversationTurn("assistant", "Hi there! Looking for something specific today?"),
                    ConversationTurn("user", "Do you have any bikes with electronic shifting?")
                ],
                expected_product_types=["bikes with electronic shifting"],
                evaluation_criteria={"focus": "feature extraction"}
            ),
            
            SearchTestScenario(
                scenario_id="conversational_refinement",
                description="Multi-turn conversation with refinements",
                conversation=[
                    ConversationTurn("assistant", "Hello! How can I help you today?"),
                    ConversationTurn("user", "I want to get into cycling"),
                    ConversationTurn("assistant", "That's great! Are you looking for road cycling, mountain biking, or casual riding?"),
                    ConversationTurn("user", "Road cycling, but I'm a beginner"),
                    ConversationTurn("assistant", "Perfect! For beginners, comfort and value are important. What's your budget?"),
                    ConversationTurn("user", "I'd like to stay under $1500 if possible")
                ],
                expected_product_types=["entry-level road bikes"],
                evaluation_criteria={"focus": "context understanding", "max_price": 1500}
            ),
            
            SearchTestScenario(
                scenario_id="similarity_search",
                description="Search for similar products",
                conversation=[
                    ConversationTurn("assistant", "Welcome back! How can I assist you?"),
                    ConversationTurn("user", "I really like the Specialized Tarmac SL7. Do you have anything similar but more affordable?")
                ],
                expected_product_types=["road bikes similar to Tarmac"],
                evaluation_criteria={"focus": "similarity matching"}
            )
        ]
    
    async def run_tests(self) -> List[Dict[str, Any]]:
        """Run all test scenarios."""
        logger.info(f"Running {len(self.test_scenarios)} test scenarios")
        
        for scenario in self.test_scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running scenario: {scenario.scenario_id}")
            logger.info(f"Description: {scenario.description}")
            
            # Simulate conversation
            simulation = await self.simulator.simulate_conversation(scenario)
            
            # Run comparisons for each search query
            for search_info in simulation['search_queries']:
                logger.info(f"\nComparing search for: '{search_info['query']}'")
                
                comparison = await self.comparator.compare_searches(
                    query=search_info['query'],
                    chat_context=search_info['context'],
                    user_state=self.simulator.user_state
                )
                
                self.results.append({
                    'scenario': scenario,
                    'search_info': search_info,
                    'comparison': comparison
                })
                
                # Log summary
                logger.info(f"Enhanced: {len(comparison.enhanced_results)} results in {comparison.enhanced_time:.3f}s")
                logger.info(f"Baseline: {len(comparison.baseline_results)} results in {comparison.baseline_time:.3f}s")
                logger.info(f"Overlap ratio: {comparison.overlap_ratio:.2f}")
                logger.info(f"Ranking correlation: {comparison.ranking_correlation:.2f}")
        
        return self.results
    
    async def generate_report(self) -> str:
        """Generate a comprehensive comparison report."""
        report = f"""# Voice Assistant Search Comparison Report

**Account**: {self.account}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Scenarios Tested**: {len(self.test_scenarios)}

## Executive Summary

This report compares the enhanced search system (separate dense/sparse indexes with reranking) 
against a baseline approach (single dense index using Product.to_markdown()).

"""
        
        # Aggregate metrics
        total_comparisons = len(self.results)
        avg_enhanced_time = np.mean([r['comparison'].enhanced_time for r in self.results])
        avg_baseline_time = np.mean([r['comparison'].baseline_time for r in self.results])
        avg_overlap = np.mean([r['comparison'].overlap_ratio for r in self.results])
        avg_correlation = np.mean([r['comparison'].ranking_correlation for r in self.results])
        
        report += f"""## Overall Metrics

- **Average Response Times**:
  - Enhanced: {avg_enhanced_time:.3f}s
  - Baseline: {avg_baseline_time:.3f}s
  - Difference: {(avg_enhanced_time - avg_baseline_time):.3f}s ({((avg_enhanced_time/avg_baseline_time - 1) * 100):.1f}%)

- **Result Similarity**:
  - Average Overlap: {avg_overlap:.2%}
  - Average Ranking Correlation: {avg_correlation:.3f}

"""
        
        # Detailed scenario results
        report += "## Detailed Scenario Results\n\n"
        
        for result in self.results:
            scenario = result['scenario']
            comparison = result['comparison']
            
            report += f"""### {scenario.scenario_id}: {scenario.description}

**Query**: "{comparison.query}"

**Performance**:
- Enhanced: {len(comparison.enhanced_results)} results in {comparison.enhanced_time:.3f}s
- Baseline: {len(comparison.baseline_results)} results in {comparison.baseline_time:.3f}s

**Result Comparison**:
- Overlap: {comparison.overlap_ratio:.2%}
- Ranking Correlation: {comparison.ranking_correlation:.3f}

**Top Results**:

Enhanced:
"""
            for i, r in enumerate(comparison.enhanced_results[:3], 1):
                report += f"{i}. {r.name} (${r.price}) - Score: {r.score:.3f}\n"
            
            report += "\nBaseline:\n"
            for i, r in enumerate(comparison.baseline_results[:3], 1):
                report += f"{i}. {r.name} (${r.price}) - Score: {r.score:.3f}\n"
            
            report += f"\n**LLM Evaluation**:\n{comparison.llm_evaluation['evaluation']}\n\n"
            report += "-" * 60 + "\n\n"
        
        return report
    
    async def save_results(self, output_dir: str = "search_comparison_results"):
        """Save test results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results_data = []
        for r in self.results:
            results_data.append({
                'scenario_id': r['scenario'].scenario_id,
                'query': r['comparison'].query,
                'enhanced_time': r['comparison'].enhanced_time,
                'baseline_time': r['comparison'].baseline_time,
                'overlap_ratio': r['comparison'].overlap_ratio,
                'ranking_correlation': r['comparison'].ranking_correlation,
                'enhanced_results': [asdict(res) for res in r['comparison'].enhanced_results],
                'baseline_results': [asdict(res) for res in r['comparison'].baseline_results],
                'llm_evaluation': r['comparison'].llm_evaluation
            })
        
        with open(f"{output_dir}/comparison_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save report
        report = await self.generate_report()
        with open(f"{output_dir}/comparison_report.md", 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to {output_dir}/")


async def main():
    """Main function to run the search comparison tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare voice assistant search approaches")
    parser.add_argument("--account", default="specialized.com", help="Brand account to test")
    parser.add_argument("--ingest-baseline", action="store_true", help="Ingest products to baseline index")
    parser.add_argument("--max-products", type=int, default=1000, help="Maximum products to ingest")
    
    args = parser.parse_args()
    
    # Initialize storage
    storage_manager = get_account_storage_provider()
    
    if args.ingest_baseline:
        logger.info("Setting up baseline index...")
        
        # Load products
        product_manager = ProductManager(storage_manager)
        products = await product_manager.fetch_products(args.account, num_products=args.max_products)
        logger.info(f"Loaded {len(products)} products")
        
        # Create and populate baseline index
        baseline_index = BaselineSearchIndex(args.account)
        await baseline_index.create_index()
        await baseline_index.ingest_products(products)
        logger.info("Baseline index ready!")
    
    # Run comparison tests
    logger.info("\nStarting search comparison tests...")
    runner = SearchTestRunner(args.account)
    await runner.setup()
    
    results = await runner.run_tests()
    await runner.save_results()
    
    # Print summary
    report = await runner.generate_report()
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(report.split("## Detailed Scenario Results")[0])


if __name__ == "__main__":
    asyncio.run(main())