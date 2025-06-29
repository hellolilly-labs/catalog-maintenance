"""
Search utilities for handling query enhancement, search execution, and speech timing.
Enhanced with intelligent query optimization and brand-specific filter extraction.
"""

import asyncio
import logging
import json
import time
from typing import Any, Dict, List, Callable, Optional, Union, Tuple
from livekit.agents import llm

from spence.llm_service import LlmService
from spence.product import Product
from spence.rag import PineconeRAG
from spence.model import UserState
from spence.account_manager import get_account_manager

# Import enhanced components from catalog-maintenance
try:
    from src.agents.query_optimization_agent import QueryOptimizationAgent
    from src.agents.catalog_filter_analyzer import CatalogFilterAnalyzer
    logger.info("Advanced RAG components loaded successfully")
except ImportError as e:
    logger.warning(f"Advanced RAG components not available: {e}")
    QueryOptimizationAgent = None
    CatalogFilterAnalyzer = None

# Langfuse for prompt management
try:
    from langfuse import Langfuse
    _langfuse_client = Langfuse()
    logger.info("Langfuse client initialized")
except Exception as e:
    logger.warning(f"Langfuse not available: {e}")
    _langfuse_client = None

logger = logging.getLogger(__name__)

class SearchService:
    """Service class for handling various types of searches and query enhancements"""
    
    # Cache for query optimizers per account
    _query_optimizers: Dict[str, QueryOptimizationAgent] = {}
    
    @staticmethod
    async def get_query_optimizer(account: str) -> Optional[QueryOptimizationAgent]:
        """Get or create a query optimizer for an account."""
        if not QueryOptimizationAgent:
            return None
            
        if account not in SearchService._query_optimizers:
            try:
                optimizer = QueryOptimizationAgent(account)
                SearchService._query_optimizers[account] = optimizer
                logger.info(f"Created query optimizer for {account}")
            except Exception as e:
                logger.error(f"Failed to create query optimizer for {account}: {e}")
                return None
                
        return SearchService._query_optimizers[account]

    @staticmethod
    async def enhance_query(
        query: str, 
        user_state: UserState,
        chat_ctx: llm.ChatContext, 
        system_prompt: str, 
        model_name: str = "gpt-4.1"
    ) -> str:
        """
        Enhance a search query using an LLM based on conversation context.
        
        Args:
            query: The original search query
            chat_ctx: Conversation context
            system_prompt: System prompt for the LLM
            model_name: LLM model to use
            
        Returns:
            Enhanced query string or original query if enhancement fails
        """
        if not query:
            return query
            
        try:
            # Create a new context for the query enhancement
            enhancement_ctx = llm.ChatContext([])
            
            # Add system prompt
            if system_prompt:
                enhancement_ctx.add_message(
                    role="system",
                    content=[system_prompt]
                )
                
            # Prepare the enhancement prompt
            enhancement_prompt = (
                f"Analyze this query and enhance it based on the conversation context. "
                f"Return only the enhanced query text with no explanations.\n\n"
                f"Query: {query}\n\nConversation:\n\"\"\""
            )
            
            # Extract relevant conversation history - use more context for better understanding
            messages = chat_ctx.copy().items[-50:] if len(chat_ctx.copy().items) > 50 else chat_ctx.copy().items
            for message in messages:
                if hasattr(message, "role") and message.role != "system":
                    content = message.content
                    if isinstance(content, list):
                        content = " ".join(content)
                    enhancement_prompt += f"{message.role}: {content}\n"
            
            enhancement_prompt += "\"\"\""
            
            # Add the prompt to the context
            enhancement_ctx.add_message(
                role="user",
                content=[enhancement_prompt]
            )
            
            # Get LLM service and generate enhanced query
            llm_model = LlmService.fetch_model_service_from_model(model_name=model_name, account=user_state.account, user=user_state.user_id, model_use="query_enhancement")
            
            enhanced_query = await LlmService.chat_wrapper(
                llm_service=llm_model,
                chat_ctx=enhancement_ctx,
            )
            
            logger.debug(f"Original query: {query}")
            logger.debug(f"Enhanced query: {enhanced_query}")
            
            # Return enhanced query if valid, otherwise return original
            return enhanced_query if enhanced_query else query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query  # Fall back to original query on error

    @staticmethod
    async def perform_search(
        query: str,
        search_function: Callable,
        query_enhancer: Optional[Callable] = None,
        search_params: Dict[str, Any] = None,
        user_state: UserState = None,
    ) -> Any:
        """
        Generic search function that handles query enhancement and search execution.
        
        Args:
            query: Original search query
            search_function: Async function that performs the actual search
            query_enhancer: Optional function to enhance the query
            search_params: Parameters to pass to search_function
            
        Returns:
            Search results from the search_function
        """
        if not query:
            return []
            
        search_params = search_params or {}
        
        # 1. Enhance query if enhancer is provided
        enhanced_query = query
        if query_enhancer:
            try:
                enhanced_query = await query_enhancer(query, user_state, **search_params.get("enhancer_params", {}))
            except Exception as e:
                logger.error(f"Error enhancing query: {e}")
        
        # 2. Run the search function with the enhanced query
        try:
            return await search_function(enhanced_query, **search_params)
        except Exception as e:
            logger.error(f"Error in search function: {e}")
            return []

    @staticmethod
    async def enhance_product_query(query: str, user_state: UserState, chat_ctx: llm.ChatContext, product_knowledge: str = "") -> str:
        """
        Enhance a product search query with domain-specific knowledge.
        
        Args:
            query: Original search query
            chat_ctx: Conversation context
            product_knowledge: Optional product domain knowledge
            
        Returns:
            Enhanced query string
        """
        system_prompt = (
            f"You are a product search assistant who is an expert at crafting product search queries. "
            f"The following is a knowledge base that will assist you in curating the user's search query. "
            f"Use this knowledge base to help you curate the perfect search query based on the initial query "
            f"as well as the context of the conversation.\n\n{product_knowledge or 'No specific product knowledge provided.'}"
        )
        return await SearchService.enhance_query(query, user_state, chat_ctx, system_prompt)
    
    @staticmethod
    async def enhance_product_query_with_filters(
        query: str, 
        user_state: UserState, 
        chat_ctx: llm.ChatContext, 
        account: str,
        product_knowledge: str = ""
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced product query with intelligent filter extraction.
        
        Args:
            query: Original search query
            user_state: User state with context
            chat_ctx: Conversation context
            account: Account domain
            product_knowledge: Optional product domain knowledge
            
        Returns:
            Tuple of (enhanced_query, extracted_filters)
        """
        try:
            # Get query optimizer for this account
            optimizer = await SearchService.get_query_optimizer(account)
            
            if not optimizer:
                # Fallback to simple enhancement
                enhanced_query = await SearchService.enhance_product_query(
                    query, user_state, chat_ctx, product_knowledge
                )
                return enhanced_query, {}
            
            # Build context from chat history - use more context
            context = {
                "recent_messages": [],
                "expressed_interests": []
            }
            
            # Extract conversation context - look further back for better understanding
            messages = chat_ctx.items[-30:] if len(chat_ctx.items) > 30 else chat_ctx.items
            for msg in messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    if msg.role == "user":
                        content = msg.content if isinstance(msg.content, str) else " ".join(msg.content)
                        context["recent_messages"].append(content)
                    elif msg.role == "assistant":
                        # Extract mentioned interests/preferences
                        content = msg.content if isinstance(msg.content, str) else " ".join(msg.content)
                        if any(term in content.lower() for term in ["looking for", "interested in", "need", "want", "recommend"]):
                            context["expressed_interests"].append(content)
            
            # Optimize query with filter extraction
            start_time = time.time()
            optimization_result = await optimizer.optimize_product_query(
                original_query=query,
                context=context,
                user_state=user_state.__dict__ if user_state else None
            )
            
            optimization_time = time.time() - start_time
            
            enhanced_query = optimization_result.get("optimized_query", query)
            extracted_filters = optimization_result.get("filters", {})
            
            # Log performance and results
            logger.info(f"Query optimization completed in {optimization_time:.3f}s")
            logger.debug(f"Original query: {query}")
            logger.debug(f"Enhanced query: {enhanced_query}")
            logger.debug(f"Extracted filters: {extracted_filters}")
            
            # Track metrics if Langfuse is available
            if _langfuse_client:
                try:
                    _langfuse_client.generation(
                        name="query_optimization",
                        input={"query": query, "context": context},
                        output={"enhanced_query": enhanced_query, "filters": extracted_filters},
                        metadata={
                            "account": account,
                            "optimization_time": optimization_time,
                            "filter_count": len(extracted_filters)
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to track Langfuse metrics: {e}")
            
            return enhanced_query, extracted_filters
            
        except Exception as e:
            logger.error(f"Error in enhanced query optimization: {e}")
            # Fallback to simple enhancement
            enhanced_query = await SearchService.enhance_product_query(
                query, user_state, chat_ctx, product_knowledge
            )
            return enhanced_query, {}

    @staticmethod
    async def enhance_knowledge_query(query: str, user_state: UserState, chat_ctx: llm.ChatContext, knowledge_base: str = "") -> str:
        """
        Enhance a knowledge search query with domain-specific context.
        
        Args:
            query: Original search query
            chat_ctx: Conversation context
            knowledge_base: Optional knowledge base context
            
        Returns:
            Enhanced query string
        """
        system_prompt = (
            f"You are a knowledge retrieval specialist. Your job is to enhance search queries to find "
            f"relevant information in documentation, guides, and other informational content. "
            f"Use this knowledge base to improve the query precision:\n\n{knowledge_base or 'No specific knowledge base provided.'}"
        )
        return await SearchService.enhance_query(query, user_state, chat_ctx, system_prompt)

    @staticmethod
    async def search_products_rag(
        query: str, 
        account: str = None, 
        top_k: int = 35, 
        top_n: int = 10, 
        min_score: float = 0.15,
        min_n: int = 0,
        **kwargs
    ) -> List[Dict]:
        """
        Search for products using RAG.
        
        Args:
            query: Search query
            account: Account identifier
            top_k: Maximum number of documents to return
            top_n: Number of top results for re-ranking
            min_score: Minimum similarity score threshold
            min_n: Minimum number of results to consider
            
        Returns:
            List of product search results
        """
        try:
            return await PineconeRAG.rag_query_pinecone(
                query=query, 
                account=account,
                use_ranked=False, 
                namespaces=["products"],
                top_k=kwargs.get('top_k', top_k),
                top_n=kwargs.get('top_n', top_n),
                min_score=kwargs.get('min_score', min_score),
                min_n=kwargs.get('min_n', min_n)
            )
        except Exception as e:
            logger.error(f"Error in search_products_rag: {e}")
            return []
    
    @staticmethod
    async def search_products_rag_with_filters(
        query: str,
        filters: Dict[str, Any],
        account: str = None,
        top_k: int = 35,
        top_n: int = 10,
        min_score: float = 0.15,
        min_n: int = 3,
        **kwargs
    ) -> List[Dict]:
        """
        Enhanced RAG search with intelligent filter application.
        
        Args:
            query: Enhanced search query
            filters: Extracted filters from query optimization
            account: Account identifier
            top_k: Maximum documents to retrieve
            top_n: Number of results after reranking
            min_score: Minimum relevance score
            min_n: Minimum results to return
            
        Returns:
            List of product search results with metadata
        """
        try:
            # Get account manager for RAG configuration
            account_manager = await get_account_manager(account)
            rag_index = account_manager.get_rag_details()
            
            if not rag_index:
                logger.warning(f"No RAG index configured for {account}")
                return []
            
            # Initialize RAG client with standard embedding model
            rag = PineconeRAG(
                account=account,
                index_name=rag_index,
                model_name="llama-text-embed-v2",
                namespace="products"
            )
            
            # Convert filters to Pinecone format
            pinecone_filters = SearchService._convert_filters_to_pinecone(filters)
            
            # Search with filters
            start_time = time.time()
            
            if pinecone_filters:
                # Use filtered search for better precision
                results = await rag.search_with_filter(
                    query=query,
                    filter_dict=pinecone_filters,
                    namespace="products",
                    top_k=top_k,
                    top_n=top_n,
                    min_score=min_score,
                    min_n=min_n,
                    timeout=10.0
                )
            else:
                # Use standard search without filters
                results = await rag.search(
                    query=query,
                    namespace="products",
                    top_k=top_k,
                    top_n=top_n,
                    min_score=min_score,
                    min_n=min_n,
                    timeout=10.0
                )
            
            search_time = time.time() - start_time
            
            # Process and enrich results
            enriched_results = []
            for result in results:
                metadata = result.get('metadata', {})
                
                # Parse metadata if it's a string
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        pass
                
                # Add relevance information
                metadata['relevance_score'] = result.get('score', 0)
                metadata['search_rank'] = len(enriched_results) + 1
                
                enriched_results.append({
                    'id': result.get('id'),
                    'score': result.get('score', 0),
                    'metadata': metadata,
                    'matched_filters': SearchService._get_matched_filters(metadata, filters)
                })
            
            logger.info(f"Enhanced RAG search completed in {search_time:.3f}s, found {len(enriched_results)} results")
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Error in filtered product search: {e}")
            return []
    
    @staticmethod
    def _convert_filters_to_pinecone(filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extracted filters to Pinecone query format."""
        pinecone_filters = {}
        
        for key, value in filters.items():
            if value is None or value == "":
                continue
                
            # Handle different filter types
            if key == "price":
                # Price is typically a range [min, max]
                if isinstance(value, list) and len(value) == 2:
                    pinecone_filters["price"] = {
                        "$gte": value[0],
                        "$lte": value[1]
                    }
            elif key == "features":
                # Features are multi-select
                if isinstance(value, list) and value:
                    pinecone_filters["features"] = {"$in": value}
            elif key in ["category", "gender", "frame_material"]:
                # Categorical filters
                if isinstance(value, str):
                    pinecone_filters[key] = value
            elif key == "intended_use":
                # Multi-value field
                if isinstance(value, list) and value:
                    pinecone_filters["intended_use"] = {"$in": value}
                    
        return pinecone_filters
    
    @staticmethod
    def _get_matched_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> List[str]:
        """Identify which filters matched this product."""
        matched = []
        
        for key, value in filters.items():
            if key == "price" and isinstance(value, list):
                product_price = metadata.get("price", 0)
                if value[0] <= product_price <= value[1]:
                    matched.append(f"price_{value[0]}-{value[1]}")
            elif key == "features" and isinstance(value, list):
                product_features = metadata.get("features", [])
                for feature in value:
                    if feature in product_features:
                        matched.append(f"feature_{feature}")
            elif key in metadata:
                if metadata[key] == value:
                    matched.append(f"{key}_{value}")
                elif isinstance(value, list) and metadata[key] in value:
                    matched.append(f"{key}_{metadata[key]}")
                    
        return matched

    @staticmethod
    async def search_products_llm(query: str, products: List[Product], user_state: UserState) -> List[Product]:
        """
        Search for products using LLM for small catalogs.
        
        Args:
            query: Search query
            products: List of available products
            account: Account identifier
            
        Returns:
            List of matching Product objects
        """
        if not products:
            return []
            
        try:
            results = []
            chat_ctx = llm.ChatContext([])
            
            system_prompt = (
                f"You are an expert sales assistant curating products for a user based on their query. "
                f"Return a JSON array of product ID's from the catalog that best match the query."
                f"Do not make up product ID's or hallucinate product ID's."
            )
            
            chat_ctx.add_message(
                role="system",
                content=[system_prompt]
            )
            
            # Build product catalog text
            product_catalog = f"Return a list of products that match: \"{query}\"\n\n"
            product_catalog += "Respond with JSON format only - an array of product ID's. Include up to 7 matches.\n\n"
            product_catalog += "# PRODUCT CATALOG:\n\n"
            
            for product in products[:500]:  # Limit to 100 products to avoid context limits
                product_catalog += f"{Product.to_markdown(depth=1, product=product)}\n\n"
            
            chat_ctx.add_message(
                role="user",
                content=[product_catalog]
            )
            
            # Use capable model for search
            llm_model = LlmService.fetch_model_service_from_model(model_name="gpt-4.1", account=user_state.account, user=user_state.user_id, model_use="query_enhancement")
            search_results = await LlmService.chat_wrapper(
                llm_service=llm_model,
                chat_ctx=chat_ctx,
            )
            
            # Parse results and find products
            product_ids = LlmService.parse_json_response(search_results)
            if product_ids:
                for product_id in product_ids:
                    product = await Product.find_by_id(productId=product_id, account=user_state.account)
                    if product:
                        results.append(product)
            
            return results
        except Exception as e:
            logger.error(f"Error in search_products_llm: {e}")
            return []

    @staticmethod
    async def search_knowledge(
        query: str, 
        account: str = None, 
        top_k: int = 20, 
        top_n: int = 5, 
        min_score: float = 0.15,
        min_n: int = 0,
        **kwargs
    ) -> List[Dict]:
        """
        Search for knowledge base articles.
        
        Args:
            query: Search query
            account: Account identifier
            top_k: Maximum number of documents to return
            top_n: Number of top results for re-ranking
            min_score: Minimum similarity score threshold
            min_n: Minimum number of results to consider
            
        Returns:
            List of knowledge base search results
        """
        try:
            return await PineconeRAG.rag_query_pinecone(
                query=query,
                account=account,
                use_ranked=True, 
                namespaces=["information"],
                top_k=kwargs.get('top_k', top_k),
                top_n=kwargs.get('top_n', top_n),
                min_score=kwargs.get('min_score', min_score),
                min_n=kwargs.get('min_n', min_n)
            )
        except Exception as e:
            logger.error(f"Error in search_knowledge: {e}")
            return []

    @staticmethod
    async def search_knowledge_rag_with_context(
        query: str,
        user_state: UserState,
        chat_ctx: llm.ChatContext,
        account: str = None,
        knowledge_base: str = "",
        top_k: int = 20,
        top_n: int = 5,
        min_score: float = 0.15,
        **kwargs
    ) -> List[Dict]:
        """
        Enhanced knowledge search with context-aware query optimization.
        
        Args:
            query: Search query
            user_state: User state
            chat_ctx: Conversation context
            account: Account identifier
            knowledge_base: Optional knowledge base context
            top_k: Maximum documents to retrieve
            top_n: Number of results after reranking
            min_score: Minimum relevance score
            
        Returns:
            List of knowledge base search results
        """
        try:
            # Get enhanced search service if available
            if QueryOptimizationAgent and _langfuse_client:
                try:
                    # Try to get enhanced query from Langfuse
                    full_prompt_name = f"liddy/catalog/{account}/knowledge_query_enhancement"
                    prompt = _langfuse_client.get_prompt(full_prompt_name)
                    
                    # Build context variables
                    recent_messages = []
                    for msg in chat_ctx.items[-5:]:
                        if hasattr(msg, "role") and msg.role == "user":
                            content = msg.content if isinstance(msg.content, str) else " ".join(msg.content)
                            recent_messages.append(content)
                    
                    enhanced_query = prompt.compile(
                        query=query,
                        knowledge_base=knowledge_base,
                        account=account,
                        conversation_context=" ".join(recent_messages[-2:])
                    )
                    
                    logger.debug(f"Enhanced knowledge query with Langfuse: {enhanced_query}")
                    
                except Exception as e:
                    logger.warning(f"Failed to enhance query with Langfuse: {e}")
                    # Fallback to basic enhancement
                    enhanced_query = await SearchService.enhance_knowledge_query(
                        query, user_state, chat_ctx, knowledge_base
                    )
            else:
                # Use existing enhancement method
                enhanced_query = await SearchService.enhance_knowledge_query(
                    query, user_state, chat_ctx, knowledge_base
                )
            
            # Use enhanced query for search
            results = await SearchService.search_knowledge(
                enhanced_query, account, top_k, top_n, min_score, **kwargs
            )
            
            # Process results with enhanced metadata
            processed_results = []
            for result in results:
                metadata = result.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        pass
                
                processed_results.append({
                    'id': result.get('id'),
                    'score': result.get('score', 0),
                    'text': result.get('text', ''),
                    'metadata': metadata,
                    'type': 'knowledge'
                })
            
            logger.info(f"Enhanced knowledge search completed, found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in enhanced knowledge search: {e}")
            return []

# Update test suite at the end of the file
if __name__ == "__main__":
    # Test suite for search_utils
    import argparse
    import json
    import time
    from unittest.mock import Mock, AsyncMock, patch
    
    # Configure argument parser for test selection
    parser = argparse.ArgumentParser(description="Test search utilities")
    parser.add_argument("--test", choices=["enhance", "search", "products", "knowledge", "all"], default="all",
                      help="Select which test to run")
    parser.add_argument("--real", action="store_true", help="Use real services instead of mocks")
    parser.add_argument("--query", type=str, default=None, help="Custom query to use for testing")
    parser.add_argument("--account", type=str, default="specialized.com", help="Account to use for testing")
    args = parser.parse_args()
    
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("search_utils_test")
    logger.setLevel(logging.DEBUG)
    
    print("\n===== SEARCH UTILITIES TEST SUITE =====")
    if args.real:
        print("⚠️ USING REAL SERVICES - This will make actual API calls! ⚠️")
    
    class MockSession:
        """Mock session for testing speech timing"""
        def __init__(self, speech_done=True, speech_interrupted=False):
            self.current_speech = Mock()
            self.current_speech.done = Mock(return_value=speech_done)
            self.current_speech.interrupted = speech_interrupted
            self.current_speech.wait_for_playout = AsyncMock()
            self.messages = []
            
        async def say(self, text, add_to_chat_ctx=True):
            logger.info(f"Session saying: '{text}'")
            if add_to_chat_ctx:
                self.messages.append({"role": "assistant", "content": text})
            return AsyncMock()
    
    class MockProduct:
        """Mock product for testing product search"""
        def __init__(self, id="test-1", name="Test Product", productUrl="https://example.com/test"):
            self.id = id
            self.name = name 
            self.productUrl = productUrl
            self.description = "Test product description"
            self.imageUrls = ["https://example.com/test.jpg"]
            self.sizes = ["S", "M", "L"]
            self.colors = ["Red", "Blue"]
            
    async def mock_slow_search(_query, **_kwargs):
        """Mock search that takes longer than the timeout"""
        await asyncio.sleep(0.8)  # Simulate slow search
        return [{"id": "result1", "score": 0.95, "text": "Test result"}]
        
    async def mock_fast_search(_query, **_kwargs):
        """Mock search that completes before the timeout"""
        await asyncio.sleep(0.1)  # Simulate fast search
        return [{"id": "result2", "score": 0.85, "text": "Fast result"}]
    
    async def test_enhance_query():
        """Test query enhancement with LLM responses"""
        print("\n----- Testing Query Enhancement -----")
        
        # Create test chat context with conversation history
        test_ctx = llm.ChatContext([])
        test_ctx.add_message(role="user", content="I need a mountain bike")
        test_ctx.add_message(role="assistant", content="What type of trails do you plan to ride?")
        test_ctx.add_message(role="user", content="Mostly downhill and some technical trails")
        
        # Use custom query if provided
        query = args.query or "mountain bike"
        print(f"Testing with query: '{query}'")
        
        if args.real:
            try:
                # Test with real LLM service
                print("\nTesting with real LLM service...")
                
                # Test basic query enhancement
                enhanced = await SearchService.enhance_query(
                    query, 
                    UserState(account=args.account, user_id="test_user"),
                    test_ctx,
                    "You are a query enhancement specialist"
                )
                print(f"Original query: '{query}'")
                print(f"Enhanced query: '{enhanced}'")
                
                # Test domain-specific enhancements
                product_enhanced = await SearchService.enhance_product_query(
                    query, 
                    UserState(account=args.account, user_id="test_user"),
                    test_ctx,
                    "Product knowledge base: Specialized offers Stumpjumper, Demo, and Enduro models for mountain biking."
                )
                print(f"Product-specific enhanced query: '{product_enhanced}'")
                
                knowledge_enhanced = await SearchService.enhance_knowledge_query(
                    query.replace("bike", "bike maintenance") if "bike" in query else query + " maintenance", 
                    test_ctx,
                    "Knowledge base covers: chain lubrication, brake adjustments, tire pressure, and suspension setup."
                )
                print(f"Knowledge-specific enhanced query: '{knowledge_enhanced}'")
                
            except Exception as e:
                print(f"❌ Error with real LLM service: {e}")
        else:
            # Use mocks for testing
            with patch("spence.llm_service.LlmService.fetch_model_service_from_model") as mock_fetch:
                with patch("spence.llm_service.LlmService.chat_wrapper") as mock_chat:
                    # Set up mocks
                    mock_fetch.return_value = AsyncMock()
                    mock_chat.return_value = f"enhanced query for {query} with additional context"
                    
                    # Test basic query enhancement
                    enhanced = await SearchService.enhance_query(
                        query, 
                        test_ctx,
                        "You are a query enhancement specialist"
                    )
                    print(f"Original query: '{query}'")
                    print(f"Enhanced query: '{enhanced}'")
                    
                    # Test domain-specific enhancements
                    product_enhanced = await SearchService.enhance_product_query(
                        query, 
                        test_ctx,
                        "Product knowledge base: Specialized offers Stumpjumper, Demo, and Enduro models for mountain biking."
                    )
                    print(f"Product-specific enhanced query: '{product_enhanced}'")
                    
                    knowledge_enhanced = await SearchService.enhance_knowledge_query(
                        query.replace("bike", "bike maintenance") if "bike" in query else query + " maintenance", 
                        test_ctx,
                        "Knowledge base covers: chain lubrication, brake adjustments, tire pressure, and suspension setup."
                    )
                    print(f"Knowledge-specific enhanced query: '{knowledge_enhanced}'")
        
        print("✅ Query enhancement tests completed")
    
    async def test_search_timing():
        """Test search timing behavior for slow and fast searches"""
        print("\n----- Testing Search Timing -----")
        
        # Test 1: Fast search should not display wait message
        print("\nTest: Fast search (should NOT display wait message)")
        fast_session = MockSession()
        _result = await SearchService.perform_search(
            query="quick search",
            search_function=mock_fast_search,
            timeout_seconds=0.5,
            wait_message="Please wait..."
        )
        if len(fast_session.messages) == 0:
            print("✅ Passed: No wait message displayed for fast search")
        else:
            print("❌ Failed: Wait message was incorrectly displayed")
        
        # Test 2: Slow search should display wait message
        print("\nTest: Slow search (SHOULD display wait message)")
        slow_session = MockSession()
        _result = await SearchService.perform_search(
            query="slow search",
            search_function=mock_slow_search,
            timeout_seconds=0.2,
            wait_message="Please wait..."
        )
        if len(slow_session.messages) == 1:
            print("✅ Passed: Wait message displayed for slow search")
        else:
            print("❌ Failed: Wait message was not displayed")
        
        # Test 3: Ongoing speech that finishes during search
        print("\nTest: Ongoing speech interaction")
        speech_session = MockSession(speech_done=False)
        
        # Create a side effect that makes the speech finish after a delay
        speech_finish_event = asyncio.Event()
        
        async def mock_wait_for_playout():
            await asyncio.sleep(0.1)
            speech_session.current_speech.done = Mock(return_value=True)
            speech_finish_event.set()
        
        speech_session.current_speech.wait_for_playout.side_effect = mock_wait_for_playout
        
        _result = await SearchService.perform_search(
            query="speech test",
            search_function=mock_slow_search,
            timeout_seconds=0.2
        )
        
        if speech_session.current_speech.wait_for_playout.called:
            print("✅ Passed: Search waited for speech to complete")
        else:
            print("❌ Failed: Search did not wait for speech")
        
        print("✅ Search timing tests completed")
    
    async def test_product_search():
        """Test product search functionality"""
        print("\n----- Testing Product Search -----")
        
        # Use custom query if provided
        query = args.query or "mountain bike"
        print(f"Testing with query: '{query}'")
        
        if args.real:
            try:
                # Test with real RAG services
                print("\nTesting with real RAG service...")
                print("This may take a moment...")
                
                # Test RAG-based product search
                start_time = time.time()
                results = await SearchService.search_products_rag(query, account=args.account)
                elapsed = time.time() - start_time
                print(f"RAG search completed in {elapsed:.2f}s")
                print(f"Found {len(results)} products via RAG")
                
                # Display top results
                for i, result in enumerate(results[:3]):
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            pass
                    
                    print(f"\nResult {i+1}:")
                    print(f"  ID: {result.get('id', 'N/A')}")
                    print(f"  Score: {result.get('score', 0):.4f}")
                    print(f"  Name: {metadata.get('name', 'N/A')}")
                    if 'description' in metadata:
                        desc = metadata['description']
                        if len(desc) > 100:
                            desc = desc[:97] + "..."
                        print(f"  Description: {desc}")
                
                # For small catalogs, also test LLM-based search
                try:
                    print("\nFetching products for LLM-based search...")
                    products = await Product.get_products_async(account=args.account)
                    if len(products) < 50:  # Only test LLM search with small catalogs
                        print(f"Testing LLM search with {len(products)} products...")
                        start_time = time.time()
                        llm_results = await SearchService.search_products_llm(query, products, user_state=args.user_state)
                        elapsed = time.time() - start_time
                        print(f"LLM search completed in {elapsed:.2f}s")
                        print(f"Found {len(llm_results)} products via LLM search")
                        
                        for i, product in enumerate(llm_results[:3]):
                            print(f"\nLLM Result {i+1}:")
                            print(f"  ID: {product.id}")
                            print(f"  Name: {product.name}")
                            print(f"  URL: {product.productUrl}")
                    else:
                        print(f"Skipping LLM search test - catalog too large ({len(products)} products)")
                except Exception as e:
                    print(f"❌ Error with LLM-based product search: {e}")
                
            except Exception as e:
                print(f"❌ Error testing real product search: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Create mock product data
            mock_products = [
                MockProduct(id="bike1", name="Trail Bike", productUrl="https://example.com/bike1"),
                MockProduct(id="bike2", name="Road Bike", productUrl="https://example.com/bike2"),
            ]
            
            # Test RAG-based product search
            print("\nTest: RAG-based product search (mocked)")
            mock_rag_results = [
                {
                    "id": "bike1", 
                    "score": 0.95, 
                    "metadata": {
                        "id": "bike1",
                        "name": "Trail Bike",
                        "description": "A great trail bike",
                        "categories": ["Mountain", "Trail"],
                        "original_price": "$2,999"
                    }
                }
            ]
            
            with patch("spence.rag.PineconeRAG.rag_query_pinecone", new=AsyncMock(return_value=mock_rag_results)):
                results = await SearchService.search_products_rag(query, account="test")
                print(f"Found {len(results)} products via RAG")
                print(f"First result: {results[0]['id'] if results else 'None'}")
            
            # Test LLM-based product search
            print("\nTest: LLM-based product search (mocked)")
            with patch("spence.product.Product.to_markdown", return_value="Mocked product markdown"):
                with patch("spence.llm_service.LlmService.fetch_model_service_from_model", return_value=AsyncMock()):
                    with patch("spence.llm_service.LlmService.chat_wrapper", new=AsyncMock(return_value='["https://example.com/bike1"]')):
                        with patch("spence.llm_service.LlmService.parse_json_response", return_value=["https://example.com/bike1"]):
                            with patch("spence.product.Product.find_by_url", return_value=mock_products[0]):
                                results = await SearchService.search_products_llm(query, mock_products, user_state=UserState(account=args.account, user_id="test_user"))
                                print(f"Found {len(results)} products via LLM search")
                                if results:
                                    print(f"First result: {results[0].name}")
        
        print("✅ Product search tests completed")
        
    async def test_knowledge_search():
        """Test knowledge search functionality"""
        print("\n----- Testing Knowledge Search -----")
        
        # Use custom query if provided
        query = args.query or "bike maintenance"
        print(f"Testing with query: '{query}'")
        
        if args.real:
            try:
                # Test with real knowledge search
                print("\nTesting with real knowledge search...")
                print("This may take a moment...")
                
                start_time = time.time()
                results = await SearchService.search_knowledge(query, account=args.account)
                elapsed = time.time() - start_time
                
                print(f"Knowledge search completed in {elapsed:.2f}s")
                print(f"Found {len(results)} knowledge articles")
                
                # Display top results
                for i, result in enumerate(results[:3]):
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            pass
                    
                    print(f"\nArticle {i+1}:")
                    print(f"  ID: {result.get('id', 'N/A')}")
                    print(f"  Score: {result.get('score', 0):.4f}")
                    print(f"  Title: {metadata.get('title', 'N/A')}")
                    
                    # Get text content if available
                    text = result.get('text', '')
                    if text and isinstance(text, str):
                        if len(text) > 100:
                            text = text[:97] + "..."
                        print(f"  Content: {text}")
                
            except Exception as e:
                print(f"❌ Error testing real knowledge search: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Test with mocked knowledge search
            mock_knowledge_results = [
                {
                    "id": "article1", 
                    "score": 0.88, 
                    "text": "How to maintain your bike chain",
                    "metadata": {
                        "title": "Bike Maintenance Guide",
                        "category": "maintenance",
                        "content_type": "information"
                    }
                }
            ]
            
            with patch("spence.rag.PineconeRAG.rag_query_pinecone", new=AsyncMock(return_value=mock_knowledge_results)):
                results = await SearchService.search_knowledge(query, account="test")
                print(f"Found {len(results)} knowledge articles")
                if results:
                    if isinstance(results[0].get('metadata'), str):
                        # Handle case where metadata might be a JSON string
                        metadata = json.loads(results[0].get('metadata'))
                    else:
                        metadata = results[0].get('metadata', {})
                        
                    print(f"First result: {metadata.get('title', 'Untitled')}")
        
        print("✅ Knowledge search tests completed")
        
    async def run_all_tests():
        """Run all test cases based on command line arguments"""
        if args.test in ["enhance", "all"]:
            await test_enhance_query()
            
        if args.test in ["search", "all"]:
            await test_search_timing()
            
        if args.test in ["products", "all"]:
            await test_product_search()
            
        if args.test in ["knowledge", "all"]:
            await test_knowledge_search()
            
        print("\n===== TEST SUITE COMPLETE =====")
    
    # Execute the tests
    asyncio.run(run_all_tests())
