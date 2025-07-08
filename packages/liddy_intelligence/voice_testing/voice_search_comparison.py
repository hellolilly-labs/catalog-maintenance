#!/usr/bin/env python3
"""
Voice-Realistic Search Comparison Framework

This framework simulates realistic voice assistant conversations with proper
tool calls and natural spoken language patterns for comparing search approaches.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import re

from pinecone import Pinecone
import numpy as np

# Import our components
from liddy.models.product import Product
from liddy.models.product_manager import ProductManager
from liddy.storage import get_account_storage_provider
from liddy.llm.simple_factory import LLMFactory
from liddy.prompt_manager import PromptManager
from liddy.search.service import SearchService, SearchMetrics
from test_search_comparison import (
    BaselineSearchIndex, 
    SearchResult, 
    ComparisonMetrics,
    ConversationTurn,
    SearchTestScenario
)

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call made by the assistant."""
    tool_name: str
    parameters: Dict[str, Any]
    timestamp: datetime = None
    result: Optional[Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AssistantResponse:
    """Assistant response with potential tool calls."""
    message: str
    tool_calls: List[ToolCall]
    thinking: Optional[str] = None


class VoiceUserSimulator:
    """
    Simulates realistic voice user inputs using an intelligent LLM.
    """
    
    def __init__(self, intelligence_model: str = "openai/o3-mini"):
        self.intelligence_model = intelligence_model
        
    async def generate_voice_query(
        self, 
        scenario: str, 
        context: Optional[List[Dict[str, str]]] = None,
        previous_products: Optional[List[str]] = None
    ) -> str:
        """Generate a realistic voice query for the given scenario."""
        
        system_prompt = """You are simulating a real customer speaking to a voice AI shopping assistant.

Generate natural, conversational speech that someone would actually say out loud. Include:
- Natural speech patterns (um, uh, like, you know)
- Conversational style (not formal writing)
- Realistic pauses and corrections
- How people actually talk when shopping

Examples of natural voice queries:
- "Um, I'm looking for a bike for my daughter, she's like 12 years old"
- "Do you have any road bikes that are, uh, good for beginners? Maybe under 2000 dollars?"
- "I saw this bike online, I think it was called the Tarmac? Do you have anything similar but cheaper?"
- "So I need something for commuting to work, maybe 5 miles each way, nothing too fancy"

Keep responses under 2 sentences as people don't speak in long paragraphs."""

        user_prompt = f"Generate a natural voice query for this scenario: {scenario}"
        
        if context:
            user_prompt += f"\n\nConversation so far:\n"
            for msg in context[-13:]:  # Last 13 messages
                user_prompt += f"{msg['role'].upper()}: {msg['content']}\n"
        
        if previous_products:
            user_prompt += f"\n\nProducts mentioned: {', '.join(previous_products[:3])}"
        
        response = await LLMFactory.chat_completion(
            task="voice_query_generation",
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.8,
            model=self.intelligence_model
        )
        
        return response.get("content", "")


class VoiceAssistantSimulator:
    """
    Simulates the voice assistant with proper tool calls and responses.
    """
    
    def __init__(self, account: str, primary_model: str = "openai/gpt-4.1"):
        self.account = account
        self.primary_model = primary_model
        self.prompt_manager = PromptManager()
        self.system_prompt = None
        self.tool_definitions = self._define_tools()
        
    def _define_tools(self):
        """Define tools matching the actual voice assistant."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "product_search",
                    "description": """Use this to search for products that the user may be interested in. Include as much depth and detail in the query as possible to get the best results. Generally, the only way for you to know about individual products is through this tool call, so feel free to liberally call this tool call.

ALWAYS say something to the user in parallel to calling this function to let them know you are looking for products (using whatever language is appropriate for the context of the conversation).
""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": """The query to use for the search. Be as descriptive and detailed as possible based on the context of the entire conversation. For example, the query "mountain bike" is not very descriptive, but "mid-range non-electric mountain bike with electronic shifting for downhill flowy trails for advanced rider" is much more descriptive. The more descriptive the query, the better the results will be. You can also use this to search for specific products by name or ID, but be sure to include as much context as possible to help narrow down the results."""
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
                    "description": """Use this to display or show exactly one product. The tool call itself will handle actually doing this. Never try to display or show products on your own, ALWAYS use this tool call. It is not possible to show multiple products at once, so you must narrow it down to a single product. For example, if the user wants to see multiple products at a time, you should show them the first product only. Then move on to the next product when the user is ready. Never make multiple calls to this function at once.

You typically would first use `product_search` to find the product ID, and then use this function to display the product.

NOTE: If you know the user is already looking at that product (because they are viewing that page or something like that) then you can silently call this function without necessarily saying anything to the user. This is useful for showing the product details without interrupting the conversation. Otherwise, it is generally a good idea to say something to the user to let them know you are displaying the product.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "The ID of the product to display. Required."
                            },
                            "variant": {
                                "type": "object",
                                "description": """The ID or name of the product variant to display. This is optional and can be used to show a specific variant of the product. If not provided, the default variant will be shown. Include the name of the variant, such as "color" or "size", and the value of the variant, such as "red" or "large". For example, "color=red" or "size=large". If you don't know the variant ID, you can use the `product_search` function to find it."""
                            },
                            "resumption_message": {
                                "type": "string",
                                "description": "The message to say back to the user after displaying the product. This is required and should be a natural language message that is appropriate for the context of the conversation."
                            }
                        },
                        "required": ["product_id", "resumption_message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "knowledge_search",
                    "description": """Search the knowledge base (restricted to the "information" namespace) using the provided query.

ALWAYS say something to the user in parallel to calling this function to let them know you are finding the relevant information (using whatever language is contextually appropriate).
""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string (required)."
                            },
                            "topic_name": {
                                "type": "string",
                                "description": "Optional topic name to refine the query."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def load_system_prompt(self):
        """Load the brand-specific system prompt."""
        try:
            prompt_name = f"{self.account}/full_instructions"
            prompt_template = await self.prompt_manager.get_prompt(
                prompt_name=prompt_name,
                prompt_type="text"
            )
            self.system_prompt = prompt_template.prompt
            logger.info(f"✅ Loaded system prompt for {self.account}")
        except Exception as e:
            logger.warning(f"Using fallback system prompt: {e}")
            self.system_prompt = f"You are a helpful sales assistant for {self.account}."
    
    async def generate_response(
        self, 
        chat_context: List[Dict[str, str]], 
        execute_search: Optional[Callable] = None,
        encourage_product_search: bool = True
    ) -> AssistantResponse:
        """Generate assistant response with tool calls using actual LLM."""
        
        # Prepare messages with system prompt
        messages = []
        
        # Enhanced system prompt to encourage product search
        system_content = self.system_prompt or f"You are a helpful sales assistant for {self.account}."
        
        if encourage_product_search:
            system_content += """

IMPORTANT: You are in a product discovery conversation. Your primary goal is to help the customer find products. 
- Always use the product_search tool when the user expresses any interest in products
- Be proactive in searching for products based on their needs
- After searching, always use display_product to show specific products
- Keep responses concise and focused on helping them find products"""
        
        messages.append({"role": "system", "content": system_content})
        messages.extend(chat_context)
        
        # Use the actual primary model with tools
        try:
            # Get the response from the LLM
            response = await LLMFactory.chat_completion(
                task="assistant_response",
                messages=messages,
                tools=self.tool_definitions,
                model=self.primary_model,
                temperature=0.7,
                stream=False
            )
            
            # Parse the response
            tool_calls = []
            message = ""
            
            # Handle OpenAI response format
            if isinstance(response, dict):
                # Check for content at top level first (direct format)
                if "content" in response:
                    message = response["content"]
                
                # Also check for tool_calls at top level
                if "tool_calls" in response:
                    for tc in response["tool_calls"]:
                        try:
                            tool_call = ToolCall(
                                tool_name=tc["function"]["name"],
                                parameters=json.loads(tc["function"]["arguments"])
                            )
                            tool_calls.append(tool_call)
                        except Exception as e:
                            logger.warning(f"Error parsing tool call: {e}")
                
                # Also check choices format for compatibility
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "message" in choice:
                        # Get content if we haven't already
                        if not message:
                            message = choice["message"].get("content", "")
                        
                        # Extract tool calls from choices format
                        if "tool_calls" in choice["message"] and not tool_calls:
                            for tc in choice["message"]["tool_calls"]:
                                try:
                                    tool_call = ToolCall(
                                        tool_name=tc["function"]["name"],
                                        parameters=json.loads(tc["function"]["arguments"])
                                    )
                                    tool_calls.append(tool_call)
                                except Exception as e:
                                    logger.warning(f"Error parsing tool call: {e}")
            
            # If no message but we have tool calls, generate a message
            if not message and tool_calls:
                if any(tc.tool_name == "product_search" for tc in tool_calls):
                    message = "Let me search for products that match your needs."
                elif any(tc.tool_name == "display_product" for tc in tool_calls):
                    message = "Here's a product that might interest you."
            
            # If still no message, provide a helpful default
            if not message:
                message = "I'd be happy to help you find the perfect product. Could you tell me more about what you're looking for?"
            
            return AssistantResponse(
                message=message,
                tool_calls=tool_calls,
                thinking=response.get("thinking") if isinstance(response, dict) else None
            )
            
        except Exception as e:
            logger.error(f"Error generating assistant response: {e}")
            # Fallback response
            return AssistantResponse(
                message="I'd be happy to help you find products. What are you looking for?",
                tool_calls=[],
                thinking=None
            )


class VoiceSearchComparator:
    """
    Compares search approaches in realistic voice conversations.
    """
    
    def __init__(self, account: str):
        self.account = account
        self.user_simulator = VoiceUserSimulator()
        self.assistant_simulator = VoiceAssistantSimulator(account)
        self.enhanced_search = None
        self.baseline_search = None
        self.product_manager = ProductManager(account=account)
        
    async def initialize(self):
        """Initialize all components."""
        # Load system prompt
        await self.assistant_simulator.load_system_prompt()
        
        # Initialize search systems
        from liddy.search.pinecone import PineconeRAG
        self.enhanced_search = PineconeRAG(
            brand_domain=self.account,
            dense_index_name=f"{self.account.replace('.', '-')}-dense",
            sparse_index_name=f"{self.account.replace('.', '-')}-sparse"
        )
        await self.enhanced_search.initialize()
        
        self.baseline_search = BaselineSearchIndex(self.account)
        await self.baseline_search.create_index()
        
        # SearchService will handle catalog intelligence loading internally
    
    async def simulate_voice_conversation(
        self, 
        scenario_description: str,
        max_turns: int = 5
    ) -> Dict[str, Any]:
        """Simulate a complete voice conversation with tool execution."""
        
        conversation_history = []
        search_queries = []
        tool_calls = []
        products_shown = []
        
        # Generate initial user query
        initial_query = await self.user_simulator.generate_voice_query(
            scenario=scenario_description
        )
        
        conversation_history.append({
            "role": "user",
            "content": initial_query
        })
        
        # Conversation loop
        for turn in range(max_turns):
            # Assistant response with potential tool calls
            assistant_response = await self.assistant_simulator.generate_response(
                chat_context=conversation_history,
                execute_search=self._execute_enhanced_search,
                encourage_product_search=False
            )
            
            # Add assistant message with tool calls if present
            assistant_msg = {
                "role": "assistant",
                "content": assistant_response.message
            }
            
            # Include tool calls in assistant message if present
            if assistant_response.tool_calls:
                assistant_msg["tool_calls"] = []
                for idx, tc in enumerate(assistant_response.tool_calls):
                    tool_call_id = f"call_{turn}_{idx}"
                    assistant_msg["tool_calls"].append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.parameters)
                        }
                    })
                    # Store the ID for later use
                    tc.id = tool_call_id
            
            conversation_history.append(assistant_msg)
            
            # Process tool calls
            for tool_call in assistant_response.tool_calls:
                tool_calls.append(tool_call)
                
                if tool_call.tool_name == "product_search":
                    # Track the search query
                    search_queries.append({
                        "query": tool_call.parameters["query"],
                        "turn": turn + 1,
                        "context": conversation_history.copy()
                    })
                    
                    # Execute the search and add results to context
                    try:
                        search_results = await self._execute_enhanced_search(tool_call.parameters["query"])
                        
                        # Add tool result to conversation
                        if search_results:
                            result_summary = f"Found {len(search_results)} products. Top results: "
                            for i, result in enumerate(search_results[:3]):
                                result_summary += f"\n# {i+1}. {Product.to_markdown(product=result, depth=1, obfuscatePricing=True)}\n"
                            tool_call.result = search_results
                        else:
                            result_summary = "No products found matching your search."
                        
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_summary
                        })
                    except Exception as e:
                        logger.error(f"Error executing search: {e}")
                        # Still need to provide a tool response even on error
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Sorry, I encountered an error while searching for products."
                        })
                    
                    # now come up with the assistant's response given the tool call results
                    assistant_response = await self.assistant_simulator.generate_response(
                        chat_context=conversation_history,
                        # execute_search=self._execute_enhanced_search,
                        encourage_product_search=False
                    )
                                # Add assistant message with tool calls if present
                    assistant_msg = {
                        "role": "assistant",
                        "content": assistant_response.message
                    }
                    conversation_history.append(assistant_msg)                    
                
                elif tool_call.tool_name == "display_product":
                    # Track displayed products
                    product_id = tool_call.parameters.get("product_id")
                    if product_id:
                        products_shown.append(product_id)
                        content = f"Displayed product {product_id}"
                    else:
                        content = "Error: No product ID provided"
                    
                    # Always add display confirmation to conversation
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": content
                    })
                
                elif tool_call.tool_name == "knowledge_search":
                    # Handle knowledge search tool
                    query = tool_call.parameters.get("query", "")
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Knowledge search for '{query}' completed."
                    })
                
                else:
                    # Handle any other unknown tools
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Tool {tool_call.tool_name} executed."
                    })
            
            # Check if conversation should continue
            if self._is_conversation_complete(assistant_response):
                break
            
            # Generate follow-up user response
            if turn < max_turns - 1 and not self._should_end_conversation(conversation_history):
                # Create context for follow-up
                follow_up_context = "The assistant "
                if tool_calls and tool_calls[-1].tool_name == "product_search":
                    follow_up_context += "just searched for products. "
                elif tool_calls and tool_calls[-1].tool_name == "display_product":
                    follow_up_context += "just showed you a product. "
                else:
                    follow_up_context += "is helping you find products. "
                
                follow_up_context += f"Their last message was: '{assistant_response.message}'"
                
                follow_up = await self.user_simulator.generate_voice_query(
                    scenario=follow_up_context,
                    context=conversation_history,
                    previous_products=products_shown
                )
                
                conversation_history.append({
                    "role": "user",
                    "content": follow_up
                })
        
        return {
            "conversation": conversation_history,
            "search_queries": search_queries,
            "tool_calls": tool_calls,
            "products_shown": products_shown
        }
    
    def _should_end_conversation(self, conversation_history: List[Dict]) -> bool:
        """Check if the conversation has enough product interactions."""
        # Count product-related interactions by checking assistant messages with tool calls
        product_searches = 0
        product_displays = 0
        
        for msg in conversation_history:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if tc["function"]["name"] == "product_search":
                        product_searches += 1
                    elif tc["function"]["name"] == "display_product":
                        product_displays += 1
        
        # End if we've done enough product discovery
        return product_searches >= 2 or product_displays >= 1
    
    async def compare_search_approaches(
        self, 
        search_query: Dict[str, Any],
        flip: bool=False
    ) -> ComparisonMetrics:
        """Compare enhanced vs baseline search for a query."""
        
        query = search_query["query"]
        context = search_query.get("context", [])
        
        # Enhanced search
        start_time = time.time()
        enhanced_results, enhanced_metrics = await SearchService.search_products(
            query=query,
            account=self.account,
            enable_enhancement=True,
            enable_filter_extraction=True,
            search_mode="hybrid",
            user_context={
                "conversation": context
            }
        )
        enhanced_time = time.time() - start_time
        
        # Baseline search
        start_time = time.time()
        baseline_results = await self.baseline_search.search(query)
        baseline_time = time.time() - start_time
        
        # Convert results
        enhanced_standardized = self._standardize_results(enhanced_results)
        
        # LLM evaluation
        if flip:
            tmp_results = enhanced_standardized
            enhanced_standardized = baseline_results
            baseline_results = tmp_results
            
            tmp_time = enhanced_time
            enhanced_time = baseline_time
            baseline_time = tmp_time

        # Calculate metrics
        overlap_ratio = self._calculate_overlap(enhanced_standardized, baseline_results)
        ranking_correlation = self._calculate_correlation(enhanced_standardized, baseline_results)
        
        llm_evaluation = await self._evaluate_results_quality(
            query=query,
            context=context,
            enhanced_results=enhanced_standardized,
            baseline_results=baseline_results
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
    
    async def _execute_enhanced_search(self, query: str) -> List[Product]:
        """Execute enhanced search for tool calls."""
        results, _ = await SearchService.search_products(
            query=query,
            account=self.account,
            top_k=10,
            enable_enhancement=True
        )
        all_products = await self.product_manager.get_products()
        products = [p for p in all_products if str(p.id) in [r.get('id', '') for r in results]]
        return products
    
    def _is_conversation_complete(self, response: AssistantResponse) -> bool:
        """Check if conversation has reached a natural end."""
        completion_indicators = [
            "is there anything else",
            "let me know if you need",
            "happy to help with anything else",
            "feel free to ask"
        ]
        
        message_lower = response.message.lower()
        return any(indicator in message_lower for indicator in completion_indicators)
    
    def _standardize_results(self, results: List[Dict]) -> List[SearchResult]:
        """Convert search results to standardized format."""
        standardized = []
        for r in results:
            metadata = r.get('metadata', {})
            standardized.append(SearchResult(
                product_id=metadata.get('product_id', r.get('id', '')),
                score=r.get('score', 0),
                name=metadata.get('name', 'Unknown'),
                price=metadata.get('price', 0),
                category=metadata.get('category', ''),
                relevance_explanation=r.get('relevance_explanation'),
                metadata=metadata
            ))
        return standardized
    
    def _calculate_overlap(self, results1: List[SearchResult], results2: List[SearchResult]) -> float:
        """Calculate overlap between result sets."""
        ids1 = {r.product_id for r in results1}
        ids2 = {r.product_id for r in results2}
        
        if not ids1 or not ids2:
            return 0.0
        
        intersection = ids1.intersection(ids2)
        union = ids1.union(ids2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_correlation(self, results1: List[SearchResult], results2: List[SearchResult]) -> float:
        """Calculate ranking correlation."""
        common_ids = {r.product_id for r in results1} & {r.product_id for r in results2}
        
        if len(common_ids) < 2:
            return 0.0
        
        rank1 = {r.product_id: i for i, r in enumerate(results1)}
        rank2 = {r.product_id: i for i, r in enumerate(results2)}
        
        ranks1 = [rank1[id] for id in common_ids]
        ranks2 = [rank2[id] for id in common_ids]
        
        return np.corrcoef(ranks1, ranks2)[0, 1]
    
    async def _evaluate_results_quality(
        self,
        query: str,
        context: List[Dict[str, str]],
        enhanced_results: List[SearchResult],
        baseline_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Use LLM to evaluate search quality."""
        
        system_prompt = """You are an expert at evaluating search results for voice commerce.

Evaluate both result sets considering:
1. Understanding of natural speech patterns and intent
2. Relevance to the conversational context
3. Quality of results for voice interaction
4. Handling of ambiguity in spoken queries

Provide scores (0-10) and detailed analysis."""
        
        # Format conversation
        conv_text = "Conversation:\n"
        for msg in context[-5:]:
            conv_text += f"{msg['role'].upper()}: {msg['content']}\n"
        
        # Format results
        def format_results(results: List[SearchResult], label: str) -> str:
            text = f"\n{label}:\n"
            for i, r in enumerate(results[:5], 1):
                text += f"{i}. {r.name} (${r.price}) - Score: {r.score:.3f}\n"
                if r.relevance_explanation:
                    text += f"   Relevance: {r.relevance_explanation}\n"
            return text
        
        user_prompt = f"""Voice Query: "{query}"

{conv_text}

{format_results(enhanced_results, "ENHANCED SEARCH RESULTS")}

{format_results(baseline_results, "BASELINE SEARCH RESULTS")}

Evaluate which approach better handles this voice commerce scenario."""
        
        response = await LLMFactory.chat_completion(
            task="search_evaluation",
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.3,
            model="openai/o3-mini"
        )
        
        return {
            'evaluation': response.get('content', ''),
            'timestamp': datetime.now().isoformat()
        }


class VoiceSearchTestRunner:
    """
    Runs voice-realistic search comparison tests.
    """
    
    def __init__(self, account: str):
        self.account = account
        self.comparator = VoiceSearchComparator(account)
        self.test_scenarios = []
        self.results = []
        
    async def setup(self):
        """Initialize test environment."""
        await self.comparator.initialize()
        self._create_voice_scenarios()
    
    def _create_voice_scenarios(self):
        """Create realistic voice search scenarios based on the brand."""
        brand_scenarios = self._get_brand_scenarios(self.account)
        self.test_scenarios = brand_scenarios
    
    def _get_brand_scenarios(self, account: str) -> List[Dict[str, str]]:
        """Generate brand-specific test scenarios."""
        brand_name = account.split('.')[0].lower()
        
        # Brand-specific scenario mappings
        if 'specialized' in brand_name or 'trek' in brand_name or 'giant' in brand_name:
            # Cycling/bike brands
            return [
                {
                    "id": "beginner_cyclist",
                    "description": "New cyclist looking for first bike",
                    "scenario": "Customer calling about getting into cycling, unsure about terminology and budget"
                },
                {
                    "id": "parent_shopping",
                    "description": "Parent shopping for child's bike",
                    "scenario": "Parent needs a bike for their 10-year-old daughter who's outgrown her current bike"
                },
                {
                    "id": "upgrade_search",
                    "description": "Experienced cyclist upgrading",
                    "scenario": "Cyclist with a 5-year-old bike looking to upgrade to something with modern features"
                },
                {
                    "id": "comparison_shopping",
                    "description": "Comparing specific models",
                    "scenario": f"Customer has researched online and wants to compare different {brand_name.title()} models"
                },
                {
                    "id": "budget_conscious",
                    "description": "Price-sensitive shopping",
                    "scenario": "Looking for best value bike under $1500 for casual weekend rides"
                }
            ]
        
        elif 'flexfits' in brand_name or 'flex' in brand_name:
            # Menstrual/feminine care products
            return [
                {
                    "id": "first_time_user",
                    "description": "First-time user exploring options",
                    "scenario": "Customer new to menstrual discs, wants to understand different options and what might work best"
                },
                {
                    "id": "off_topic_customer",
                    "description": "Customer veers off topic",
                    "scenario": "Customer persists in asking off topic questions and for products that the brand does not offer such as shoes, laptops, etc."
                },
                {
                    "id": "comfort_concerns",
                    "description": "Comfort and fit questions",
                    "scenario": "Customer concerned about comfort and proper fit, has had issues with other products"
                },
                {
                    "id": "active_lifestyle",
                    "description": "Active lifestyle compatibility",
                    "scenario": "Customer who exercises regularly and needs products that work during workouts and swimming"
                },
                {
                    "id": "sensitivity_issues",
                    "description": "Sensitive skin concerns",
                    "scenario": "Customer with sensitive skin looking for hypoallergenic, chemical-free options"
                },
                {
                    "id": "travel_convenience",
                    "description": "Travel and convenience needs",
                    "scenario": "Frequent traveler looking for convenient, eco-friendly period products for long trips"
                }
            ]
        
        elif 'nike' in brand_name or 'adidas' in brand_name or 'athletic' in brand_name:
            # Athletic/sportswear brands
            return [
                {
                    "id": "workout_gear",
                    "description": "Workout gear for beginners",
                    "scenario": "Customer starting a fitness routine and needs appropriate workout clothing"
                },
                {
                    "id": "sport_specific",
                    "description": "Sport-specific equipment",
                    "scenario": "Customer looking for gear specific to their sport or activity"
                },
                {
                    "id": "performance_upgrade",
                    "description": "Performance equipment upgrade",
                    "scenario": "Experienced athlete looking to upgrade their gear for better performance"
                },
                {
                    "id": "seasonal_needs",
                    "description": "Seasonal clothing needs",
                    "scenario": "Customer preparing for seasonal weather changes and needs appropriate gear"
                },
                {
                    "id": "size_fit_concerns",
                    "description": "Size and fit questions",
                    "scenario": "Customer unsure about sizing and fit, especially for performance gear"
                }
            ]
        
        elif 'beauty' in brand_name or 'cosmetic' in brand_name or 'skincare' in brand_name:
            # Beauty/cosmetics brands
            return [
                {
                    "id": "skin_type_match",
                    "description": "Finding products for skin type",
                    "scenario": "Customer with specific skin type looking for products that won't cause breakouts"
                },
                {
                    "id": "routine_building",
                    "description": "Building a skincare routine",
                    "scenario": "Customer new to skincare wanting to build an effective daily routine"
                },
                {
                    "id": "problem_solving",
                    "description": "Addressing specific skin concerns",
                    "scenario": "Customer dealing with specific skin issues and seeking targeted solutions"
                },
                {
                    "id": "ingredient_conscious",
                    "description": "Ingredient-conscious shopping",
                    "scenario": "Customer who wants clean, natural ingredients and avoids certain chemicals"
                },
                {
                    "id": "gift_shopping",
                    "description": "Gift shopping for others",
                    "scenario": "Customer shopping for skincare gifts but unsure what would work for the recipient"
                }
            ]
        
        else:
            # Generic e-commerce scenarios
            return [
                {
                    "id": "first_time_customer",
                    "description": "First-time customer exploration",
                    "scenario": "Customer new to the brand, exploring what products are available"
                },
                {
                    "id": "specific_need",
                    "description": "Specific product need",
                    "scenario": "Customer with a specific need looking for the right product solution"
                },
                {
                    "id": "quality_value",
                    "description": "Quality vs value comparison",
                    "scenario": "Customer comparing different products to find the best quality for their budget"
                },
                {
                    "id": "recommendation_seeking",
                    "description": "Seeking recommendations",
                    "scenario": "Customer looking for expert recommendations based on their specific situation"
                },
                {
                    "id": "problem_solving",
                    "description": "Problem-solving purchase",
                    "scenario": "Customer trying to solve a specific problem and needs the right product"
                }
            ]
    
    async def run_voice_tests(self) -> List[Dict[str, Any]]:
        """Run all voice test scenarios."""
        
        for scenario in self.test_scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running voice scenario: {scenario['id']}")
            logger.info(f"Description: {scenario['description']}")
            
            # Simulate conversation
            conversation_result = await self.comparator.simulate_voice_conversation(
                scenario_description=scenario['scenario'],
                max_turns=4
            )
            
            # Track all comparisons for this scenario
            scenario_comparisons = []
            
            # Compare search approaches for EACH product_search query
            logger.info(f"\n📊 Found {len(conversation_result['search_queries'])} product searches to compare")
            
            for i, search_query in enumerate(conversation_result['search_queries'], 1):
                logger.info(f"\n🔍 Comparing search #{i}: '{search_query['query']}'")
                
                # Run comparison for both search methods
                comparison = await self.comparator.compare_search_approaches(search_query)
                
                scenario_comparisons.append(comparison)
                
                # comparison_flipped = await self.comparator.compare_search_approaches(search_query, flip=True)
                # scenario_comparisons.append(comparison_flipped)
                
                # Log detailed comparison
                logger.info(f"✅ Enhanced: {len(comparison.enhanced_results)} results in {comparison.enhanced_time:.3f}s")
                logger.info(f"✅ Baseline: {len(comparison.baseline_results)} results in {comparison.baseline_time:.3f}s")
                logger.info(f"📈 Overlap: {comparison.overlap_ratio:.2%}, Correlation: {comparison.ranking_correlation:.3f}")
                
                # Log top 3 from each
                logger.info("Top 3 Enhanced:")
                for j, r in enumerate(comparison.enhanced_results[:3], 1):
                    logger.info(f"  {j}. {r.name} - ${r.price}")
                
                logger.info("Top 3 Baseline:")
                for j, r in enumerate(comparison.baseline_results[:3], 1):
                    logger.info(f"  {j}. {r.name} - ${r.price}")
            
            # Store complete results for this scenario
            self.results.append({
                'scenario': scenario,
                'conversation': conversation_result['conversation'],
                'tool_calls': conversation_result['tool_calls'],
                'products_shown': conversation_result.get('products_shown', []),
                'search_queries': conversation_result['search_queries'],
                'comparisons': scenario_comparisons,  # All comparisons for this scenario
                'summary': {
                    'total_searches': len(conversation_result['search_queries']),
                    'total_tool_calls': len(conversation_result['tool_calls']),
                    'avg_enhanced_time': np.mean([c.enhanced_time for c in scenario_comparisons]) if scenario_comparisons else 0,
                    'avg_baseline_time': np.mean([c.baseline_time for c in scenario_comparisons]) if scenario_comparisons else 0,
                    'avg_overlap': np.mean([c.overlap_ratio for c in scenario_comparisons]) if scenario_comparisons else 0
                }
            })
        
        return self.results
    
    async def generate_voice_report(self) -> str:
        """Generate report focused on voice interaction quality."""
        
        report = f"""# Voice Assistant Search Comparison Report

**Account**: {self.account}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Scenarios**: {len(self.test_scenarios)}

## Executive Summary

This report compares search performance in realistic voice assistant conversations,
focusing on natural language understanding and conversational context handling.

"""
        
        # Aggregate metrics by conversation style
        for result in self.results:
            scenario = result['scenario']
            
            # Handle multiple comparisons per scenario
            for i, comparison in enumerate(result['comparisons']):
                search_query = result['search_queries'][i] if i < len(result['search_queries']) else result['search_queries'][0]
                
                report += f"""
## Scenario: {scenario['description']} (Search #{i+1})

**Voice Query**: "{search_query['query']}"

### Conversation Context:
"""
                # Show last few turns
                for msg in result['conversation'][-4:]:
                    report += f"- **{msg['role'].upper()}**: {msg['content']}\n"
                
                report += f"""
### Search Performance:
- **Enhanced Search**: {len(comparison.enhanced_results)} results in {comparison.enhanced_time:.3f}s
- **Baseline Search**: {len(comparison.baseline_results)} results in {comparison.baseline_time:.3f}s
- **Result Overlap**: {comparison.overlap_ratio:.2%}

### Top Results Comparison:
"""
                # Show top 3 from each
                report += "**Enhanced (with context understanding):**\n"
                for j, r in enumerate(comparison.enhanced_results[:3], 1):
                    report += f"{j}. {r.name} - ${r.price}\n"
                    if r.relevance_explanation:
                        report += f"   *{r.relevance_explanation}*\n"
                
                report += "\n**Baseline (simple text matching):**\n"
                for j, r in enumerate(comparison.baseline_results[:3], 1):
                    report += f"{j}. {r.name} - ${r.price}\n"
                
                report += f"\n### Voice Interaction Analysis:\n{comparison.llm_evaluation['evaluation']}\n"
                report += "\n" + "-"*60 + "\n"
        
        return report
    
    async def save_results(self, output_dir: str = "voice_search_results"):
        """Save voice test results."""
        output_dir = f"{output_dir}/{self.account}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        results_data = []
        for r in self.results:
            # Handle multiple comparisons per scenario
            for i, comparison in enumerate(r['comparisons']):
                # Get the corresponding search query
                search_query = r['search_queries'][i] if i < len(r['search_queries']) else r['search_queries'][0]
                
                results_data.append({
                    'scenario_id': r['scenario']['id'],
                    'conversation': r['conversation'],
                    'search_query': search_query['query'],
                    'turn_number': search_query['turn'],
                    'comparison': {
                        'enhanced_time': comparison.enhanced_time,
                        'baseline_time': comparison.baseline_time,
                        'overlap_ratio': comparison.overlap_ratio,
                        'enhanced_results': [asdict(res) for res in comparison.enhanced_results[:5]],
                        'baseline_results': [asdict(res) for res in comparison.baseline_results[:5]],
                        'evaluation': comparison.llm_evaluation
                    }
                })
        
        with open(f"{output_dir}/voice_comparison_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save report
        report = await self.generate_voice_report()
        with open(f"{output_dir}/voice_comparison_report.md", 'w') as f:
            f.write(report)
        
        logger.info(f"Voice test results saved to {output_dir}/")


async def main():
    """Run voice-realistic search comparison tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice-realistic search comparison")
    parser.add_argument("account", nargs='?', default="flexfits.com", help="Brand account (default: flexfits.com)")
    parser.add_argument("--setup-baseline", action="store_true", help="Setup baseline index")
    
    args = parser.parse_args()
    
    try:
        # Setup baseline if needed
        if args.setup_baseline:
            logger.info("Setting up baseline index...")
            
            # Load products directly from storage
            storage_manager = get_account_storage_provider()
            products_data = await storage_manager.get_product_catalog(args.account)
            products = [Product.from_dict(product=item) for item in products_data[:1000]]
            logger.info(f"Loaded {len(products)} products")
            
            baseline = BaselineSearchIndex(args.account)
            await baseline.create_index()
            await baseline.ingest_products(products)
            logger.info("Baseline index ready!")
        
        # Run voice tests
        logger.info("\nStarting voice-realistic search comparison...")
        runner = VoiceSearchTestRunner(args.account)
        await runner.setup()
        
        results = await runner.run_voice_tests()
        await runner.save_results()
        
        # Print summary
        print("\n" + "="*80)
        print("VOICE SEARCH COMPARISON COMPLETE")
        print("="*80)
        print(f"Tested {len(results)} search queries across {len(runner.test_scenarios)} scenarios")
        print(f"Results saved to voice_search_results/")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())