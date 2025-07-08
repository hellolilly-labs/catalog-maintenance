"""
Voice Search Wrapper

Wraps the unified SearchService to add voice-specific features like:
- Voice-optimized query enhancement
- Speech timing and pacing
- Conversation context awareness
- Real-time streaming considerations
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from livekit.agents import llm

from liddy.search.service import SearchService as UnifiedSearchService
from liddy.models.product import Product
from liddy_voice.llm_service import LlmService
from liddy.model import UserState, BasicChatMessage
import numpy as np

logger = logging.getLogger(__name__)


class VoiceSearchWrapper:
    """
    Voice-specific wrapper around the unified SearchService.
    
    Adds voice interaction features while delegating core search to the unified service.
    """
    
    @staticmethod
    async def search_products(
        query: str,
        user_state: UserState,
        chat_ctx: Optional[llm.ChatContext] = None,
        enhance_for_voice: bool = True,
        limit: int = 7,
        top_k: int = 50,
        rerank: bool = True,
        enable_confidence_analysis: bool = True,
        **kwargs
    ) -> Union[List[Product], Dict[str, Any]]:
        """
        Voice-optimized product search using the unified SearchService.
        
        Args:
            query: Search query
            user_state: Current user state with account info
            chat_ctx: Conversation context for enhancement
            enhance_for_voice: Whether to apply voice-specific query enhancement
            limit: Maximum number of results
            rerank: Whether to rerank results
            enable_confidence_analysis: Whether to return confidence analysis
            **kwargs: Additional parameters for SearchService
            
        Returns:
            List of Product objects or Dict with confidence analysis
        """
        # Apply voice-specific query enhancement if requested
        enhanced_query = query
        if enhance_for_voice and chat_ctx:
            enhanced_query = await VoiceSearchWrapper.enhance_query_for_voice(
                query, user_state, chat_ctx
            )
        
        # Use the unified SearchService
        try:
            # Get results with metrics
            from liddy.search.service import SearchService as UnifiedSearchServiceClass
            # if account not in kwargs, t
            if "account" not in kwargs:
                kwargs["account"] = user_state.account
            
            results, metrics = await UnifiedSearchServiceClass.search_products(
                # account=user_state.account,
                query=enhanced_query,
                top_k=top_k,
                top_n=limit,  # Note: SearchService uses top_k, not limit
                enable_enhancement=False,  # We already enhanced above
                user_state=user_state,
                chat_ctx=[BasicChatMessage(role=msg.role, content=msg.content, timestamp=msg.created_at) for msg in chat_ctx.items if msg.role] if chat_ctx else None,
                rerank=rerank,
                **kwargs
            )
            
            # Apply voice-specific post-processing
            # voice_optimized = VoiceSearchWrapper._optimize_for_voice_response(results)
            voice_optimized = results
            
            # If confidence analysis requested, perform it
            if enable_confidence_analysis:
                # Extract scores from results
                scores = []
                for r in results:
                    if hasattr(r, 'score'):
                        scores.append(r.score)
                    elif isinstance(r, dict) and 'score' in r:
                        scores.append(r['score'])
                    else:
                        scores.append(0.5)  # Default score if not available
                
                # Analyze confidence
                confidence_analysis = VoiceSearchWrapper._analyze_search_confidence(voice_optimized, scores[:len(voice_optimized)])
                
                # Apply strategy
                strategy_result = VoiceSearchWrapper._apply_confidence_strategy(voice_optimized, confidence_analysis)
                
                return {
                    "results": strategy_result["results"],
                    "agent_instructions": strategy_result["agent_instructions"],
                    "confidence_metadata": strategy_result["confidence_metadata"],
                    "original_count": len(results),
                    "query": query,
                    "enhanced_query": enhanced_query,
                    "search_metrics": {
                        "search_time": metrics.search_time,
                        "enhancements_used": metrics.enhancements_used,
                        "backend": metrics.search_backend
                    }
                }
            else:
                # Return voice-optimized results for backward compatibility
                return voice_optimized
            
        except Exception as e:
            logger.error(f"Error in voice product search: {e}")
            if enable_confidence_analysis:
                return {
                    "results": [],
                    "agent_instructions": "I encountered an error while searching. Please try rephrasing your request.",
                    "confidence_metadata": {"confidence": "error", "pattern": "error"},
                    "error": str(e)
                }
            else:
                return []
    
    @staticmethod
    async def enhance_query_for_voice(
        query: str,
        user_state: UserState,
        chat_ctx: llm.ChatContext,
        model_name: str = "gpt-4.1-mini"
    ) -> str:
        """
        Enhance query specifically for voice interactions.
        
        Voice-specific enhancements include:
        - Understanding conversational references ("that one", "the blue one")
        - Handling speech disfluencies and corrections
        - Incorporating recent conversation context more heavily
        - Simplifying complex spoken queries
        
        Args:
            query: Original spoken query
            user_state: User state with account info
            chat_ctx: Conversation context
            model_name: LLM model to use
            
        Returns:
            Enhanced query optimized for voice search
        """
        if not query:
            return query
            
        try:
            # Create enhancement context
            enhancement_ctx = llm.ChatContext([])
            
            # Voice-specific system prompt
            system_prompt = """You are a voice search query enhancer. Your job is to transform spoken queries into effective search queries.

Focus on:
1. Resolving conversational references (e.g., "that one" → specific product mentioned earlier)
2. Removing speech disfluencies (um, uh, you know)
3. Correcting common speech recognition errors
4. Expanding abbreviated or casual spoken terms
5. Including context from the recent conversation

Return ONLY the enhanced search query with no explanation."""
            
            enhancement_ctx.add_message(role="system", content=[system_prompt])
            
            # Build context from recent conversation (voice needs more context)
            enhancement_prompt = f"Original query: \"{query}\"\n\nRecent conversation:\n"
            
            # Get more context for voice (last 10 turns instead of 5)
            recent_messages = chat_ctx.copy().items[-20:] if len(chat_ctx.copy().items) > 20 else chat_ctx.copy().items
            
            for msg in recent_messages:
                if hasattr(msg, "role") and msg.role != "system":
                    content = msg.content
                    if isinstance(content, list):
                        content = " ".join(str(c) for c in content)
                    enhancement_prompt += f"{msg.role}: {content}\n"
            
            enhancement_prompt += "\nEnhanced search query:"
            enhancement_ctx.add_message(role="user", content=[enhancement_prompt])
            
            # Get enhancement from LLM
            llm_model = LlmService.fetch_model_service_from_model(
                model_name=model_name,
                account=user_state.account,
                user=user_state.user_id,
                model_use="voice_query_enhancement"
            )
            
            enhanced = await LlmService.chat_wrapper(
                llm_service=llm_model,
                chat_ctx=enhancement_ctx
            )
            
            logger.info(f"Voice query enhancement: '{query}' → '{enhanced}'")
            return enhanced if enhanced else query
            
        except Exception as e:
            logger.error(f"Error in voice query enhancement: {e}")
            return query
    
    @staticmethod
    def _analyze_search_confidence(results: List[Product], scores: List[float]) -> Dict[str, Any]:
        """
        Analyze search result confidence based on score distribution.
        
        Args:
            results: List of products from search
            scores: List of corresponding scores
            
        Returns:
            Dict with confidence level and analysis
        """
        if not scores:
            return {
                "confidence": "low",
                "score_gap": 0,
                "top_score": 0,
                "mean_score": 0,
                "std_dev": 0,
                "pattern": "no_results"
            }
        
        scores_array = np.array(scores)
        
        # Calculate metrics
        top_score = scores_array[0] if len(scores_array) > 0 else 0
        score_gap = scores_array[0] - scores_array[1] if len(scores_array) > 1 else 0
        mean_score = np.mean(scores_array)
        std_dev = np.std(scores_array)
        
        # Determine pattern
        pattern = "unknown"
        if len(scores) == 0:
            pattern = "no_results"
        elif score_gap > 0.3 and top_score > 0.8:
            pattern = "clear_winner"
        elif std_dev < 0.1 and mean_score > 0.6:
            pattern = "multiple_good_matches"
        elif top_score < 0.5:
            pattern = "poor_matches"
        elif score_gap < 0.1 and len(scores) > 3:
            pattern = "ambiguous"
        else:
            pattern = "moderate_matches"
        
        # Determine confidence level
        if pattern == "clear_winner":
            confidence = "high"
        elif pattern in ["multiple_good_matches", "moderate_matches"]:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "confidence": confidence,
            "score_gap": score_gap,
            "top_score": top_score,
            "mean_score": mean_score,
            "std_dev": std_dev,
            "pattern": pattern
        }
    
    @staticmethod
    def _apply_confidence_strategy(results: List[Product], confidence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply adaptive strategy based on confidence analysis.
        
        Uses mathematical thresholds to determine which results are "good enough" to return,
        rather than arbitrary cutoffs.
        
        Args:
            results: Search results
            confidence_analysis: Analysis from _analyze_search_confidence
            
        Returns:
            Dict with filtered results and agent instructions
        """
        pattern = confidence_analysis["pattern"]
        
        # Get scores for mathematical filtering
        scores = []
        for r in results:
            if hasattr(r, 'score'):
                scores.append(r.score)
            elif isinstance(r, dict) and 'score' in r:
                scores.append(r['score'])
            else:
                scores.append(0.0)
        
        # Determine cutoff thresholds based on score distribution
        if len(scores) > 0:
            top_score = confidence_analysis["top_score"]
            mean_score = confidence_analysis["mean_score"]
            std_dev = confidence_analysis["std_dev"]
            
            # Calculate dynamic thresholds
            if pattern == "clear_winner":
                # If there's a clear winner, include only results within 10% of top score
                threshold = top_score * 0.9
                filtered_results = [r for r, s in zip(results, scores) if s >= threshold]
                agent_instructions = "Present this product confidently as the best match."
                if len(filtered_results) > 1:
                    agent_instructions = f"Found {len(filtered_results)} excellent matches. Present the top option but mention the alternatives."
            
            elif pattern == "multiple_good_matches":
                # Include all results above mean score (they're all good)
                threshold = mean_score
                filtered_results = [r for r, s in zip(results, scores) if s >= threshold]
                if len(filtered_results) <= 1:
                    threshold = min(0.9, mean_score)
                    filtered_results = [r for r, s in zip(results, scores) if s >= threshold]

                if len(filtered_results) <= 1:
                    filtered_results = results[:3]
                agent_instructions = f"Found {len(filtered_results)} good matches. Present these options and help the user choose between them by asking about their specific needs."
            
            elif pattern == "ambiguous":
                # Include results within 1 standard deviation of mean
                threshold = max(mean_score - std_dev * 2, 0.3)  # At least 0.3 score
                filtered_results = [r for r, s in zip(results, scores) if s >= threshold]
                # But cap at 10 results for practicality
                if len(filtered_results) > 10:
                    filtered_results = filtered_results[:10]
                agent_instructions = f"The search found {len(filtered_results)} possible matches with similar relevance. Ask clarifying questions about specific features or aspects to help narrow down the options."
            
            elif pattern == "poor_matches":
                # Include only results above a minimal threshold
                threshold = 0.25
                filtered_results = [r for r, s in zip(results, scores) if s >= threshold]
                # If nothing above threshold, take top 3
                if not filtered_results and results:
                    filtered_results = results[:3]
                agent_instructions = "These results have low relevance scores. Ask the user to provide more specific details or suggest alternative search terms."
            
            else:  # moderate_matches or unknown
                # Include results above 80% of mean score
                threshold = mean_score * 0.8
                filtered_results = [r for r, s in zip(results, scores) if s >= threshold]
                # Ensure at least 3 results if available
                if len(filtered_results) < 3 and len(results) >= 3:
                    filtered_results = results[:3]
                agent_instructions = f"Found {len(filtered_results)} relevant products. Present these options and be ready to provide more details if needed."
        
        else:
            # No scores available - fall back to pattern-based approach
            filtered_results = []
            agent_instructions = "No products were found. Suggest the user try different search terms or ask about their needs to help formulate a better search."
        
        # Add specific guidance based on score distribution
        if confidence_analysis["top_score"] > 0.9 and len(filtered_results) == 1:
            agent_instructions += " This product has an exceptionally high relevance score (90%+ match)."
        elif confidence_analysis["mean_score"] < 0.4:
            agent_instructions += " Consider asking if the user would like to broaden their search or try different terms."
        elif len(filtered_results) > 5:
            agent_instructions += f" Focus on the top {min(3, len(filtered_results))} options unless the user wants to see more."
        
        return {
            "results": filtered_results,
            "agent_instructions": agent_instructions,
            "confidence_metadata": confidence_analysis,
            "filtering_threshold": threshold if 'threshold' in locals() else None
        }
    
    @staticmethod
    def _optimize_for_voice_response(results: List[Product]) -> List[Product]:
        """
        Optimize search results for voice presentation.
        
        - Prioritize products with good voice descriptions
        - Limit results to avoid overwhelming spoken responses
        - Sort by relevance for voice (considering voice_summary quality)
        
        Args:
            results: Raw search results
            
        Returns:
            Optimized results for voice
        """
        # Filter out products without good voice summaries
        voice_ready = [
            p for p in results 
            if hasattr(p, 'voice_summary') and p.voice_summary and len(p.voice_summary) > 20
        ]
        
        # If we filtered out too many, include some without voice summaries
        if len(voice_ready) < 3 and len(results) > len(voice_ready):
            voice_ready.extend([p for p in results if p not in voice_ready])
        
        # Limit to reasonable number for voice (5-7 items max)
        return voice_ready[:7]
    
    @staticmethod
    async def search_with_speech_timing(
        query: str,
        user_state: UserState,
        chat_ctx: Optional[llm.ChatContext] = None,
        speak_while_searching: bool = True,
        **search_kwargs
    ) -> Dict[str, Any]:
        """
        Perform search with speech timing considerations.
        
        Returns both results and timing information for optimal voice UX.
        
        Args:
            query: Search query
            user_state: User state
            chat_ctx: Conversation context
            speak_while_searching: Whether to return filler speech
            **search_kwargs: Additional search parameters
            
        Returns:
            Dict with 'results', 'search_time', and optional 'filler_speech'
        """
        import time
        start_time = time.time()
        
        # Prepare filler speech if needed
        filler_speech = None
        if speak_while_searching:
            filler_speech = VoiceSearchWrapper._generate_search_filler(query)
        
        # Perform the search
        results = await VoiceSearchWrapper.search_products(
            query=query,
            user_state=user_state,
            chat_ctx=chat_ctx,
            **search_kwargs
        )
        
        search_time = time.time() - start_time
        
        return {
            "results": results,
            "search_time": search_time,
            "filler_speech": filler_speech,
            "result_count": len(results)
        }
    
    @staticmethod
    def _generate_search_filler(query: str) -> str:
        """Generate natural filler speech while searching."""
        fillers = [
            f"Let me search for {query}...",
            f"Looking for {query} in our catalog...",
            f"Searching for the best matches...",
            f"Let me find that for you...",
            "One moment while I search..."
        ]
        
        # Simple hash to consistently pick a filler for similar queries
        import hashlib
        hash_val = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        return fillers[hash_val % len(fillers)]


# Convenience class to maintain backward compatibility
class VoiceSearchService:
    """
    Backward-compatible interface that mimics the old SearchService.
    
    This allows existing code to work while we migrate to the wrapper pattern.
    """
    
    @staticmethod
    async def search_products_rag(query: str, user_state: UserState, chat_ctx: Optional[llm.ChatContext] = None, **kwargs) -> List[Product]:
        """Backward compatible product search."""
        return await VoiceSearchWrapper.search_products(
            query=query,
            user_state=user_state,
            chat_ctx=chat_ctx,
            **kwargs
        )
    
    @staticmethod
    async def enhance_query(query: str, user_state: UserState, chat_ctx: llm.ChatContext, **kwargs) -> str:
        """Backward compatible query enhancement."""
        return await VoiceSearchWrapper.enhance_query_for_voice(
            query=query,
            user_state=user_state,
            chat_ctx=chat_ctx
        )
    
    @staticmethod
    async def perform_search(
        query: str,
        search_function: Callable,
        query_enhancer: Optional[Callable] = None,
        search_params: Dict[str, Any] = None,
        user_state: UserState = None,
        chat_ctx: Optional[llm.ChatContext] = None,
        enable_confidence_analysis: bool = True,
    ) -> Union[List[Product], Dict[str, Any]]:
        """Perform search with optional confidence analysis.
        
        Args:
            query: Search query
            search_function: Function to perform the search
            query_enhancer: Optional function to enhance the query
            search_params: Parameters for search function
            user_state: User state
            chat_ctx: Chat context
            enable_confidence_analysis: Whether to perform confidence analysis
            
        Returns:
            Either list of products (backward compatible) or dict with confidence analysis
        """
        search_params = search_params or {}
        
        enhanced_query = query
        if query_enhancer and user_state:
            try:
                enhanced_query = await query_enhancer(query, user_state, **search_params.get("enhancer_params", {}))
            except Exception as e:
                logger.error(f"Error enhancing query: {e}")
        
        try:
            # Call the search function
            results = await search_function(enhanced_query, **search_params, user_state=user_state, chat_ctx=chat_ctx)
            
            # If confidence analysis is disabled, return results as-is (backward compatible)
            if not enable_confidence_analysis:
                return results
            
            # Extract scores if available
            scores = []
            if isinstance(results, list) and results and hasattr(results[0], 'score'):
                scores = [getattr(r, 'score', 0.0) for r in results]
            elif isinstance(results, list) and results and isinstance(results[0], dict) and 'score' in results[0]:
                scores = [r.get('score', 0.0) for r in results]
            
            # If no scores available, return results as-is
            if not scores:
                logger.debug("No scores available for confidence analysis")
                return results
            
            # Perform confidence analysis
            confidence_analysis = VoiceSearchWrapper._analyze_search_confidence(results, scores)
            
            # Apply confidence strategy
            strategy_result = VoiceSearchWrapper._apply_confidence_strategy(results, confidence_analysis)
            
            logger.info(
                f"Search confidence analysis: pattern={confidence_analysis['pattern']}, "
                f"confidence={confidence_analysis['confidence']}, "
                f"returning {len(strategy_result['results'])} of {len(results)} results"
            )
            
            # Return enhanced result with confidence metadata
            return {
                "results": strategy_result["results"],
                "agent_instructions": strategy_result["agent_instructions"],
                "confidence_metadata": strategy_result["confidence_metadata"],
                "original_count": len(results),
                "query": query,
                "enhanced_query": enhanced_query
            }
            
        except Exception as e:
            logger.error(f"Error in search function: {e}")
            if enable_confidence_analysis:
                return {
                    "results": [],
                    "agent_instructions": "Search encountered an error. Please try rephrasing your search or contact support.",
                    "confidence_metadata": {"confidence": "error", "pattern": "error"},
                    "error": str(e)
                }
            else:
                return []