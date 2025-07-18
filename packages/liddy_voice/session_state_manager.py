import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# from livekit.agents import llm, utils

from .markdown_utils import obj_to_markdown
from liddy_voice.user_manager import UserManager
from liddy.models.product import Product
from liddy.model import BasicChatMessage, UrlTracking, UserState, ConversationExitState, ConversationResumptionState
from liddy.models.product_manager import ProductManager, get_product_manager


logger = logging.getLogger("session-state-manager")

# pick up from the environment variable
CONVERSATION_RESUMPTION_TTL = int(os.getenv("CONVERSATION_RESUMPTION_TTL", 60 * 60 * 24 * 10))  # 10 days



class SessionStateManager:
    """
    Manages both persistent and ephemeral state across sessions.
    
    This class handles:
    1. Persistent user data in Redis
    2. Ephemeral conversation analysis and context
    3. Real-time information extraction
    """
    
    def __init__(
        self,
        # llm_model: any
    ):
        # self.llm_model = llm_model
        pass

    
    # In-memory cache for fast access to ephemeral state
    _ephemeral_states = {}
    
    @classmethod
    def get_user_recent_history(cls, user_id: str, limit: int = 10) -> List[UrlTracking]:
        """Get user history from Redis"""
        return UserManager.get_user_recent_history(user_id, limit)

    @classmethod
    async def get_user_recent_products(cls, user_id: str, product_manager:ProductManager, limit: int=15) -> List[Tuple[float, Product]]:
        """Get user product history from Redis"""
        if not product_manager:
            raise ValueError("Product manager is required")
        
        history = UserManager.get_user_recent_history(user_id)
        
        if not history or len(history) == 0:
            return []
        product_identifiers: List[str] = ["product_id", "productId", "product_identifier", "productIdentifier", "id"]
        products: List[Product] = []
        for urlTracking in history:
            if urlTracking.url:
                product = None
                product_details = urlTracking.product_details
                # see if we have a product id in the urlTracking
                if product_details:
                    product_id = None
                    for identifier in product_identifiers:
                        product_id = product_details.get(identifier)
                        if product_id:
                            break
                    
                    if product_id:
                        product = await product_manager.find_product_by_id(product_id)
                        if product and isinstance(product, Product):
                            products.append((urlTracking.timestamp, product))
                            # exit the loop if we have enough products
                            if len(products) >= limit:
                                break
                        
        # sort the products by timestamp descending
        products.sort(key=lambda x: x[0], reverse=True)
        # limit the number of products to the specified limit
        if len(products) > limit:
            products = products[:limit]
        # return the products
        return products
                
    @classmethod
    def update_search_context(cls, user_id: str, search_context: str) -> bool:
        """Update search context in user state"""
        # return cls.update_user_state(user_id, {
        #     "search_context": search_context
        # })
        pass
    
    # @classmethod
    # def update_current_page(cls, user_id: str, page_info: Dict[str, Any] | str) -> bool:
    #     """
    #     Update current page in user state
        
    #     Expected page_info structure:
    #     {
    #         "url": "https://www.example.com/product/123",
    #         "title": "Product Name",
    #         "product_id": "123",
    #         "category": "Category",
    #         "description": "Short description",
    #         "timestamp": 1234567890,
    #         "additional_info": {
    #             "key1": "value1",
    #             "key2": "value2"
    #         }
    #     }
    #     """
    #     return cls.update_user_state(user_id, {
    #         "current_page": page_info
    #     })
    
    @classmethod
    def get_ephemeral_state(cls, user_id: str) -> Dict[str, Any]:
        """Get ephemeral (in-memory) state for analysis and context"""
        if user_id not in cls._ephemeral_states:
            cls._ephemeral_states[user_id] = {
                "preferences": {},
                "mentioned_categories": set(),
                "mentioned_features": set(),
                "price_info": None,
                "last_analyzed_at": 0,
            }
        return cls._ephemeral_states[user_id]
    
    @classmethod
    def store_conversation_exit_reason(cls, user_state: UserState, reason: str):
        """
        Store conversation exit reason in user state.
        
        Args:
            user_id: The user identifier
            reason: The reason for exit (e.g., "user disconnected")
        
        Returns:
            bool: Success status
        """
        try:
            # Get current state and conversation history
            if not user_state:
                return False
            conversation_exit_state = user_state.conversation_exit_state
            if not conversation_exit_state:
                conversation_exit_state = ConversationExitState(
                    last_interaction_time=time.time(),
                    exit_reason=reason,
                )
                user_state.conversation_exit_state = conversation_exit_state
            else:
                conversation_exit_state.exit_reason = reason
                conversation_exit_state.last_interaction_time = time.time()
            
            return UserManager.save_user_state(user_state)
        except Exception as e:
            logger.error(f"Error storing conversation exit reason: {e}")
            return False
    
    @classmethod
    def get_conversation_resumption(cls, user_state: UserState) -> Optional[ConversationResumptionState]:
        """
        Get conversation resumption data if it exists and hasn't expired.
        
        Args:
            user_id: The user identifier
        
        Returns:
            Dict or None: Resumption data if available, None otherwise
        """
        try:
            if not user_state:
                return None
            # resumption_data = user_state.get('conversation_exit_state')
            # if not resumption_data:
            #     return None

            # # Check if the resumption data has expired
            # last_interaction_time = resumption_data.get('last_interaction_time')
            # if not last_interaction_time:
            #     return None
            # elif (datetime.now() - datetime.fromisoformat(last_interaction_time)).total_seconds() > CONVERSATION_RESUMPTION_TTL:
            #     return None
            

            transcript: Optional[List[BasicChatMessage]] = None
            conversation_exit_state = user_state.conversation_exit_state
            last_interaction_time = None
            resumption_message = None
            transcript_summary = None
            if conversation_exit_state:
                last_interaction_time = conversation_exit_state.last_interaction_time
                resumption_message = conversation_exit_state.resumption_message
                if last_interaction_time:
                    if isinstance(last_interaction_time, str):
                        last_interaction_time = datetime.fromisoformat(last_interaction_time).timestamp()
                    if isinstance(last_interaction_time, int):
                        last_interaction_time = float(last_interaction_time)
                    if time.time() - last_interaction_time > CONVERSATION_RESUMPTION_TTL:
                        return None
                transcript_summary = conversation_exit_state.transcript_summary
                
            # last_conversation_messages = user_state.chat_messages if user_state.chat_messages else None
            last_conversation_messages = None
            is_resumable = False
            if last_conversation_messages:
                if isinstance(last_conversation_messages, List):
                    # if the transcript is a list, then convert it to a list of tuples
                    transcript = last_conversation_messages                    
                    is_resumable = len(transcript) > 0

            user_state_message = None
            if last_interaction_time:
                current_time = time.time()
                time_difference = round(current_time - last_interaction_time)
                # get a rough approximation of the time difference
                if time_difference > 31536000:
                    time_difference_approx = f"{time_difference / 31536000:.1f} years"
                elif time_difference > 2592000:
                    time_difference_approx = f"{time_difference / 2592000:.1f} months"
                elif time_difference > 604800:
                    time_difference_approx = f"{time_difference / 604800:.1f} weeks"
                elif time_difference > 86400:
                    time_difference_approx = f"{time_difference / 86400:.1f} days"
                elif time_difference > 3600:
                    time_difference_approx = f"{time_difference / 3600:.1f} hours"
                elif time_difference > 60:
                    time_difference_approx = f"{time_difference / 60:.0f} minutes"
                else:
                    time_difference_approx = "less than a minute"

                # if last conversation was less than CONVERSATION_RESUMPTION_TTL seconds ago, then we can resume
                if time_difference < CONVERSATION_RESUMPTION_TTL:
                    is_resumable = True 
                    # this should a continuation of the previous conversation
                    user_state_message = f"Conversation Resumption:\nThis is a continuation of the previous conversation. The last interaction with the user was roughly {time_difference_approx} ago."
                    if transcript_summary:
                        user_state_message += f"\nHere is a summary of that conversation:\n{transcript_summary}"
                else:
                    is_resumable = False
                    # this should be a new conversation
                    user_state_message = f"Last Conversation:\nThe last interaction with the user was roughly {time_difference_approx} ago."
                    if transcript_summary:
                        user_state_message += f"\nHere is a summary of that conversation:\n{transcript_summary}"

            result = ConversationResumptionState(
                user_id=user_state.user_id,
                last_interaction_time=last_interaction_time,
                is_resumable=is_resumable,
                resumption_message=resumption_message,
                chat_messages=transcript,
                user_state_message=user_state_message,
            )

            return result
            
        except Exception as e:
            logger.error(f"Error getting conversation resumption data: {e}")
            return None

    @classmethod
    def clear_conversation_resumption(cls, user_id: str) -> bool:
        """Clear conversation resumption data after it's been used"""
        try:
            user_state = UserManager.get_user_state(user_id)
            if user_state:
                user_state.conversation_exit_state = None
                UserManager.save_user_state(user_state)
            return True
            
        except Exception as e:
            logger.error(f"Error clearing conversation resumption: {e}")
            return False

    @classmethod
    async def build_user_state_message(cls, user_state:UserState, product_manager:ProductManager, include_communication_directive: bool=False, include_resumption:bool=True, include_product_history: bool=True, include_browsing_history_depth: int=5) -> str:
        """
        Build a user state message for the agent.
        
        Args:
            user_id: The user identifier
        
        Returns:
            str: The user state message
        """
        if not user_state:
            raise ValueError("User state is required")
        
        if not product_manager:
            raise ValueError("Product manager is required")
        
        # the user_state should already have the current_url and current_title and current_product
        current_url = user_state.current_url
        current_url_title = user_state.current_url_title
        current_url_timestamp = user_state.current_url_timestamp
        current_product = user_state.current_product
        current_product_id = user_state.current_product_id
        current_product_timestamp = user_state.current_product_timestamp
        recent_product_ids = user_state.recent_product_ids or []
        
        # Only fetch history if include_browsing_history_depth > 1,
        # or if current_url from user_state is empty (None or "")
        if (include_browsing_history_depth > 1) or (not user_state.current_url):
            limit = 10 if include_browsing_history_depth == -1 else max(1, include_browsing_history_depth + 1)
            history = cls.get_user_recent_history(user_id=user_state.user_id, limit=limit)
            if history and len(history) > 0:
                most_recent_history = history[0]
                if most_recent_history and not current_url:
                    current_url = most_recent_history.url
                    current_url_title = most_recent_history.title if most_recent_history.title else current_url
                    current_url_timestamp = most_recent_history.timestamp
                    current_product = await product_manager.find_product_from_url_smart(url=current_url, fallback_to_url_lookup=False)
                    if current_product and isinstance(current_product, Product):
                        current_product_id = current_product.id
                        current_product_timestamp = time.time()
                        user_state.current_product = current_product
                        user_state.current_product_id = current_product_id
                        user_state.current_product_timestamp = current_product_timestamp
                    else:
                        current_product = None
                        current_product_id = None
                        current_product_timestamp = None
                        user_state.current_product = None
                        user_state.current_product_id = None
                        user_state.current_product_timestamp = None
        else:
            history = None
            if current_url:
                history = [UrlTracking(url=current_url, title=current_url_title, timestamp=current_product_timestamp)]

        # Add user state to context if available
        user_state_message = ""
        current_page_message = None
        
        if include_browsing_history_depth >= 1:
            if current_url:
                current_page_message = f"# Current Page\n\nThe user is currently looking at the following page:\n\n[{current_url_title or current_url}]({current_url})\n\n"
                if current_product and isinstance(current_product, Product):
                    current_page_message += f"## Current Product\n\n"
                    current_page_message += f"*Note: The user is viewing this product - you already have its details below*\n\n"
                    current_page_message += f"{Product.to_markdown(product=current_product, depth=2, obfuscatePricing=True)}\n"
        
        if history and len(history) > 1 and include_browsing_history_depth > 1:
            # Determine which items to include
            # Skip the first item (current page) if we're already showing it
            history_items = history[1:include_browsing_history_depth+1] if include_browsing_history_depth != -1 else history[1:]
            
            if history_items:
                if not current_page_message:
                    current_page_message = ""
                current_page_message += f"\n\n## Browsing History\n\n"
                for url in history_items:
                    if url.url:
                        # Add time context if available
                        time_context = ""
                        if hasattr(url, 'timestamp') and url.timestamp:
                            time_diff = time.time() - url.timestamp
                            if time_diff < 60:
                                time_context = " (just now)"
                            elif time_diff < 3600:
                                time_context = f" ({int(time_diff/60)}m ago)"
                            elif time_diff < 86400:
                                time_context = f" ({time_diff/3600:.1f}h ago)"
                        current_page_message += f"- [{url.title or url.url}]({url.url}){time_context}\n"
                current_page_message += "\n"

        if current_page_message:
            user_state_message = current_page_message
        
        if include_product_history:
            product_history = []
            
            for timestamp, product_id in recent_product_ids:   
                product = await product_manager.find_product_by_id(product_id)
                if product and isinstance(product, Product):
                    product_history.append((timestamp, product))
            
            if product_history and len(product_history) > 0:
                # get a count of each product in the history
                products_by_count = {}
                products_to_include = []
                for _, product in product_history:
                    if product and isinstance(product, Product):
                        product_id = product.id
                        if product_id:
                            if product_id in products_by_count:
                                products_by_count[product_id] += 1
                            else:
                                products_by_count[product_id] = 1
                                products_to_include.append(product)

                # Only add the section if there are products to show
                if products_to_include:
                    product_history_message = "# Product History\n\nRecently viewed products:\n\n"
                    # get the top 5 products
                    products_to_include = products_to_include[:5]
                    # now add the top products to the message, including the count as additional information
                    for product in products_to_include:
                        count = products_by_count.get(product.id, 1)                 
                        count_str = f"Viewed {count}x" if count > 1 else ""
                        product_history_message += f"{Product.to_markdown_short(product=product, depth=1, additional_info=count_str)}\n"
                    user_state_message += f"\n\n{product_history_message}"
        
        if include_communication_directive:
            communication_directive = user_state.communication_directive
            if communication_directive and communication_directive.directive:
                user_state_message += f"\n\nCommunication Directive: {communication_directive.directive}"

        # check to see if there is a previous conversation exit state
        if include_resumption:
            if user_state.conversation_exit_state:
                conversation_exit_state = user_state.conversation_exit_state
                # get the last interaction time and the transcript summary
                should_resume = conversation_exit_state.resumption_message is not None and len(conversation_exit_state.resumption_message) > 0 and conversation_exit_state.resumption_message.lower() != "false"

                # get the resumption flag
                last_interaction_time = None
                transcript_summary = None
                last_interaction_time = conversation_exit_state.last_interaction_time
                
                transcript_summary = conversation_exit_state.transcript_summary

                if last_interaction_time:
                    current_time = time.time()
                    time_difference = current_time - last_interaction_time
                    # get a rough approximation of the time difference
                    if time_difference > 31536000:
                        time_difference_approx = f"{time_difference / 31536000:.1f} years"
                    elif time_difference > 2592000:
                        time_difference_approx = f"{time_difference / 2592000:.1f} months"
                    elif time_difference > 604800:
                        time_difference_approx = f"{time_difference / 604800:.1f} weeks"
                    elif time_difference > 86400:
                        time_difference_approx = f"{time_difference / 86400:.1f} days"
                    elif time_difference > 3600:
                        time_difference_approx = f"{time_difference / 3600:.1f} hours"
                    elif time_difference > 60:
                        time_difference_approx = f"{time_difference / 60:.0f} minutes"
                    else:
                        time_difference_approx = "less than a minute"

                    if should_resume:
                        # this should a continuation of the previous conversation
                        user_state_message += f"\n\nConversation Resumption:\nThis is a continuation of the previous conversation. The last interaction with the user was roughly {time_difference_approx} ago."
                        if transcript_summary:
                            user_state_message += f"\nHere is a summary of that conversation:\n{transcript_summary}"
                    else:
                        # this should be a new conversation
                        user_state_message += f"\n\nLast Conversation:\nThe last interaction with the user was roughly {time_difference_approx} ago."
                        if transcript_summary:
                            user_state_message += f"\nHere is a summary of that conversation:\n{transcript_summary}"

        return user_state_message

    @classmethod
    async def build_current_page_message(cls, user_id: str) -> tuple[str, str] | None:
        """
        Build a message about the current page for the user.
        
        Args:
            user_id: The user identifier
        
        Returns:
            tuple: (url, message) or None
        """
        if not user_id:
            return None

        url_context = await cls.get_current_url_context(user_id)
        if not url_context:
            return None

        message = "User is now browsing the following page:\n\n"
        page_url = None

        if isinstance(url_context, Product):
            page_url = url_context.productUrl
            message += Product.to_markdown(url_context, obfuscatePricing=True)
        elif isinstance(url_context, dict):
            page_url = url_context.get("url")
            message += obj_to_markdown(url_context, "Current Page Information", level=2)
        elif isinstance(url_context, str):
            page_url = url_context
            message += url_context
        else:
            return None

        if not page_url or not message:
            return None

        return page_url, message

    @classmethod
    async def get_current_url_context(cls, user_id) -> Product | dict | None:
        """
        Get the current URL context for a user from Redis.
        
        Args:
            user_id: The user ID to get URL context for
            
        Returns:
            dict: The URL context information or None if not found
        """
        try:
            history = cls.get_user_recent_history(user_id)
            if not history or len(history) == 0:            
                return None

            most_recent_history = history[0]
            # if this is a product, then get the details
            if isinstance(most_recent_history, dict):
                product_details = most_recent_history.get("product_details")
                if product_details:
                    product = product_details.get("product")
                    if product:
                        # Return the product from history
                        # Note: With Redis-backed products, we can't look up without account
                        # The product in history should be sufficient for URL context
                        return product
                
            return most_recent_history

        except Exception as e:
            logger.warning(f"Error getting URL context for user {user_id}: {e}")
            return None