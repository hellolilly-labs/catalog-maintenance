import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# from livekit.agents import llm, utils

from .markdown_utils import obj_to_markdown
from redis_client import get_user_state, save_user_state, get_user_recent_history
from spence.product import Product
from spence.model import BasicChatMessage, UrlTracking, UserState, ConversationExitState, ConversationResumptionState


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
    def get_user_recent_history(cls, user_id: str) -> List[UrlTracking]:
        """Get user history from Redis"""
        return get_user_recent_history(user_id)

    @classmethod
    async def get_user_recent_products(cls, user_id: str, account:str, limit: int=25) -> List[Tuple[float, Product]]:
        """Get user product history from Redis"""
        history = get_user_recent_history(user_id)
        
        if not history or len(history) == 0:
            return []
        products = []
        for urlTracking in history:
            if urlTracking.url:
                base_url = urlTracking.url.split("?")[0]
                product = await Product.find_by_url(product_url=urlTracking.url, account=account)
                
                if product and isinstance(product, Product):
                    # get the details from the products database
                    product_id = product.id
                    if product_id:
                        products.append((urlTracking.timestamp, product))
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
            
            return save_user_state(user_state=user_state)
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
                time_difference_approx = "less than a minute"
                if time_difference > 31536000:
                    time_difference_approx = f"{time_difference / 31536000} years"
                elif time_difference > 2592000:
                    time_difference_approx = f"{time_difference / 2592000} months"
                elif time_difference > 604800:
                    time_difference_approx = f"{time_difference / 604800} weeks"
                elif time_difference > 86400:
                    time_difference_approx = f"{time_difference / 86400} days"
                elif time_difference > 3600:
                    time_difference_approx = f"{time_difference / 3600} hours"
                elif time_difference > 60:
                    time_difference_approx = f"{time_difference / 60} minutes"
                else:
                    time_difference_approx = f"{time_difference} seconds"

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
            user_state = get_user_state(user_id)
            if user_state:
                user_state.conversation_exit_state = None
                save_user_state(user_state=user_state)
            return True
            
        except Exception as e:
            logger.error(f"Error clearing conversation resumption: {e}")
            return False

    @classmethod
    async def build_user_state_message(cls, user_id: str, user_state:UserState=None, include_current_page: bool=True, include_communication_directive: bool=False, include_resumption:bool=True, include_product_history: bool=True, include_browsing_history:bool=False) -> str:
        """
        Build a user state message for the agent.
        
        Args:
            user_id: The user identifier
        
        Returns:
            str: The user state message
        """
        if not user_id:
            return ""

        user_state = user_state or get_user_state(user_id)
        if not user_state:
            user_state: UserState = UserState(user_id=user_id)
        
        if (include_current_page or include_browsing_history):
            history = cls.get_user_recent_history(user_id)
        else:
            history = None

        # Add user state to context if available
        user_state_message = ""
        current_page_message = None
        if history and len(history) > 0 and (include_current_page or include_browsing_history):
            browsing_history_message = None
            most_recent_history = history[0]
            # if there are more than 1 history, then pull the title and url from each one into the browsing history message
            if include_browsing_history and len(history) > 1:
                browsing_history_message = "Browsing History (in descending order):\n\n"
                for url in history:
                    if url.title and url.url:
                        # grab the details if they exist
                        details = None
                        if url.pr:
                            if 'product' in url.product_details and 'details' in url.product_details['product']:
                                details = url.product_details['product']['details']
                        browsing_history_message += f"- [{url.title}]({url.url}): {details if details else ''}\n\n"
            
            if include_current_page and most_recent_history:
                current_url = None
                details = None

                product = None
                if most_recent_history.url:
                    current_url = most_recent_history.url
                    current_title = most_recent_history.title if most_recent_history.title else current_url

                    current_page_message = f"# Current Page\n\nThe user is currently looking at the following page:\n\n[{current_title}]({current_url})\n\n"

                    product = await Product.find_by_url(current_url, account=user_state.account)
                    if product and isinstance(product, Product):
                        current_page_message += f"## Current Product\n\n{Product.to_markdown(product, depth=2, obfuscatePricing=True)}\n"
                

            if include_browsing_history and len(history) > 0:
                if include_current_page:
                    if len(history) > 1:
                        current_page_message += f"\n\n## Browsing History\n\n"
                        # append brief browsing history to the current page message
                        for url in history[1:]:
                            if url.title and url.url:
                                current_page_message += f"- [{url.title}]({url.url})\n"
                        current_page_message += "\n"
                else:
                    # if we are not including the current page, then just use the browsing history message
                    current_page_message += f"\n\n## Browsing History\n\n"
                    # append brief browsing history to the current page message
                    for url in history:
                        if url.title and url.url:
                            current_page_message += f"- [{url.title}]({url.url})\n"
                    current_page_message += "\n"

            if current_page_message:
                user_state_message = f"\n\n{current_page_message}"
        
        if include_product_history:
            product_history_message = "# Product History\n\nThe user has looked at these products (in the following order):\n\n"
            product_history = await cls.get_user_recent_products(user_id=user_id, account=user_state.account)
            if product_history and len(product_history) > 0:
                # get a count of each product in the history
                products_by_count = {}
                products_to_include = []
                for timestamp, product in product_history:
                    if product and isinstance(product, Product):
                        product_id = product.id
                        if product_id:
                            if product_id in products_by_count:
                                products_by_count[product_id] += 1
                            else:
                                products_by_count[product_id] = 1
                                products_to_include.append(product)

                # get the top 7 products
                products_to_include = products_to_include[:7]
                # now add the top products to the message, including the count as additional information
                for product in products_to_include:
                    count = products_by_count.get(product.id, 1)                 
                    count_str = f"Number of times viewed: {count}"
                    # product_history_message += f"{Product.to_markdown(product=product, depth=1, obfuscatePricing=True, additional_info=count_str)}\n"
                    product_history_message += f"{Product.to_markdown_short(product=product, depth=1, additional_info=count_str)}\n"
                user_state_message += f"\n\n{product_history_message}\n"
        
        if include_communication_directive:
            communication_directive = user_state.communication_directive
            if communication_directive and communication_directive.directive:
                user_state_message += f"\n\nCommunication Directive: {communication_directive.directive}"

        # check to see if there is a previous conversation exit state
        if include_resumption:
            if user_state.conversation_exit_state:
                conversation_exit_state = user_state.conversation_exit_state
                # get the last interaction time and the transcript summary
                should_resume = conversation_exit_state.use_resumption_message

                # get the resumption flag
                last_interaction_time = None
                transcript_summary = None
                last_interaction_time = conversation_exit_state.last_interaction_time
                
                transcript_summary = conversation_exit_state.transcript_summary

                if last_interaction_time:
                    current_time = time.time()
                    time_difference = current_time - last_interaction_time
                    # get a rough approximation of the time difference
                    time_difference_approx = "less than a minute"
                    if time_difference > 31536000:
                        time_difference_approx = f"{time_difference / 31536000} years"
                    if time_difference > 2592000:
                        time_difference_approx = f"{time_difference / 2592000} months"
                    if time_difference > 604800:
                        time_difference_approx = f"{time_difference / 604800} weeks"
                    if time_difference_approx > 86400:
                        time_difference_approx = f"{time_difference / 86400} days"
                    if time_difference > 3600:
                        time_difference_approx = f"{time_difference / 3600} hours"
                    if time_difference > 60:
                        time_difference_approx = f"{time_difference / 60} minutes"

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
                        # get the details from the products database
                        existing_product = await Product.find_by_id(product["id"])
                        if existing_product:
                            return existing_product
                
            return most_recent_history

        except Exception as e:
            logger.warning(f"Error getting URL context for user {user_id}: {e}")
            return None