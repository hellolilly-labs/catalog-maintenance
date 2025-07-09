import logging
import os
import json
import re
import shutil
import time
from datetime import datetime
from typing import Dict, Any, Optional
from liddy.models.product import Product

from langfuse import get_client
from langfuse.types import PromptClient
from liddy.account_config_loader import get_account_config_loader
from liddy import AccountManager
import asyncio
 
 
logger = logging.getLogger("prompt-manager")
_account_prompt_managers: Dict[str, "AccountPromptManager"] = {}

class AccountPromptManager:
    """
    Manages system prompts using local files.
    
    This class handles:
    1. Loading prompts from files
    2. Personalizing prompts for specific users
    3. Managing RAG content injection
    """
    
    def __init__(self, account:str):
        """
        Initialize the AccountPromptManager.
        
        Args:
            account: The account/brand domain to manage prompts for
        """
        init_start = time.time()
        timing_breakdown = {}
        
        # Phase 1: Initialize attributes
        attr_start = time.time()
        self.account = account
        self.full_instructions = ""
        self.prompt_additions: Dict[str, str] = {}
        self._full_instructions_langfuse_prompt = None
        timing_breakdown['attributes'] = time.time() - attr_start
        
        # Phase 2: Initialize external clients
        clients_start = time.time()
        self.langfuse = get_client()
        self.account_config_loader = get_account_config_loader()
        self._cached_account_config = None
        timing_breakdown['external_clients'] = time.time() - clients_start
                
        # # Phase 3: Load base prompts
        # prompts_load_start = time.time()
        # self._load_base_prompts()
        # timing_breakdown['load_prompts'] = time.time() - prompts_load_start
        
        # Log timing summary
        total_time = time.time() - init_start
        timing_breakdown['total'] = total_time
        
        logger.info(f"📊 PromptManager init timing for {account}: "
                   f"total={total_time:.3f}s "
                   f"(clients={timing_breakdown['external_clients']:.3f}s, "
                #    f"prompts={timing_breakdown['load_prompts']:.3f}s)"
        )
    
    async def load_account_config_async(self):
        """Preload account configuration asynchronously for better performance"""
        try:
            config = await self.account_config_loader.get_account_config(self.account)
            if config:
                self._cached_account_config = config
                logger.info(f"Preloaded account config for {self.account}")
            else:
                logger.info(f"No account config found for {self.account}")
        except Exception as e:
            logger.warning(f"Error preloading account config for {self.account}: {e}")
    
    def get_full_instructions_langfuse_prompt(self):
        if self._full_instructions_langfuse_prompt:
            return self._full_instructions_langfuse_prompt
        else:
            return None
    
    def update_current_generation(self):
        self.langfuse.update_current_generation(prompt=self._full_instructions_langfuse_prompt)
    
    def _load_prompt_from_langfuse(self, prompt_name: str) -> Optional[PromptClient]:
        start_time = time.time()
        
        # Phase 1: Try to get from Langfuse
        langfuse_start = time.time()
        try:
            langfuse_prompt = self.langfuse.get_prompt(f"{self.account}/{prompt_name}")
            if langfuse_prompt:
                logger.debug(f"✅ Loaded {prompt_name} from Langfuse in {time.time() - langfuse_start:.3f}s")
                return langfuse_prompt

        except Exception as e:
            logger.error(f"Error getting prompt from Langfuse: {e}")
        langfuse_time = time.time() - langfuse_start

        # Phase 2: Try to load from file
        file_start = time.time()
        file_prompt = None
        account_path = os.path.join(self.prompts_dir, self.account)
        prompt_path = os.path.join(account_path, f"{prompt_name}.md")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                file_prompt = f.read()
            logger.debug(f"📄 Loaded {prompt_name} from file in {time.time() - file_start:.3f}s")
        else:
            logger.debug(f"❌ {prompt_name} not found (checked Langfuse: {langfuse_time:.3f}s)")
            return None
    
        # Phase 3: Create in Langfuse if loaded from file
        if file_prompt:
            create_start = time.time()
            try:
                result = self.langfuse.create_prompt(name=f"{self.account}/{prompt_name}", prompt=file_prompt, type="text", labels=[self.account, "production"])
                logger.debug(f"📤 Created {prompt_name} in Langfuse in {time.time() - create_start:.3f}s")
                return result
            except Exception as e:
                logger.error(f"Error creating prompt in Langfuse: {e}")
                return None
        else:
            return None            
    
    def _load_base_prompts(self):
        """Load base prompts from files"""
        method_start = time.time()
        timing_breakdown = {}
        
        try:
            # Load the base system prompt
            
            # Phase 1: Load full instructions
            full_instructions_start = time.time()
            try:
                # Get current `production` version of a text prompt
                self._full_instructions_langfuse_prompt = self._load_prompt_from_langfuse("full_instructions")
                if self._full_instructions_langfuse_prompt:
                    # self.langfuse.update_current_generation(prompt=self._full_instructions_langfuse_prompt)
                
                    # Insert variables into prompt template
                    compiled_prompt = ""
                    compiled_prompt = self._full_instructions_langfuse_prompt.compile(SLIDER_BLOCK=self.slider_block())

                    if compiled_prompt:
                        self.full_instructions = compiled_prompt
                        self.system_prompt = self.full_instructions
                        # return compiled_prompt
                    else:
                        logger.error(f"Error compiling prompt: {e}")
                        # return None
            except Exception as e:
                logger.error(f"Error getting prompt: {e}")
                raise e
            timing_breakdown['full_instructions'] = time.time() - full_instructions_start
            
            # Log timing summary
            total_time = time.time() - method_start
            logger.info(f"📊 _load_base_prompts timing: total={total_time:.3f}s "
                       f"(full_instructions={timing_breakdown.get('full_instructions', 0):.3f}s, "
                       f"knowledge={timing_breakdown.get('knowledge_prompts', 0):.3f}s)")
            
            logger.info("Loaded system prompts successfully")
                
        except Exception as e:
            logger.error(f"Error loading system prompts: {e}")
            # Provide fallback prompts for testing
            self.full_instructions = f"You are an AI sales assistant for the {self.account} brand. Help the user with their questions."
    
    def update_prompt_additions(self, prompt_additions: Dict[str, str], overwrite: bool=False):
        """Update the prompt additions for the user"""
        if prompt_additions:
            if overwrite:
                self.prompt_additions = prompt_additions
            else:
                # if the key already exists and is a list, append the new value(s)
                for key, value in prompt_additions.items():
                    if key in self.prompt_additions:
                        if isinstance(self.prompt_additions[key], list) and self.prompt_additions[key]:
                            if isinstance(value, list):
                                self.prompt_additions[key].extend(value)
                            else:
                                self.prompt_additions[key].append(value)
                        else:
                            self.prompt_additions[key] = value
                    else:
                        self.prompt_additions[key] = value
                # self.prompt_additions = {**self.prompt_additions, **prompt_additions}
        else:
            if overwrite:
                self.prompt_additions = {}

    def _get_account_personality_config(self) -> Optional[Dict[str, Any]]:
        """Get account personality configuration from account.json"""
        if self._cached_account_config:
            return self._cached_account_config
        
        try:
            # Run async call in current event loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context and no cached config, use hardcoded values
                    # The config should be preloaded via load_account_config_async()
                    if not self._cached_account_config:
                        logger.debug(f"Using hardcoded personality config for {self.account} (async context)")
                    return None
                else:
                    config = loop.run_until_complete(self.account_config_loader.get_account_config(self.account))
            except RuntimeError:
                # No event loop, create one
                config = asyncio.run(self.account_config_loader.get_account_config(self.account))
            
            if config:
                self._cached_account_config = config
                logger.info(f"Loaded personality config for {self.account}")
                return config
            else:
                logger.info(f"No account config found for {self.account}, using fallback")
                return None
                
        except Exception as e:
            logger.warning(f"Error loading account config for {self.account}: {e}, using fallback")
            return None
    
    def slider_block(self) -> str:
        """Convert slider values into a markdown block for system prompt.
        
        Loads personality slider values from account configuration (account.json)
        and falls back to hardcoded defaults if not available.
        """
        
        # Try to load dynamic configuration
        account_config = self._get_account_personality_config()
        persona_config = None
        
        if account_config:
            persona_config = account_config.get('agent', {}).get('persona', {})
            logger.info(f"Using dynamic personality sliders for {self.account}")
        else:
            logger.debug(f"Using hardcoded personality sliders for {self.account}")
        
        # Map personality traits from account.json to slider format
        # 6 Core Personality Traits (1-10 scale) + 4 Sales-Specific Traits
        sliders = [
            {
                "id": "formality_level",
                "value": persona_config.get("formality_level", 5) if persona_config else 5,
                "instruction": "1-3: Very casual, conversational; 4-6: Balanced professional yet approachable; 7-10: Formal, professional"
            },
            {
                "id": "warmth_empathy", 
                "value": persona_config.get("warmth_empathy", 7) if persona_config else 7,
                "instruction": "1-3: Professional but neutral; 4-6: Friendly and understanding; 7-10: Highly nurturing, deeply empathetic"
            },
            {
                "id": "response_length",
                "value": persona_config.get("response_length", 5) if persona_config else 5,
                "instruction": "1-3: Concise (1-2 sentences); 4-6: Balanced detail (2-4 sentences); 7-10: Comprehensive (4+ sentences)"
            },
            {
                "id": "confidence_level",
                "value": persona_config.get("confidence_level", 7) if persona_config else 7,
                "instruction": "1-3: Tentative, uses qualifiers; 4-6: Balanced confidence; 7-10: Highly confident, assertive"
            },
            {
                "id": "enthusiasm_energy",
                "value": persona_config.get("enthusiasm_energy", 6) if persona_config else 6,
                "instruction": "1-3: Calm, measured; 4-6: Positive but controlled; 7-10: High energy, very animated"
            },
            {
                "id": "humor_playfulness",
                "value": persona_config.get("humor_playfulness", 4) if persona_config else 4,
                "instruction": "1-3: Serious, information-focused; 4-6: Occasionally light; 7-10: Playful, frequently humorous"
            },
            {
                "id": "sales_push",
                "value": persona_config.get("sales_push", 4.0) if persona_config else 4.0,
                "instruction": "Controls how often to suggest add-ons or move to checkout"
            },
            {
                "id": "policy_emphasis",
                "value": persona_config.get("policy_emphasis", 4.0) if persona_config else 4.0,
                "instruction": "How proactively to mention returns, lead-time, or safety policies"
            },
            {
                "id": "brand_flourish",
                "value": persona_config.get("brand_flourish", 3.0) if persona_config else 3.0,
                "instruction": "How often to use brand-specific phrases and terminology"
            },
            {
                "id": "on_topic_strictness",
                "value": persona_config.get("on_topic_strictness", 5.5) if persona_config else 5.5,
                "instruction": "Speed of pivoting back to brand products when conversation drifts"
            }
        ]
        
        # Format as markdown block for prompt injection
        lines = ["### Personality & Sales Configuration (Dynamic from Account Settings)"]
        lines.append("**Core Personality Traits (1-10 scale):**")
        
        # Core personality traits (1-10)
        for slider in sliders[:6]:  # First 6 are core personality
            name = slider["id"].replace("_", " ").title()
            val = slider["value"]
            desc = slider["instruction"]
            lines.append(f"* **{name}**: {val} — {desc}")
        
        lines.append("")
        lines.append("**Sales-Specific Behavior (0-10 scale):**")
        
        # Sales-specific traits (0-10)  
        for slider in sliders[6:]:  # Last 4 are sales-specific
            name = slider["id"].replace("_", " ").title()
            val = slider["value"]
            desc = slider["instruction"]
            lines.append(f"* **{name}**: {val} — {desc}")
        
        return "\n".join(lines)

    # Example CLI call:
    # build_prompt("flex_prompt_template.md", "flex_sliders.json", "flex_prompt_live.md")
    
    # def get_prompt_for_user(self, user_id: str, prompt_additions: Optional[Dict[str, str]] = None) -> str:
    def build_system_instruction_prompt(self) -> str:
        if self.full_instructions:
            prompt = self.full_instructions
            return prompt

        try:
            start_time = time.time()
            # Get current `production` version of a text prompt
            prompt = self.langfuse.get_prompt(f"{self.account}/full_instructions")
            logger.debug(f"📄 Loaded full_instructions from Langfuse in {time.time() - start_time:.3f}s")
            # Insert variables into prompt template
            compiled_prompt = ""
            compiled_prompt = prompt.compile(SLIDER_BLOCK=self.slider_block())
            logger.debug(f"📄 Compiled full_instructions in {time.time() - start_time:.3f}s")
            if compiled_prompt:
                self.full_instructions = compiled_prompt
                logger.debug(f"📄 Updated full_instructions in {time.time() - start_time:.3f}s")
                return compiled_prompt
        except Exception as e:
            logger.error(f"Error getting prompt: {e}")
        logger.debug(f"📄 Finished getting prompt in {time.time() - start_time:.3f}s")
        
        if self.full_instructions:
            prompt = self.full_instructions
        else:
            prompt = f"You are an AI sales assistant for the {self.account} brand. Help the user with their questions."
        
        return prompt
    
    def _insert_knowledge(self, prompt: str, key: str, knowledge: str, force: bool=False) -> str:
        """
        Insert knowledge content into a prompt, handling different placeholder formats.
        
        Args:
            prompt: The original prompt content
            key: The tag/key name to insert knowledge into
            knowledge: The content to insert
            force: If True, insert the knowledge even if the key is not found in the prompt
        
        Placeholder formats supported:
            - <key></key>: Empty XML-style tag - replaces with <key>content</key>
            - <key>existing</key>: XML tag with content - replaces existing content
            - <key>: Single XML opening tag - adds content and closing tag
            - {key}: Template variable style - one-time replacement (content only, no tags)
        
        Returns:
            Updated prompt with knowledge inserted
        
        Note: 
            - XML-style tags (<tag>) are designed for multiple replacements in a session
            - Template variables ({tag}) are for one-time insertion without tags
        """
        updated_prompt = prompt
        
        # if knowledge is something other than a string, convert to string
        if not isinstance(knowledge, str):
            knowledge = str(knowledge)
        
        if knowledge:
            if f"<{key}></{key}>" in updated_prompt:
                updated_prompt = updated_prompt.replace(f"<{key}></{key}>", f"<{key}>{knowledge}</{key}>")
            elif f"<{key}/>" in prompt:
                updated_prompt = updated_prompt.replace(f"<{key}/>", f"<{key}>{knowledge}</{key}>")
            elif re.search(rf"<{key}>.*?</{key}>", prompt, re.DOTALL):
                updated_prompt = re.sub(
                    rf"<{key}>.*?</{key}>", 
                    f"<{key}>{knowledge}</{key}>", 
                    updated_prompt, 
                    flags=re.DOTALL
                )
            elif f"<{key}>" in updated_prompt:
                updated_prompt = updated_prompt.replace(f"<{key}>", f"<{key}>{knowledge}</{key}>")
            # else if {key}{/key} is found, replace with {key}knowledge{/key}
            elif f"{{{key}}}{{/{key}}}" in updated_prompt:
                updated_prompt = updated_prompt.replace(f"{{{key}}}{{/{key}}}", f"{{{key}}}{knowledge}{{/{key}}}")
            # else if {key/} is found, replace with {key}knowledge{/key}
            elif f"{{{key}/}}" in updated_prompt:
                updated_prompt = updated_prompt.replace(f"{{{key}/}}", f"{{{key}}}{knowledge}{{/{key}}}")
            elif f"{{{key}}}" in updated_prompt:
                updated_prompt = updated_prompt.replace(f"{{{key}}}", knowledge)
            elif force:
                updated_prompt = f"{updated_prompt}\n\n<{key}>{knowledge}</{key}>"
            #     # updated_prompt = f"{prompt}\n\n<{key}>{knowledge}</{key}>"
            elif key == "resumption":
                pass
            else:
                logger.warning(f"Key '{key}' not found in prompt")
        # else:
        #     updated_prompt = f"<{key}>{knowledge}</{key}>"
            
        return updated_prompt 
    
    def update_rag_content(self, rag_content: str) -> None:
        """Update the prompt additions with RAG content"""
        self.update_prompt_additions(prompt_additions={"rag_content": rag_content})
    
    def update_valid_product_urls(self, valid_product_urls: str) -> None:
        """Update the prompt additions with valid product URLs"""
        self.update_prompt_additions(prompt_additions={"valid_product_urls": valid_product_urls})
    
    def update_conversation_analysis(self, conversation_analysis: Dict[str, Any]) -> None:
        """Update the prompt additions with conversation analysis"""
        self.update_prompt_additions(prompt_additions={"conversation_analysis": json.dumps(conversation_analysis)})
    
    def update_conversation_knowledge(self, conversation_knowledge: str) -> None:
        """Update the prompt additions with conversation knowledge"""
        self.update_prompt_additions(prompt_additions={"conversation_knowledge": conversation_knowledge})
    
    def clear_cache(self):
        """Clear prompt prompt_additions cache for a specific user or all users"""
        self.prompt_additions = {}
            
    def build_chat_agent_instructions(self, account: str) -> str:
        """
        Build ChatAgent-specific prompt with delegation awareness.
        
        This creates a modified version of the full instructions that:
        - Maintains persona and brand voice
        - Replaces tool instructions with delegation patterns
        - Adds local knowledge handling capabilities
        - Includes immediate feedback instructions
        """
        base_instructions = self.build_system_instruction_prompt(account)
        
        # Transform for ChatAgent delegation pattern
        chat_agent_instructions = self._apply_chat_agent_template(base_instructions)
        
        return chat_agent_instructions

    def build_supervisor_instructions(self, account: str) -> str:
        """
        Build SupervisorAssistant prompt with full capabilities.
        
        This returns the existing full instructions with additional
        context about structured response format for ChatAgent integration.
        """
        base_instructions = self.build_system_instruction_prompt(account)
        
        # Add supervisor-specific integration instructions
        supervisor_additions = """

## SUPERVISOR MODE - Structured Response Integration

**CRITICAL**: When handling delegated requests from ChatAgent, provide structured responses to enable seamless integration.

### Response Format
Return responses in this structured format when appropriate:

```json
{
  "suggested_message": "Your response in Spence brand voice",
  "context_updates": {
    "conversation_phase": "focus|consideration|support|closing",
    "active_products": ["product_id"],
    "can_handle_locally": ["sizing questions", "color options"],
    "user_preferences": {"type": "road", "budget": "mid-range"}
  },
  "follow_up_suggestions": ["Ask about sizing", "Show colors"],
  "confidence": 0.95
}
```

### Integration Guidelines
- ChatAgent will process and potentially enhance your suggested_message
- Provide detailed context_updates for ChatAgent's future local handling
- Include natural follow_up_suggestions that match conversation flow
- Use confidence score to help ChatAgent decide how to use your response
- For display_product calls, put resumption content in suggested_message field

### Brand Voice Consistency
Maintain the same Spence persona and style guidelines as ChatAgent:
- Length: one-to-three sentences when responding
- Use opener discipline and brand phrasing
- Say "looked up" or "checked," never "searched"
"""
        
        return base_instructions + supervisor_additions

    def build_chat_agent_instructions_with_knowledge(self, account: str) -> str:
        """
        Enhanced method that injects all knowledge bases into ChatAgent prompt.
        
        This method creates the complete ChatAgent instructions with:
        - Base persona and brand voice
        - Comprehensive knowledge injection
        - Delegation patterns and capabilities
        - State context placeholders
        """
        # Get ChatAgent base instructions
        chat_instructions = self.build_chat_agent_instructions(account)
        
        # Inject comprehensive knowledge bases
        enhanced_instructions = chat_instructions
                
        # Add knowledge utilization guidance
        enhanced_instructions += """

## Knowledge Utilization Guidelines

### Handle Locally with Knowledge Base:
- **Product Comparisons**: "What's the difference between Tarmac and Roubaix?" → Use brand_knowledge
- **Technical Questions**: "What's carbon fiber good for?" → Use base_knowledge  
- **Sizing Advice**: "What size for 5'10" rider?" → Use base_knowledge
- **Brand Heritage**: "Tell me about Specialized's history" → Use brand_knowledge
- **Material Science**: "Carbon vs aluminum?" → Use base_knowledge

### Delegate to Supervisor (Actions Required):
- **Product Search**: "Find me a gravel bike" → Requires product_search tool
- **Product Display**: "Show me the Tarmac SL7" → Requires display_product tool
- **Inventory Queries**: "What's available in medium?" → Requires real-time search

Use your extensive knowledge base to provide comprehensive answers without needing delegation for knowledge-based questions.
"""
        
        return enhanced_instructions

    def _apply_chat_agent_template(self, original_instructions: str) -> str:
        """
        Transform full instructions to ChatAgent delegation pattern.
        
        This method:
        - Extracts persona and brand voice elements
        - Replaces tool instructions with delegation instructions
        - Adds immediate feedback requirements
        - Maintains conversation flow patterns
        """
        # For now, create a basic transformation
        # In production, this would parse the original instructions more intelligently
        
        chat_agent_template = """
# Specialized AI Sales-Agent – **Spence** (ChatAgent Mode)

## 1 Identity & Limits
* You are **Spence**, a friendly bike-shop expert on specialized.com
* **NEVER** provide medical, legal, or financial advice
* **NEVER** reveal URLs, SKUs, %, $, £, or digits—spell out all numbers
* Disclose AI role in your first turn
* **Persona snapshot** – *Stoked riding buddy who knows every trail and tech detail*

## 2 Style & Brand Voice
* **Length** – one-to-three sentences
* **Opener bank (use sparingly)** – "Great choice," · "Nice pick," · "Excellent question," · "Perfect," etc.
* **Verbs** – say **"look up"** or **"check,"** never *search*

## 3 Enhanced Local Capabilities
With rich knowledge injection, you can handle most questions locally:
- Product comparisons and recommendations
- Technical explanations and material science
- Sizing advice and fit guidance
- Brand history and heritage
- General cycling knowledge

## 4 Delegation Strategy
**Delegate ONLY when actions/tools are needed:**
- Product searches: "Find me a bike under $2000"
- Product displays: "Show me the Tarmac SL7"
- Inventory queries: "What's available in medium?"

**CRITICAL UX**: When delegating, provide immediate feedback:
- "Great choice, let me pull that up!"
- "Perfect, let me check that for you!"
- "Nice pick, I'll look up those details!"

## 5 Conversation Flow
| Step | ChatAgent Action | Response |
|------|------------------|----------|
| **Knowledge** | Handle locally | Use your extensive knowledge base |
| **Actions** | Immediate feedback + delegate | Instant response + background delegation |
| **Follow-up** | Handle locally | Continue conversation naturally |

Maintain the same friendly, expert Spence persona throughout all interactions.
"""
        
        return chat_agent_template


# =============================================================================
# SINGLETON FACTORY FUNCTIONS
# =============================================================================

def get_account_prompt_manager(account: str) -> AccountPromptManager:
    """
    Get or create AccountPromptManager instance for an account.
    
    Uses singleton pattern for efficiency.
    
    Args:
        account: Account/brand domain
        
    Returns:
        AccountPromptManager instance
    """
    normalized_account = AccountManager._normalize_account(None, account)
    
    if normalized_account not in _account_prompt_managers:
        instance = AccountPromptManager(normalized_account)
        _account_prompt_managers[normalized_account] = instance
        # # Pre-load account config
        # await instance.load_account_config_async()
        logger.info(f"Created AccountPromptManager for {normalized_account}")
    
    return _account_prompt_managers[normalized_account]


def clear_account_prompt_managers():
    """Clear all cached AccountPromptManager instances."""
    _account_prompt_managers.clear()
    logger.info("Cleared all AccountPromptManager instances")