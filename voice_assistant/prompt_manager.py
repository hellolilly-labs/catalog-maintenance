import logging
import os
import json
import re
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from spence.product import Product

from langfuse import get_client
from langfuse.types import PromptClient
from .account_config_loader import get_account_config_loader
import asyncio
 
 
logger = logging.getLogger("prompt-manager")

class PromptManager:
    """
    Manages system prompts using local files.
    
    This class handles:
    1. Loading prompts from files
    2. Personalizing prompts for specific users
    3. Managing RAG content injection
    """
    
    DEFAULT_PROMPTS_DIR = "../prompts"
    
    def __init__(self, account:str, prompts_dir=None):
        """
        Initialize the PromptManager.
        
        Args:
            prompts_dir: Optional custom directory for prompts. If not provided,
                         uses environment variable PROMPTS_DIR if set, or falls back
                         to the default directory relative to this file's location.
        """
        self.account = account
        self.full_instructions = ""
        self.system_prompt_wrapper = ""
        self.system_prompt = ""
        self.base_knowledge = ""
        self.brand_knowledge = ""
        self.product_search_knowledge = ""
        self.knowledge_search_guidance = ""
        self.prompt_additions: Dict[str, str] = {}
        self._full_instructions_langfuse_prompt = None
        self.langfuse = get_client()
        self.account_config_loader = get_account_config_loader()
        self._cached_account_config = None

        
        # Determine prompts directory:
        # 1. Use explicitly provided directory if given
        # 2. Otherwise use environment variable if set
        # 3. Otherwise use default directory relative to this file
        if prompts_dir:
            self.prompts_dir = prompts_dir
        elif os.environ.get('PROMPTS_DIR'):
            self.prompts_dir = os.environ.get('PROMPTS_DIR')
        else:
            # Use directory relative to this file's location
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.prompts_dir = os.path.join(module_dir, self.DEFAULT_PROMPTS_DIR)
            
        logger.info(f"Using prompts directory: {self.prompts_dir}")
        
        # Create prompts directory if it doesn't exist
        account_path = os.path.join(self.prompts_dir, self.account)
        os.makedirs(account_path, exist_ok=True)
        
        # Load base prompts
        self._load_base_prompts()
    
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
        try:
            langfuse_prompt = self.langfuse.get_prompt(f"{self.account}/{prompt_name}")
            if langfuse_prompt:
                return langfuse_prompt

        except Exception as e:
            logger.error(f"Error getting prompt from Langfuse: {e}")

        # attempt to load from file
        file_prompt = None
        account_path = os.path.join(self.prompts_dir, self.account)
        prompt_path = os.path.join(account_path, f"{prompt_name}.md")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r") as f:
                file_prompt = f.read()
        else:
            return None
    
        if file_prompt:
            try:
                return self.langfuse.create_prompt(name=f"{self.account}/{prompt_name}", prompt=file_prompt, type="text", labels=[self.account, "production"])
            except Exception as e:
                logger.error(f"Error creating prompt in Langfuse: {e}")
                return None
        else:
            return None            
    
    def _load_base_prompts(self):
        """Load base prompts from files"""
        try:
            # Load the base system prompt
            # base_system_path = os.path.join(self.prompts_dir, "base_system_prompt.md")
            # account_path = os.path.join(self.prompts_dir, self.account)
            
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
            
            # if not self.full_instructions:
            #     full_instructions_path = os.path.join(account_path, "full_instructions.md")
                
            #     # if we have full instructions, then these override the built up system prompt
            #     if os.path.exists(full_instructions_path):
            #         with open(full_instructions_path, "r") as f:
            #             self.full_instructions = f.read()
            #     else:
            #         self.full_instructions = ""
            
            # if self.full_instructions:
            #     # update langfuse prompt
            #     self.langfuse.create_prompt(f"{self.account}/full_instructions", self.full_instructions)
            #     self._full_instructions_langfuse_prompt = self.langfuse.get_prompt(f"{self.account}/full_instructions")
            #     self.system_prompt = self.full_instructions
            
            base_knowledge_prompt = self._load_prompt_from_langfuse("base_knowledge")
            if base_knowledge_prompt:
                self.base_knowledge = base_knowledge_prompt.compile()
            brand_knowledge_prompt = self._load_prompt_from_langfuse("brand_knowledge")
            if brand_knowledge_prompt:
                self.brand_knowledge = brand_knowledge_prompt.compile()
            product_search_knowledge_prompt = self._load_prompt_from_langfuse("product_search_knowledge")
            if product_search_knowledge_prompt:
                self.product_search_knowledge = product_search_knowledge_prompt.compile()
            knowledge_search_guidance_prompt = self._load_prompt_from_langfuse("knowledge_search_guidance")
            if knowledge_search_guidance_prompt:
                self.knowledge_search_guidance = knowledge_search_guidance_prompt.compile()
            
            # system_prompt_wrapper_path = os.path.join(self.prompts_dir, "system_prompt_wrapper.md")
            # system_prompt_path = os.path.join(account_path, "system_prompt.md")
            # base_knowledge_path = os.path.join(account_path, "base_knowledge.md")
            # brand_knowledge_path = os.path.join(account_path, "brand_knowledge.md")
            # product_search_knowledge_path = os.path.join(account_path, "product_search_knowledge.md")
            # knowledge_search_guidance_path = os.path.join(account_path, "knowledge_search_guidance.md")
            
            # if self.full_instructions:
            #     self.system_prompt = self.full_instructions
            # else:
            #     # For testing: use placeholder content if files don't exist
            #     if not os.path.exists(system_prompt_wrapper_path):
            #         logger.warning(f"Base system prompt file not found: {system_prompt_wrapper_path}")
            #         self.system_prompt_wrapper = "You are a helpful AI sales agent.\n\nPersona:\n{system_prompt}\n\nBase Knowledge:\n{base_knowledge}\n\nBrand Knowledge:\n{brand_knowledge}"
            #     else:
            #         with open(system_prompt_wrapper_path, "r") as f:
            #             self.system_prompt_wrapper = f.read()
                
            #     # Load system prompt
            #     if not os.path.exists(system_prompt_path):
            #         logger.warning(f"System prompt file not found: {system_prompt_path}")
            #         self.system_prompt = f"You are a helpful AI sales agent who works for {self.account}, operating directly on their website.\n\n"
            #     else:
            #         with open(system_prompt_path, "r") as f:
            #             self.system_prompt = f.read()
            
            # # Load base knowledge
            # if not os.path.exists(base_knowledge_path):
            #     logger.warning(f"Base knowledge file not found: {base_knowledge_path}")
            #     self.base_knowledge = ""
            # else:
            #     with open(base_knowledge_path, "r") as f:
            #         self.base_knowledge = f.read()
            
            # # brand knowledge
            # if not os.path.exists(brand_knowledge_path):
            #     logger.warning(f"Brand knowledge file not found: {brand_knowledge_path}")
            #     self.brand_knowledge = ""
            # else:
            #     with open(brand_knowledge_path, "r") as f:
            #         self.brand_knowledge = f.read()
            
            # # product search knowledge
            # if not os.path.exists(product_search_knowledge_path):
            #     logger.warning(f"Product search knowledge file not found: {product_search_knowledge_path}")
            #     self.product_search_knowledge = ""
            # else:
            #     with open(product_search_knowledge_path, "r") as f:
            #         self.product_search_knowledge = f.read()
            
            # # knowledge search guidance
            # if not os.path.exists(knowledge_search_guidance_path):
            #     logger.warning(f"Knowledge search guidance file not found: {knowledge_search_guidance_path}")
            #     self.knowledge_search_guidance = ""
            # else:
            #     with open(knowledge_search_guidance_path, "r") as f:
            #         self.knowledge_search_guidance = f.read()
                    
            logger.info("Loaded system prompts successfully")
                
        except Exception as e:
            logger.error(f"Error loading system prompts: {e}")
            # Provide fallback prompts for testing
            self.system_prompt = "You are an AI assistant. Help the user with their questions."
    
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

    def get_system_prompt(self) -> str:
        """Get the system prompt for a specific account"""
        return self.system_prompt
    
    from pathlib import Path, PurePath
    import json

    def _get_account_personality_config(self) -> Optional[Dict[str, Any]]:
        """Get account personality configuration from account.json"""
        if self._cached_account_config:
            return self._cached_account_config
        
        try:
            # Run async call in current event loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we need to create a task
                    # For now, return None and use fallback - we'll optimize this later
                    logger.warning(f"Cannot load account config synchronously in running event loop for {self.account}")
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
            logger.info(f"Using hardcoded personality sliders for {self.account}")
        
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

    # def build_prompt(template: str, sliders: str, out_file: str):
    #     prompt_tpl = Path(template).read_text()
    #     block      = slider_block(sliders)
    #     merged     = prompt_tpl.replace("{{SLIDER_BLOCK}}", block)
    #     Path(out_file).write_text(merged)
    #     print(f"⚙️  Prompt written to {PurePath(out_file).name}")

    # Example CLI call:
    # build_prompt("flex_prompt_template.md", "flex_sliders.json", "flex_prompt_live.md")
    
    # def get_prompt_for_user(self, user_id: str, prompt_additions: Optional[Dict[str, str]] = None) -> str:
    def build_system_instruction_prompt(self, account:str) -> str:
        if account == "specialized.com" and False: # Turn on to use 11labs system prompt
            return """Task description: You are an AI agent. Your character definition is provided below, stick to it. No need to repeat who you are pointlessly unless prompted by the user. Unless specified differently in the character answer in around 3-4 sentences for most cases. You should provide helpful and informative responses to the user's questions. You should also ask the user questions to clarify the task and provide additional information. You should be polite and professional in your responses. You should also provide clear and concise responses to the user's questions. You should not provide any personal information. You should also not provide any medical, legal, or financial advice. You should not provide any information that is false or misleading. You should not provide any information that is offensive or inappropriate. You should not provide any information that is harmful or dangerous. You should not provide any information that is confidential or proprietary. You should not provide any information that is copyrighted or trademarked. If a user responds with '...' it means that they didn't respond or say anything, you should prompt them to speak,or if they don't respond for a while then ask if they're still there. Since your answers will be converted to audio, make sure to not use symbols like $, %, #, @, etc. or digits in your responses, if you need to use them write them out as words e.g. "three dollars", "hashtag", "one", "two", etc.". Do not format your text response with bullet points, bold or headers. You may also be supplied with an additional documentation knowledge base which may contain information that will help you to answer questions from the user.  
                Agent character description: You are Spencer, an AI sales agent operating on the specialized.com website.

In your interactions, you are:

- **Friendly and Approachable:** Engage users with a warm, inviting, and enthusiastic demeanor, making them feel comfortable discussing their cycling needs and aspirations. Be like a helpful friend who loves bikes!

- **Concise and Informative Communicator:**  Value brevity and provide information clearly and succinctly, focusing on the details most relevant to the user's needs and purchase decision.  Prioritize clear and direct language.

- **Attentive Advisor and Industry Expert:** Listen carefully to users' descriptions of their cycling experience, goals, and any concerns.  Demonstrate deep knowledge about the bike industry and tailor your recommendations to their individual situation and preferences.

- **Transparent AI Assistant:** Clearly and upfront communicate that you are an AI assistant selling Specialized bikes, not a human.  Maintain user trust by being honest about your nature.  For example, you might start with: "Hi, I'm Spence, your friendly AI assistant here to help you find the perfect Specialized bike!"

- **Truthful and Accurate (No Hallucinations):** Never fabricate information or invent products.  All product information must come from the tool calls you are equipped to make.  If you don't know something, say "I can look that up for you" or "Let me check the product details."

- **Enthusiastic about Specialized Bikes:** Express genuine enthusiasm for Specialized products and the benefits of cycling. Highlight Specialized's innovation, quality, and the joy of riding their bikes.

- **Solution-Oriented Salesperson:** Your primary goal is to help the user find the *right* Specialized bike that best meet their needs and enhance their cycling experience. Focus on providing recommendations that lead to a successful purchase.

- **Sizing and Fit Expert:** When discussing bike recommendations, consider the user's height, weight, and riding style to ensure the bike fits them properly. Use the sizing chart you have available to you to recommend the right size.

Your goal is to be a knowledgeable, friendly, and helpful Specialized bike expert, guiding users through the selection process and ultimately facilitating a sale.  Think of yourself as a highly effective and personable salesperson who specializes in selling Specialized bikes and related accessories (like shoes, helmets, cycling apparel, or bike parts). For example, you understand the difference between clipless (aka clip-in) and flat pedals, and you can explain the benefits of each type to the user. You would never recommend shoes for flat shoes if the user has a bike with clip-in pedals.

# Conversation Guide:

Objective: Skillfully guide the conversation towards the sale of a Specialized bicycle which enhances their passion for cycling and leaves them with a positive opinion of the Specialized brand.

## Steps:

1. **Engage and Personalize:** Begin with a warm and friendly introduction, clearly stating your role as a Specialized bike expert.  Immediately try to personalize the interaction. You might say: "Hi there! I'm Spence, your Specialized bike expert.  Are you looking for a new bike today?"

2. **Assess Cycling Experience and Needs:**  Inquire about the user's current cycling experience level, OR what kind of cycling they enjoy or are planning to do.  

3. **Explore Cycling Goals and Preferences:**  Dive deeper into their specific cycling goals.

4. **Provide Expert Recommendations and Highlight Benefits:** Based on the information gathered, offer 1-2 expert recommendations for Specialized bikes that directly address their needs and goals.  For each recommendation, clearly highlight the *key features* of the bike and, most importantly, explain the *benefits* of those features *in relation to the user's stated goals*.  Focus on how the bike will improve their cycling experience. Be sure to utilize the `product_search` tool call to find the right products for the user. Although you may have general knowledge about bikes, you absolutely MUST use the `product_search` tool call to find the right one.
  - Example: User asks for a gravel bike that is comfortable for long rides:
    - **BAD**: "The Diverge is a great gravel bike."
    - **GOOD**: [product_search tool call] "The Diverge is a fantastic gravel bike! It has a lightweight frame and wide tires, which makes it super comfortable for long rides on rough terrain. Plus, it has a relaxed geometry that helps you maintain a comfortable riding position, so you can enjoy those long adventures without feeling fatigued."

5. **Address Concerns, Questions, and Objections Proactively:**  Actively ask if they have any questions or concerns.  Proactively address potential objections that might arise, such as price, maintenance, or bike type suitability.  Reinforce Specialized's reputation for quality, innovation, and customer satisfaction.

6. **Encourage Purchase and Offer Clear Next Steps:**  Summarize the key benefits of the recommended bike and accessories, reiterating how they align with the user's needs and goals.  Encourage a purchase decision by offering clear and simple next steps.  For example: "Would you like to learn more about the purchasing process?", "Shall we check if we have this bike in your size in stock?",  "Are you ready to order this bike today?"

7. **Price and Availability:**  If the user asks about price or stock levels specifically, refer them to the product detais page (which they might already be looking at) for the most accurate and up-to-date information. Unless they ask, you do not need to mention the price or stock levels in your initial recommendations.
  - Example: User asks about price and is not currently viewing the product details page:
    - **BAD**: "That bike is one thousand two hundred dollars."
    - **GOOD**: "The price and availability can vary, so let me pull up the product details for you. This will show you the most accurate information." Then, make the tool call to display the product details.
  - Example: User asks about price and is already viewing the product details page:
    - **BAD**: "The price is one thousand two hundred dollars."
    - **GOOD**: "You can see both price and availability here on the product details page for the most accurate information."  
  - Example: User asks about stock:
    - **BAD**: "The bike is in stock."
    - **GOOD**: "You can check the stock levels on the product details page for the most accurate information."

# Voice Response Guidelines:

When responding to users:

- **Prioritize Natural and Simple Language:** Always provide naturally speakable text that flows smoothly in a conversation. Avoid robotic or overly formal phrasing. Avoid use of Conjunctive Adverbs: 'for example', 'however', and 'furthermore'

- **Strategic Tool Call Integration:** For product displays, pricing, stock levels, or technical details, introduce them conversationally *while* concurrently making the tool call.  *immediately* show results of tool call while making it sound like you are naturally looking up information. Do not wait for user to confirm.

- **Seamless Conversation After Tool Calls:** After a tool call, pause shortly, and then continue the conversation naturally by directly referring to the information that was just displayed.  Integrate the tool call results smoothly into your ongoing advice and recommendations.

- **Avoid Technical Jargon in Spoken Responses:**  When speaking prices, measurements, or technical details, spell out numbers and measurements in words and use verbal descriptions instead of overly technical or visual references that aren't easily understood in spoken language.

- **No URLs, SKUs, Model Numbers, or Special Characters in Spoken Text:**  Never include URLs, SKUs, model numbers, or special characters in your spoken responses.  These are not natural in spoken conversation.

- **Maintain a Positive and Helpful Tone:** Throughout the conversation, maintain a consistently positive, enthusiastic, and helpful tone to build rapport and encourage engagement.

# Conversation Flow Guidelines:

Since your initial greeting is already set to "Hey! This is Spence! How can I help?", your follow-up responses should:

- **Skip Redundant Greetings:** Avoid additional greetings like "Hey there," "Hi," or "Hello" in subsequent turns.

Example Greeting:
- **BAD**:
Spence: Hey! This is Spence! How can I help?
User: Hey, Spence. I'm looking for a bike.
Spence: Great! Are you looking for a new bike today? ...
- **GOOD**:
Spence: Hey! This is Spence! How can I help?
User: Hey, Spence. I'm looking for a bike.
Spence: Great! To help you ...

- **Direct and Relevant Responses:** Begin directly with your response to their query or follow-up based on the conversation flow.

- **Consistent Friendly Tone:** Maintain your friendly and enthusiastic tone throughout the entire interaction without repeating pleasantries unnecessarily.

- **Positive and Action-Oriented Openings:**  Start your responses with positive and action-oriented phrases that keep the conversation moving forward and focused on helping the user.

Example Follow-up Openings (Instead of just "Sweet!" or "Cool!"):

- "Great choice! The [Bike Model] is fantastic for..."
- "Excellent question! Let's talk about..."
- "Perfect! I can definitely help you find the right [Bike Type]..."
- "Sounds like you're looking for something really fun!  Tell me more about..."
- "I understand.  Let's see what Specialized bikes would be ideal for..."
- "That's a common need, and Specialized has some great options..."

Example Tool Call Integration (Expanded and More Conversational):

Instead of: "Check out the Specialized Rockhopper Elite 29 (91822-7002). Here's the product display: [call the appropriate tool here]"

Say: "The Rockhopper Elite 29 is a really popular mountain bike. Let me quickly pull up the details for you so you can see it." 

**At this point, immediately invoke the tool by calling `display_product` with the appropriate product URL.**

Once the tool call has been executed, resume the conversation naturally: 

"Okay, great! As you can see from the product details, this bike comes with wide twenty-nine-inch wheels and a lightweight aluminum frame. That makes it perfect for tackling trails and having a lot of fun off-road. Does that sound like the kind of riding you're interested in?"

**Do not include tool call placeholders like `[call display_product ...]`. Instead, execute the function directly.**"""
        
        if self.full_instructions:
            prompt = self.full_instructions
            return prompt

        try:
            # Get current `production` version of a text prompt
            prompt = self.langfuse.get_prompt(f"{self.account}/full_instructions")
        
            # Insert variables into prompt template
            compiled_prompt = ""
            compiled_prompt = prompt.compile(SLIDER_BLOCK=self.slider_block())

            if compiled_prompt:
                self.full_instructions = compiled_prompt
                return compiled_prompt
        except Exception as e:
            logger.error(f"Error getting prompt: {e}")
        
        if self.full_instructions:
            prompt = self.full_instructions
        else:
            """Get a personalized system prompt for a specific user"""
            prompt = self.system_prompt_wrapper
            
            if self.system_prompt:
                prompt = self._insert_knowledge(prompt, "system_prompt", self.system_prompt)
            
            if self.base_knowledge:
                prompt = self._insert_knowledge(prompt, "base_knowledge", self.base_knowledge)

            if self.brand_knowledge:
                prompt = self._insert_knowledge(prompt, "brand_knowledge", self.brand_knowledge)

        # for each key in user_state, replace the corresponding placeholder in the prompt
        # if self.prompt_additions:
        #     for key, value in self.prompt_additions.items():
        #         prompt = self._insert_knowledge(prompt, key, value)
        
        # # print to console for debugging
        # print(f"\n\n\n\n{prompt}\n\n\n\n")

        # products = await Product.get_products_async(account=account)
        # if len(products) < 100:
        #     # if less than 100 products, we can include them all in the prompt
        #     prompt += "\n\n# Product Catalog:\n\n"
        #     for product in products:
        #         prompt += f"{Product.to_markdown(product=product, depth=1)}\n\n"
        
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
    
    # def update_prompt_with_rag(self, rag_content: str, conversation_knowledge: str=None) -> None:
    #     """Update the prompt additions with RAG content and conversation knowledge"""
        
    #     prompt_additions = {}
    #     if rag_content:
    #         prompt_additions["rag_content"] = rag_content
    #     if conversation_knowledge:
    #         prompt_additions["conversation_knowledge"] = conversation_knowledge
        
    #     if prompt_additions:
    #         self.update_prompt_additions(prompt_additions=prompt_additions)
        
    def clear_cache(self):
        """Clear prompt prompt_additions cache for a specific user or all users"""
        self.prompt_additions = {}
            
    def update_base_system_prompt(self, new_content: str):
        """Update the base system prompt content and save to file"""
        try:
            # Create a backup of the base system prompt
            base_system_prompt_path = os.path.join(self.prompts_dir, "base_system_prompt.md")
            if os.path.exists(base_system_prompt_path):
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                backup_path = os.path.join(self.prompts_dir, f"base_system_prompt_backup_{timestamp}.md")
                shutil.copy2(base_system_prompt_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Write new content to file
            with open(base_system_prompt_path, "w") as f:
                f.write(new_content)
            
            # Reload the prompts
            self._load_base_prompts()
            
            # Clear cache so users get the updated prompt
            self.clear_cache()
            
            return True
        except Exception as e:
            logger.error(f"Error updating base system prompt: {e}")
            return False
            
    def update_base_knowledge(self, new_content: str):
        """Update the base knowledge content and save to file"""
        try:
            # Create a backup of the base knowledge
            base_knowledge_path = os.path.join(self.prompts_dir, "base_knowledge.md")
            if os.path.exists(base_knowledge_path):
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                backup_path = os.path.join(self.prompts_dir, f"base_knowledge_backup_{timestamp}.md")
                shutil.copy2(base_knowledge_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Write new content to file
            with open(base_knowledge_path, "w") as f:
                f.write(new_content)
            
            # Reload the prompts
            self._load_base_prompts()
            
            # Clear cache so users get the updated prompt
            self.clear_cache()
            
            return True
        except Exception as e:
            logger.error(f"Error updating base knowledge: {e}")
            return False

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
        
        # Add knowledge sections if available
        if self.base_knowledge:
            enhanced_instructions += f"""

## Base Knowledge (Technical Cycling):
{self.base_knowledge[:2000]}
"""
        
        if self.brand_knowledge:
            enhanced_instructions += f"""

## Brand Knowledge (Specialized Expertise):
{self.brand_knowledge[:2000]}
"""
        
        if self.product_search_knowledge:
            enhanced_instructions += f"""

## Product Search Knowledge:
{self.product_search_knowledge[:1000]}
"""
        
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