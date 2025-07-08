import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from openai import OpenAI
from openai.types.responses.response_input_param import Message
from livekit.agents import llm

if __name__ == "__main__":
    # add parent directory to path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    # add parent's parent directory to path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from liddy_voice.config_service import ConfigService

# Import your existing services
# These would need to be converted or adapted from TypeScript
from liddy_voice.llm_service import LlmService
from liddy_voice.model import BasicChatMessage
from liddy_voice.prompt_manager import AccountPromptManager
from liddy_voice.conversation.model import ConversationDetails


logger = logging.getLogger("conversation-evaluator")



class ConversationScrubber:
    def __init__(self):
        pass

    def load_conversation(self, account: str, conversation_id: str) -> Optional[ConversationDetails]:
        conversation_file = f"local/conversation_storage/{account}/{conversation_id}.json"
        # search for the conversation file in the conversation_storage directory and its subdirectories
        if not os.path.exists('local/conversation_storage'):
            return None
        for root, dirs, files in os.walk('local/conversation_storage'):
            for file in files:
                if file == f"{conversation_id}.json":
                    conversation_file = os.path.join(root, file)
                    break

        if not os.path.exists(conversation_file):
            return None
            
        try:
            with open(conversation_file, 'r') as f:
                conversation_data = json.load(f)

            if not conversation_data:
                return None
            
            transcript = [];
            if 'messages' in conversation_data:
                for message in conversation_data['messages']:
                    if 'role' in message and 'content' in message:
                        transcript.append(BasicChatMessage(role=message['role'], content=message['content'], timestamp=message.get('timestamp', 0)))

            conversation = ConversationDetails(
                conversation_id=conversation_id,
                transcript=transcript,
                agent_id=conversation_data['agent_id'] if 'agent_id' in conversation_data else '',
                start_time_unix_secs=conversation_data['start_time'] if 'start_time' in conversation_data else 0,
                call_duration_secs=conversation_data['metrics']['total_duration'] if 'metrics' in conversation_data and 'total_duration' in conversation_data['metrics'] else 0,
                message_count=len(conversation_data['messages']) if 'messages' in conversation_data else 0,
                status=conversation_data['status'] if 'status' in conversation_data else '',
                call_successful=conversation_data['call_successful'] if 'call_successful' in conversation_data else False,
                agent_name=conversation_data['agent_name'] if 'agent_name' in conversation_data else ''
            )
            return conversation
        except Exception as e:
            print(f'Error loading conversation: {e}')
            return None
      

    async def scrub_conversation(self, account:str, conversation: ConversationDetails, force: bool = False):
        print(f'Scrubbing conversation: {conversation}')
        
        # Check if conversation has already been scrubbed
        conversation_file = f"local/conversation_processed/{account}/scrubbed/{conversation.conversation_id}.json"
        if not force and os.path.exists(conversation_file):
            try:
                with open(conversation_file, 'r') as f:
                    scrubbed_conversation = json.load(f)
                print(f'Conversation already scrubbed: {scrubbed_conversation}')
                return scrubbed_conversation
            except Exception as e:
                print(f'Error reading conversation scrub: {e}')
        
        user_prompt = f"{ConversationScrubber.SCRUBBER_USER_PROMPT}"
        user_prompt = user_prompt.replace('$TRANSCRIPT', '\n'.join([m.content for m in conversation.transcript]))
        
        try:
            # model = "o3"
            model = "o4-mini"
            # model = "gpt-4.1-nano"
            # model = "claude-3-7-sonnet-20250219"
            response = await self.create_completion(
                model=model,
                messages=[
                    {"role": "system", "content": self.build_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ]
            )
            # client = OpenAI(
            #   api_key=os.getenv("OPENAI_API_KEY", "sk-lilly-dev-CNn9IbrmNAOvaskOOvWET3BlbkFJfGk7YH7HNFVakZL0oLAu"),
            # )
            # response = client.responses.create(
            #     model="o3",
            #     # model="gpt-4.1",
            #     # model="gpt-4.1-nano",
            #     # model="gpt-4.1-mini",
            #     input=[
            #       Message(role="system", content=self.build_system_prompt()),
            #       Message(role="user", content=user_prompt)
            #     ]
            # )

            print(response)
            scrubbed_response = LlmService.parse_json_response(response)
            
            print(f'Scrubbed conversation: {scrubbed_response}')
            
            # store the scrubbed conversation
            scrubbed_conversation_file = f"local/conversation_processed/{account}/scrubbed/{conversation.conversation_id}.json"
            os.makedirs(f"local/conversation_processed/{account}/scrubbed", exist_ok=True)
            with open(scrubbed_conversation_file, 'w') as f:
                json.dump(scrubbed_response, f, indent=2)
            
            return scrubbed_response
        except Exception as e:
            print(f'Error scrubbing conversation: {e}')
            return None


    async def create_completion(self, model: str, messages: List[BasicChatMessage], max_tokens: int=1024*8) -> str:
        if model.startswith("claude"):
            return await self.create_anthropic(model, messages, max_tokens)
        elif model.startswith("gpt") or model.startswith("o"):
            return await self.create_openai(model, messages, max_tokens)
        else:
            raise ValueError(f"Unsupported model: {model}")

    async def create_openai(self, model: str, messages: List[BasicChatMessage], max_tokens: int=1024*8) -> str:
        # Initialize OpenAI client
        openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "sk-lilly-dev-CNn9IbrmNAOvaskOOvWET3BlbkFJfGk7YH7HNFVakZL0oLAu"),
        )
        
        # response = openai_client.responses.create(
        #     model=model,
        #     reasoning={"effort": "medium"},
        #     input=messages,
        #     # temperature=0.7,
        #     # max_tokens=4096,
        #     # top_p=1.0,
        #     # frequency_penalty=0.0,
        #     # presence_penalty=0.0,
        #     # stop=None
        # )
        
        # return response.output_text
        if model.startswith("o"):
            completion = openai_client.chat.completions.create(
                model=model,
                # model="o3",
                # model="gpt-4.1",
                # model="gpt-4.1-nano",
                # model="gpt-4.1-mini",
                reasoning_effort="medium",
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_completion_tokens=max_tokens
            )
        else:
            completion = openai_client.chat.completions.create(
                model=model if model else "gpt-4.1",
                # model="o3",
                # model="gpt-4.1",
                # model="gpt-4.1-nano",
                # model="gpt-4.1-mini",
                reasoning_effort="medium",
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens
            )

        # print(completion.choices[0].message.content)
        # result = LlmService.parse_json_response(completion.choices[0].message.content)
        return completion.choices[0].message.content
      
    async def create_anthropic(self, model: str, messages: List[BasicChatMessage], max_tokens: int=1024*8) -> str:
        client = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
        )
        
        # pull out the system prompt, if it exists
        system_prompt = next((m for m in messages if m.role == 'system'), None)
        if system_prompt:
            messages.remove(system_prompt)

        async with client.messages.stream(
            system=system_prompt.content if system_prompt else None,          
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            model=model if model else "claude-3-5-sonnet-latest",
            thinking={
                "budget_tokens": min(max_tokens / 2, 1024 * 8),
                "type": "enabled",
            },
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
            print()

        message = await stream.get_final_message()
        print(message.to_json())

        # message = client.messages.create(
        #     system=system_prompt['content'] if system_prompt else None,          
        #     max_tokens=4096 * 2,
        #     messages=messages,
        #     model=model if model else "claude-3-5-sonnet-latest",
        # )
        # print(message.content)
        
        # content is the first message with type == "text"
        content = [m for m in message.content if m.type == "text"]
        if len(content) == 0:
            raise ValueError("No text content found in the response.")
        content = content[0]
        
        
        return content.text if content else ""


    SCRUBBER_SYSTEM_PROMPT = f"""You are an expert data analyst. Your job is to extract any and all personal preferences, data, and identifying information from the following conversation with the user as well as scrub the transcript of the conversation.

Your level of scrubbing should be very high. You should not leave any personal identification information in the transcript. You should also not leave any personal preferences in the transcript. The user should not be able to identify themselves or their preferences from the transcript. This includes scrubbing any system or assistant messages that may contain personal identification information or personal preferences.

Personal identification information includes, but is not limited to:
- Name
- Email
- Phone number
- Address
- Height
- Weight
- Measurements
- Age
- Date of birth
- Social security number

Personal data includes, but is not limited to:
- Product preferences
- Products owned
- Product usage

Personal preferences include, but are not limited to:
- Favorite things like color, food, etc.
- Favorite brands
- Type of products owned or used
- Type of products interested in
- Hobbies
- Interests
- Activities
- Lifestyle

Beyond extracting user data, you should also extract any brand data that is relevant to the conversation. The goal is to help the brand identify product trends, preferences, and other intangible information. This includes, but is not limited to:
- Product preferences
- Product usage
- Product ownership
- Product interests
- Product recommendations
- Product features
- Product reviews
"""

    SCRUBBER_USER_PROMPT = """Please extract any and all personal preferences and information from the following conversation with the user. Then, please provide a scrubbed transcript of the same conversation with any personal identification information scrubbed out:
\"\"\"
$TRANSCRIPT
\"\"\"

Respond in the following JSON format:
{
  "extracted_data": {
    "user": {
      "personal_identification_information": { ...key-value pairs... },
      "personal_preferences": { ...key-value pairs... },
      "personal_data": { ...key-value pairs... }
    },
    "brand": {
      "product_preferences": { ...key-value pairs... },
      "product_usage": { ...key-value pairs... },
      "product_ownership": { ...key-value pairs... },
      "product_interests": { ...key-value pairs... }
    }
  },
  "scrubbed_transcript": [
    // The transcript of the conversation with any personal identification information scrubbed out
    {
      "role": string,
      "content": string,
      "timestamp": number
    }
  ]
}
"""

if __name__ == "__main__":
    # fetch all conversation ids from the conversation_storage directory and its subdirectories
    conversation_ids = []
    for root, dirs, files in os.walk('local/conversation_storage/specialized.com'):
        for file in files:
            if file.endswith('.json'):
                conversation_ids.append(file[:-5])
    # remove duplicates
    conversation_ids = list(set(conversation_ids))
    
    # load the system prompt from the prompt manager
    prompt_manager = AccountPromptManager(account="specialized.com")
    system_prompt = prompt_manager.build_system_instruction_prompt(account="specialized.com")
    scrubber = ConversationScrubber()
    
    for conversation_id in conversation_ids:
        conversation = scrubber.load_conversation("specialized.com", conversation_id)
        if conversation:
            asyncio.run(scrubber.scrub_conversation(account="specialized.com", conversation=conversation, force=False))
            print(f"Conversation {conversation_id} evaluated successfully.")
        else:
            print(f"Conversation {conversation_id} not found.")

    print("Evaluation completed.")