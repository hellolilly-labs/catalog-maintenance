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

from liddy_voice.spence.config_service import ConfigService
from liddy_voice.spence.conversation.model import ConversationDetails

# Import your existing services
# These would need to be converted or adapted from TypeScript
from liddy_voice.spence.llm_service import LlmService
from liddy_voice.spence.model import BasicChatMessage
from liddy_voice.spence.prompt_manager import PromptManager


logger = logging.getLogger("conversation-evaluator")


class SystemPromptEvaluation(TypedDict):
    is_changed: bool
    reformatted_system_prompt: Optional[str]
    justification: str
    modifications: str

class EvaluationScore(TypedDict):
    score: int
    explanation: str

class EvaluationDetails(TypedDict):
    relevance: EvaluationScore
    accuracy: EvaluationScore
    helpfulness: EvaluationScore
    conciseness: EvaluationScore
    overall: EvaluationScore
    suggestions: List[str]
    recommended_features: List[str]

class ConversationEvaluation(TypedDict):
    conversation_id: str
    agent_id: str
    evaluation: EvaluationDetails



class ConversationEvaluator:
    def __init__(self):
        pass

    def build_system_prompt(self) -> str:
        system_prompt = f"{ConversationEvaluator.EVALUATOR_SYSTEM_PROMPT}"
        system_prompt = system_prompt.replace('$GUIDE', ConversationEvaluator.SALES_GUIDE)
        return system_prompt
    
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
      

    async def evaluate_conversation(self, account: str, conversation: ConversationDetails, force_reevaluation: bool = False) -> Optional[ConversationEvaluation]:
        print(f'Evaluating conversation: {conversation}')
        user_prompt = f"{ConversationEvaluator.EVALUATOR_USER_PROMPT}"
        user_prompt = user_prompt.replace('$TRANSCRIPT', '\n'.join([m.content for m in conversation.transcript]))
        print(user_prompt)
        
        # Check if conversation has already been evaluated
        conversation_file = f"local/conversation_processed/{account}/evaluation/{conversation.conversation_id}.json"
        if not force_reevaluation and os.path.exists(conversation_file):
            try:
                with open(conversation_file, 'r') as f:
                    evaluation = json.load(f)
                print(f'Conversation already evaluated: {evaluation}')
                return evaluation
            except Exception as e:
                print(f'Error reading conversation evaluation: {e}')
        
        try:
            # Initialize the Anthropic service similar to your existing Python services
            # llm_service = LlmService.fetch_model_service_from_model('claude-3-7-sonnet-20250219')
            # llm_service = LlmService.fetch_model_service_from_model('gpt-4.1')
            
            client = OpenAI(
              api_key=os.getenv("OPENAI_API_KEY", "sk-lilly-dev-CNn9IbrmNAOvaskOOvWET3BlbkFJfGk7YH7HNFVakZL0oLAu"),
            )

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

            # print(response.output_text)
            # conversation_evaluation = LlmService.parse_json_response(response.output_text)
            completion = client.chat.completions.create(
                model="o4-mini",
                # model="o3",
                # model="gpt-4.1",
                # model="gpt-4.1-nano",
                # model="gpt-4.1-mini",
                reasoning_effort="medium",
                messages=[
                  Message(role="developer", content=self.build_system_prompt()),
                  Message(role="user", content=user_prompt)
                ]
            )

            print(completion.choices[0].message.content)
            conversation_evaluation = LlmService.parse_json_response(completion.choices[0].message.content)
            
            # # Create chat context
            # chat_ctx = llm.ChatContext([
            #     llm.ChatMessage(role="system", content=[self.build_system_prompt()]),
            #     llm.ChatMessage(role="user", content=[user_prompt])
            # ])
            
            # # Get response
            # response = await LlmService.chat_wrapper(
            #     llm_service=llm_service,
            #     chat_ctx=chat_ctx,
            #     # max_tokens=4096,
            #     # temperature=0.7,
            #     # thinking_budget=4096
            # )
            
            # # Parse JSON response
            # conversation_evaluation = LlmService.parse_json_response(response)
            print(f'Conversation evaluation: {conversation_evaluation}')
            
            # Save to file
            os.makedirs(f"local/conversation_processed/{account}/evaluation", exist_ok=True)
            with open(conversation_file, 'w') as f:
                json.dump(conversation_evaluation, f, indent=2)
                
            return conversation_evaluation
        except Exception as e:
            print(f'Error evaluating conversation: {e}')
            return None

    async def evaluate_conversations(self, conversations: List[ConversationDetails]):
        for conversation in conversations:
            await self.evaluate_conversation(conversation)

    async def load_all_evaluations(self) -> List[ConversationEvaluation]:
        evaluations = []
        if not os.path.exists('conversations'):
            return evaluations
            
        for file in os.listdir('conversations'):
            try:
                with open(f'conversation_evaluation/{file}', 'r') as f:
                    evaluation = json.load(f)
                    evaluations.append(evaluation)
            except Exception as e:
                print(f'Error reading conversation evaluation: {e}')
        
        return evaluations

    async def summarize_conversation_evaluations(self, evaluations: List[ConversationEvaluation], 
                                               current_system_prompt: str, retry_count: int = 0) -> SystemPromptEvaluation:
        system_prompt = f"{ConversationEvaluator.EVALUATOR_SYSTEM_PROMPT}"
        system_prompt = system_prompt.replace('$GUIDE', system_prompt)
        
        # Check for cached prompt
        if os.path.exists('new_system_prompt.json'):
            try:
                with open('new_system_prompt.json', 'r') as f:
                    new_system_prompt = json.load(f)
                system_prompt_evaluation = SystemPromptEvaluation(
                    is_changed=new_system_prompt['is_changed'] if 'is_changed' in new_system_prompt else False,
                    reformatted_system_prompt=new_system_prompt['reformatted_system_prompt'] if 'reformatted_system_prompt' in new_system_prompt else SystemError,
                    justification=new_system_prompt['justification'] if 'justification' in new_system_prompt else '',
                    modifications=new_system_prompt['modifications'] if 'modifications' in new_system_prompt else ''
                )
                return system_prompt_evaluation
            except Exception as e:
                print(f'Error reading new system prompt: {e}')
        
        try:
            user_prompt = f"""Modify the following system prompt given the feedback from the evaluations. Remember, we do not want drastic changes to the system prompt, but we do want to make it more effective. In fact, no changes at all is the ultimate goal. Another option is to simply reformat the system prompt to promote better adherence to its directives. The system prompt is as follows:
\"\"\"
{current_system_prompt}
\"\"\"

"""
            
            for evaluation in evaluations:
                user_prompt += f"""Conversation ID: {evaluation['conversation_id']}
Suggestions: {' '.join(evaluation['evaluation']['suggestions'])}

"""
            
            user_prompt += """Respond in the following JSON format:
{
  "is_changed": boolean,
  "reformatted_system_prompt": string,
  "justification": string,
  "modifications": string
}"""

            # Initialize the Anthropic service
            # llm_service = LlmService.fetch_model_service_from_model('claude-3-7-sonnet-20250219')
            
            # Create chat context
            chat_ctx_messages = [
                BasicChatMessage(role="system", content=system_prompt),
                BasicChatMessage(role="user", content=user_prompt)
            ]
            
            response = await self.create_completion(
                # model="claude-3-7-sonnet-20250219",
                model="o4-mini",
                messages=chat_ctx_messages,
                max_tokens=8192 * 4,
            )
            
            # # Get response
            # response = await LlmService.chat_wrapper(
            #     llm_service=llm_service,
            #     chat_ctx=chat_ctx,
            #     max_tokens=8192,
            #     temperature=0.7,
            #     thinking_budget=8192
            # )
            
            # Parse JSON response
            prompt_response = LlmService.parse_json_response(response)
            print(f'Summarized conversation evaluations: {prompt_response}')
            
            # Save to file
            with open('new_system_prompt.json', 'w') as f:
                json.dump(prompt_response, f, indent=2)
            
            system_prompt_evaluation = SystemPromptEvaluation(
                is_changed=prompt_response['is_changed'],
                reformatted_system_prompt=prompt_response['reformatted_system_prompt'],
                justification=prompt_response['justification'],
                modifications=prompt_response['modifications']
            )  

            return system_prompt_evaluation
        except Exception as e:
            if retry_count < 3:
                return await self.summarize_conversation_evaluations(evaluations, current_system_prompt, retry_count + 1)
            print(f'Error summarizing conversation evaluations: {e}')
            return None

    async def rerun_conversation(self, conversation: ConversationDetails, system_prompt: str) -> Optional[ConversationEvaluation]:
        print(f'Rerunning conversation: {conversation}')
                
        # Format original transcript
        original_transcript = '\n'.join([f"{m.role}: {m.message}" for m in conversation.transcript])
        
        # Find first messages
        first_assistant_message = next((m for m in conversation.transcript if m.role == 'agent'), None)
        first_user_message = next((m for m in conversation.transcript if m.role == 'user'), None)
        
        # Initialize conversation with first messages
        current_conversation_messages: List[BasicChatMessage] = [BasicChatMessage(role='user', content='hello')]
        if first_assistant_message:
            current_conversation_messages.append(
                BasicChatMessage(role='assistant', content=first_assistant_message.message)
            )
        else:
            current_conversation_messages.append(
                BasicChatMessage(role='assistant', content='Hi! How can I help?')
            )
        
        user_system_prompt = """You are attempting to analyze the effectiveness of a new system prompt on a previous conversation. Given the transcript from a previous conversation, please provide a response that is as close to the original conversation as possible given the current state of this conversation. Give the user response only so that it can be inserted into the conversation."""
        
        # Simulate the conversation
        while True:
            # Create context for user's next response
            user_message = f"""Please provide the next user message in the conversation. When the conversation should end, reply simply with the response "STOP". The original conversation transcript is as follows:
\"\"\"
{original_transcript}
\"\"\"

The current conversation is as follows:
\"\"\"
{''.join([f"{m.role}: {m.content}\n" for m in current_conversation_messages])}
\"\"\"
"""
            
            # Get next user message
            user_ctx_messages = [
                BasicChatMessage(role="system", content=user_system_prompt),
                BasicChatMessage(role="user", content=user_message)
            ]
            
            next_user_prompt = await self.create_completion(
                model="claude-3-7-sonnet-20250219",
                messages=user_ctx_messages
            )
            # next_user_prompt = await LlmService.chat_wrapper(
            #     llm_service=anthropic_service,
            #     chat_ctx=user_ctx,
            #     max_tokens=2048,
            #     temperature=0.7
            # )
            
            if next_user_prompt.strip() == "STOP":
                break
                
            # Add user message to conversation
            current_conversation_messages.append(
                BasicChatMessage(role='user', content=next_user_prompt),
            )
            
            # Get agent response
            agent_ctx_messages = [
                BasicChatMessage(role="system", content=system_prompt),
                *[BasicChatMessage(role=m.role, content=[m.message]) for m in current_conversation_messages]
            ]
            # agent_ctx = llm.ChatContext([
            #     llm.ChatMessage(role="system", content=[system_prompt]),
            #     *[llm.ChatMessage(role=m.role, content=[m.message]) for m in current_conversation_messages]
            # ])
            
            next_assistant_message = await self.create_completion(
                model="gpt-4.1",
                messages=agent_ctx_messages
            )
            # next_assistant_message = await LlmService.chat_wrapper(
            #     llm_service=gemini_service,
            #     chat_ctx=agent_ctx, 
            #     max_tokens=2048,
            #     temperature=0.7
            # )
            
            # Add assistant message to conversation
            current_conversation_messages.append(
                # llm.ChatMessage(role='assistant', message=[next_assistant_message], time_in_call_secs=0)
                BasicChatMessage(role='assistant', content=next_assistant_message)
            )
        
        # Evaluate the simulated conversation
        evaluation = await self.evaluate_conversation(
            ConversationDetails(
                conversation_id=f"{conversation.conversation_id}_rerun",
                transcript=current_conversation_messages,
                agent_id=conversation.agent_id,
                start_time_unix_secs=0,
                call_duration_secs=0,
                message_count=len(current_conversation_messages),
                status="",
                call_successful=conversation.call_successful,
                agent_name=conversation.agent_name
            ),
            True
        )
        
        print(f'Rerun conversation evaluation: {evaluation}')
        return evaluation


    async def run_system_prompt_evaluation(self, account: str, system_prompt: str, conversations: List[ConversationDetails]):
        # load the conversations
        # loader = ConversationLoader(account=account)
        # conversations = await loader.loadSavedConversations()
        # print(conversations)
        # filter for conversations that have more than 25 messages
        longConversations = [
            conversation for conversation in conversations if conversation.message_count > 25
        ]
        print(longConversations)

        # evaluate the conversation
        evaluations = []
        evaluator = ConversationEvaluator()
        for conversation in longConversations:
            try:
                evaluation = await evaluator.evaluate_conversation(account=account, conversation=conversation)
                print(evaluation)
                evaluations.append(evaluation)
            except Exception as error:
                print(f'Error evaluating conversation: {error}')


        try:
            new_system_prompt: SystemPromptEvaluation = await evaluator.summarize_conversation_evaluations(current_system_prompt=system_prompt, evaluations=evaluations)
            print(new_system_prompt)

            if new_system_prompt.get('reformatted_system_prompt') and new_system_prompt.get('is_changed'):
              # now rerun the conversations with the new system prompt
                for conversation in longConversations:
                    try:
                        evaluation = await evaluator.rerun_conversation(conversation=conversation, system_prompt=new_system_prompt.get('reformatted_system_prompt') if new_system_prompt.get('is_changed') else system_prompt)
                        print(evaluation)
                        evaluations.push(evaluation)
                    except Exception as error:
                        print(f'Error rerunning conversation: {error}')
            else:
                print('No new system prompt');
        except Exception as error:
            print(f'Error running system prompt evaluation: {error}')

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

    # Define static class properties with prompts
    SALES_GUIDE = """# The Ultimate Guide to Training Sales Agents for Considered Purchases

## Core Philosophy
The best sales agents for considered purchases aren't "closers"—they're trusted advisors who create value throughout a complex decision journey. Train your agents to embrace this fundamental truth: meaningful success comes from helping customers make confident decisions that genuinely serve their interests.

## Essential Training Elements
1. Customer Psychology Understanding

Train agents to recognize the emotional weight of major purchases—fear, uncertainty, and desire for validation
Teach them to identify and address the "job to be done" beyond surface-level features
Help them understand decision-making biases that affect high-stakes purchases

2. Consultative Selling Approach

Focus on asking powerful questions rather than presenting solutions prematurely
Teach active listening techniques that uncover unstated needs and concerns
Develop skills in translating technical features into meaningful outcomes

3. Trust Building Mastery

Emphasize radical transparency—acknowledging limitations builds more trust than exaggerating strengths
Train agents to demonstrate expertise without intimidating customers
Develop authentic personal connection skills without appearing manipulative

4. Knowledge Architecture

Ensure deep product knowledge is complemented by competitive landscape understanding
Train agents to discuss alternatives honestly, including when your solution isn't ideal
Develop frameworks for explaining complex information in accessible ways

5. Process Management

Teach patience with longer sales cycles—rushing creates pressure that kills trust
Develop clear next-step strategies that maintain momentum without pushiness
Train on documentation practices that prevent details from falling through cracks

6. Objection Navigation

Reframe objection handling as collaborative problem-solving
Teach validation techniques before addressing concerns
Develop skills for distinguishing between real blockers and information requests

7. Value Demonstration

Train agents to create personalized ROI models relevant to each customer's situation
Develop storytelling skills that make abstract benefits concrete through relevant examples
Teach appropriate use of social proof without relying on it exclusively

## Implementation Guidance

Role-play extensively with realistic scenarios that include unexpected challenges
Provide continuous coaching, not just initial training
Develop metrics that reward quality relationships and optimal decisions, not just closed deals
Create systems for capturing and sharing customer insights across the organization

Remember: In considered purchases, the agent who helps customers make the best decision (even when that occasionally means not buying) creates the greatest long-term value for all parties."""

    EVALUATOR_SYSTEM_PROMPT = f"""You are an expert sales coach evaluating AI chat conversations. The following is the guide you use when training your sales agents:

\"\"\"
$GUIDE
\"\"\"


A few things to keep in mind when evaluating the conversation:
- This is a voice conversation (as opposed to a text chat), so the AI agent's responses should be appropriate for a voice conversation. For example, the agent should never reference a URL or a link such as "https://www.example.com". Also, conversational elements such as "um" and "uh" are acceptable in voice conversations but should be used sparingly. Sometimes these elements can make the conversation sound more natural, but they should not be overused. Slang and contractions are also acceptable in voice conversations.
- Availability of products is generally dependent on size. However, the AI agent does not currently have the ability to check stock or availability.
- Asking too many questions at once can be overwhelming for the user.
- The AI agent does not currently have the ability to do anything other than show products.
- It tends to be that features sell the product more than the price. However, budget is a concern for many users -- the AI agent should suss out the importance of budget to the user.
- The AI agent does not have access to or information about upcoming sales or promotions.
- If the user cuts off the agent, it may appear that the agent has stopped mid-sentence. This is actually desired behavior as the agent should be able to handle interruptions gracefully.
- The AI sales agents are able to navigate around the site on behalf of the user, so there will likely be messages from the AI agent that reference pulling up information or navigating to different parts of the site. The AI agent may also reference things on the page that the user is looking at, which can be very helpful (assuming the AI agent is accurate).  Please evaluate the conversation based on the user's query and the AI agent's responses.
- Currently, the AI agent is only able to show one product at a time.

The AI agent should be helpful, accurate, and relevant to the user's query. The driving motto is: is it true, is it kind, is it helpful?
"""

    EVALUATOR_USER_PROMPT = """Please evaluate the following conversation with the user. Pay attention to the "system" prompt that sets up the conversation:
\"\"\"
$TRANSCRIPT
\"\"\"

Evaluate this response based on:
1. Relevance to the user query (1-10)
2. Accuracy of information (1-10)
3. Helpfulness for the specific use case (1-10)
4. Conciseness and clarity (1-10)

For each criterion, provide a score and a brief explanation. Then provide an overall score (1-10) with a summary explanation.

Format your evaluation as:
Relevance: [score] - [brief explanation]
Accuracy: [score] - [brief explanation]
Helpfulness: [score] - [brief explanation]
Conciseness: [score] - [brief explanation]
OVERALL: [score] - [summary explanation]

Also, please provide detailed suggestions for improvement. Also, please provide a list of recommended features that would improve the conversation (if any). Recommended features would be anything that the agent is not currently able to do that would be helpful for the user.

Respond in the following JSON format:
{
  "conversation_id": string,
  "agent_id": string,
  "evaluation": {
    "relevance": {
      "score": number,
      "explanation": string
    },
    "accuracy": {
      "score": number,
      "explanation": string
    },
    "helpfulness": {
      "score": number,
      "explanation": string
    },
    "conciseness": {
      "score": number,
      "explanation": string
    },
    "overall": {
      "score": number,
      "explanation": string
    },
    "suggestions": string[],
    "recommended_features": string[]
  }
}
"""


if __name__ == "__main__":
    evaluator = ConversationEvaluator()
    # Example usage
    conversation_ids = [
        "cf8ea811-9e56-4357-b1ff-e4e917171034_1745362341",
        "cf8ea811-9e56-4357-b1ff-e4e917171034_1745354581",
    ]
    
    # fetch all conversation ids from the conversation_storage directory and its subdirectories
    conversation_ids = []
    for root, dirs, files in os.walk('local/conversation_storage/specialized.com'):
        for file in files:
            if file.endswith('.json'):
                conversation_ids.append(file[:-5])
    # remove duplicates
    conversation_ids = list(set(conversation_ids))
    
    # load the system prompt from the prompt manager
    prompt_manager = PromptManager(account="specialized.com")
    system_prompt = prompt_manager.build_system_instruction_prompt(account="specialized.com")
    
    conversations = []
    for conversation_id in conversation_ids:
        conversation = evaluator.load_conversation("specialized.com", conversation_id)
        if conversation:
            conversations.append(conversation)
    result = asyncio.run(evaluator.run_system_prompt_evaluation(account="specialized.com", system_prompt=system_prompt, conversations=conversations))
    print(result)

    for conversation_id in conversation_ids:
        conversation = evaluator.load_conversation("specialized.com", conversation_id)
        if conversation:
            asyncio.run(evaluator.evaluate_conversation(account="specialized.com", conversation=conversation, force_reevaluation=False))
            print(f"Conversation {conversation_id} evaluated successfully.")
        else:
            print(f"Conversation {conversation_id} not found.")

    print("Evaluation completed.")