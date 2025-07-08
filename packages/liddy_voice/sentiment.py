import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

from livekit.agents import llm

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from liddy_voice.config_service import ConfigService
from liddy.model import BasicChatMessage, SentimentAnalysis, Sentiment, CommunicationDirective, UserState
from liddy_voice.user_manager import UserManager
from liddy_voice.llm_service import LlmService

logger = logging.getLogger("sentiment")


class SentimentService:
    BASE_INSTRUCTIONS = """Analyze the overall sentiment, emotional tone, and communication style of the user in the given snippets, with specific attention to enhancing engagementLevel and trustLevel. Assess trustLevel based on consistent positive interactions or reliability over time, and engagementLevel based on the user’s responsiveness and interest in the conversation.\n\nIn the to fields, provide brief, actionable instructions to the AI assistant that will help improve both engagement and trust with the user. For engagementLevel, suggest ways to encourage interaction, such as asking thoughtful questions or showing interest in the user’s needs. For trustLevel, suggest behaviors that convey reliability, consistency, and transparency, like following through on commitments and acknowledging feedback.\n\nAll values should be very brief, direct instructions on how to communicate with the user. When possible, include multiple values for each field. For example, "toEmotionalTones" might include both "empathetic" and "enthusiastic".\n\nDo not include actual content in your analysis, this is simply meant to analyze tone and style. For example, if the content references a particular article, your response should NOT have any reference to that article. Or if the content references an event, your response should NOT reference that event in any way.\n\n## Output Format:\n```json
{
  "keyObservations": ["..."],
  "trustLevel": {
    "value": "0..1",
    "weight": "0..1",
    "reason": "..."
  },
  "engagementLevel": {
    "value": "0..1",
    "weight": "0..1",
    "reason": "..."
  },
  "fromEmotionalTones": [{
    "value": "...",
    "weight": "0..1",
    "reason": "..."
  }],
  "fromSentiments": [{
    "value": "...",
    "weight": "0..1",
    "reason": "..."
  }],
  "fromStyles": [{
    "value": "...",
    "weight": "0..1",
    "reason": "..."
  }],
  "toEmotionalTones": [{
    "value": "...",
    "weight": "0..1",
    "reason": "..."
  }],
  "toSentiments": [{
    "value": "...",
    "weight": "0..1",
    "reason": "..."
  }],
  "toStyles": [{
    "value": "...",
    "weight": "0..1",
    "reason": "..."
  }]
}
```"""

    # one day in seconds
    SENTIMENT_ANALYSIS_TIME_INTERVAL = 86400

    def __init__(self):
        self.max_input_tokens = 8192
        self.SENTIMENT_ANALYSIS_TIME_INTERVAL = int(ConfigService.get('SENTIMENT_ANALYSIS_TIME_INTERVAL', 120))
        self.logger = logging.getLogger("SentimentService")

    async def analyze_sentiment(self, user_state: UserState, source: str, source_id: str, source_date_time: float, texts: Optional[List[BasicChatMessage]] = None, model: Optional[str] = None, force: bool = False) -> SentimentAnalysis:
        if not user_state:
            self.logger.error(f"User state is None.")
            return None
            
        model = model or ConfigService.get('SENTIMENT_ANALYSIS_MODEL')
        texts = texts
        sentiment_analysis = user_state.sentiment_analysis if user_state else None
        
        if not texts:
            self.logger.warning(f"No texts provided for sentiment analysis for user {user_state.user_id}.")
            return None
        
        if sentiment_analysis and sentiment_analysis.sentiments and not force:
            # if the last sentiment_analysis was less than 2 minutes ago, then don't do anything
            # first see if the source_sentiment for this source and source_id already exists
            source_sentiment = next((s for s in sentiment_analysis.sentiments if s.source == source and s.sourceId == source_id), None)
            now = time.time()
            print(f"Now: {now}")
            next_analysis_time = source_sentiment.analysisDateTime + self.SENTIMENT_ANALYSIS_TIME_INTERVAL if source_sentiment and source_sentiment.analysisDateTime else now
            print(f"Next analysis time: {next_analysis_time}")
            if next_analysis_time > now:
                print(f"Skipping sentiment analysis for {source} {source_id} for user {user_state.user_id} as it was analyzed less than {self.SENTIMENT_ANALYSIS_TIME_INTERVAL} ago.")
                return sentiment_analysis
        else:
            sentiment_analysis = SentimentAnalysis(user_id=user_state.user_id, sentiments=[], details=None)
        
        print(f"Analyzing sentiment for {source} {source_id} for user {user_state.user_id}")
        sentiment_analysis = await self._sync_sentiment(model, user_state, sentiment_analysis, texts, source, source_id, source_date_time=time.time())
        user_state.sentiment_analysis = sentiment_analysis
        await self.generate_communication_directive(user_state, model)
        return user_state.sentiment_analysis

    async def _sync_sentiment(self, model: str, sentiment_analysis: SentimentAnalysis, texts: List[BasicChatMessage], source: str, source_id: str, source_date_time: float) -> SentimentAnalysis:
        sentiment = await self._analyze_sentiment(model, texts, source, source_id, source_date_time)
        sentiment_analysis.upsert_sentiment(sentiment=sentiment, source=source, source_id=source_id, source_date_time=source_date_time)
        sentiment_analysis.sentiments.sort(key=lambda x: x.sourceDateTime, reverse=True)
        
        return sentiment_analysis

    async def _analyze_sentiment(self, model: str, user_state: UserState, texts: List[BasicChatMessage], source: str, source_id: str, source_date_time: float) -> Sentiment:
        
        transcript = ""
        for text in texts:
            if text.role != "system":
                transcript += f"{text.role}: {text.content}\n"
        
        instructions = f"""{SentimentService.BASE_INSTRUCTIONS}"""
        instructions += f"\nSource: {source}\n\n" if source else ""
        instructions += f"Transcript:\n{transcript}"

        llm_service = LlmService.fetch_model_service_from_model(model, account=user_state.account, user=user_state.user_id, model_use="sentiment_analysis")
        self.max_input_tokens = LlmService.max_input_tokens(model_name=model)
        max_length = int((self.max_input_tokens * 4) * 0.6)

        chat_ctx = llm.ChatContext([
            llm.ChatMessage(role="user", content=[instructions])
        ])
        
        json_result = LlmService.parse_json_response(await LlmService.chat_wrapper(llm_service, chat_ctx))

        return Sentiment.from_dict(json_result)

    async def generate_communication_directive(self, user_state: UserState, model: str) -> CommunicationDirective:
        
        if not user_state:
            self.logger.error(f"User state is None.")
            return None
        
        if not model:
            model = ConfigService.get('SENTIMENT_ANALYSIS_MODEL')
        sentiment_analysis = user_state.sentiment_analysis
        if not sentiment_analysis or not sentiment_analysis.sentiments:
            self.logger.warning(f"No sentiment analysis found for user {user_state.user_id}.")
            return None

        instructions = """Given the following sentiment analyses for the user, provide a communication directive for yourself as the user's AI advocate for interacting with the user. Make your response as short and succinct as possible while still maintaining data integrity. As an emotionally intelligent advocate, you focus on understanding the nuances of human behavior and communicate with warmth and empathy while simultaneously communicating in a style and tone that is most impactful and engaging to THIS user. Your guiding principle for all interactions is: "Is it true, is it kind, is it helpful?" Your goal is to maximize user engagement and trust.\n\n

## Output Format
```json
{
  "directive": "String. The communication directive for the AI advocate.",
  "formality": {
    "score": "Score the user's formality on a scale of 0.1 to 1.0.",
    "reasoning": "String. The reasoning behind the formality score."
  }
}
`"""
        if user_state.communication_directive:
            instructions += f"Current Communication Directive:\n{user_state.communication_directive.__repr__()}\n\n"

        instructions += """\n## Sentiment Analyses:\n"""
        max_length = int((self.max_input_tokens * 4) * 0.8)
        sentiment_analysis.sentiments.sort(key=lambda x: x.sourceDateTime, reverse=True)
        for sa in sentiment_analysis.sentiments[:5]:
            sa_str = sa.__repr__()
            if len(instructions) + len(sa_str) > max_length:
                break
            instructions += f"{sa_str}\n"

        llm_service = LlmService.fetch_model_service_from_model(model_name=model, account=user_state.account, user=user_state.user_id, model_use="sentiment_analysis")
        chat_ctx=llm.ChatContext([
            llm.ChatMessage(role="user", content=[instructions])
        ])
        json_result = LlmService.parse_json_response(await LlmService.chat_wrapper(llm_service, chat_ctx))

        user_state.communication_directive = CommunicationDirective.from_dict(json_result)
        return user_state.communication_directive

async def test_analyze_chat_sentiment():
    sentiment_service = SentimentService()
    await sentiment_service.analyze_sentiment(
        user_id="test_user",
        source="chat",
        source_id="chat_123",
        source_date_time=time.time(),
        texts=["Assistant: Hello! How are you?", "User: What a dumb question!", "Assistant: I'm sorry to hear that. How can I help?", "User: You're not sorry! You're just a computer program!"]
    )



if __name__ == "__main__":
    # Example usage
    sentiment_service = SentimentService()
    # user_id="4d1b1f89-563e-483f-9d9d-a17c6524c633"
    user_id="cf8ea811-9e56-4357-b1ff-e4e917171034"
    user_state = UserManager.get_user_state(user_id=user_id)
    sentiment_analysis = asyncio.run(sentiment_service.analyze_sentiment(
        user_state=user_state,
        source="chat",
        source_id="chat_123",
        source_date_time=time.time(),
        force=True
    ))
    user_state.sentiment_analysis = sentiment_analysis
    UserManager.save_user_state(user_state)
    print(sentiment_analysis)