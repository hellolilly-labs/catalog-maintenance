from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from datetime import datetime
import time
import json

if TYPE_CHECKING:
    from liddy.models.product import Product

@dataclass
class UserRoom:
    account: str
    user_id: str
    room_id: str
    last_connection_time: Optional[float] = None
    auto_reconnect_until: Optional[float] = None
    resumption_message: Optional[str] = None
    
    @staticmethod
    def from_json(json_str: str) -> "UserRoom":
        json_object = json.loads(json_str)
        return UserRoom.from_dict(json_object)
    
    @staticmethod
    def from_dict(data: dict) -> "UserRoom":
        # Create an instance of UserRoom from a dictionary
        user_room = UserRoom(
            account=data.get("account"),
            user_id=data.get("userId") or data.get("user_id"),
            room_id=data.get("roomId") or data.get("room_id"),
            last_connection_time=data.get("lastConnectionTime") or data.get("last_connection_time"),
            auto_reconnect_until=data.get("autoReconnectUntil") or data.get("auto_reconnect_until"),
            resumption_message=data.get("resumptionMessage") or data.get("resumption_message")
        )
        
        # Convert string timestamps to float if needed
        if user_room.last_connection_time and isinstance(user_room.last_connection_time, str):
            user_room.last_connection_time = float(user_room.last_connection_time)
        
        if user_room.auto_reconnect_until and isinstance(user_room.auto_reconnect_until, str):
            user_room.auto_reconnect_until = float(user_room.auto_reconnect_until)
        
        return user_room
    
    def to_dict(self) -> dict:
        # Convert the object to a dictionary
        return {
            "account": self.account,
            "user_id": self.user_id,
            "room_id": self.room_id,
            "last_connection_time": self.last_connection_time,
            "auto_reconnect_until": self.auto_reconnect_until,
            "resumption_message": self.resumption_message
        }
    
    def to_json(self) -> str:
        # Convert the object to a dictionary
        user_room_dict = self.to_dict()
        
        # Convert the dictionary to a JSON string
        json_str = json.dumps(user_room_dict)
        
        return json_str


@dataclass
class ConversationResumptionState:
    """
    Class to manage conversation resumption state.
    
    This class handles:
    1. Storing and retrieving conversation resumption state
    2. Setting a TTL for the resumption state
    """
    
    user_id: str
    last_interaction_time: Optional[float] = None
    is_resumable: bool = False
    resumption_message: Optional[Tuple[str, str]] = None
    chat_messages: Optional[List["BasicChatMessage"]] = None
    user_state_message: Optional[str] = None
    


@dataclass
class UrlTracking:
    url: str
    title: str
    state: str
    product_details: Optional[any]
    content: Optional[str]
    timestamp: Optional[float] = None
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        elif isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp).timestamp()
            except ValueError:
                self.timestamp = time.time()
    
    
    @staticmethod
    def from_json(json_str: str | dict) -> "UrlTracking":
        if isinstance(json_str, dict):
            return UrlTracking.from_dict(json_str)
        json_object = json.loads(json_str)
        return UrlTracking.from_dict(json_object)
    @staticmethod
    def from_dict(data: dict) -> "UrlTracking":
        # Create an instance of UrlTracking from a dictionary
        return UrlTracking(
            url=data.get("url"),
            title=data.get("title"),
            state=data.get("state"),
            product_details=data.get("product_details"),
            content=data.get("content"),
            timestamp=data.get("timestamp")
        )
    
    def to_dict(self) -> dict:
        # Convert the object to a dictionary
        return {
            "url": self.url,
            "title": self.title,
            "state": self.state,
            "product_details": self.product_details,
            "content": self.content,
            "timestamp": self.timestamp
        }
            
    def to_json(self) -> str:
        # Convert the object to a dictionary
        url_tracking_dict = self.to_dict()
        
        # Convert the dictionary to a JSON string
        json_str = json.dumps(url_tracking_dict)
        
        return json_str


@dataclass
class BasicChatMessage:
    role: str
    content: str
    timestamp: Optional[float] = None
    
    @staticmethod
    def from_json(json_str: str) -> "BasicChatMessage":
        json_object = json.loads(json_str)
        return BasicChatMessage.from_dict(json_object)
    @staticmethod
    def from_dict(data: dict) -> "BasicChatMessage":
        # Create an instance of BasicChatMessage from a dictionary
        return BasicChatMessage(
            role=data.get("role"),
            content=data.get("content"),
            timestamp=data.get("timestamp")
        )
    
    def to_dict(self) -> dict:
        # Convert the object to a dictionary
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        # Convert the object to a dictionary
        chat_message_dict = self.to_dict()
        
        # Convert the dictionary to a JSON string
        json_str = json.dumps(chat_message_dict)
        
        return json_str

@dataclass
class ConversationExitState:
    last_interaction_time: Optional[float] = None
    transcript_summary: Optional[str] = None
    resumption_message: Optional[str] = None
    exit_reason: Optional[str] = None
    
    @staticmethod
    def from_json(json_str: str) -> "ConversationExitState":
        json_object = json.loads(json_str)
        return ConversationExitState.from_dict(json_object)

    @staticmethod
    def from_dict(data: dict) -> "ConversationExitState":
        # Create an instance of ConversationExitState from a dictionary
        return ConversationExitState(
            last_interaction_time=data.get("last_interaction_time"),
            transcript_summary=data.get("transcript_summary"),
            exit_reason=data.get("exit_reason"),
            resumption_message=data.get("resumption_message")
        )
    
    def to_dict(self) -> dict:
        # Convert the object to a dictionary
        return {
            "last_interaction_time": self.last_interaction_time,
            "transcript_summary": self.transcript_summary,
            "exit_reason": self.exit_reason,
            "resumption_message": self.resumption_message
        }
    
    def to_json(self) -> str:
        # Convert the object to a dictionary
        exit_state_dict = self.to_dict()
        
        # Convert the dictionary to a JSON string
        json_str = json.dumps(exit_state_dict)
        
        return json_str

@dataclass
class ScoreReasoning:
    score: float
    reasoning: str
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ScoreReasoning":
        return ScoreReasoning(
            score=float(data["score"]),
            reasoning=data["reasoning"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "reasoning": self.reasoning
        }

@dataclass
class AnalysisWeight:
    value: str
    weight: float
    reason: str

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AnalysisWeight":
        return AnalysisWeight(
            value=data["value"],
            weight=float(data["weight"]),
            reason=data["reason"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "weight": self.weight,
            "reason": self.reason
        }

@dataclass
class CommunicationDirective:
    directive: List[str] = field(default_factory=list)
    formality: Optional[ScoreReasoning] = None
    
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CommunicationDirective":
        return CommunicationDirective(
            directive=data["directive"],
            formality=ScoreReasoning.from_dict(data["formality"])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "directive": self.directive,
            "formality": self.formality.to_dict()
        }
    

@dataclass
class Sentiment:
    trustLevel: AnalysisWeight
    engagementLevel: AnalysisWeight
    keyObservations: List[str] = field(default_factory=list)
    fromEmotionalTones: List[AnalysisWeight] = field(default_factory=list)
    fromSentiments: List[AnalysisWeight] = field(default_factory=list)
    fromStyles: List[AnalysisWeight] = field(default_factory=list)
    toEmotionalTones: List[AnalysisWeight] = field(default_factory=list)
    toSentiments: List[AnalysisWeight] = field(default_factory=list)
    toStyles: List[AnalysisWeight] = field(default_factory=list)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Sentiment":
        return Sentiment(
            keyObservations=data["keyObservations"],
            trustLevel=AnalysisWeight.from_dict(data["trustLevel"]),
            engagementLevel=AnalysisWeight.from_dict(data["engagementLevel"]),
            fromEmotionalTones=[AnalysisWeight.from_dict(i) for i in data["fromEmotionalTones"]],
            fromSentiments=[AnalysisWeight.from_dict(i) for i in data["fromSentiments"]],
            fromStyles=[AnalysisWeight.from_dict(i) for i in data["fromStyles"]],
            toEmotionalTones=[AnalysisWeight.from_dict(i) for i in data["toEmotionalTones"]],
            toSentiments=[AnalysisWeight.from_dict(i) for i in data["toSentiments"]],
            toStyles=[AnalysisWeight.from_dict(i) for i in data["toStyles"]]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keyObservations": self.keyObservations,
            "trustLevel": self.trustLevel.to_dict(),
            "engagementLevel": self.engagementLevel.to_dict(),
            "fromEmotionalTones": [i.to_dict() for i in self.fromEmotionalTones],
            "fromSentiments": [i.to_dict() for i in self.fromSentiments],
            "fromStyles": [i.to_dict() for i in self.fromStyles],
            "toEmotionalTones": [i.to_dict() for i in self.toEmotionalTones],
            "toSentiments": [i.to_dict() for i in self.toSentiments],
            "toStyles": [i.to_dict() for i in self.toStyles]
        }

@dataclass
class SourceSentiment:
    source: str
    sourceId: str
    sourceDateTime: float
    sentiment: Sentiment
    analysisDateTime: Optional[float] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SourceSentiment":
        return SourceSentiment(
            source=data["source"],
            sourceId=data["sourceId"],
            sourceDateTime=data["sourceDateTime"] if data.get("sourceDateTime") and isinstance(data["sourceDateTime"], (int, float)) else time.time(),
            sentiment=Sentiment.from_dict(data["sentiment"]),
            analysisDateTime=data["analysisDateTime"] if data.get("analysisDateTime") and isinstance(data["analysisDateTime"], (int, float)) else time.time()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "sourceId": self.sourceId,
            "sourceDateTime": self.sourceDateTime,
            "sentiment": self.sentiment.to_dict(),
            "analysisDateTime": self.analysisDateTime if self.analysisDateTime else None
        }

# @dataclass
# class UserSentimentData:
#     userId: str
#     sentiments: List[SourceSentiment] = field(default_factory=list)
#     details: Optional[Any] = None

#     @staticmethod
#     def from_dict(data: Dict[str, Any]) -> "UserSentimentData":
#         return UserSentimentData(
#             userId=data["userId"],
#             sentiments=[SourceSentiment.from_dict(i) for i in data.get("sentiments", [])],
#             details=data.get("details")
#         )

#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "userId": self.userId,
#             "sentiments": [i.to_dict() for i in self.sentiments],
#             "details": self.details
#         }

class SentimentAnalysis:
    def __init__(self, user_id: str, sentiments: List[SourceSentiment] = None, details: Optional[Any] = None):
        self.user_id = user_id
        self.sentiments = sentiments if sentiments is not None else []
        self.details = details

    def upsert_sentiment(self, source: str, source_id: str, source_date_time: float, sentiment: Sentiment):
        existing_sentiment = next((s for s in self.sentiments if s.source == source and s.sourceId == source_id), None)
        if existing_sentiment:
            existing_sentiment.sentiment = sentiment
            existing_sentiment.source_date_time = source_date_time if source_date_time else existing_sentiment.source_date_time
            existing_sentiment.analysisDateTime = time.time()
        else:
            self.sentiments.append(SourceSentiment(source, source_id, source_date_time, sentiment, time.time()))

    def to_dict(self):
        return {
            'userId': self.user_id,
            'sentiments': [s.to_dict() for s in self.sentiments],
            'details': self.details
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        user_id = data['userId'] if 'userId' in data else data["user_id"] # Handle both snake_case and camelCase
        return SentimentAnalysis(
            user_id=user_id,
            sentiments=[SourceSentiment.from_dict(s) for s in data.get('sentiments', [])],
            details=data.get('details')
        )

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __json__(self) -> Dict[str, Any]:
        return self.to_dict()
@dataclass
class UserState:
    user_id: str
    account: str = ""
    conversation_exit_state: Optional[ConversationExitState] = None
    communication_directive: Optional[CommunicationDirective] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    voice_session_id: Optional[str] = None
    interaction_start_time: float = field(default_factory=time.time)
    last_interaction_time: Optional[float] = None
    current_url: Optional[str] = None
    current_url_timestamp: Optional[float] = None
    current_url_title: Optional[str] = None
    current_product: Optional["Product"] = None
    current_product_id: Optional[str] = None
    current_product_timestamp: Optional[float] = None
    recent_product_ids: List[Tuple[float, str]] = field(default_factory=list)
    
    @staticmethod
    def _parse_recent_product_ids(data: List) -> List[Tuple[float, str]]:
        """Parse recent_product_ids from various formats for backward compatibility."""
        if not data:
            return []
        
        result = []
        for item in data:
            if isinstance(item, str):
                # Legacy format: just product_id string, use current time
                result.append((time.time(), item))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # New format: [timestamp, product_id]
                timestamp = float(item[0]) if item[0] is not None else time.time()
                product_id = str(item[1])
                result.append((timestamp, product_id))
            # Skip invalid entries
        
        return result
    
    @staticmethod
    def from_json(json_str: str | dict) -> "UserState":
        if isinstance(json_str, dict):
            return UserState.from_dict(json_str)
        json_object = json.loads(json_str)
        
        return UserState.from_dict(json_object)

    @staticmethod
    def from_dict(data: dict) -> "UserState":
        # Convert the dictionary to a UserState object
        sentiment_analysis_dict = data.get("sentiment_analysis")
        if sentiment_analysis_dict:
            sentiment_analysis_dict['user_id'] = data.get("user_id")
        user_state = UserState(
            user_id=data.get("user_id"),
            account=data.get("account", ""),
            conversation_exit_state=ConversationExitState.from_dict(data.get("conversation_exit_state", {})) if data.get("conversation_exit_state") else None,
            communication_directive=CommunicationDirective.from_dict(data.get("communication_directive", {})) if data.get("communication_directive") else None,
            sentiment_analysis=SentimentAnalysis.from_dict(sentiment_analysis_dict) if sentiment_analysis_dict else None,
            voice_session_id=data.get("voice_session_id"),
            interaction_start_time=data.get("interaction_start_time", time.time()),
            last_interaction_time=data.get("last_interaction_time"),
            current_url=data.get("current_url"),
            current_url_timestamp=data.get("current_url_timestamp"),
            current_url_title=data.get("current_title"),
            current_product=None,  # Product object not serialized
            current_product_id=data.get("current_product_id"),
            current_product_timestamp=data.get("current_product_timestamp"),
            recent_product_ids=UserState._parse_recent_product_ids(data.get("recent_product_ids", []))
        )
        return user_state
    
    def to_dict(self) -> dict:
        # Convert the UserState object to a dictionary
        return {
            "user_id": self.user_id,
            "account": self.account,
            "conversation_exit_state": self.conversation_exit_state.to_dict() if self.conversation_exit_state else None,
            "communication_directive": self.communication_directive.to_dict() if self.communication_directive else None,
            "sentiment_analysis": self.sentiment_analysis.to_dict() if self.sentiment_analysis else None,
            "voice_session_id": self.voice_session_id,
            "interaction_start_time": self.interaction_start_time,
            "last_interaction_time": self.last_interaction_time,
            "current_url": self.current_url,
            "current_url_timestamp": self.current_url_timestamp,
            "current_url_title": self.current_url_title,
            "current_product_id": self.current_product_id,
            "current_product_timestamp": self.current_product_timestamp,
            "recent_product_ids": [[timestamp, product_id] for timestamp, product_id in self.recent_product_ids]
        }
    
    def to_json(self) -> str:
        # Convert the object to a dictionary
        user_state_dict = self.to_dict()
        
        # Convert the dictionary to a JSON string
        json_str = json.dumps(user_state_dict)
        
        return json_str
