from dataclasses import dataclass

from typing import List, Optional

from liddy_voice.spence.model import BasicChatMessage

@dataclass
class ConversationDetails:
    conversation_id: str
    transcript: List[BasicChatMessage]
    agent_id: str
    start_time_unix_secs: int
    call_duration_secs: int
    message_count: int
    status: str
    call_successful: bool
    agent_name: str
