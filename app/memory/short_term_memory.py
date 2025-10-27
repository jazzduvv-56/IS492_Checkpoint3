"""
Short-term memory - persistent, DB-based recent conversation context
Fetches last 8-10 messages directly from the database
"""

from typing import List, Dict
from datetime import datetime
from app.database.crud import ConversationCRUD


class ShortTermMemory:
    """Manages short-term conversation context from database"""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize short-term memory
        
        Args:
            max_size: Maximum number of recent messages to fetch (default: 10)
        """
        self.max_size = max_size
    
    def get_recent_context(self, user_id: int, num_exchanges: int = None) -> List[Dict]:
        """
        Get recent conversation exchanges from database
        
        Args:
            user_id: User ID
            num_exchanges: Number of recent exchanges to retrieve (default: max_size)
        
        Returns:
            List of conversation exchanges
        """
        if num_exchanges is None:
            num_exchanges = self.max_size
        
        # Fetch from database
        conversations = ConversationCRUD.get_user_conversations(user_id, limit=num_exchanges)
        
        if not conversations:
            return []
        
        # Convert to exchange format
        exchanges = []
        for conv in reversed(conversations):  # Reverse to get chronological order
            exchanges.append({
                "user_message": conv.message,
                "assistant_response": conv.response,
                "timestamp": conv.timestamp
            })
        
        return exchanges
    
    def get_formatted_context(self, user_id: int, num_exchanges: int = None) -> str:
        """
        Get formatted string of recent context for AI prompts (compact, ~500 tokens)
        
        Args:
            user_id: User ID
            num_exchanges: Number of recent exchanges to include (default: 8)
        
        Returns:
            Formatted conversation context
        """
        if num_exchanges is None:
            num_exchanges = min(8, self.max_size)  # Default to 8 for token efficiency
        
        exchanges = self.get_recent_context(user_id, num_exchanges)
        
        if not exchanges:
            return "No recent conversation history."
        
        # Compact format to save tokens
        context_lines = []
        for exchange in exchanges:
            # Truncate long messages to keep under ~500 tokens total
            user_msg = exchange['user_message'][:150]
            asst_msg = exchange['assistant_response'][:150]
            context_lines.append(f"User: {user_msg}")
            context_lines.append(f"Carely: {asst_msg}")
        
        return "\n".join(context_lines)
    
    def clear(self, user_id: int = None):
        """
        Clear short-term memory (no-op for DB-based implementation)
        Memory is persistent in database
        
        Args:
            user_id: User ID (optional, for compatibility)
        """
        # No action needed - memory is in database
        pass
    
    def get_size(self, user_id: int) -> int:
        """
        Get current number of exchanges available
        
        Args:
            user_id: User ID
        
        Returns:
            Number of recent exchanges
        """
        exchanges = self.get_recent_context(user_id, self.max_size)
        return len(exchanges)
    
    def add_exchange(self, user_message: str, assistant_response: str, 
                    timestamp: datetime = None):
        """
        Legacy method for compatibility (no-op - conversations saved via CRUD)
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            timestamp: Timestamp
        """
        # No action needed - conversations are saved via ConversationCRUD
        pass
