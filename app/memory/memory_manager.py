from utils.timezone_utils import now_central
"""
Unified Memory Manager orchestrating all four memory layers
Provides a single interface for context-aware, personalized responses
"""

from typing import Dict, List, Optional
from datetime import datetime

from app.memory.short_term_memory import ShortTermMemory
from app.memory.long_term_memory import LongTermMemory
from app.memory.episodic_memory import EpisodicMemory
from app.memory.structured_memory import StructuredMemory


class MemoryManager:
    """Unified interface for all memory layers"""
    
    def __init__(self):
        """Initialize all memory layers"""
        self.short_term = ShortTermMemory(max_size=10)  # DB-based, fetches last 10
        self.long_term = LongTermMemory()  # ChromaDB-based embeddings
        self.episodic = EpisodicMemory()
        self.structured = StructuredMemory()
    
    def add_conversation(self, user_id: int, conversation_id: int, user_message: str, 
                        assistant_response: str, timestamp: datetime = None):
        """
        Add a conversation to memory system (incremental vector store update)
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID from database
            user_message: User's message
            assistant_response: Assistant's response
            timestamp: Timestamp of conversation
        """
        # Short-term memory is DB-based (no action needed)
        # Long-term memory: add to vector store incrementally
        if timestamp is None:
            timestamp = now_central()
        
        try:
            self.long_term.add_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_response=assistant_response,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"Warning: Could not add conversation to vector store: {e}")
    
    def get_full_context(self, user_id: int, current_query: str) -> str:
        """
        Get comprehensive context from all memory layers
        
        Args:
            user_id: User ID
            current_query: Current user query
        
        Returns:
            Complete context string for AI prompt
        """
        context_parts = []
        
        # 1. Structured Memory - User Profile and Preferences
        profile = self.structured.get_formatted_profile(user_id)
        if profile:
            context_parts.append("=== USER PROFILE ===")
            context_parts.append(profile)
        
        # 2. Short-Term Memory - Recent conversation (DB-based, last 8 messages)
        short_term_context = self.short_term.get_formatted_context(user_id, num_exchanges=8)
        if short_term_context and "No recent" not in short_term_context:
            context_parts.append("\n=== RECENT CONVERSATION ===")
            context_parts.append(short_term_context)
        
        # 3. Long-Term Memory - Semantically similar past context
        # Retrieves top-1 conversation + top-2 summaries/facts (max 3 total, ≤2 sentences each)
        try:
            similar_context = self.long_term.get_formatted_similar_context(
                current_query, user_id, top_k=3
            )
            if similar_context:
                context_parts.append("\n=== RELEVANT PAST CONTEXT ===")
                context_parts.append(similar_context)
        except Exception as e:
            # Gracefully handle vector store errors
            print(f"Warning: Long-term memory retrieval failed: {e}")
        
        return "\n".join(context_parts)
    
    def recall_information(self, user_id: int, query: str) -> str:
        """
        Intelligently recall specific information based on query
        
        Args:
            user_id: User ID
            query: User's query
        
        Returns:
            Relevant information
        """
        query_lower = query.lower()
        
        # Check for specific query types
        if any(word in query_lower for word in ['medication', 'medicine', 'pill', 'schedule']):
            return self.structured.get_medication_schedule(user_id)
        
        elif any(word in query_lower for word in ['breakfast', 'lunch', 'dinner', 'meal', 'eat']):
            # First, check if asking about meal TIME (e.g., "what time is lunch?")
            is_time_query = any(phrase in query_lower for phrase in ['what time', 'when is', 'time for'])
            
            if is_time_query:
                # Determine which meal they're asking about
                meal_name = None
                if 'breakfast' in query_lower:
                    meal_name = 'breakfast'
                elif 'lunch' in query_lower:
                    meal_name = 'lunch'
                elif 'dinner' in query_lower:
                    meal_name = 'dinner'
                
                if meal_name:
                    # Look up configured meal time
                    meal_time = self.structured.get_meal_time(user_id, meal_name)
                    if meal_time:
                        return f"Your {meal_name} is usually at {meal_time}."
                    else:
                        return f"I don't have a time set for {meal_name}. What time do you usually have it?"
            
            # Otherwise, asking about today's meals - use daily summary only when explicitly requested
            # Do NOT return "Today you mentioned..." for simple meal time queries
            if any(word in query_lower for word in ['today', 'summary', 'what did i']):
                logs = self.structured.get_daily_logs(user_id, exclude_message=query, max_topics=3)
                if logs["meals"]:
                    return f"Today you mentioned: {', '.join(logs['meals'])}"
            
            # If not a time query or summary request, fall through to general recall
            return "I can help you track meals or set meal times. What would you like to know?"
        
        elif any(word in query_lower for word in ['remember', 'talked about', 'discussed', 'said']):
            # Recall from long-term memory
            similar = self.long_term.retrieve_similar_conversations(query, user_id, top_k=3, exclude_query=query)
            if similar:
                response = "Yes, I remember we talked about:\n"
                for conv in similar[:2]:
                    date_str = conv['timestamp'].strftime('%B %d')
                    response += f"\n[{date_str}] You: {conv['user_message'][:100]}...\n"
                return response
            else:
                return "I'm not finding a specific memory of that. Could you give me more details?"
        
        elif any(word in query_lower for word in ['today', 'yesterday', 'summary']):
            # Get episodic summary
            date = now_central()
            if 'yesterday' in query_lower:
                from datetime import timedelta
                date = date - timedelta(days=1)
            
            summary = self.episodic.get_formatted_summary(user_id, date)
            return summary
        
        else:
            # General recall from structured memory
            result = self.structured.recall_specific_info(user_id, query, exclude_message=query)
            if result:
                return result
            # Fallback if nothing matches
            return "I'm here to help. Could you tell me more about what you're looking for?"
    
    def generate_daily_summary(self, user_id: int):
        """
        Generate daily summary for episodic memory
        
        Args:
            user_id: User ID
        """
        return self.episodic.generate_daily_summary(user_id)
    
    def update_long_term_index(self, user_id: int):
        """
        Update long-term memory index
        
        Args:
            user_id: User ID
        """
        self.long_term.build_memory_index(user_id)
    
    def clear_short_term(self, user_id: int = None):
        """
        Clear short-term memory buffer (no-op for DB-based implementation)
        
        Args:
            user_id: User ID (optional)
        """
        self.short_term.clear(user_id)
    
    def get_memory_stats(self, user_id: int) -> Dict:
        """
        Get statistics about memory system
        
        Args:
            user_id: User ID
        
        Returns:
            Dictionary with memory stats
        """
        return {
            "short_term_size": self.short_term.get_size(user_id),
            "recent_summaries": len(self.episodic.get_recent_summaries(user_id, days=7)),
            "medications_count": len(self.structured.get_medication_schedule(user_id).split('\n')) - 2
        }
    
    def add_daily_summary(self, user_id: int, summary_text: str, date: datetime = None):
        """
        Add a daily summary to the vector store
        
        Args:
            user_id: User ID
            summary_text: Summary text (3-6 lines)
            date: Date of the summary (defaults to today)
        """
        if date is None:
            date = now_central()
        
        try:
            self.long_term.add_summary(user_id, summary_text, date)
        except Exception as e:
            print(f"Warning: Could not add summary to vector store: {e}")
    
    def add_profile_fact(self, user_id: int, fact: str, fact_type: str = "general"):
        """
        Add a profile fact to the vector store
        
        Args:
            user_id: User ID
            fact: One-liner fact about the user
            fact_type: Type of fact (e.g., "meal_time", "preference")
        """
        try:
            self.long_term.add_profile_fact(user_id, fact, fact_type)
        except Exception as e:
            print(f"Warning: Could not add profile fact to vector store: {e}")
