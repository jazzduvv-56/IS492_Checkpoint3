from utils.timezone_utils import now_central
"""
Long-term semantic memory using TF-IDF vectorization
Retrieves semantically similar past conversations without external APIs
"""

import pickle
import os
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.database.crud import ConversationCRUD


class LongTermMemory:
    """Manages long-term semantic memory using local TF-IDF"""
    
    def __init__(self, storage_path: str = "data/long_term_memory.pkl"):
        """
        Initialize long-term memory system
        
        Args:
            storage_path: Path to store vectorizer and embeddings
        """
        self.storage_path = storage_path
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.conversation_vectors = None
        self.conversation_ids = []
        self.last_update = None
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        # Load existing memory if available
        self._load_memory()
    
    def build_memory_index(self, user_id: int, limit: int = 100):
        """
        Build or update the memory index from conversation history
        
        Args:
            user_id: User ID to build memory for
            limit: Maximum number of conversations to index
        """
        # Get recent conversations from database
        conversations = ConversationCRUD.get_user_conversations(user_id, limit=limit)
        
        if not conversations:
            return
        
        # Prepare texts and IDs
        texts = []
        self.conversation_ids = []
        
        for conv in conversations:
            # Combine user message and assistant response for better context
            combined_text = f"{conv.message} {conv.response}"
            texts.append(combined_text)
            self.conversation_ids.append(conv.id)
        
        # Create TF-IDF vectors
        if texts:
            self.conversation_vectors = self.vectorizer.fit_transform(texts)
            self.last_update = now_central()
            self._save_memory()
    
    def retrieve_similar_conversations(self, query: str, user_id: int, 
                                      top_k: int = 3, exclude_query: str = None) -> List[Dict]:
        """
        Retrieve semantically similar past conversations
        
        Args:
            query: User's current query
            user_id: User ID
            top_k: Number of similar conversations to retrieve
            exclude_query: Query text to exclude from results (avoids echoing current message)
        
        Returns:
            List of similar conversation dictionaries
        """
        # Rebuild index if empty or outdated
        if self.conversation_vectors is None or self._needs_update():
            self.build_memory_index(user_id)
        
        if self.conversation_vectors is None or len(self.conversation_ids) == 0:
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity scores
            similarities = cosine_similarity(query_vector, self.conversation_vectors)[0]
            
            # Get top-k similar conversations
            top_indices = np.argsort(similarities)[-top_k * 2:][::-1]  # Get more candidates for filtering
            
            # Retrieve full conversation details
            similar_conversations = []
            all_convs = ConversationCRUD.get_user_conversations(user_id, limit=100)
            conv_dict = {conv.id: conv for conv in all_convs}
            
            exclude_lower = exclude_query.lower().strip() if exclude_query else ""
            
            for idx in top_indices:
                if len(similar_conversations) >= top_k:
                    break
                    
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    conv_id = self.conversation_ids[idx]
                    if conv_id in conv_dict:
                        conv = conv_dict[conv_id]
                        
                        # Skip if this is the current query (avoid echoing)
                        if exclude_query and conv.message.lower().strip() == exclude_lower:
                            continue
                        
                        # Skip if message is too similar to current query (>95% similar)
                        if exclude_query and similarities[idx] > 0.95:
                            continue
                        
                        similar_conversations.append({
                            "id": conv.id,
                            "user_message": conv.message,
                            "assistant_response": conv.response,
                            "timestamp": conv.timestamp,
                            "similarity_score": float(similarities[idx])
                        })
            
            return similar_conversations
        
        except Exception as e:
            print(f"Error retrieving similar conversations: {e}")
            return []
    
    def get_formatted_similar_context(self, query: str, user_id: int, 
                                     top_k: int = 3) -> str:
        """
        Get formatted string of similar past conversations
        
        Args:
            query: Current user query
            user_id: User ID
            top_k: Number of similar conversations to retrieve
        
        Returns:
            Formatted context string
        """
        similar_convs = self.retrieve_similar_conversations(query, user_id, top_k)
        
        if not similar_convs:
            return ""
        
        context = "Relevant past conversations:\n"
        for conv in similar_convs:
            time_str = conv['timestamp'].strftime('%B %d, %Y')
            context += f"[{time_str}]\n"
            context += f"User: {conv['user_message']}\n"
            context += f"Carely: {conv['assistant_response']}\n"
            context += "---\n"
        
        return context
    
    def _needs_update(self) -> bool:
        """Check if memory index needs updating"""
        if self.last_update is None:
            return True
        # Update if more than 1 hour old
        time_diff = now_central() - self.last_update
        return time_diff.total_seconds() > 3600
    
    def _save_memory(self):
        """Save vectorizer and memory state to disk"""
        try:
            memory_state = {
                'vectorizer': self.vectorizer,
                'conversation_ids': self.conversation_ids,
                'last_update': self.last_update
            }
            with open(self.storage_path, 'wb') as f:
                pickle.dump(memory_state, f)
        except Exception as e:
            print(f"Error saving long-term memory: {e}")
    
    def _load_memory(self):
        """Load vectorizer and memory state from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'rb') as f:
                    memory_state = pickle.load(f)
                    self.vectorizer = memory_state['vectorizer']
                    self.conversation_ids = memory_state['conversation_ids']
                    self.last_update = memory_state['last_update']
        except Exception as e:
            print(f"Error loading long-term memory: {e}")
