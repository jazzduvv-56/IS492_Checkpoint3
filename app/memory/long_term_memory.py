"""
Long-term semantic memory using ChromaDB with embeddings
Retrieves semantically similar past conversations, summaries, and profile facts
"""

import os
import uuid
from typing import List, Dict, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from utils.timezone_utils import now_central
from app.database.crud import ConversationCRUD


class LongTermMemory:
    """Manages long-term semantic memory using ChromaDB embeddings"""
    
    def __init__(self, storage_path: str = "data/vectors"):
        """
        Initialize long-term memory system with ChromaDB
        
        Args:
            storage_path: Path to store vector database
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=storage_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection (uses default embedding function)
        self.collection = self.client.get_or_create_collection(
            name="carely_memory",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.last_update = None
    
    def add_conversation(self, user_id: int, conversation_id: int, 
                        user_message: str, assistant_response: str, 
                        timestamp: datetime) -> None:
        """
        Add a single conversation to the vector store (incremental update)
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID from database
            user_message: User's message
            assistant_response: Assistant's response
            timestamp: Timestamp of conversation
        """
        try:
            # Combine user message and response for richer context
            combined_text = f"{user_message} {assistant_response}"
            
            # Create unique ID for this entry
            doc_id = f"user_{user_id}_conv_{conversation_id}"
            
            # Add to collection
            self.collection.upsert(
                ids=[doc_id],
                documents=[combined_text],
                metadatas=[{
                    "user_id": user_id,
                    "type": "conversation",
                    "source_id": conversation_id,
                    "timestamp": timestamp.isoformat(),
                    "user_message": user_message[:200],  # Truncate for metadata
                    "assistant_response": assistant_response[:200]
                }]
            )
            
        except Exception as e:
            print(f"Error adding conversation to vector store: {e}")
    
    def add_summary(self, user_id: int, summary_text: str, date: datetime) -> None:
        """
        Add or update a daily summary in the vector store
        
        Args:
            user_id: User ID
            summary_text: Summary text (3-6 lines)
            date: Date of the summary
        """
        try:
            # Keep summaries concise (≤2 sentences for retrieval)
            sentences = summary_text.split('.')[:2]
            concise_summary = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
            
            doc_id = f"user_{user_id}_summary_{date.strftime('%Y%m%d')}"
            
            self.collection.upsert(
                ids=[doc_id],
                documents=[concise_summary],
                metadatas=[{
                    "user_id": user_id,
                    "type": "summary",
                    "date": date.strftime('%Y-%m-%d'),
                    "timestamp": date.isoformat()
                }]
            )
            
        except Exception as e:
            print(f"Error adding summary to vector store: {e}")
    
    def add_profile_fact(self, user_id: int, fact: str, fact_type: str = "general") -> None:
        """
        Add a profile fact (preferences, meal times, etc.) to the vector store
        
        Args:
            user_id: User ID
            fact: One-liner fact about the user
            fact_type: Type of fact (e.g., "meal_time", "preference")
        """
        try:
            doc_id = f"user_{user_id}_fact_{fact_type}_{uuid.uuid4().hex[:8]}"
            
            self.collection.upsert(
                ids=[doc_id],
                documents=[fact],
                metadatas=[{
                    "user_id": user_id,
                    "type": "profile_fact",
                    "fact_type": fact_type,
                    "timestamp": now_central().isoformat()
                }]
            )
            
        except Exception as e:
            print(f"Error adding profile fact to vector store: {e}")
    
    def retrieve_similar_conversations(self, query: str, user_id: int, 
                                      top_k: int = 3, exclude_query: str = None) -> List[Dict]:
        """
        Retrieve semantically similar past conversations
        
        Args:
            query: User's current query
            user_id: User ID
            top_k: Total number of snippets to retrieve (max 3)
            exclude_query: Query text to exclude from results
        
        Returns:
            List of similar items (conversations, summaries, facts)
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 3, 10),  # Get more candidates for filtering
                where={"user_id": user_id}
            )
            
            if not results or not results['ids'] or not results['ids'][0]:
                return []
            
            similar_items = []
            exclude_lower = exclude_query.lower().strip() if exclude_query else ""
            
            # Track how many of each type we've added
            conv_count = 0
            other_count = 0
            
            for idx, doc_id in enumerate(results['ids'][0]):
                if len(similar_items) >= top_k:
                    break
                
                metadata = results['metadatas'][0][idx]
                document = results['documents'][0][idx]
                distance = results['distances'][0][idx] if 'distances' in results else None
                
                # Skip if too similar to current query (avoid echoing)
                if exclude_query:
                    if metadata.get('user_message', '').lower().strip() == exclude_lower:
                        continue
                    # Skip if distance is too small (too similar)
                    if distance is not None and distance < 0.1:
                        continue
                
                item_type = metadata.get('type', 'conversation')
                
                # Enforce limits: max 1 conversation, max 2 summaries/facts
                if item_type == 'conversation':
                    if conv_count >= 1:
                        continue
                    conv_count += 1
                else:
                    if other_count >= 2:
                        continue
                    other_count += 1
                
                # Truncate to ≤2 sentences
                sentences = document.split('.')[:2]
                concise_text = '. '.join(s.strip() for s in sentences if s.strip())
                if concise_text and not concise_text.endswith('.'):
                    concise_text += '.'
                
                similar_item = {
                    "type": item_type,
                    "text": concise_text,
                    "metadata": metadata,
                    "relevance": 1.0 - distance if distance is not None else 0.5
                }
                
                # Add specific fields based on type
                if item_type == 'conversation':
                    similar_item['user_message'] = metadata.get('user_message', '')
                    similar_item['assistant_response'] = metadata.get('assistant_response', '')
                    if 'timestamp' in metadata:
                        try:
                            similar_item['timestamp'] = datetime.fromisoformat(metadata['timestamp'])
                        except:
                            pass
                
                similar_items.append(similar_item)
            
            return similar_items[:top_k]  # Ensure we return max top_k items
            
        except Exception as e:
            print(f"Error retrieving similar conversations: {e}")
            return []
    
    def get_formatted_similar_context(self, query: str, user_id: int, 
                                     top_k: int = 3) -> str:
        """
        Get formatted string of similar past context
        
        Args:
            query: Current user query
            user_id: User ID
            top_k: Number of items to retrieve (max 3)
        
        Returns:
            Formatted context string
        """
        similar_items = self.retrieve_similar_conversations(query, user_id, top_k)
        
        if not similar_items:
            return ""
        
        context_parts = []
        
        for item in similar_items:
            if item['type'] == 'conversation' and 'timestamp' in item:
                time_str = item['timestamp'].strftime('%B %d')
                context_parts.append(f"[{time_str}] {item['text']}")
            elif item['type'] == 'summary':
                date_str = item['metadata'].get('date', 'Recent')
                context_parts.append(f"[Summary {date_str}] {item['text']}")
            elif item['type'] == 'profile_fact':
                context_parts.append(f"[Profile] {item['text']}")
            else:
                context_parts.append(item['text'])
        
        return "\n".join(context_parts)
    
    def build_memory_index(self, user_id: int, limit: int = 100):
        """
        Build or rebuild the memory index from conversation history
        (For initial migration or full rebuild only)
        
        Args:
            user_id: User ID to build memory for
            limit: Maximum number of conversations to index
        """
        try:
            # Get recent conversations from database
            conversations = ConversationCRUD.get_user_conversations(user_id, limit=limit)
            
            if not conversations:
                return
            
            # Add each conversation to the vector store
            for conv in conversations:
                self.add_conversation(
                    user_id=user_id,
                    conversation_id=conv.id,
                    user_message=conv.message,
                    assistant_response=conv.response,
                    timestamp=conv.timestamp
                )
            
            self.last_update = now_central()
            print(f"Indexed {len(conversations)} conversations for user {user_id}")
            
        except Exception as e:
            print(f"Error building memory index: {e}")
    
    def clear_user_memory(self, user_id: int):
        """
        Clear all memory for a specific user
        
        Args:
            user_id: User ID
        """
        try:
            # Query all documents for this user
            results = self.collection.get(
                where={"user_id": user_id}
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Cleared {len(results['ids'])} memory items for user {user_id}")
                
        except Exception as e:
            print(f"Error clearing user memory: {e}")
