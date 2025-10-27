import os
import json
import re
from datetime import datetime, timedelta
from utils.timezone_utils import now_central, to_central
from typing import Dict, Any, List
from groq import Groq
from app.database.crud import (ConversationCRUD, MedicationCRUD,
                               MedicationLogCRUD, CaregiverAlertCRUD, UserCRUD,
                               PersonalEventCRUD)
from utils.sentiment_analysis import analyze_sentiment
from utils.emergency_detection import detect_emergency
from app.memory.memory_manager import MemoryManager


class CompanionAgent:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"  # Using Groq model
        self.memory_manager = MemoryManager()  # Initialize memory system

        # System prompt for elderly care companion
        self.system_prompt = """You are Carely, a warm, empathetic AI companion for elderly care.

CRITICAL RULES:
1. Answer in ≤4 short sentences maximum. Be concise and direct.
2. NO repetition, filler, or rambling. Every word must add value.
3. If uncertain or missing information, ask EXACTLY 1 clarifying question.
4. For time questions: use available tools/context to get local time. NEVER guess or make up times.
5. For medications, schedules, and appointments: call tools or check database and quote results EXACTLY as provided.
6. Warm, caring tone but BRIEF. Think of a caring friend who values your time.

YOUR ROLE:
- Medication reminders and tracking
- Daily wellness check-ins  
- Emotional support and companionship
- Alert caregivers when needed
- Remember personal details

Be gentle, patient, and use simple everyday language. Never use medical jargon."""

    def _limit_to_sentences(self, text: str, max_sentences: int = 4) -> str:
        """
        Limit text to at most max_sentences. If exceeds, reduce to first 3 concise sentences.
        """
        if not text:
            return text
        
        # Split on sentence boundaries (., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # If exceeds max_sentences, return first 3 sentences
        return ' '.join(sentences[:3])

    def get_conversation_context(self, user_id: int, limit: int = 5) -> str:
        """Get recent conversation context for memory"""
        conversations = ConversationCRUD.get_user_conversations(user_id, limit)
        if not conversations:
            return "No previous conversations."

        context = "Recent conversation history:\n"
        for conv in reversed(conversations):  # Show chronologically
            context += f"User: {conv.message}\nCarely: {conv.response}\n---\n"
        return context

    def get_personal_events_context(self, user_id: int) -> str:
        """Get upcoming personal events for personalized conversation"""
        upcoming_events = PersonalEventCRUD.get_upcoming_events(user_id,
                                                                days=30)

        if not upcoming_events:
            return "No upcoming events stored."

        context = "Upcoming important events to remember:\n"
        for event in upcoming_events:
            days_until = (to_central(event.event_date) - now_central()).days
            context += f"- {event.title} ({event.event_type}) in {days_until} days"
            if event.description:
                context += f": {event.description}"
            context += "\n"

        return context

    def log_medication_tool(self,
                            user_id: int,
                            medication_name: str = None,
                            notes: str = "",
                            medication_id: int = None) -> str:
        """Tool to log medication intake with duplicate detection"""
        try:
            # Find medication either by ID or name
            if medication_id:
                medications = MedicationCRUD.get_user_medications(user_id)
                medication = next((med for med in medications if med.id == medication_id), None)
            elif medication_name:
                medications = MedicationCRUD.get_user_medications(user_id)
                medication = next((med for med in medications
                                   if medication_name.lower() in med.name.lower()),
                                  None)
            else:
                return "Please specify which medication you took."

            if not medication:
                return f"I couldn't find that medication in your schedule. Please check the spelling or ask your caregiver to add it."

            # Check for recent duplicate logs (within last 6 hours)
            recent_log = MedicationLogCRUD.check_recent_medication_log(
                user_id=user_id,
                medication_id=medication.id,
                hours=6
            )
            
            if recent_log:
                time_str = recent_log.taken_time.strftime('%I:%M %p')
                return f"I already logged your {medication.name} earlier today at {time_str}. Would you like me to update that entry, or did you take another dose?"

            # Log the medication as taken
            MedicationLogCRUD.log_medication_taken(
                user_id=user_id,
                medication_id=medication.id,
                scheduled_time=now_central(),
                status="taken",
                notes=notes)

            return f"Got it! I've logged your {medication.name} for today. You're staying on track! 🌟"

        except Exception as e:
            return f"I had trouble logging your medication. Please try again or contact your caregiver."

    def check_schedule_tool(self, user_id: int) -> str:
        """Tool to check upcoming medications and reminders"""
        try:
            medications = MedicationCRUD.get_user_medications(user_id)
            if not medications:
                return "You don't have any medications scheduled right now."

            schedule_info = "Here's your medication schedule:\n\n"
            for med in medications:
                times = json.loads(
                    med.schedule_times) if med.schedule_times else []
                schedule_info += f"• {med.name} ({med.dosage}) - {med.frequency}\n"
                if times:
                    schedule_info += f"  Times: {', '.join(times)}\n"
                schedule_info += "\n"

            return schedule_info

        except Exception as e:
            return "I had trouble checking your schedule. Please try again later."

    def alert_caregiver_tool(self,
                             user_id: int,
                             alert_type: str,
                             description: str,
                             severity: str = "medium") -> str:
        """Tool to alert caregivers about concerning patterns"""
        try:
            user = UserCRUD.get_user(user_id)
            title = f"Alert for {user.name if user else 'Patient'}"

            CaregiverAlertCRUD.create_alert(user_id=user_id,
                                            alert_type=alert_type,
                                            title=title,
                                            description=description,
                                            severity=severity)

            return "I've notified your caregiver about this. They'll be in touch soon to check on you."

        except Exception as e:
            return "I had trouble sending the alert. Please contact your caregiver directly if this is urgent."

    def determine_quick_actions(self, user_message: str, user_id: int) -> List[str]:
        """Determine 2-3 relevant quick action buttons based on context"""
        message_lower = user_message.lower()
        actions = []
        
        # Medication-related keywords
        med_keywords = ["medication", "med", "pill", "medicine", "take", "took", "dose"]
        if any(keyword in message_lower for keyword in med_keywords):
            actions.append("log_medication")
        
        # Boredom or entertainment keywords
        bored_keywords = ["bored", "lonely", "nothing to do", "entertain", "fun"]
        if any(keyword in message_lower for keyword in bored_keywords):
            actions.extend(["play_music", "fun_corner"])
        
        # Music keywords
        music_keywords = ["music", "song", "relax", "calming", "peaceful"]
        if any(keyword in message_lower for keyword in music_keywords):
            if "play_music" not in actions:
                actions.append("play_music")
        
        # Default fallback: add variety if no specific context
        if not actions:
            medications = MedicationCRUD.get_user_medications(user_id)
            if medications:
                actions.append("log_medication")
            actions.append("play_music")
        
        # Always consider memory cue as a gentle engagement option
        if len(actions) < 3 and "memory_cue" not in actions:
            actions.append("memory_cue")
        
        # If still need more, add fun corner
        if len(actions) < 3 and "fun_corner" not in actions:
            actions.append("fun_corner")
        
        return actions[:3]
    
    def handle_play_music(self) -> Dict[str, Any]:
        """Return a relaxing music recommendation"""
        music_options = [
            {"title": "Peaceful Piano Music", "url": "https://www.youtube.com/watch?v=7emS3ye3cIY"},
            {"title": "Calming Nature Sounds", "url": "https://www.youtube.com/watch?v=eKFTSSKCzWA"},
            {"title": "Gentle Classical Music", "url": "https://www.youtube.com/watch?v=jgpJVI3tDbY"},
            {"title": "Relaxing Guitar Music", "url": "https://www.youtube.com/watch?v=4bMr6vKXnkw"},
        ]
        
        import random
        selected = random.choice(music_options)
        
        return {
            "message": f"Here's something relaxing 🎵\n\n🎶 {selected['title']}\n{selected['url']}",
            "music_url": selected['url'],
            "music_title": selected['title']
        }
    
    def handle_fun_corner(self, corner_type: str = "joke") -> str:
        """Return either a joke or puzzle"""
        import random
        
        if corner_type == "joke":
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What do you call a fish wearing a bowtie? So-fish-ticated!",
                "Why did the scarecrow win an award? He was outstanding in his field!",
                "What do you call a bear with no teeth? A gummy bear!"
            ]
            return random.choice(jokes)
        else:
            puzzles = [
                "What has keys but no locks, space but no room, and you can enter but can't go in? (Answer: A keyboard)",
                "I have cities but no houses, forests but no trees, and water but no fish. What am I? (Answer: A map)",
                "What gets wet while drying? (Answer: A towel)",
                "What can you catch but not throw? (Answer: A cold)"
            ]
            return random.choice(puzzles)
    
    def generate_memory_cue(self, user_id: int) -> str:
        """Generate a gentle memory recall question from personal data"""
        user = UserCRUD.get_user(user_id)
        personal_events = PersonalEventCRUD.get_user_events(user_id, limit=10)
        medications = MedicationCRUD.get_user_medications(user_id)
        
        questions = []
        
        # Questions about medications
        if medications:
            med_names = [med.name for med in medications[:3]]
            questions.extend([
                f"Do you remember what medication you take in the morning?",
                f"Can you tell me the name of your heart medication?",
            ])
        
        # Questions about personal events
        if personal_events:
            for event in personal_events[:2]:
                if event.event_type == "family":
                    questions.append(f"Do you remember {event.title}?")
                elif event.event_type == "birthday":
                    questions.append(f"Can you tell me when {event.title} is?")
        
        # General questions
        questions.extend([
            "What did you have for breakfast this morning?",
            "Can you tell me what day of the week it is today?",
            "Do you remember what we talked about earlier today?"
        ])
        
        import random
        return random.choice(questions) if questions else "What's your favorite memory from this week?"

    def _local_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Local keyword-based sentiment analysis (no API call)"""
        text_lower = text.lower()
        
        positive_words = ["good", "great", "happy", "wonderful", "excellent", "love", "enjoy",
                         "better", "fine", "well", "nice", "pleasant", "comfortable", "peaceful"]
        
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "pain", "hurt", "sad",
                         "worried", "anxious", "confused", "lost", "dizzy", "sick", "tired",
                         "lonely", "scared", "frightened", "depressed", "upset"]
        
        concern_words = ["pain", "hurt", "dizzy", "fall", "emergency", "help", "confused",
                        "memory", "forgot", "lost", "scared", "can't", "unable", "difficult"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        concern_count = sum(1 for word in concern_words if word in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return {"score": 0, "label": "neutral", "confidence": 0.6, "emotions": []}
        
        score = (positive_count - negative_count - (concern_count * 1.5)) / total_words
        score = max(-1, min(1, score))
        
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        emotions = []
        if concern_count > 0:
            emotions.append("concern")
        if any(word in text_lower for word in ["pain", "hurt", "sick"]):
            emotions.append("discomfort")
        if any(word in text_lower for word in ["lonely", "alone", "miss"]):
            emotions.append("loneliness")
        if any(word in text_lower for word in ["happy", "good", "great"]):
            emotions.append("contentment")
        
        return {"score": score, "label": label, "confidence": 0.6, "emotions": emotions}
    
    def _local_emergency_detection(self, message: str, user_id: int) -> Dict[str, Any]:
        """Local keyword-based emergency detection (no API call)"""
        message_lower = message.lower()
        
        # Critical emergency keywords
        critical_keywords = ["chest pain", "can't breathe", "cannot breathe", "heart attack", 
                            "stroke", "fell", "bleeding", "unconscious", "dizzy", "fainted"]
        
        # High severity keywords
        high_keywords = ["pain", "hurt", "emergency", "help me", "fallen", "can't move",
                        "difficulty breathing", "severe", "blood"]
        
        # Medium severity keywords
        medium_keywords = ["dizzy", "confused", "nausea", "headache", "weak", "tired", 
                          "worried", "scared", "anxious"]
        
        is_critical = any(keyword in message_lower for keyword in critical_keywords)
        is_high = any(keyword in message_lower for keyword in high_keywords)
        is_medium = any(keyword in message_lower for keyword in medium_keywords)
        
        if is_critical:
            return {
                "is_emergency": True,
                "severity": "critical",
                "concerns": ["Possible medical emergency detected"],
                "should_alert": True
            }
        elif is_high:
            return {
                "is_emergency": True,
                "severity": "high",
                "concerns": ["Concerning symptoms reported"],
                "should_alert": True
            }
        elif is_medium:
            return {
                "is_emergency": False,
                "severity": "medium",
                "concerns": ["Minor health concern mentioned"],
                "should_alert": False
            }
        
        return {
            "is_emergency": False,
            "severity": "none",
            "concerns": [],
            "should_alert": False
        }

    def should_alert_caregiver(self, user_id: int, sentiment_score: float,
                               message: str) -> bool:
        """Determine if caregiver should be alerted based on conversation"""
        # Alert for very negative sentiment
        if sentiment_score < -0.7:
            return True

        # Check for concerning keywords
        concerning_keywords = [
            "pain", "hurt", "dizzy", "fall", "fell", "emergency", "help",
            "can't breathe", "chest pain", "confused", "lost", "scared"
        ]

        message_lower = message.lower()
        return any(keyword in message_lower for keyword in concerning_keywords)

    def _get_next_medication_time(self, user_id: int) -> str:
        """Get the next scheduled medication time for a user"""
        medications = MedicationCRUD.get_user_medications(user_id, active_only=True)
        if not medications:
            return "You don't have any medications scheduled right now."
        
        current_time = now_central()
        next_med = None
        next_time = None
        
        for med in medications:
            try:
                schedule_times = json.loads(med.schedule_times)
                for time_str in schedule_times:
                    # Parse time string (HH:MM format)
                    hour, minute = map(int, time_str.split(':'))
                    
                    # Create datetime for today
                    scheduled_dt = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # If time has passed today, check tomorrow
                    if scheduled_dt <= current_time:
                        scheduled_dt = scheduled_dt + timedelta(days=1)
                    
                    # Track the earliest next medication
                    if next_time is None or scheduled_dt < next_time:
                        next_time = scheduled_dt
                        next_med = med
            except (json.JSONDecodeError, ValueError):
                continue
        
        if next_med and next_time:
            time_str = next_time.strftime("%I:%M %p %Z")
            return f"Your next {next_med.name} is due at {time_str}."
        
        return "I couldn't find your next medication time."

    def generate_response(
            self,
            user_id: int,
            user_message: str,
            conversation_type: str = "general") -> Dict[str, Any]:
        """Generate AI response with context and tools using memory system"""
        try:
            message_lower = user_message.lower()
            
            # FIRST: Check if this is a medication timing query (handle without LLM)
            medication_timing_keywords = ['when should i take', 'next medication', 'next dose', 
                                         'meds due', 'medication due', 'what time are my meds',
                                         'when is my medication', 'medication schedule',
                                         'next pill', 'when do i take']
            is_med_timing_query = any(keyword in message_lower for keyword in medication_timing_keywords)
            
            if is_med_timing_query:
                med_response = self._get_next_medication_time(user_id)
                
                # Save this simple interaction
                ConversationCRUD.save_conversation(
                    user_id=user_id,
                    message=user_message,
                    response=med_response,
                    conversation_type="medication_schedule_query"
                )
                
                return {
                    "response": med_response,
                    "sentiment_score": 0.5,
                    "sentiment_label": "helpful",
                    "alert_sent": False,
                    "quick_actions": ["log_medication"],
                    "is_emergency": False
                }
            
            # SECOND: Check if this is a current time query (handle without LLM)
            # Exclude medication-related queries by checking they're not about meds/pills/dose
            time_query_keywords = ['time now', 'current time', 'what\'s the time', 
                                   'tell me the time', 'time is it']
            # Only match if not asking about medication
            is_time_query = (any(keyword in message_lower for keyword in time_query_keywords) and
                           not any(med_word in message_lower for med_word in ['med', 'pill', 'dose', 'medication']))
            
            # Also handle "what time" if it's clearly about current time, not meds
            if 'what time' in message_lower and not is_med_timing_query:
                # If "what time" is followed by "is it" or similar, it's asking current time
                if any(phrase in message_lower for phrase in ['what time is it', 'what time it is']):
                    is_time_query = True
            
            if is_time_query:
                # Get current time using timezone utility
                current_time = now_central()
                time_str = current_time.strftime("%I:%M %p %Z")
                time_response = f"It's {time_str} right now."
                
                # Save this simple interaction
                ConversationCRUD.save_conversation(
                    user_id=user_id,
                    message=user_message,
                    response=time_response,
                    conversation_type="time_query"
                )
                
                return {
                    "response": time_response,
                    "sentiment_score": 0.5,
                    "sentiment_label": "neutral",
                    "alert_sent": False,
                    "quick_actions": [],
                    "is_emergency": False
                }
            
            # Check if this is a memory-specific query
            memory_query_keywords = ['remember', 'talked about', 'medication schedule', 
                                   'breakfast', 'lunch', 'dinner', 'meal', 
                                   'yesterday', 'summary', 'discussed']
            is_memory_query = any(keyword in user_message.lower() for keyword in memory_query_keywords)
            
            # Use memory manager for memory-specific queries
            if is_memory_query:
                memory_response = self.memory_manager.recall_information(user_id, user_message)
                if memory_response and len(memory_response) > 20:
                    # Save to database first
                    conversation = ConversationCRUD.save_conversation(
                        user_id=user_id,
                        message=user_message,
                        response=memory_response,
                        conversation_type="memory_query"
                    )
                    
                    # Add to vector store incrementally
                    self.memory_manager.add_conversation(
                        user_id=user_id,
                        conversation_id=conversation.id,
                        user_message=user_message,
                        assistant_response=memory_response,
                        timestamp=conversation.timestamp
                    )
                    
                    return {
                        "response": memory_response,
                        "sentiment_score": 0.5,
                        "sentiment_label": "helpful",
                        "alert_sent": False,
                        "quick_actions": [],
                        "is_emergency": False
                    }
            
            # Get full context from all memory layers
            memory_context = self.memory_manager.get_full_context(user_id, user_message)

            # Get user info
            user = UserCRUD.get_user(user_id)
            user_name = user.name if user else "there"

            # Use local fallback sentiment analysis (no API call)
            sentiment_result = self._local_sentiment_analysis(user_message)
            sentiment_score = sentiment_result.get("score", 0)
            sentiment_label = sentiment_result.get("label", "neutral")
            
            # Use local keyword-based emergency detection (no API call)
            emergency_result = self._local_emergency_detection(user_message, user_id)
            is_emergency = emergency_result.get("is_emergency", False)
            emergency_severity = emergency_result.get("severity", "manageable")
            emergency_concerns = emergency_result.get("concerns", [])
            should_alert = emergency_result.get("should_alert", False)
            
            # If emergency detected, prepare reassurance message
            emergency_context = ""
            if is_emergency:
                emergency_context = "\nIMPORTANT: The user is experiencing emergency symptoms. Provide immediate reassurance and comfort."

            # Build the prompt with comprehensive memory context
            prompt = f"""{memory_context}

User's name: {user_name}
Conversation type: {conversation_type}
Current message: {user_message}{emergency_context}

Please respond as Carely, keeping in mind:
- The user's profile, medications, and preferences from the context above
- Recent conversation history and past relevant conversations
- Daily summaries and patterns
- Be warm, caring, and supportive
- If they ask about medications, reference their actual schedule
- If they ask about past conversations, use the relevant past conversations provided
- Reference their personal information naturally when relevant
- If this is an emergency situation, provide immediate reassurance and comfort

Respond naturally and warmly based on ALL the context provided."""

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": self.system_prompt
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=230,
                stop=["\n\n", "\n\n\n"])

            ai_response = response.choices[0].message.content
            
            # Post-trim: limit to at most 4 short sentences
            ai_response = self._limit_to_sentences(ai_response, max_sentences=4)
            
            # If emergency, prepend reassurance message
            if is_emergency:
                reassurance = "I'm here with you. I'm notifying your caregiver now so help can reach you quickly. Try to sit comfortably and focus on slow breaths. You're not alone.\n\n"
                ai_response = reassurance + ai_response

            # Save conversation to database
            conversation = ConversationCRUD.save_conversation(
                user_id=user_id,
                message=user_message,
                response=ai_response,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                conversation_type=conversation_type)
            
            # Add conversation to vector store incrementally
            self.memory_manager.add_conversation(
                user_id=user_id,
                conversation_id=conversation.id,
                user_message=user_message,
                assistant_response=ai_response,
                timestamp=conversation.timestamp
            )

            # Check if caregiver alert is needed
            alert_sent = False
            if self.should_alert_caregiver(user_id, sentiment_score,
                                           user_message):
                self.alert_caregiver_tool(
                    user_id=user_id,
                    alert_type="mood_concern",
                    description=
                    f"User expressed concerning sentiment: '{user_message}' (sentiment: {sentiment_label})",
                    severity="medium" if sentiment_score > -0.8 else "high")
                alert_sent = True

            # Determine quick action buttons (2-3 relevant buttons)
            quick_actions = self.determine_quick_actions(user_message, user_id)
            
            return {
                "response": ai_response,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "alert_sent": alert_sent,
                "conversation_id": conversation.id,
                "is_emergency": is_emergency,
                "emergency_severity": emergency_severity,
                "emergency_concerns": emergency_concerns,
                "should_alert": should_alert,
                "quick_actions": quick_actions
            }

        except Exception as e:
            # Check if it's a rate limit error
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                error_response = "I'm getting a lot of requests right now and need a moment to catch my breath! Please wait just a minute and try again. I'm still here for you!"
            else:
                error_response = f"I'm sorry, I'm having a bit of trouble right now. But I'm here for you! Is there anything specific you'd like to talk about or any way I can help you today?"

            # Still save the conversation attempt
            ConversationCRUD.save_conversation(
                user_id=user_id,
                message=user_message,
                response=error_response,
                conversation_type=conversation_type)

            return {
                "response": error_response,
                "sentiment_score": 0,
                "sentiment_label": "neutral",
                "alert_sent": False,
                "error": str(e)
            }

    def conduct_daily_checkin(self,
                              user_id: int,
                              checkin_type: str = "morning") -> Dict[str, Any]:
        """Conduct a daily check-in with the user"""
        user = UserCRUD.get_user(user_id)
        user_name = user.name if user else "there"

        checkin_prompts = {
            "morning":
            f"Good morning, {user_name}! I hope you slept well. How are you feeling this morning? Did you take your morning medications?",
            "afternoon":
            f"Good afternoon, {user_name}! How has your day been so far? Are you feeling alright?",
            "evening":
            f"Good evening, {user_name}! How was your day? Did you remember to take all your medications today?"
        }

        prompt = checkin_prompts.get(checkin_type, checkin_prompts["morning"])

        # For check-ins, we don't wait for user response - we just send the prompt
        # The user can respond through the normal chat interface
        return {
            "prompt": prompt,
            "checkin_type": checkin_type,
            "scheduled_time": now_central()
        }
