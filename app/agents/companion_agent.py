import os
import json
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.timezone_utils import now_central, to_central
from typing import Dict, Any, List
from groq import Groq

# Load environment variables
load_dotenv()
from app.database.crud import (ConversationCRUD, MedicationCRUD,
                               MedicationLogCRUD, CaregiverAlertCRUD, UserCRUD,
                               PersonalEventCRUD)
from utils.sentiment_analysis import analyze_sentiment
from utils.emergency_detection import detect_emergency
from app.memory.memory_manager import MemoryManager
from utils.pii_redaction import PIIRedactor, sanitize_before_storage, generate_safe_response_prompt
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["TELEGRAM_BOT_TOKEN"] = st.secrets["TELEGRAM_BOT_TOKEN"]
    os.environ["TELEGRAM_CHAT_ID"] = st.secrets["TELEGRAM_CHAT_ID"]

class CompanionAgent:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"  # Using Groq model
        self.memory_manager = MemoryManager()  # Initialize memory system

    def _get_system_prompt(self) -> str:
        """Generate system prompt with current time context"""
        # Get current time in Central Time
        current_time = now_central()
        hour = current_time.hour
        
        # Determine time of day
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Format current time
        time_str = current_time.strftime("%I:%M %p %Z")
        
        # System prompt for elderly care companion with time context
        return f"""You are Carely, a warm, empathetic AI companion for elderly care.

CURRENT TIME CONTEXT:
- Current time: {time_str}
- Time of day: {time_of_day}
- User location: Chicago, IL (Central Time)

ADAPTIVE RESPONSE STYLE:
- Default to concise responses (~4 short sentences) for casual conversation and simple questions.
- If the user asks for something complex, story-like, explanations, or multi-step instructions, provide a fuller, structured answer.
- If unsure about the detail level needed, ask for clarification.

CRITICAL RULES:
1. NO repetition, filler, or rambling. Every word must add value.
2. If uncertain or missing information, ask EXACTLY 1 clarifying question.
3. For time questions: use the current time provided above ({time_str}). NEVER guess or make up times.
4. For medications, schedules, and appointments: call tools or check database and quote results EXACTLY as provided.
5. Warm, caring tone. Think of a caring friend who adapts to your needs.
6. DO NOT repeat greetings like "Good morning/afternoon/evening" in every message.
7. DO NOT say "It's lovely to chat with you at [time]" repeatedly.
8. Continue conversations naturally without re-introducing yourself or stating the time.
9. Only greet the user at the very start of a new conversation, not in follow-up messages.

YOUR ROLE:
- Medication reminders and tracking
- Daily wellness check-ins  
- Emotional support and companionship
- Alert caregivers when needed
- Remember personal details

Be gentle, patient, and use simple everyday language. Never use medical jargon.
Focus on continuing the conversation naturally, as if you're already in the middle of a friendly chat.
"""

    def _limit_to_sentences(self, text: str, max_sentences: int = 4) -> str:
        """
        Limit text to at most max_sentences. If exceeds, reduce to fit max_sentences.
        """
        if not text:
            return text
        
        # Split on sentence boundaries (., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # If exceeds max_sentences, return first max_sentences
        return ' '.join(sentences[:max_sentences])
    
    def _decide_verbosity(self, user_text: str) -> str:
        """
        Decide response verbosity level based on user intent and complexity.
        Returns: "SHORT", "MEDIUM", or "LONG"
        """
        try:
            # Use LLM to classify verbosity need
            classification_prompt = f"""Analyze this user message and classify the required response detail level.

User message: "{user_text}"

Consider:
- SHORT: casual chat, greetings, simple yes/no, quick factual answers, small talk
- MEDIUM: moderate explanations, summaries, basic how-to, short reasoning
- LONG: detailed instructions, multi-part questions, stories, complex explanations, step-by-step guides

Respond ONLY with valid JSON in this exact format:
{{"verbosity":"SHORT"}}
or {{"verbosity":"MEDIUM"}}
or {{"verbosity":"LONG"}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": "You are a verbosity classifier. Return only JSON."
                }, {
                    "role": "user",
                    "content": classification_prompt
                }],
                temperature=0.1,
                max_tokens=50)
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result_json = json.loads(result_text)
            verbosity = result_json.get("verbosity", "SHORT")
            
            # Validate
            if verbosity in ["SHORT", "MEDIUM", "LONG"]:
                return verbosity
            else:
                return "SHORT"
                
        except Exception as e:
            # Fallback heuristic if LLM classification fails
            text_lower = user_text.lower()
            
            # LONG indicators
            long_indicators = ["step by step", "step-by-step", "explain", "tell me about", 
                             "how do i", "how can i", "story", "describe", "what happened",
                             "walk me through", "detail", "instruction"]
            
            # MEDIUM indicators  
            medium_indicators = ["why", "how", "what is", "what are", "summary", 
                               "summarize", "compare", "difference"]
            
            # Check for multiple questions (indicates complexity)
            question_marks = user_text.count("?")
            
            if any(indicator in text_lower for indicator in long_indicators) or question_marks >= 2:
                return "LONG"
            elif any(indicator in text_lower for indicator in medium_indicators):
                return "MEDIUM"
            else:
                return "SHORT"

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

    def _detect_user_intent(self, user_input: str) -> Dict[str, Any]:
        """Use AI to detect what user wants to do"""
        
        prompt = f"""Analyze this user message and determine their intent with high accuracy:
Message: "{user_input}"

Possible intents:
- log_medication: User is CONFIRMING they took/have taken their medication (e.g., "I took my pill", "Just had my medication", "I already logged it")
- ask_medication: User is ASKING about medication, not confirming they took it (e.g., "Did I take my pill?", "What's my medication?", "Should I take it?")
- ask_schedule: User asking about their schedule or appointments
- emergency: User needs urgent help or expressing pain/distress
- mood_check: User expressing emotions or feelings
- general_chat: Normal conversation

IMPORTANT: Only classify as "log_medication" if the user is CLEARLY STATING they took the medication, not asking questions about it.
Questions like "Did I take..." or "Should I take..." are "ask_medication", NOT "log_medication".

Return JSON only:
{{
    "type": "intent_type",
    "confidence": 0.95,
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert intent classifier. You must distinguish between statements (user took medication) and questions (user asking about medication). Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except Exception as e:
            # If AI fails, default to general chat - don't auto-log anything
            return {"type": "general_chat", "confidence": 0.5, "reasoning": "AI classification failed, defaulting to safe option"}

    def _extract_medication_details(self, user_id: int, user_input: str) -> Dict[str, Any]:
        """Extract which medication and any notes from natural language"""
        
        # Get user's medications
        medications = MedicationCRUD.get_user_medications(user_id, active_only=True)
        if not medications:
            return {"medication_id": None, "medication_name": None, "notes": "", "confidence": 0.0}
        
        med_list = "\n".join([f"- {med.name} (ID: {med.id})" for med in medications])
        
        prompt = f"""User said: "{user_input}"

Available medications:
{med_list}

Extract:
1. Which medication they took (match to list above, use closest match if not exact)
2. Any additional notes (side effects, timing, feelings, etc.)

Return JSON only:
{{
    "medication_id": 1,
    "medication_name": "name",
    "notes": "extracted notes or empty string",
    "confidence": 0.95
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical information extractor. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except Exception as e:
            # Fallback: try to match medication name in user input
            user_lower = user_input.lower()
            for med in medications:
                if med.name.lower() in user_lower:
                    return {
                        "medication_id": med.id,
                        "medication_name": med.name,
                        "notes": "",
                        "confidence": 0.7
                    }
            
            return {"medication_id": None, "medication_name": None, "notes": "", "confidence": 0.0}

    def _get_pending_medications(self, user_id: int) -> List[Dict[str, Any]]:
        """Get medications not yet taken today"""
        current_time = now_central()
        medications = MedicationCRUD.get_user_medications(user_id, active_only=True)
        
        pending = []
        for med in medications:
            try:
                schedule_times = json.loads(med.schedule_times) if med.schedule_times else []
                for time_str in schedule_times:
                    hour, minute = map(int, time_str.split(':'))
                    scheduled_time = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # Check if time has passed and not logged
                    if scheduled_time <= current_time:
                        # Check if already logged
                        recent_log = MedicationLogCRUD.check_recent_medication_log(
                            user_id=user_id,
                            medication_id=med.id,
                            hours=6
                        )
                        if not recent_log:
                            pending.append({
                                "medication": med,
                                "scheduled_time": scheduled_time,
                                "time_str": time_str
                            })
                            break  # Only add once per medication
            except Exception:
                continue
        
        return pending

    def generate_proactive_greeting(self, user_id: int) -> str:
        """Generate a contextual proactive greeting when user opens chat"""
        current_time = now_central()
        hour = current_time.hour
        
        # Determine time of day
        if 5 <= hour < 12:
            time_of_day = "morning"
            greeting = "Good morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
            greeting = "Good afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
            greeting = "Good evening"
        else:
            time_of_day = "night"
            greeting = "Good evening"
        
        # Get user context
        user = UserCRUD.get_user(user_id)
        recent_convs = ConversationCRUD.get_user_conversations(user_id, limit=3)
        pending_meds = self._get_pending_medications(user_id)
        
        # Get upcoming events today
        try:
            from app.database.models import PersonalEvent, get_session
            from sqlmodel import select
            
            with get_session() as session:
                today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                today_end = today_start + timedelta(days=1)
                
                query = select(PersonalEvent).where(
                    PersonalEvent.user_id == user_id,
                    PersonalEvent.event_date.isnot(None)
                )
                all_events = session.exec(query).all()
                
                upcoming_events = []
                for event in all_events:
                    event_time = to_central(event.event_date)
                    if today_start <= event_time < today_end and event_time >= current_time:
                        upcoming_events.append(event)
        except Exception:
            upcoming_events = []
        
        # Build context for AI
        recent_mood = "unknown"
        if recent_convs and recent_convs[0].sentiment_score is not None:
            score = recent_convs[0].sentiment_score
            if score > 0.3:
                recent_mood = "positive"
            elif score < -0.3:
                recent_mood = "negative"
            else:
                recent_mood = "neutral"
        
        context = f"""Generate a brief, warm, proactive greeting (2-3 sentences) for {user.name}.

Current context:
- Time: {current_time.strftime('%I:%M %p')} {time_of_day}
- Recent mood: {recent_mood}
- Pending medications today: {len(pending_meds)}
- Upcoming events today: {len(upcoming_events)}

Guidelines:
1. Start with appropriate time-based greeting ({greeting})
2. If there are pending medications or upcoming events, mention ONE of them briefly
3. Ask how they're doing or offer help
4. Keep it warm, natural, and conversational
5. Do NOT sound robotic or repetitive

Example good greetings:
- "Good morning, Dorothy! ☀️ I see you have your blood pressure medication due soon. How are you feeling today?"
- "Good afternoon! Hope you're having a nice day. Just a reminder you have an appointment later. Need anything?"
- "Good evening! How has your day been? I'm here if you need to chat or check your schedule."

Generate greeting:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback greeting
            if pending_meds:
                return f"{greeting}, {user.name}! I see you have {len(pending_meds)} medication{'s' if len(pending_meds) > 1 else ''} due today. How are you feeling?"
            elif upcoming_events:
                return f"{greeting}, {user.name}! You have {len(upcoming_events)} event{'s' if len(upcoming_events) > 1 else ''} scheduled today. Need anything?"
            else:
                return f"{greeting}, {user.name}! How are you doing today? I'm here to help with anything you need. 😊"

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
            {"title": "Music", "url": "https://www.youtube.com/watch?v=D1lH55N72U0&list=RDD1lH55N72U0&start_radio=1"},
            {"title": "Music", "url": "https://www.youtube.com/watch?v=6FOUqQt3Kg0"},
            {"title": "Music", "url": "https://www.youtube.com/watch?v=8jCFzreP1ng&list=RD8jCFzreP1ng&start_radio=1"},
            {"title": "Music", "url": "https://www.youtube.com/watch?v=9Qp_SrTgBBs&list=RD9Qp_SrTgBBs&start_radio=1"},
            {"title": "Music", "url": "https://www.youtube.com/watch?v=Ms4KTpdx1wY&list=RDMs4KTpdx1wY&start_radio=1"},
            {"title": "Music", "url": "https://www.youtube.com/watch?v=nsCwpwGi9uE&list=RDnsCwpwGi9uE&start_radio=1"},
            {"title": "Music", "url": "https://www.youtube.com/watch?v=uA4mfu_5TyI&list=RDuA4mfu_5TyI&start_radio=1"},
        ]
        
        import random
        selected = random.choice(music_options)
        
        return {
            "message": f"Here's your favorite music 🎵\n\n{selected['url']}",
            "music_url": selected['url'],
            "music_title": "Music"
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
            
            # SECOND: Check if this is a current time/date query (handle without LLM)
            # Exclude medication-related queries by checking they're not about meds/pills/dose
            time_query_keywords = ['time now', 'current time', 'what\'s the time', 
                                   'tell me the time', 'time is it', 'what is the time',
                                   'what time is', 'what\'s time']
            date_query_keywords = ['what is the date', 'what\'s the date', 'what date',
                                   'what day is it', 'what\'s the day', 'what is the day',
                                   'date today', 'day today', 'today\'s date']
            datetime_query_keywords = ['day, time and date', 'date and time', 'time and date',
                                       'day and time', 'time, date', 'date, time']
            
            # Check if asking about date and/or time (not medication-related)
            is_datetime_query = any(keyword in message_lower for keyword in datetime_query_keywords)
            is_time_query = (any(keyword in message_lower for keyword in time_query_keywords) and
                           not any(med_word in message_lower for med_word in ['med', 'pill', 'dose', 'medication']))
            is_date_query = any(keyword in message_lower for keyword in date_query_keywords)
            
            # Also handle "what time" if it's clearly about current time, not meds
            if 'what time' in message_lower and not is_med_timing_query:
                # If "what time" is followed by "is it" or similar, it's asking current time
                if any(phrase in message_lower for phrase in ['what time is it', 'what time it is']):
                    is_time_query = True
            
            # Handle date/time queries deterministically
            if is_datetime_query or is_time_query or is_date_query:
                # Get current time using timezone utility
                current_time = now_central()
                
                # Build appropriate response based on what was asked
                if is_datetime_query:
                    # Full date and time
                    day_name = current_time.strftime("%A")  # e.g., "Friday"
                    time_str = current_time.strftime("%I:%M %p %Z")  # e.g., "8:15 PM CST"
                    date_str = current_time.strftime("%B %d, %Y")  # e.g., "November 1, 2025"
                    time_response = f"It's currently {time_str} on {day_name}, {date_str}."
                elif is_date_query:
                    # Just date
                    day_name = current_time.strftime("%A")
                    date_str = current_time.strftime("%B %d, %Y")
                    time_response = f"Today is {day_name}, {date_str}."
                else:
                    # Just time
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
            
            # AI-Driven Intent Detection (do once for all medication handling)
            intent = self._detect_user_intent(user_message)
            
            # THIRD: AI-Driven Medication Information Queries
            # Handle questions about medications (not logging) with AI + log data
            if intent["type"] == "ask_medication" and intent["confidence"] > 0.6:
                # Get user's medications and today's logs
                medications = MedicationCRUD.get_user_medications(user_id, active_only=True)
                all_logs = MedicationLogCRUD.get_user_logs(user_id, limit=20)
                
                # Filter today's logs (handle timezone-aware/naive comparison)
                today_start = now_central().replace(hour=0, minute=0, second=0, microsecond=0)
                today_logs = []
                for log in all_logs:
                    if log['taken_at']:
                        # Convert to timezone-aware if needed
                        taken_at = to_central(log['taken_at']) if log['taken_at'].tzinfo is None else log['taken_at']
                        if taken_at >= today_start:
                            today_logs.append(log)
                
                # Build context for AI
                med_list = "\n".join([f"- {med.name} (Schedule: {med.schedule_times})" for med in medications]) if medications else "No medications prescribed"
                
                if today_logs:
                    log_list = "\n".join([
                        f"- {log['medication_name']} at {to_central(log['taken_at']).strftime('%I:%M %p')}" 
                        for log in today_logs
                    ])
                    log_context = f"Medications taken today:\n{log_list}"
                else:
                    log_context = "No medications logged today yet."
                
                # Let AI generate personalized response with context
                ai_prompt = f"""The user asked: "{user_message}"

User's prescribed medications:
{med_list}

{log_context}

Provide a helpful, conversational response that:
1. Answers their question accurately based on the data above
2. Is warm and supportive
3. Keeps it brief (2-3 sentences max)
4. If they're asking if they took something, check the logs and tell them
5. If they're asking how many medications, count from the prescribed list"""

                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful medical companion. Answer questions about medications based on the provided data."},
                            {"role": "user", "content": ai_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=200
                    )
                    response_text = response.choices[0].message.content.strip()
                except Exception as e:
                    # Fallback response
                    med_count = len(medications)
                    if med_count > 0:
                        response_text = f"You have {med_count} medications prescribed: {', '.join([m.name for m in medications])}."
                    else:
                        response_text = "You don't have any medications prescribed in the system yet."
                
                # Save conversation
                ConversationCRUD.save_conversation(
                    user_id=user_id,
                    message=user_message,
                    response=response_text,
                    conversation_type="medication_inquiry"
                )
                
                return {
                    "response": response_text,
                    "sentiment_score": 0.5,
                    "sentiment_label": "helpful",
                    "alert_sent": False,
                    "quick_actions": [],
                    "is_emergency": False
                }
            
            # FOURTH: AI-Driven Medication Logging Detection
            # Let AI intelligently detect if user is confirming they took medication
            
            # Only proceed if AI is confident this is medication logging (not asking)
            if intent["type"] == "log_medication" and intent["confidence"] > 0.75:
                # Extract medication details using AI
                med_details = self._extract_medication_details(user_id, user_message)
                
                # Require high confidence to auto-log
                if med_details["medication_id"] and med_details["confidence"] > 0.7:
                    # Automatically log the medication
                    log_result = self.log_medication_tool(
                        user_id=user_id,
                        medication_id=med_details["medication_id"],
                        notes=med_details["notes"]
                    )
                    
                    # Generate a natural response
                    response_text = log_result
                    
                    # Save conversation
                    ConversationCRUD.save_conversation(
                        user_id=user_id,
                        message=user_message,
                        response=response_text,
                        conversation_type="medication_logging"
                    )
                    
                    return {
                        "response": response_text,
                        "sentiment_score": 0.5,
                        "sentiment_label": "positive",
                        "alert_sent": False,
                        "quick_actions": [],
                        "is_emergency": False,
                        "action_taken": "medication_logged",
                        "medication_id": med_details["medication_id"]
                    }
                elif med_details["confidence"] <= 0.6 and med_details["medication_id"]:
                    # Low confidence, ask for confirmation
                    response_text = f"Just to confirm - did you take your {med_details['medication_name']}? I can log that for you."
                    
                    return {
                        "response": response_text,
                        "sentiment_score": 0.5,
                        "sentiment_label": "neutral",
                        "alert_sent": False,
                        "quick_actions": ["log_medication"],
                        "is_emergency": False
                    }
                else:
                    # Couldn't identify medication, ask which one
                    medications = MedicationCRUD.get_user_medications(user_id, active_only=True)
                    if medications:
                        med_list = ", ".join([med.name for med in medications[:3]])
                        response_text = f"I'd be happy to log your medication! Which one did you take? Your medications include: {med_list}."
                    else:
                        response_text = "I'd love to help log your medication, but I don't see any medications in your schedule. Would you like to add one?"
                    
                    return {
                        "response": response_text,
                        "sentiment_score": 0.5,
                        "sentiment_label": "helpful",
                        "alert_sent": False,
                        "quick_actions": ["log_medication"],
                        "is_emergency": False
                    }
            
            # FOURTH: Deterministic "yesterday/day before" summary handling
            # Broaden detection to cover common phrasings
            yesterday_keywords = ['yesterday', 'day before yesterday', 'two days ago']
            is_yesterday_query = any(keyword in message_lower for keyword in yesterday_keywords)
            
            # Expanded talk/summary indicators
            talk_indicators = ['talk', 'discuss', 'chat', 'conversation', 'tell me about', 
                             'what did', 'what happened', 'summary', 'recap']
            is_talk_query = any(indicator in message_lower for indicator in talk_indicators)
            
            if is_yesterday_query and is_talk_query:
                # Determine offset
                offset_days = 1 if 'yesterday' in message_lower and 'day before' not in message_lower else 2
                
                # Fetch summary using Central Time boundaries
                summary_data = self.memory_manager.fetch_summary_for_relative_day(user_id, offset_days)
                
                if summary_data:
                    # Bypass LLM - return only summary text + key topics
                    key_topics_str = ", ".join(summary_data['key_topics']) if summary_data['key_topics'] else "general conversation"
                    response_text = f"{summary_data['summary_text']}\n\nKey topics: {key_topics_str}"
                else:
                    offset_label = "yesterday" if offset_days == 1 else "the day before yesterday"
                    response_text = f"I don't have a summary from {offset_label} yet."
                
                # Save this simple interaction
                ConversationCRUD.save_conversation(
                    user_id=user_id,
                    message=user_message,
                    response=response_text,
                    conversation_type="yesterday_summary"
                )
                
                return {
                    "response": response_text,
                    "sentiment_score": 0.5,
                    "sentiment_label": "helpful",
                    "alert_sent": False,
                    "quick_actions": [],
                    "is_emergency": False
                }
            
            # FIFTH: Partial entity resolution (e.g., "meeting with Mary")
            # Check if message mentions partial event names
            event_mention_keywords = ['meeting', 'appointment', 'doctor', 'event', 'visit']
            has_event_mention = any(keyword in message_lower for keyword in event_mention_keywords)
            is_question = any(q in message_lower for q in ['when', 'what time', 'where', 'remind'])
            
            if has_event_mention and is_question:
                # Extract and sanitize potential event names
                import string
                words = user_message.split()
                potential_names = []
                
                for keyword in event_mention_keywords:
                    if keyword in message_lower:
                        try:
                            idx = [w.lower() for w in words].index(keyword)
                            # Get next 2-4 words after keyword
                            phrase = " ".join(words[idx:min(idx+4, len(words))])
                            
                            # Strip punctuation from the phrase
                            phrase = phrase.translate(str.maketrans('', '', string.punctuation))
                            
                            # Remove common stopwords
                            stopwords = ['with', 'the', 'a', 'an', 'at', 'on', 'in', 'for']
                            cleaned_words = [w for w in phrase.split() if w.lower() not in stopwords or w.lower() == keyword]
                            phrase = " ".join(cleaned_words)
                            
                            if phrase:
                                potential_names.append(phrase)
                        except ValueError:
                            pass
                
                # Try to find matching events
                if potential_names:
                    for potential_name in potential_names:
                        matches = PersonalEventCRUD.find_event_by_name(user_id, potential_name, window_days=7)
                        
                        if matches:
                            if len(matches) == 1:
                                # Single match - answer with date/time
                                event = matches[0]
                                event_time = to_central(event.event_date)
                                time_str = event_time.strftime("%B %d at %I:%M %p %Z")
                                response_text = f"Your {event.title} is scheduled for {time_str}."
                                
                                if event.description:
                                    response_text += f" {event.description}"
                                
                                # Save this simple interaction
                                ConversationCRUD.save_conversation(
                                    user_id=user_id,
                                    message=user_message,
                                    response=response_text,
                                    conversation_type="event_lookup"
                                )
                                
                                return {
                                    "response": response_text,
                                    "sentiment_score": 0.5,
                                    "sentiment_label": "helpful",
                                    "alert_sent": False,
                                    "quick_actions": [],
                                    "is_emergency": False
                                }
                            else:
                                # Multiple matches - ask clarifier
                                event_list = "\n".join([f"- {e.title} on {to_central(e.event_date).strftime('%B %d')}" 
                                                      for e in matches[:3]])
                                response_text = f"I found a few events. Which one did you mean?\n{event_list}"
                                
                                # Save this simple interaction
                                ConversationCRUD.save_conversation(
                                    user_id=user_id,
                                    message=user_message,
                                    response=response_text,
                                    conversation_type="event_clarification"
                                )
                                
                                return {
                                    "response": response_text,
                                    "sentiment_score": 0.5,
                                    "sentiment_label": "helpful",
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

            # Decide verbosity level based on user intent
            verbosity_level = self._decide_verbosity(user_message)
            
            # Set parameters based on verbosity
            if verbosity_level == "SHORT":
                max_tokens = 220
                sentence_limit = 4
            elif verbosity_level == "MEDIUM":
                max_tokens = 600
                sentence_limit = 8
            else:  # LONG
                max_tokens = 1200
                sentence_limit = None  # No sentence limit for detailed responses
            
            # Check for PII in user message BEFORE sending to AI
            detected_pii = PIIRedactor.detect_pii(user_message)
            pii_privacy_notice = generate_safe_response_prompt(detected_pii) if detected_pii else ""
            
            # Build the prompt with comprehensive memory context
            prompt = f"""{memory_context}

User's name: {user_name}
Conversation type: {conversation_type}
Current message: {user_message}{emergency_context}
{pii_privacy_notice}

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

            # Generate response with dynamic max_tokens
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": self._get_system_prompt()
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=max_tokens,
                stop=["\n\n", "\n\n\n"])

            ai_response = response.choices[0].message.content
            
            # Apply dynamic sentence limiting (only for SHORT and MEDIUM)
            if sentence_limit:
                ai_response = self._limit_to_sentences(ai_response, max_sentences=sentence_limit)
            
            # If emergency, prepend reassurance message
            if is_emergency:
                reassurance = "I'm here with you. I'm notifying your caregiver now so help can reach you quickly. Try to sit comfortably and focus on slow breaths. You're not alone.\n\n"
                ai_response = reassurance + ai_response

            # PII/PHI Detection and Redaction before storage
            user_msg_redacted, ai_response_redacted, contains_pii, pii_warning = sanitize_before_storage(
                user_message, ai_response
            )
            
            # If PII was detected, append warning to response (for user to see)
            ai_response_display = ai_response
            if contains_pii:
                ai_response_display = ai_response + "\n\n" + pii_warning

            # Save conversation to database with REDACTED versions
            conversation = ConversationCRUD.save_conversation(
                user_id=user_id,
                message=user_msg_redacted,  # Store redacted version
                response=ai_response_redacted,  # Store redacted version
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
                "response": ai_response_display,  # Return display version with PII warning
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "alert_sent": alert_sent,
                "conversation_id": conversation.id,
                "is_emergency": is_emergency,
                "emergency_severity": emergency_severity,
                "emergency_concerns": emergency_concerns,
                "should_alert": should_alert,
                "quick_actions": quick_actions,
                "contains_pii": contains_pii  # Flag for UI to show warning
            }

        except Exception as e:
            # Check if it's a rate limit error
            error_str = str(e)
            print(f"ERROR in generate_response: {error_str}")  # Debug logging
            import traceback
            traceback.print_exc()  # Print full traceback
            
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
