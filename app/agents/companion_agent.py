import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from groq import Groq
from app.database.crud import (ConversationCRUD, MedicationCRUD,
                               MedicationLogCRUD, CaregiverAlertCRUD, UserCRUD,
                               PersonalEventCRUD)
from utils.sentiment_analysis import analyze_sentiment
from utils.emergency_detection import detect_emergency


class CompanionAgent:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"  # Using Groq model

        # System prompt for elderly care companion
        self.system_prompt = """You are Carely, a warm, empathetic AI companion designed specifically for elderly care. Your role is to:

1. PERSONALITY: Be gentle, patient, understanding, and warm. Use a conversational tone that feels like talking to a caring friend or family member.

2. COMMUNICATION STYLE:
   - Keep responses clear and not too long
   - Avoid medical jargon - use simple, everyday language
   - Show genuine interest in their wellbeing
   - Remember and reference past conversations when appropriate
   - Be encouraging and supportive

3. CORE RESPONSIBILITIES:
   - Help with medication reminders and tracking
   - Conduct daily wellness check-ins
   - Provide emotional support and companionship
   - Alert caregivers when concerning patterns emerge
   - Remember personal details and preferences

4. SAFETY: If you detect signs of medical emergency, severe depression, or immediate danger, recommend contacting emergency services or their caregiver immediately.

5. TOOLS AVAILABLE:
   - log_medication: Record when medications are taken
   - check_schedule: View upcoming medications and reminders
   - alert_caregiver: Send alerts to designated caregivers for concerning patterns

Always respond with empathy and care, as if you're genuinely concerned about their wellbeing."""

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
            days_until = (event.event_date - datetime.now()).days
            context += f"- {event.title} ({event.event_type}) in {days_until} days"
            if event.description:
                context += f": {event.description}"
            context += "\n"

        return context

    def log_medication_tool(self,
                            user_id: int,
                            medication_name: str,
                            notes: str = "") -> str:
        """Tool to log medication intake"""
        try:
            medications = MedicationCRUD.get_user_medications(user_id)
            medication = next((med for med in medications
                               if medication_name.lower() in med.name.lower()),
                              None)

            if not medication:
                return f"I couldn't find a medication named '{medication_name}' in your schedule. Please check the spelling or ask your caregiver to add it."

            # Log the medication as taken
            MedicationLogCRUD.log_medication_taken(
                user_id=user_id,
                medication_id=medication.id,
                scheduled_time=datetime.now(),
                status="taken",
                notes=notes)

            return f"Great! I've recorded that you took your {medication.name} ({medication.dosage}) at {datetime.now().strftime('%I:%M %p')}."

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

    def generate_response(
            self,
            user_id: int,
            user_message: str,
            conversation_type: str = "general") -> Dict[str, Any]:
        """Generate AI response with context and tools"""
        try:
            # Get conversation context
            context = self.get_conversation_context(user_id)

            # Get personal events context
            events_context = self.get_personal_events_context(user_id)

            # Get user info
            user = UserCRUD.get_user(user_id)
            user_name = user.name if user else "there"

            # Analyze sentiment of user message
            sentiment_result = analyze_sentiment(user_message)
            sentiment_score = sentiment_result.get("score", 0)
            sentiment_label = sentiment_result.get("label", "neutral")
            
            # Detect emergency situations
            emergency_result = detect_emergency(user_message)
            is_emergency = emergency_result.get("is_emergency", False)
            emergency_severity = emergency_result.get("severity", "low")
            emergency_concerns = emergency_result.get("concerns", [])

            # Build the prompt
            prompt = f"""Context: {context}

{events_context}

User's name: {user_name}
Conversation type: {conversation_type}
Current message: {user_message}

Please respond as Carely, keeping in mind:
- This person's conversation history
- Their upcoming events and important dates
- The type of conversation (general chat, check-in, etc.)
- Be warm, caring, and supportive
- If they mention medications, offer to help log them
- If they seem distressed, offer appropriate support
- Reference their upcoming events naturally when relevant (e.g., "How are you feeling about your grandson's birthday coming up?")

Respond naturally and warmly."""

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
                max_tokens=512)

            ai_response = response.choices[0].message.content

            # Save conversation
            conversation = ConversationCRUD.save_conversation(
                user_id=user_id,
                message=user_message,
                response=ai_response,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                conversation_type=conversation_type)

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

            return {
                "response": ai_response,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "alert_sent": alert_sent,
                "conversation_id": conversation.id,
                "is_emergency": is_emergency,
                "emergency_severity": emergency_severity,
                "emergency_concerns": emergency_concerns
            }

        except Exception as e:
            # Fallback response
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
            "scheduled_time": datetime.now()
        }
