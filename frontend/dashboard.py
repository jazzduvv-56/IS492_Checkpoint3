import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import json
import os
from typing import List, Dict, Any
from streamlit_mic_recorder import speech_to_text

from app.database.crud import (
    UserCRUD, MedicationCRUD, ConversationCRUD, ReminderCRUD,
    MedicationLogCRUD, CaregiverAlertCRUD, CaregiverPatientCRUD
)
from app.agents.companion_agent import CompanionAgent
from utils.sentiment_analysis import analyze_sentiment, get_sentiment_emoji, get_sentiment_color
from utils.telegram_notification import send_emergency_alert
from utils.tts_helper import generate_speech_audio

def run_dashboard():
    """Main dashboard function"""
    st.title("🏥 Carely - AI Companion Dashboard")
    st.markdown("*Your caring AI companion for elderly care*")
    
    # Initialize session state
    if 'companion_agent' not in st.session_state:
        st.session_state.companion_agent = CompanionAgent()
    
    # Sidebar for user selection
    with st.sidebar:
        st.header("👤 Select User")
        users = UserCRUD.get_all_users()
        
        if not users:
            st.error("No users found. Please add users first.")
            show_user_management()
            return
        
        user_options = {f"{user.name} (ID: {user.id})": user.id for user in users}
        selected_user_key = st.selectbox("Choose a user:", list(user_options.keys()))
        selected_user_id = user_options[selected_user_key]
        
        st.divider()
        
        # Navigation
        st.header("📱 Navigation")
        page = st.radio(
            "Choose a section:",
            [
                "🏠 Overview",
                "💬 Chat with Carely",
                "💊 Medications",
                "📊 Health Insights",
                "🚨 Alerts & Reminders",
                "👥 User Management"
            ]
        )
    
    # Main content based on selected page
    if page == "🏠 Overview":
        show_overview(selected_user_id)
    elif page == "💬 Chat with Carely":
        show_chat_interface(selected_user_id)
    elif page == "💊 Medications":
        show_medication_management(selected_user_id)
    elif page == "📊 Health Insights":
        show_health_insights(selected_user_id)
    elif page == "🚨 Alerts & Reminders":
        show_alerts_and_reminders(selected_user_id)
    elif page == "👥 User Management":
        show_user_management()

def show_overview(user_id: int):
    """Show overview dashboard"""
    user = UserCRUD.get_user(user_id)
    if not user:
        st.error("User not found")
        return
    
    st.header(f"Overview for {user.name}")
    
    # Today's summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 User", user.name)
    
    with col2:
        # Medication adherence today
        adherence = MedicationLogCRUD.get_medication_adherence(user_id, days=1)
        adherence_rate = adherence.get("adherence_rate", 0)
        st.metric(
            "💊 Today's Adherence", 
            f"{adherence_rate:.0f}%",
            delta=f"{adherence.get('taken', 0)}/{adherence.get('total', 0)} doses"
        )
    
    with col3:
        # Recent mood
        conversations = ConversationCRUD.get_recent_sentiment_data(user_id, days=1)
        if conversations:
            avg_mood = sum(c.sentiment_score for c in conversations if c.sentiment_score) / len([c for c in conversations if c.sentiment_score])
            mood_emoji = get_sentiment_emoji(avg_mood)
            st.metric("😊 Today's Mood", f"{mood_emoji} {avg_mood:.2f}")
        else:
            st.metric("😊 Today's Mood", "No data")
    
    with col4:
        # Unresolved alerts
        alerts = CaregiverAlertCRUD.get_unresolved_alerts(user_id)
        alert_count = len(alerts)
        st.metric("🚨 Active Alerts", alert_count)
    
    st.divider()
    
    # Today's schedule and recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Today's Schedule")
        
        # Get pending reminders
        reminders = ReminderCRUD.get_pending_reminders(user_id)
        today_reminders = [
            r for r in reminders 
            if r.scheduled_time.date() == datetime.now().date()
        ]
        
        if today_reminders:
            for reminder in today_reminders[:5]:
                with st.container():
                    st.write(f"**{reminder.scheduled_time.strftime('%I:%M %p')}** - {reminder.title}")
                    st.write(f"_{reminder.message}_")
                    st.divider()
        else:
            st.info("No pending reminders for today")
    
    with col2:
        st.subheader("💬 Recent Conversations")
        
        recent_conversations = ConversationCRUD.get_user_conversations(user_id, limit=3)
        
        if recent_conversations:
            for conv in recent_conversations:
                with st.container():
                    # Show sentiment with color
                    sentiment_color = get_sentiment_color(conv.sentiment_score or 0)
                    sentiment_emoji = get_sentiment_emoji(conv.sentiment_score or 0)
                    
                    st.write(f"**{conv.timestamp.strftime('%I:%M %p')}** {sentiment_emoji}")
                    st.write(f"You: {conv.message}")
                    st.write(f"Carely: {conv.response}")
                    st.divider()
        else:
            st.info("No recent conversations")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💊 Log Medication Taken", use_container_width=True):
            st.session_state.show_medication_log = True
    
    with col2:
        if st.button("💬 Chat with Carely", use_container_width=True):
            st.session_state.current_page = "chat"
    
    with col3:
        if st.button("📊 View Health Report", use_container_width=True):
            st.session_state.current_page = "health"
    
    # Quick medication logging
    if st.session_state.get('show_medication_log', False):
        st.subheader("💊 Quick Medication Log")
        
        medications = MedicationCRUD.get_user_medications(user_id)
        if medications:
            med_options = {f"{med.name} ({med.dosage})": med.id for med in medications}
            selected_med = st.selectbox("Select medication:", list(med_options.keys()))
            notes = st.text_area("Notes (optional):")
            
            if st.button("Log as Taken"):
                MedicationLogCRUD.log_medication_taken(
                    user_id=user_id,
                    medication_id=med_options[selected_med],
                    scheduled_time=datetime.now(),
                    status="taken",
                    notes=notes or None
                )
                st.success("Medication logged successfully!")
                st.session_state.show_medication_log = False
                st.rerun()

def show_emergency_safety_sheet(user_id: int, concerns: list, severity: str, message: str):
    """Display emergency safety sheet with two-step flow"""
    user = UserCRUD.get_user(user_id)
    
    st.error("🚨 **EMERGENCY ALERT DETECTED**")
    
    st.markdown("### Safety Check")
    st.warning(f"We noticed you mentioned: {', '.join(concerns)}")
    
    st.markdown("---")
    st.markdown("### 📋 **What would you like to do?**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔴 **Notify Caregiver** (very urgent)", use_container_width=True, type="primary"):
            caregivers = CaregiverPatientCRUD.get_patient_caregivers(user_id)
            alert_sent = False
            
            if caregivers:
                for caregiver in caregivers:
                    if caregiver.telegram_chat_id:
                        result = send_emergency_alert(
                            chat_id=caregiver.telegram_chat_id,
                            patient_name=user.name,
                            concerns=concerns,
                            severity=severity,
                            message=message
                        )
                        if result.get("success"):
                            alert_sent = True
            
            if not alert_sent and os.getenv("TELEGRAM_CHAT_ID"):
                result = send_emergency_alert(
                    chat_id=os.getenv("TELEGRAM_CHAT_ID"),
                    patient_name=user.name,
                    concerns=concerns,
                    severity=severity,
                    message=message
                )
                if result.get("success"):
                    alert_sent = True
            
            if alert_sent:
                st.success("✅ **Help is on the way!**")
                st.info("Your caregiver has been notified via Telegram and will be with you shortly.")
                st.session_state.emergency_handled = True
            else:
                st.warning("⚠️ We couldn't send the Telegram alert. Please call your caregiver directly.")
                st.info("📞 Emergency: 911")
    
    with col2:
        if st.button("🟢 **I feel OK** (manageable)", use_container_width=True):
            st.success("**Feeling better!**")
            st.info("That's great to hear! Are you feeling better?")
            st.session_state.emergency_handled = True
    
    st.markdown("---")
    st.caption("💡 If this is a medical emergency, please call 911 immediately.")

def show_chat_interface(user_id: int):
    """Show chat interface with Carely"""
    user = UserCRUD.get_user(user_id)
    st.header(f"💬 Chat with Carely - {user.name}")
    
    # Display emergency safety sheet if emergency detected
    if st.session_state.get("emergency_data") and not st.session_state.get("emergency_handled"):
        emergency_data = st.session_state.emergency_data
        show_emergency_safety_sheet(
            user_id=user_id,
            concerns=emergency_data.get("concerns", []),
            severity=emergency_data.get("severity", "medium"),
            message=emergency_data.get("message", "")
        )
        
        if st.session_state.get("emergency_handled"):
            st.session_state.emergency_data = None
            st.rerun()
        
        return
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Load recent conversations
    if not st.session_state.chat_history:
        recent_convs = ConversationCRUD.get_user_conversations(user_id, limit=10)
        for conv in reversed(recent_convs):
            st.session_state.chat_history.append({"role": "user", "content": conv.message, "timestamp": conv.timestamp})
            st.session_state.chat_history.append({"role": "assistant", "content": conv.response, "timestamp": conv.timestamp})
    
    # Check for pending reminders and display them proactively
    if 'reminders_displayed' not in st.session_state:
        st.session_state.reminders_displayed = set()
    
    pending_reminders = ReminderCRUD.get_pending_reminders(user_id)
    for reminder in pending_reminders:
        if reminder.id not in st.session_state.reminders_displayed:
            # Add reminder to chat as an assistant message
            reminder_content = reminder.message
            
            # Add quick action button for medication reminders
            quick_actions = []
            if reminder.reminder_type == "medication":
                quick_actions = ["log_medication"]
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": reminder_content,
                "timestamp": reminder.scheduled_time,
                "quick_actions": quick_actions,
                "reminder_id": reminder.id,
                "medication_id": reminder.medication_id
            })
            
            st.session_state.reminders_displayed.add(reminder.id)
            
            # Mark reminder as displayed (completed)
            ReminderCRUD.complete_reminder(reminder.id)
    
    # Chat container
    chat_container = st.container()
    
    # Initialize TTS state
    if 'playing_audio' not in st.session_state:
        st.session_state.playing_audio = None
    
    with chat_container:
        # Display chat history
        for idx, message in enumerate(st.session_state.chat_history[-10:]):  # Show last 10 messages
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant", avatar="🏥"):
                    msg_col1, msg_col2 = st.columns([9, 1])
                    
                    with msg_col1:
                        content = message["content"]
                        
                        # Check if message contains YouTube URL and embed it
                        if "youtube.com/watch?v=" in content or "youtu.be/" in content:
                            lines = content.split("\n")
                            text_lines = []
                            video_url = None
                            
                            for line in lines:
                                if "youtube.com/watch?v=" in line or "youtu.be/" in line:
                                    video_url = line.strip()
                                else:
                                    text_lines.append(line)
                            
                            # Display text first
                            st.write("\n".join(text_lines))
                            
                            # Embed YouTube video
                            if video_url:
                                st.video(video_url)
                        else:
                            st.write(content)
                    
                    with msg_col2:
                        if st.button("🔊", key=f"listen_{idx}", help="Listen to this message"):
                            st.session_state.playing_audio = idx
                            st.rerun()
                    
                    if st.session_state.playing_audio == idx:
                        audio_bytes = generate_speech_audio(message["content"], slow=True)
                        if audio_bytes:
                            st.audio(audio_bytes, format='audio/mp3', autoplay=True)
                            st.session_state.playing_audio = None
                    
                    # Show quick action buttons for the most recent message
                    if idx == len(st.session_state.chat_history[-10:]) - 1 and message.get("quick_actions"):
                        st.markdown("---")
                        st.markdown("**Quick Actions:**")
                        action_cols = st.columns(len(message["quick_actions"]))
                        
                        for i, action in enumerate(message["quick_actions"]):
                            with action_cols[i]:
                                if action == "log_medication":
                                    if st.button("🕐 Log Medication", key=f"action_log_med_{idx}"):
                                        st.session_state.pending_action = "log_medication"
                                        st.rerun()
                                elif action == "play_music":
                                    if st.button("🎵 Play Music", key=f"action_music_{idx}"):
                                        st.session_state.pending_action = "play_music"
                                        st.rerun()
                                elif action == "fun_corner":
                                    if st.button("🧩 Fun Corner", key=f"action_fun_{idx}"):
                                        st.session_state.pending_action = "fun_corner"
                                        st.rerun()
                                elif action == "memory_cue":
                                    if st.button("🧠 Memory Cue", key=f"action_memory_{idx}"):
                                        st.session_state.pending_action = "memory_cue"
                                        st.rerun()
    
    # Handle quick action button clicks
    if st.session_state.get("pending_action"):
        action = st.session_state.pending_action
        st.session_state.pending_action = None
        
        if action == "log_medication":
            # Find the most recent reminder with medication_id
            medication_id = None
            for msg in reversed(st.session_state.chat_history):
                if msg.get("reminder_id") and msg.get("medication_id"):
                    medication_id = msg.get("medication_id")
                    break
            
            # Log medication with duplicate detection
            response_text = st.session_state.companion_agent.log_medication_tool(
                user_id=user_id,
                medication_id=medication_id
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now(),
                "quick_actions": []
            })
            st.rerun()
        
        elif action == "play_music":
            music_data = st.session_state.companion_agent.handle_play_music()
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": music_data["message"],
                "timestamp": datetime.now(),
                "quick_actions": ["fun_corner", "memory_cue"]
            })
            st.rerun()
        
        elif action == "fun_corner":
            joke_or_puzzle = st.session_state.companion_agent.handle_fun_corner("joke")
            message_text = f"Here's a joke for you! 😊\n\n{joke_or_puzzle}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": message_text,
                "timestamp": datetime.now(),
                "quick_actions": ["play_music", "memory_cue"]
            })
            st.rerun()
        
        elif action == "memory_cue":
            memory_question = st.session_state.companion_agent.generate_memory_cue(user_id)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": memory_question,
                "timestamp": datetime.now(),
                "quick_actions": ["fun_corner", "play_music"]
            })
            st.rerun()
    
    # Integrated input bar with voice and text
    st.markdown("---")
    st.markdown("##### Type or speak your message:")
    
    # Voice and text input side-by-side
    input_col1, input_col2 = st.columns([1, 9])
    
    with input_col1:
        # Compact voice button
        voice_text = speech_to_text(
            language='en',
            start_prompt="🎤 Voice",
            stop_prompt="⏹️ Stop",
            just_once=False,
            use_container_width=True,
            key=f'voice_input_{user_id}'
        )
    
    # Process voice input immediately when received
    if voice_text:
        prompt = voice_text
        is_voice = True
    else:
        prompt = None
        is_voice = False
    
    # Chat text input (appears alongside voice button)
    if text_prompt := st.chat_input(f"💬 Type your message here, {user.name}... (or use the voice button above)"):
        prompt = text_prompt
        is_voice = False
    
    # Process input (from either voice or text)
    if prompt:
        # Add user message to chat (with mic emoji for voice input)
        display_message = f"🎤 {prompt}" if is_voice else prompt
        
        with st.chat_message("user"):
            st.write(display_message)
        
        # Generate AI response
        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("Carely is thinking..."):
                response_data = st.session_state.companion_agent.generate_response(
                    user_id=user_id,
                    user_message=prompt
                )
            
            st.write(response_data["response"])
            
            # Show sentiment if available
            if response_data.get("sentiment_score") is not None:
                sentiment_emoji = get_sentiment_emoji(response_data["sentiment_score"])
                st.caption(f"Detected mood: {sentiment_emoji} {response_data['sentiment_label']}")
        
        # Update session state
        st.session_state.chat_history.append({"role": "user", "content": display_message, "timestamp": datetime.now()})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_data["response"],
            "timestamp": datetime.now(),
            "quick_actions": response_data.get("quick_actions", [])
        })
        
        # Check for emergency
        if response_data.get("is_emergency") and not st.session_state.get("emergency_handled"):
            st.session_state.emergency_data = {
                "concerns": response_data.get("emergency_concerns", []),
                "severity": response_data.get("emergency_severity", "medium"),
                "message": prompt
            }
        
        # Rerun to show the new messages
        st.rerun()
    
    # Chat actions
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📊 View Mood Trends"):
            st.session_state.show_mood_analysis = True
    
    with col3:
        if st.button("💊 Quick Med Check"):
            # Quick medication check
            medications = MedicationCRUD.get_user_medications(user_id)
            if medications:
                med_list = ", ".join([med.name for med in medications])
                quick_prompt = f"Can you remind me about my medications? I take: {med_list}"
                
                response_data = st.session_state.companion_agent.generate_response(
                    user_id=user_id,
                    user_message=quick_prompt,
                    conversation_type="medication"
                )
                
                st.session_state.chat_history.append({"role": "user", "content": quick_prompt, "timestamp": datetime.now()})
                st.session_state.chat_history.append({"role": "assistant", "content": response_data["response"], "timestamp": datetime.now()})
                st.rerun()
    
    # Mood analysis
    if st.session_state.get('show_mood_analysis', False):
        st.subheader("📈 Conversation Mood Analysis")
        
        conversations = ConversationCRUD.get_recent_sentiment_data(user_id, days=7)
        if conversations:
            # Create sentiment chart
            df = pd.DataFrame([
                {
                    "timestamp": conv.timestamp,
                    "sentiment_score": conv.sentiment_score,
                    "sentiment_label": conv.sentiment_label
                }
                for conv in conversations
                if conv.sentiment_score is not None
            ])
            
            fig = px.line(
                df, x="timestamp", y="sentiment_score",
                title="Mood Trends Over Time",
                color_discrete_sequence=["#1f77b4"]
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                yaxis_title="Mood Score",
                xaxis_title="Time",
                yaxis_range=[-1, 1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No mood data available yet. Keep chatting with Carely!")

def show_medication_management(user_id: int):
    """Show medication management interface"""
    user = UserCRUD.get_user(user_id)
    st.header(f"💊 Medication Management - {user.name}")
    
    # Medication overview
    medications = MedicationCRUD.get_user_medications(user_id)
    
    if medications:
        st.subheader("Current Medications")
        
        for med in medications:
            with st.expander(f"{med.name} - {med.dosage}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Frequency:** {med.frequency}")
                    if med.schedule_times:
                        times = json.loads(med.schedule_times)
                        st.write(f"**Times:** {', '.join(times)}")
                    if med.instructions:
                        st.write(f"**Instructions:** {med.instructions}")
                    st.write(f"**Active:** {'Yes' if med.active else 'No'}")
                
                with col2:
                    # Recent logs for this medication
                    adherence = MedicationLogCRUD.get_medication_adherence(user_id, days=7)
                    med_logs = [log for log in adherence.get("logs", []) if log.medication_id == med.id]
                    
                    if med_logs:
                        st.write("**Recent Activity:**")
                        for log in med_logs[-3:]:  # Last 3 logs
                            status_emoji = "✅" if log.status == "taken" else "❌" if log.status == "missed" else "⏸️"
                            st.write(f"{status_emoji} {log.scheduled_time.strftime('%m/%d %I:%M %p')} - {log.status}")
                    
                    # Quick log button
                    if st.button(f"Log {med.name} as Taken", key=f"log_{med.id}"):
                        MedicationLogCRUD.log_medication_taken(
                            user_id=user_id,
                            medication_id=med.id,
                            scheduled_time=datetime.now(),
                            status="taken"
                        )
                        st.success(f"{med.name} logged as taken!")
                        st.rerun()
    
    else:
        st.info("No medications found. Add medications below.")
    
    st.divider()
    
    # Medication adherence chart
    st.subheader("📊 Adherence Overview")
    
    # Time period selector
    period = st.selectbox("Select period:", ["Last 7 days", "Last 30 days"], key="adherence_period")
    days = 7 if period == "Last 7 days" else 30
    
    adherence = MedicationLogCRUD.get_medication_adherence(user_id, days=days)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Doses", adherence.get("total", 0))
    with col2:
        st.metric("Doses Taken", adherence.get("taken", 0))
    with col3:
        st.metric("Adherence Rate", f"{adherence.get('adherence_rate', 0):.1f}%")
    
    # Adherence chart
    if adherence.get("logs"):
        df = pd.DataFrame([
            {
                "date": log.scheduled_time.date(),
                "status": log.status,
                "medication": next((med.name for med in medications if med.id == log.medication_id), "Unknown")
            }
            for log in adherence["logs"]
        ])
        
        # Group by date and calculate daily adherence
        daily_adherence = df.groupby("date").apply(
            lambda x: (x["status"] == "taken").sum() / len(x) * 100
        ).reset_index(name="adherence_rate")
        daily_adherence["date"] = pd.to_datetime(daily_adherence["date"])
        
        fig = px.line(
            daily_adherence, 
            x="date", 
            y="adherence_rate",
            title=f"Daily Adherence Rate ({period})",
            range_y=[0, 100]
        )
        fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Target: 80%")
        fig.update_layout(yaxis_title="Adherence Rate (%)", xaxis_title="Date")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Add new medication
    st.subheader("➕ Add New Medication")
    
    with st.form("add_medication"):
        col1, col2 = st.columns(2)
        
        with col1:
            med_name = st.text_input("Medication Name*")
            dosage = st.text_input("Dosage*", placeholder="e.g., 10mg, 1 tablet")
            frequency = st.selectbox("Frequency", ["daily", "twice_daily", "three_times_daily", "weekly", "as_needed"])
        
        with col2:
            # Schedule times
            if frequency == "daily":
                times = [st.time_input("Time", value=time(9, 0)).strftime("%H:%M")]
            elif frequency == "twice_daily":
                time1 = st.time_input("Morning", value=time(9, 0)).strftime("%H:%M")
                time2 = st.time_input("Evening", value=time(21, 0)).strftime("%H:%M")
                times = [time1, time2]
            elif frequency == "three_times_daily":
                time1 = st.time_input("Morning", value=time(8, 0)).strftime("%H:%M")
                time2 = st.time_input("Afternoon", value=time(14, 0)).strftime("%H:%M")
                time3 = st.time_input("Evening", value=time(20, 0)).strftime("%H:%M")
                times = [time1, time2, time3]
            else:
                times = []
        
        instructions = st.text_area("Instructions (optional)", placeholder="Take with food, etc.")
        
        if st.form_submit_button("Add Medication"):
            if med_name and dosage:
                try:
                    MedicationCRUD.create_medication(
                        user_id=user_id,
                        name=med_name,
                        dosage=dosage,
                        frequency=frequency,
                        schedule_times=times,
                        instructions=instructions or None
                    )
                    st.success(f"Added {med_name} successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding medication: {e}")
            else:
                st.error("Please fill in required fields (Name and Dosage)")

def show_health_insights(user_id: int):
    """Show health insights and analytics"""
    user = UserCRUD.get_user(user_id)
    st.header(f"📊 Health Insights - {user.name}")
    
    # Time period selector
    col1, col2 = st.columns([1, 3])
    with col1:
        period = st.selectbox("Time Period:", ["7 days", "30 days", "90 days"])
        days = int(period.split()[0])
    
    # Get data
    conversations = ConversationCRUD.get_recent_sentiment_data(user_id, days=days)
    adherence = MedicationLogCRUD.get_medication_adherence(user_id, days=days)
    
    # Summary metrics
    st.subheader("📈 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if conversations:
            avg_mood = sum(c.sentiment_score for c in conversations if c.sentiment_score) / len([c for c in conversations if c.sentiment_score])
            mood_emoji = get_sentiment_emoji(avg_mood)
            st.metric("Average Mood", f"{mood_emoji} {avg_mood:.2f}")
        else:
            st.metric("Average Mood", "No data")
    
    with col2:
        st.metric("Medication Adherence", f"{adherence.get('adherence_rate', 0):.1f}%")
    
    with col3:
        st.metric("Total Conversations", len(conversations))
    
    with col4:
        alerts = CaregiverAlertCRUD.get_unresolved_alerts(user_id)
        st.metric("Active Concerns", len(alerts))
    
    st.divider()
    
    # Charts
    if conversations:
        # Mood trend chart
        st.subheader("😊 Mood Trends")
        
        df_mood = pd.DataFrame([
            {
                "date": conv.timestamp.date(),
                "sentiment_score": conv.sentiment_score,
                "sentiment_label": conv.sentiment_label
            }
            for conv in conversations
            if conv.sentiment_score is not None
        ])
        
        # Daily average mood
        daily_mood = df_mood.groupby("date")["sentiment_score"].mean().reset_index()
        daily_mood["date"] = pd.to_datetime(daily_mood["date"])
        
        fig_mood = px.line(
            daily_mood, 
            x="date", 
            y="sentiment_score",
            title="Daily Average Mood",
            range_y=[-1, 1]
        )
        fig_mood.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_mood.add_hline(y=0.3, line_dash="dot", line_color="green", annotation_text="Good mood")
        fig_mood.add_hline(y=-0.3, line_dash="dot", line_color="red", annotation_text="Concerning")
        
        st.plotly_chart(fig_mood, use_container_width=True)
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df_mood["sentiment_label"].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Mood Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Most active times
            df_mood_with_hour = pd.DataFrame([
                {
                    "hour": conv.timestamp.hour,
                    "sentiment_score": conv.sentiment_score
                }
                for conv in conversations
                if conv.sentiment_score is not None
            ])
            
            if not df_mood_with_hour.empty:
                hourly_mood = df_mood_with_hour.groupby("hour")["sentiment_score"].mean().reset_index()
                fig_hourly = px.bar(
                    hourly_mood,
                    x="hour",
                    y="sentiment_score",
                    title="Mood by Time of Day"
                )
                fig_hourly.update_xaxis(dtick=1)
                st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Medication insights
    if adherence.get("logs"):
        st.subheader("💊 Medication Insights")
        
        # Weekly adherence pattern
        df_med = pd.DataFrame([
            {
                "date": log.scheduled_time.date(),
                "day_of_week": log.scheduled_time.strftime("%A"),
                "status": log.status,
                "hour": log.scheduled_time.hour
            }
            for log in adherence["logs"]
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Adherence by day of week
            weekly_adherence = df_med.groupby("day_of_week").apply(
                lambda x: (x["status"] == "taken").sum() / len(x) * 100
            ).reset_index(name="adherence_rate")
            
            # Order by day of week
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekly_adherence["day_of_week"] = pd.Categorical(weekly_adherence["day_of_week"], categories=day_order, ordered=True)
            weekly_adherence = weekly_adherence.sort_values("day_of_week")
            
            fig_weekly = px.bar(
                weekly_adherence,
                x="day_of_week",
                y="adherence_rate",
                title="Adherence by Day of Week"
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        with col2:
            # Adherence by time of day
            hourly_adherence = df_med.groupby("hour").apply(
                lambda x: (x["status"] == "taken").sum() / len(x) * 100
            ).reset_index(name="adherence_rate")
            
            fig_hourly_med = px.bar(
                hourly_adherence,
                x="hour",
                y="adherence_rate",
                title="Adherence by Time of Day"
            )
            st.plotly_chart(fig_hourly_med, use_container_width=True)
    
    # Health recommendations
    st.subheader("💡 Health Recommendations")
    
    recommendations = []
    
    if conversations:
        recent_mood = [c.sentiment_score for c in conversations[-7:] if c.sentiment_score is not None]
        if recent_mood:
            avg_recent_mood = sum(recent_mood) / len(recent_mood)
            if avg_recent_mood < -0.3:
                recommendations.append("🟡 Recent mood trends show concern. Consider scheduling a check-in with healthcare provider.")
            elif avg_recent_mood > 0.3:
                recommendations.append("🟢 Mood trends are positive! Keep up the good routine.")
    
    if adherence.get("adherence_rate", 100) < 80:
        recommendations.append("🔴 Medication adherence is below 80%. Consider setting more reminders or reviewing medication schedule.")
    elif adherence.get("adherence_rate", 100) > 90:
        recommendations.append("🟢 Excellent medication adherence! Keep up the great work.")
    
    if len(conversations) < 7 and days >= 7:
        recommendations.append("🟡 Consider chatting with Carely more regularly for better mood tracking.")
    
    if not recommendations:
        recommendations.append("🟢 All health metrics look good! Continue current routine.")
    
    for rec in recommendations:
        st.write(rec)

def show_alerts_and_reminders(user_id: int):
    """Show alerts and reminders interface"""
    user = UserCRUD.get_user(user_id)
    st.header(f"🚨 Alerts & Reminders - {user.name}")
    
    # Tabs for different types
    tab1, tab2, tab3 = st.tabs(["🔔 Pending Reminders", "🚨 Active Alerts", "📋 Reminder History"])
    
    with tab1:
        st.subheader("Pending Reminders")
        
        reminders = ReminderCRUD.get_pending_reminders(user_id)
        
        if reminders:
            for reminder in reminders:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Color code by type
                        if reminder.reminder_type == "medication":
                            st.markdown(f"💊 **{reminder.title}**")
                        elif reminder.reminder_type == "checkin":
                            st.markdown(f"💬 **{reminder.title}**")
                        else:
                            st.markdown(f"📅 **{reminder.title}**")
                        
                        st.write(reminder.message)
                        st.caption(f"Scheduled: {reminder.scheduled_time.strftime('%m/%d/%Y %I:%M %p')}")
                    
                    with col2:
                        if st.button("✅ Complete", key=f"complete_{reminder.id}"):
                            ReminderCRUD.complete_reminder(reminder.id)
                            st.success("Reminder completed!")
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No pending reminders")
    
    with tab2:
        st.subheader("Active Alerts")
        
        alerts = CaregiverAlertCRUD.get_unresolved_alerts(user_id)
        
        if alerts:
            for alert in alerts:
                with st.container():
                    # Color code by severity
                    if alert.severity == "high":
                        st.error(f"🔴 **{alert.title}**")
                    elif alert.severity == "medium":
                        st.warning(f"🟡 **{alert.title}**")
                    else:
                        st.info(f"🟢 **{alert.title}**")
                    
                    st.write(alert.description)
                    st.caption(f"Created: {alert.created_at.strftime('%m/%d/%Y %I:%M %p')} | Type: {alert.alert_type}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("✅ Resolve", key=f"resolve_{alert.id}"):
                            CaregiverAlertCRUD.resolve_alert(alert.id)
                            st.success("Alert resolved!")
                            st.rerun()
                    
                    st.divider()
        else:
            st.success("No active alerts - all good! ✨")
    
    with tab3:
        st.subheader("Reminder History")
        
        # This would show completed reminders and resolved alerts
        # For now, showing a simple message
        st.info("Reminder history feature coming soon!")
        
        # Could add filters for date range, type, etc.
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Filter by Type:", ["All", "Medication", "Check-in", "Custom"])
        with col2:
            st.selectbox("Filter by Status:", ["All", "Completed", "Missed"])
        with col3:
            st.selectbox("Time Period:", ["Last 7 days", "Last 30 days", "All time"])

def show_user_management():
    """Show user management interface"""
    st.header("👥 User Management")
    
    # Current users
    users = UserCRUD.get_all_users()
    
    if users:
        st.subheader("Current Users")
        
        for user in users:
            with st.expander(f"{user.name} (ID: {user.id})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Name:** {user.name}")
                    st.write(f"**Email:** {user.email or 'Not provided'}")
                    st.write(f"**Phone:** {user.phone or 'Not provided'}")
                    st.write(f"**Emergency Contact:** {user.emergency_contact or 'Not provided'}")
                
                with col2:
                    st.write(f"**Created:** {user.created_at.strftime('%m/%d/%Y')}")
                    if user.preferences:
                        try:
                            prefs = json.loads(user.preferences)
                            st.write("**Preferences:**")
                            for key, value in prefs.items():
                                st.write(f"- {key}: {value}")
                        except:
                            st.write("**Preferences:** Invalid format")
                    
                    # Quick stats
                    conversations = ConversationCRUD.get_user_conversations(user.id, limit=1)
                    medications = MedicationCRUD.get_user_medications(user.id)
                    st.write(f"**Medications:** {len(medications)}")
                    st.write(f"**Last Chat:** {conversations[0].timestamp.strftime('%m/%d/%Y') if conversations else 'Never'}")
    
    st.divider()
    
    # Add new user
    st.subheader("➕ Add New User")
    
    with st.form("add_user"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*")
            email = st.text_input("Email")
            phone = st.text_input("Phone Number")
        
        with col2:
            emergency_contact = st.text_input("Emergency Contact")
            
            # Preferences
            st.write("**Preferences:**")
            pref_language = st.selectbox("Preferred Language:", ["English", "Spanish", "French", "Other"])
            pref_time = st.selectbox("Preferred Contact Time:", ["Morning", "Afternoon", "Evening", "Any"])
            pref_reminders = st.checkbox("Enable Reminders", value=True)
        
        if st.form_submit_button("Add User"):
            if name:
                try:
                    preferences = {
                        "language": pref_language,
                        "contact_time": pref_time,
                        "reminders_enabled": pref_reminders
                    }
                    
                    new_user = UserCRUD.create_user(
                        name=name,
                        email=email or None,
                        phone=phone or None,
                        preferences=preferences,
                        emergency_contact=emergency_contact or None
                    )
                    
                    st.success(f"Added user {name} successfully! (ID: {new_user.id})")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error adding user: {e}")
            else:
                st.error("Please enter a name")
    
    # User statistics
    if users:
        st.divider()
        st.subheader("📊 User Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users))
        
        with col2:
            # Users with medications
            users_with_meds = 0
            for user in users:
                if MedicationCRUD.get_user_medications(user.id):
                    users_with_meds += 1
            st.metric("Users with Medications", users_with_meds)
        
        with col3:
            # Users with recent activity (last 7 days)
            active_users = 0
            for user in users:
                conversations = ConversationCRUD.get_user_conversations(user.id, limit=1)
                if conversations and (datetime.now() - conversations[0].timestamp).days <= 7:
                    active_users += 1
            st.metric("Active Users (7d)", active_users)
        
        with col4:
            # Users with alerts
            users_with_alerts = 0
            for user in users:
                alerts = CaregiverAlertCRUD.get_unresolved_alerts(user.id)
                if alerts:
                    users_with_alerts += 1
            st.metric("Users with Alerts", users_with_alerts)
