from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from datetime import datetime, timedelta, time
import json
import logging
from typing import List, Dict, Any

from app.database.crud import (
    ReminderCRUD, MedicationCRUD, MedicationLogCRUD, 
    CaregiverAlertCRUD, UserCRUD, PersonalEventCRUD
)
from app.agents.companion_agent import CompanionAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReminderScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.companion_agent = CompanionAgent()
        self.is_running = False
    
    def start(self):
        """Start the scheduler with all recurring jobs"""
        if self.is_running:
            return
        
        try:
            # Schedule daily check-ins
            self.schedule_daily_checkins()
            
            # Schedule medication reminders
            self.schedule_medication_reminders()
            
            # Schedule appointment reminders
            self.schedule_appointment_reminders()
            
            # Schedule weekly reports
            self.schedule_weekly_reports()
            
            # Schedule adherence monitoring
            self.schedule_adherence_monitoring()
            
            self.scheduler.start()
            self.is_running = True
            logger.info("Reminder scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Reminder scheduler stopped")
    
    def schedule_daily_checkins(self):
        """Schedule daily check-ins at fixed times"""
        # Morning check-in at 9:00 AM
        self.scheduler.add_job(
            func=self.morning_checkin,
            trigger=CronTrigger(hour=9, minute=0),
            id='morning_checkin',
            name='Morning Check-in',
            replace_existing=True
        )
        
        # Afternoon check-in at 2:00 PM
        self.scheduler.add_job(
            func=self.afternoon_checkin,
            trigger=CronTrigger(hour=14, minute=0),
            id='afternoon_checkin',
            name='Afternoon Check-in',
            replace_existing=True
        )
        
        # Evening check-in at 7:00 PM
        self.scheduler.add_job(
            func=self.evening_checkin,
            trigger=CronTrigger(hour=19, minute=0),
            id='evening_checkin',
            name='Evening Check-in',
            replace_existing=True
        )
        
        logger.info("Daily check-ins scheduled")
    
    def schedule_medication_reminders(self):
        """Schedule medication reminders for all users"""
        try:
            users = UserCRUD.get_all_users()
            
            for user in users:
                medications = MedicationCRUD.get_user_medications(user.id)
                
                for medication in medications:
                    if not medication.active or not medication.schedule_times:
                        continue
                    
                    try:
                        schedule_times = json.loads(medication.schedule_times)
                        
                        for time_str in schedule_times:
                            # Parse time string (expected format: "HH:MM")
                            hour, minute = map(int, time_str.split(':'))
                            
                            job_id = f'med_reminder_{medication.id}_{time_str.replace(":", "")}'
                            
                            self.scheduler.add_job(
                                func=self.medication_reminder,
                                trigger=CronTrigger(hour=hour, minute=minute),
                                args=[user.id, medication.id],
                                id=job_id,
                                name=f'Medication reminder for {medication.name}',
                                replace_existing=True
                            )
                            
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Invalid schedule format for medication {medication.id}: {e}")
            
            logger.info("Medication reminders scheduled for all users")
            
        except Exception as e:
            logger.error(f"Failed to schedule medication reminders: {e}")
    
    def schedule_appointment_reminders(self):
        """Schedule reminders for upcoming appointments (1 hour before)"""
        try:
            users = UserCRUD.get_all_users()
            
            for user in users:
                # Get appointments in the next 7 days
                upcoming_events = PersonalEventCRUD.get_upcoming_events(user.id, days=7)
                appointments = [e for e in upcoming_events if e.event_type == "appointment"]
                
                for appointment in appointments:
                    if not appointment.event_date:
                        continue
                    
                    # Schedule reminder 1 hour before appointment
                    reminder_time = appointment.event_date - timedelta(hours=1)
                    
                    # Only schedule if reminder time is in the future
                    if reminder_time > datetime.now():
                        job_id = f'appointment_reminder_{appointment.id}'
                        
                        self.scheduler.add_job(
                            func=self.appointment_reminder,
                            trigger=DateTrigger(run_date=reminder_time),
                            args=[user.id, appointment.id],
                            id=job_id,
                            name=f'Appointment reminder for {appointment.title}',
                            replace_existing=True
                        )
            
            logger.info("Appointment reminders scheduled for all users")
            
        except Exception as e:
            logger.error(f"Failed to schedule appointment reminders: {e}")
    
    def appointment_reminder(self, user_id: int, appointment_id: int):
        """Send appointment reminder to specific user"""
        try:
            user = UserCRUD.get_user(user_id)
            events = PersonalEventCRUD.get_user_events(user_id)
            appointment = next((e for e in events if e.id == appointment_id), None)
            
            if not appointment:
                logger.error(f"Appointment {appointment_id} not found for user {user_id}")
                return
            
            # Create conversational, supportive reminder message
            time_str = appointment.event_date.strftime('%I:%M %p')
            reminder_message = f"You have an appointment with {appointment.title} today at {time_str}."
            
            if appointment.description:
                reminder_message += f" {appointment.description}"
            
            reminder_message += " Would you like directions or to confirm attendance?"
            
            ReminderCRUD.create_reminder(
                user_id=user_id,
                reminder_type="appointment",
                title=f"Upcoming: {appointment.title}",
                message=reminder_message,
                scheduled_time=datetime.now()
            )
            
            logger.info(f"Appointment reminder sent for {appointment.title} to user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to send appointment reminder: {e}")
    
    def schedule_weekly_reports(self):
        """Schedule weekly summary reports"""
        self.scheduler.add_job(
            func=self.generate_weekly_report,
            trigger=CronTrigger(day_of_week=0, hour=8, minute=0),  # Monday at 8 AM
            id='weekly_report',
            name='Weekly Summary Report',
            replace_existing=True
        )
        
        logger.info("Weekly reports scheduled")
    
    def schedule_adherence_monitoring(self):
        """Schedule medication adherence monitoring"""
        # Check for missed medications every 2 hours
        self.scheduler.add_job(
            func=self.check_missed_medications,
            trigger=CronTrigger(minute=0, second=0),  # Every hour
            id='adherence_monitoring',
            name='Medication Adherence Monitoring',
            replace_existing=True
        )
        
        logger.info("Adherence monitoring scheduled")
    
    def morning_checkin(self):
        """Perform morning check-in for all users"""
        try:
            users = UserCRUD.get_all_users()
            
            for user in users:
                # Create a reminder for morning check-in
                checkin = self.companion_agent.conduct_daily_checkin(user.id, "morning")
                
                ReminderCRUD.create_reminder(
                    user_id=user.id,
                    reminder_type="checkin",
                    title="Morning Check-in",
                    message=checkin["prompt"],
                    scheduled_time=datetime.now()
                )
            
            logger.info(f"Morning check-in completed for {len(users)} users")
            
        except Exception as e:
            logger.error(f"Morning check-in failed: {e}")
    
    def afternoon_checkin(self):
        """Perform afternoon check-in for all users"""
        try:
            users = UserCRUD.get_all_users()
            
            for user in users:
                checkin = self.companion_agent.conduct_daily_checkin(user.id, "afternoon")
                
                ReminderCRUD.create_reminder(
                    user_id=user.id,
                    reminder_type="checkin",
                    title="Afternoon Check-in",
                    message=checkin["prompt"],
                    scheduled_time=datetime.now()
                )
            
            logger.info(f"Afternoon check-in completed for {len(users)} users")
            
        except Exception as e:
            logger.error(f"Afternoon check-in failed: {e}")
    
    def evening_checkin(self):
        """Perform evening check-in for all users"""
        try:
            users = UserCRUD.get_all_users()
            
            for user in users:
                checkin = self.companion_agent.conduct_daily_checkin(user.id, "evening")
                
                ReminderCRUD.create_reminder(
                    user_id=user.id,
                    reminder_type="checkin",
                    title="Evening Check-in",
                    message=checkin["prompt"],
                    scheduled_time=datetime.now()
                )
            
            logger.info(f"Evening check-in completed for {len(users)} users")
            
        except Exception as e:
            logger.error(f"Evening check-in failed: {e}")
    
    def medication_reminder(self, user_id: int, medication_id: int):
        """Send medication reminder to specific user"""
        try:
            medication = MedicationCRUD.get_user_medications(user_id)
            medication = next((med for med in medication if med.id == medication_id), None)
            
            if not medication:
                logger.error(f"Medication {medication_id} not found for user {user_id}")
                return
            
            user = UserCRUD.get_user(user_id)
            
            # Create conversational, supportive reminder message
            instruction_text = f" {medication.instructions}" if medication.instructions else ""
            reminder_message = f"It's time for your {medication.name} {medication.dosage}.{instruction_text} Tap '🕐 Log Medication' once you've taken it."
            
            ReminderCRUD.create_reminder(
                user_id=user_id,
                reminder_type="medication",
                title=f"Time for {medication.name}",
                message=reminder_message,
                scheduled_time=datetime.now(),
                medication_id=medication_id
            )
            
            logger.info(f"Medication reminder sent for {medication.name} to user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to send medication reminder: {e}")
    
    def check_missed_medications(self):
        """Check for missed medications and create alerts"""
        try:
            users = UserCRUD.get_all_users()
            current_time = datetime.now()
            
            for user in users:
                # Get medication adherence for last 24 hours
                adherence = MedicationLogCRUD.get_medication_adherence(user.id, days=1)
                
                # Check for missed medications in the last 2 hours
                recent_missed = 0
                for log in adherence.get("logs", []):
                    if (log.status == "missed" and 
                        current_time - log.scheduled_time < timedelta(hours=2)):
                        recent_missed += 1
                
                # Alert if adherence is below 80% or recent missed doses
                if adherence.get("adherence_rate", 100) < 80 or recent_missed > 0:
                    alert_description = f"Medication adherence concern: {adherence.get('adherence_rate', 0):.1f}% adherence rate"
                    if recent_missed > 0:
                        alert_description += f", {recent_missed} missed doses in last 2 hours"
                    
                    CaregiverAlertCRUD.create_alert(
                        user_id=user.id,
                        alert_type="medication_missed",
                        title="Medication Adherence Alert",
                        description=alert_description,
                        severity="high" if recent_missed > 1 else "medium"
                    )
            
            logger.info("Medication adherence monitoring completed")
            
        except Exception as e:
            logger.error(f"Failed to check missed medications: {e}")
    
    def generate_weekly_report(self):
        """Generate weekly summary reports for caregivers"""
        try:
            users = UserCRUD.get_all_users()
            
            for user in users:
                # Get adherence data
                adherence = MedicationLogCRUD.get_medication_adherence(user.id, days=7)
                
                # Get mood data
                from app.database.crud import ConversationCRUD
                conversations = ConversationCRUD.get_recent_sentiment_data(user.id, days=7)
                
                mood_scores = [c.sentiment_score for c in conversations if c.sentiment_score is not None]
                avg_mood = sum(mood_scores) / len(mood_scores) if mood_scores else 0
                
                report = f"""Weekly Report for {user.name}:
                
Medication Adherence:
- Total doses: {adherence.get('total', 0)}
- Doses taken: {adherence.get('taken', 0)}
- Adherence rate: {adherence.get('adherence_rate', 0):.1f}%

Mood & Wellbeing:
- Average mood: {avg_mood:.2f} (scale: -1 to 1)
- Total conversations: {len(conversations)}
- Mood trend: {'Positive' if avg_mood > 0.2 else 'Neutral' if avg_mood > -0.2 else 'Concerning'}

Recommendations:
{self._generate_recommendations(user.id, adherence, avg_mood)}
"""
                
                # Create alert with weekly report
                CaregiverAlertCRUD.create_alert(
                    user_id=user.id,
                    alert_type="weekly_report",
                    title=f"Weekly Report - {user.name}",
                    description=report,
                    severity="low"
                )
            
            logger.info(f"Weekly reports generated for {len(users)} users")
            
        except Exception as e:
            logger.error(f"Failed to generate weekly reports: {e}")
    
    def _generate_recommendations(self, user_id: int, adherence: Dict, avg_mood: float) -> str:
        """Generate personalized recommendations based on data"""
        recommendations = []
        
        if adherence.get("adherence_rate", 100) < 90:
            recommendations.append("- Consider medication reminder system improvements")
        
        if avg_mood < -0.3:
            recommendations.append("- Monitor mood closely, consider professional consultation")
        
        if not recommendations:
            recommendations.append("- Continue current care routine, all metrics look good")
        
        return "\n".join(recommendations)
    
    def add_custom_reminder(self, user_id: int, title: str, message: str, 
                          scheduled_time: datetime):
        """Add a custom one-time reminder"""
        try:
            # Create database entry
            ReminderCRUD.create_reminder(
                user_id=user_id,
                reminder_type="custom",
                title=title,
                message=message,
                scheduled_time=scheduled_time
            )
            
            # Schedule the reminder
            job_id = f"custom_reminder_{user_id}_{int(scheduled_time.timestamp())}"
            
            self.scheduler.add_job(
                func=self._send_custom_reminder,
                trigger=DateTrigger(run_date=scheduled_time),
                args=[user_id, title, message],
                id=job_id,
                name=f"Custom reminder: {title}",
                replace_existing=True
            )
            
            logger.info(f"Custom reminder scheduled for user {user_id} at {scheduled_time}")
            
        except Exception as e:
            logger.error(f"Failed to add custom reminder: {e}")
    
    def _send_custom_reminder(self, user_id: int, title: str, message: str):
        """Send a custom reminder"""
        try:
            # This would typically send a notification
            # For now, we'll just log it
            logger.info(f"Custom reminder for user {user_id}: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to send custom reminder: {e}")
