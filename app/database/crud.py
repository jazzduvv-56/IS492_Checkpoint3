from sqlmodel import Session, select
from datetime import datetime, timedelta
from typing import List, Optional
import json
from app.database.models import (
    get_session, User, Medication, Conversation, Reminder, 
    MedicationLog, CaregiverAlert, CaregiverPatientAssignment, PersonalEvent
)

class UserCRUD:
    @staticmethod
    def create_user(name: str, email: str = None, phone: str = None, 
                   preferences: dict = None, emergency_contact: str = None,
                   user_type: str = "patient", password: str = None) -> User:
        """Create a new user"""
        with get_session() as session:
            preferences_json = json.dumps(preferences) if preferences else None
            password_hash = None
            if password:
                import hashlib
                password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            user = User(
                name=name,
                email=email,
                phone=phone,
                preferences=preferences_json,
                emergency_contact=emergency_contact,
                user_type=user_type,
                password_hash=password_hash
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
    
    @staticmethod
    def get_user(user_id: int) -> Optional[User]:
        """Get user by ID"""
        with get_session() as session:
            return session.get(User, user_id)
    
    @staticmethod
    def get_all_users() -> List[User]:
        """Get all users"""
        with get_session() as session:
            return session.exec(select(User)).all()

class MedicationCRUD:
    @staticmethod
    def create_medication(user_id: int, name: str, dosage: str, frequency: str,
                         schedule_times: List[str], instructions: str = None) -> Medication:
        """Create a new medication"""
        with get_session() as session:
            medication = Medication(
                user_id=user_id,
                name=name,
                dosage=dosage,
                frequency=frequency,
                schedule_times=json.dumps(schedule_times),
                instructions=instructions
            )
            session.add(medication)
            session.commit()
            session.refresh(medication)
            return medication
    
    @staticmethod
    def get_user_medications(user_id: int, active_only: bool = True) -> List[Medication]:
        """Get all medications for a user"""
        with get_session() as session:
            query = select(Medication).where(Medication.user_id == user_id)
            if active_only:
                query = query.where(Medication.active == True)
            return session.exec(query).all()
    
    @staticmethod
    def update_medication(medication_id: int, **kwargs) -> Optional[Medication]:
        """Update medication"""
        with get_session() as session:
            medication = session.get(Medication, medication_id)
            if medication:
                for key, value in kwargs.items():
                    setattr(medication, key, value)
                session.add(medication)
                session.commit()
                session.refresh(medication)
            return medication

class ConversationCRUD:
    @staticmethod
    def save_conversation(user_id: int, message: str, response: str,
                         sentiment_score: float = None, sentiment_label: str = None,
                         conversation_type: str = "general") -> Conversation:
        """Save a conversation"""
        with get_session() as session:
            conversation = Conversation(
                user_id=user_id,
                message=message,
                response=response,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                conversation_type=conversation_type
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation
    
    @staticmethod
    def get_user_conversations(user_id: int, limit: int = 50) -> List[Conversation]:
        """Get recent conversations for a user"""
        with get_session() as session:
            query = select(Conversation).where(
                Conversation.user_id == user_id
            ).order_by(Conversation.timestamp.desc()).limit(limit)
            return session.exec(query).all()
    
    @staticmethod
    def get_recent_sentiment_data(user_id: int, days: int = 7) -> List[Conversation]:
        """Get recent conversations with sentiment data"""
        with get_session() as session:
            cutoff_date = datetime.now() - timedelta(days=days)
            query = select(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.timestamp >= cutoff_date,
                Conversation.sentiment_score.isnot(None)
            ).order_by(Conversation.timestamp.desc())
            return session.exec(query).all()

class ReminderCRUD:
    @staticmethod
    def create_reminder(user_id: int, reminder_type: str, title: str, message: str,
                       scheduled_time: datetime, medication_id: int = None) -> Reminder:
        """Create a new reminder"""
        with get_session() as session:
            reminder = Reminder(
                user_id=user_id,
                reminder_type=reminder_type,
                title=title,
                message=message,
                scheduled_time=scheduled_time,
                medication_id=medication_id
            )
            session.add(reminder)
            session.commit()
            session.refresh(reminder)
            return reminder
    
    @staticmethod
    def get_pending_reminders(user_id: int = None) -> List[Reminder]:
        """Get pending reminders"""
        with get_session() as session:
            query = select(Reminder).where(
                Reminder.completed == False,
                Reminder.scheduled_time <= datetime.now()
            )
            if user_id:
                query = query.where(Reminder.user_id == user_id)
            return session.exec(query).all()
    
    @staticmethod
    def complete_reminder(reminder_id: int) -> Optional[Reminder]:
        """Mark reminder as completed"""
        with get_session() as session:
            reminder = session.get(Reminder, reminder_id)
            if reminder:
                reminder.completed = True
                reminder.completed_at = datetime.now()
                session.add(reminder)
                session.commit()
                session.refresh(reminder)
            return reminder

class MedicationLogCRUD:
    @staticmethod
    def log_medication_taken(user_id: int, medication_id: int, scheduled_time: datetime,
                           taken_time: datetime = None, status: str = "taken",
                           notes: str = None) -> MedicationLog:
        """Log medication intake"""
        with get_session() as session:
            log = MedicationLog(
                user_id=user_id,
                medication_id=medication_id,
                scheduled_time=scheduled_time,
                taken_time=taken_time or datetime.now(),
                status=status,
                notes=notes
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log
    
    @staticmethod
    def get_medication_adherence(user_id: int, days: int = 7) -> dict:
        """Get medication adherence statistics"""
        with get_session() as session:
            cutoff_date = datetime.now() - timedelta(days=days)
            query = select(MedicationLog).where(
                MedicationLog.user_id == user_id,
                MedicationLog.scheduled_time >= cutoff_date
            )
            logs = session.exec(query).all()
            
            total = len(logs)
            taken = len([log for log in logs if log.status == "taken"])
            missed = len([log for log in logs if log.status == "missed"])
            
            return {
                "total": total,
                "taken": taken,
                "missed": missed,
                "adherence_rate": (taken / total * 100) if total > 0 else 0,
                "logs": logs
            }
    
    @staticmethod
    def check_recent_medication_log(user_id: int, medication_id: int, hours: int = 24) -> Optional[MedicationLog]:
        """Check if medication was logged recently (for duplicate detection)"""
        with get_session() as session:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            query = select(MedicationLog).where(
                MedicationLog.user_id == user_id,
                MedicationLog.medication_id == medication_id,
                MedicationLog.taken_time >= cutoff_time,
                MedicationLog.status == "taken"
            ).order_by(MedicationLog.taken_time.desc())
            return session.exec(query).first()
    
    @staticmethod
    def get_today_medication_logs(user_id: int, medication_id: int) -> List[MedicationLog]:
        """Get all medication logs for today"""
        with get_session() as session:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            query = select(MedicationLog).where(
                MedicationLog.user_id == user_id,
                MedicationLog.medication_id == medication_id,
                MedicationLog.taken_time >= today_start
            ).order_by(MedicationLog.taken_time.desc())
            return session.exec(query).all()

class CaregiverAlertCRUD:
    @staticmethod
    def create_alert(user_id: int, alert_type: str, title: str, description: str,
                    severity: str = "medium") -> CaregiverAlert:
        """Create a caregiver alert"""
        with get_session() as session:
            alert = CaregiverAlert(
                user_id=user_id,
                alert_type=alert_type,
                title=title,
                description=description,
                severity=severity
            )
            session.add(alert)
            session.commit()
            session.refresh(alert)
            return alert
    
    @staticmethod
    def get_unresolved_alerts(user_id: int = None) -> List[CaregiverAlert]:
        """Get unresolved alerts"""
        with get_session() as session:
            query = select(CaregiverAlert).where(CaregiverAlert.resolved == False)
            if user_id:
                query = query.where(CaregiverAlert.user_id == user_id)
            query = query.order_by(CaregiverAlert.created_at.desc())
            return session.exec(query).all()
    
    @staticmethod
    def resolve_alert(alert_id: int) -> Optional[CaregiverAlert]:
        """Resolve an alert"""
        with get_session() as session:
            alert = session.get(CaregiverAlert, alert_id)
            if alert:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                session.add(alert)
                session.commit()
                session.refresh(alert)
            return alert

class CaregiverPatientCRUD:
    @staticmethod
    def assign_patient(caregiver_id: int, patient_id: int, relationship: str = None, 
                      notification_preferences: dict = None) -> CaregiverPatientAssignment:
        """Assign a patient to a caregiver"""
        with get_session() as session:
            assignment = CaregiverPatientAssignment(
                caregiver_id=caregiver_id,
                patient_id=patient_id,
                relationship=relationship,
                notification_preferences=json.dumps(notification_preferences) if notification_preferences else None
            )
            session.add(assignment)
            session.commit()
            session.refresh(assignment)
            return assignment
    
    @staticmethod
    def get_caregiver_patients(caregiver_id: int) -> List[User]:
        """Get all patients assigned to a caregiver"""
        with get_session() as session:
            query = select(CaregiverPatientAssignment).where(
                CaregiverPatientAssignment.caregiver_id == caregiver_id
            )
            assignments = session.exec(query).all()
            
            patients = []
            for assignment in assignments:
                patient = session.get(User, assignment.patient_id)
                if patient:
                    patients.append(patient)
            return patients
    
    @staticmethod
    def get_patient_caregivers(patient_id: int) -> List[User]:
        """Get all caregivers assigned to a patient"""
        with get_session() as session:
            query = select(CaregiverPatientAssignment).where(
                CaregiverPatientAssignment.patient_id == patient_id
            )
            assignments = session.exec(query).all()
            
            caregivers = []
            for assignment in assignments:
                caregiver = session.get(User, assignment.caregiver_id)
                if caregiver:
                    caregivers.append(caregiver)
            return caregivers
    
    @staticmethod
    def remove_assignment(caregiver_id: int, patient_id: int) -> bool:
        """Remove patient assignment from caregiver"""
        with get_session() as session:
            query = select(CaregiverPatientAssignment).where(
                CaregiverPatientAssignment.caregiver_id == caregiver_id,
                CaregiverPatientAssignment.patient_id == patient_id
            )
            assignment = session.exec(query).first()
            if assignment:
                session.delete(assignment)
                session.commit()
                return True
            return False

class PersonalEventCRUD:
    @staticmethod
    def create_event(user_id: int, event_type: str, title: str, description: str = None,
                    event_date: datetime = None, recurring: bool = False, 
                    importance: str = "medium") -> PersonalEvent:
        """Create a personal event for memory tracking"""
        with get_session() as session:
            event = PersonalEvent(
                user_id=user_id,
                event_type=event_type,
                title=title,
                description=description,
                event_date=event_date,
                recurring=recurring,
                importance=importance
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event
    
    @staticmethod
    def get_user_events(user_id: int, limit: int = 50) -> List[PersonalEvent]:
        """Get personal events for a user"""
        with get_session() as session:
            query = select(PersonalEvent).where(
                PersonalEvent.user_id == user_id
            ).order_by(PersonalEvent.created_at.desc()).limit(limit)
            return session.exec(query).all()
    
    @staticmethod
    def get_upcoming_events(user_id: int, days: int = 30) -> List[PersonalEvent]:
        """Get upcoming events in the next N days"""
        with get_session() as session:
            future_date = datetime.now() + timedelta(days=days)
            query = select(PersonalEvent).where(
                PersonalEvent.user_id == user_id,
                PersonalEvent.event_date.isnot(None),
                PersonalEvent.event_date <= future_date,
                PersonalEvent.event_date >= datetime.now()
            ).order_by(PersonalEvent.event_date)
            return session.exec(query).all()
    
    @staticmethod
    def delete_event(event_id: int) -> bool:
        """Delete a personal event"""
        with get_session() as session:
            event = session.get(PersonalEvent, event_id)
            if event:
                session.delete(event)
                session.commit()
                return True
            return False
