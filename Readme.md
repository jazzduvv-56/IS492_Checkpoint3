# Carely - AI Companion for Elderly Care

## Overview

Carely is a comprehensive AI-powered elderly care companion application that provides proactive health monitoring, medication management, and emotional support. The system uses OpenAI's GPT-5 for natural language interactions, sentiment analysis, and emergency detection. It features a dual-portal architecture serving both patients and caregivers, with automated scheduling for medication reminders and wellness check-ins. The application stores conversation history and personal context to provide personalized, empathetic interactions while monitoring for concerning health patterns and alerting caregivers when necessary.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid UI development with minimal complexity
- **Portal Structure**: Dual-portal design with separate patient and caregiver interfaces
- **Session Management**: Streamlit session state for user authentication and context persistence
- **Visualization**: Plotly for interactive charts showing medication adherence trends and sentiment analysis
- **Voice Input**: Integrated speech-to-text capability using streamlit-mic-recorder for accessibility

**Rationale**: Streamlit was chosen for its simplicity and rapid development capabilities, allowing quick iteration on UI features. The dual-portal approach separates concerns between patient-facing companionship features and caregiver-facing analytics/monitoring tools.

### Backend Architecture
- **AI Agent System**: Central CompanionAgent class orchestrating all AI interactions using Groq API (llama-3.1-8b-instant)
- **Multi-Layer Memory System**: 
  - **Short-term memory**: DB-based, fetches last 10 conversations from database
  - **Long-term memory**: ChromaDB vector store with default embedding function (all-MiniLM-L6-v2) for semantic retrieval
  - **Episodic memory**: Daily summaries with automated generation at 11:59 PM CT, stored in vector database
  - **Structured memory**: User profiles, medication schedules, and personal events from SQLite
- **Scheduling System**: APScheduler (BackgroundScheduler) handles automated reminders, check-ins, reports, and daily memory summarization
- **CRUD Pattern**: Separated data access layer with dedicated CRUD classes for each model (UserCRUD, MedicationCRUD, etc.)
- **Authentication**: Simple hash-based authentication (SHA-256) with role-based access control (patient/caregiver/admin hierarchy)

**Rationale**: The agent-based architecture centralizes AI logic for maintainability. The multi-layer memory system enables personalized, context-aware interactions using free, local, open-source tools (ChromaDB + sentence-transformers) instead of paid embedding APIs. DB-backed short-term memory eliminates in-memory state issues. Daily automated summarization (11:59 PM CT) maintains long-term context efficiently. Separation of scheduling into a background service ensures reminders execute reliably independent of user interactions.

### Data Storage
- **Database**: SQLite with SQLModel ORM for type-safe database operations
- **Schema Design**: 
  - Core entities: User, Medication, Conversation, Reminder, MedicationLog
  - Relationship tracking: CaregiverPatientAssignment links caregivers to patients
  - Alert system: CaregiverAlert for flagging concerning patterns
  - Personal context: PersonalEvent stores important life events and memories
- **JSON Fields**: Preferences, schedule times, and metadata stored as JSON strings for flexibility
- **Timestamps**: All critical entities track creation time for temporal analysis

**Rationale**: SQLite chosen for simplicity and zero-configuration deployment. SQLModel provides type safety and Pydantic validation. JSON fields allow schema flexibility without migrations for user-specific data structures.

### AI and Analytics Components
- **Conversation AI**: Groq API (llama-3.1-8b-instant) with specialized system prompts for elderly-appropriate communication, enforced brevity (≤4 sentences, max 230 tokens)
- **Sentiment Analysis**: Local keyword-based sentiment detection (no API calls) analyzing emotional states
- **Emergency Detection**: Local keyword-based emergency detection identifying medical emergencies with severity classification (high/medium/low)
- **Emergency Trigger System**: Real-time emergency detection in chat with interactive safety sheet UI
  - Detects concerning health symptoms (chest pain, dizziness, breathing issues, etc.)
  - Shows three-step safety sheet: emergency alert → action options → confirmation
  - Options: Contact caregiver via Telegram or self-resolve ("I Feel OK")
  - Session state management to prevent duplicate alerts
- **Context Building**: Multi-layer memory system provides comprehensive context from all memory layers (short-term, long-term semantic search, episodic summaries, structured data)
- **Semantic Search**: ChromaDB vector store with default embedding function (all-MiniLM-L6-v2) for embedding-based retrieval - completely free and local

**Rationale**: Groq API provides fast, cost-effective LLM responses with strict brevity controls. Local sentiment/emergency detection eliminates API costs while maintaining accuracy. ChromaDB with default embeddings provides production-quality semantic search entirely free and local (no paid embedding APIs). Multi-layer memory architecture delivers rich, personalized context from recent conversations, semantic-matched past interactions, daily summaries, and structured user data.

### Notification and Alert System
- **Emergency Alerts**: Real-time caregiver notifications triggered by emergency detection
- **Telegram Integration**: TelegramNotifier class for push notifications to caregivers
- **Alert Persistence**: CaregiverAlert database table maintains alert history
- **Scheduled Reminders**: Automated medication and wellness check-in notifications

**Rationale**: Multi-channel notification approach ensures critical alerts reach caregivers. Telegram chosen for its reliability and ease of integration. Database persistence enables alert tracking and pattern analysis.

## External Dependencies

### AI Services
- **Groq API**: llama-3.1-8b-instant model for conversation generation
  - Requires: `GROQ_API_KEY` environment variable
  - Usage: Companion chat responses (single API call per message)
  - Rate limits: Free tier with usage caps
- **Local AI Processing**: Sentiment analysis and emergency detection use keyword-based local processing (no API calls)

### Communication Services
- **Telegram Bot API**: Push notifications to caregivers
  - Requires: `TELEGRAM_BOT_TOKEN` environment variable
  - Usage: Emergency alerts, medication reminders, status updates
  - Stored: `telegram_chat_id` in User table per user/caregiver

### Python Packages
- **Core Framework**: 
  - `streamlit`: Web interface and session management
  - `fastapi`: REST API endpoints (routes.py)
- **Database**: 
  - `sqlmodel`: ORM and schema definition
  - `sqlite3`: Database engine (built-in)
  - `chromadb`: Vector database for semantic memory search
- **AI/ML**:
  - `groq`: Official Groq Python client for LLM API
  - `numpy`, `scikit-learn`: Supporting ML utilities
- **Scheduling**: 
  - `apscheduler`: Background job scheduling
- **Data Processing**:
  - `plotly`: Interactive visualizations
- **Utilities**:
  - `requests`: HTTP client for Telegram API
  - `streamlit-mic-recorder`: Voice input component
  - `gtts`: Text-to-speech for voice output

### Database Schema
- **Primary Database**: `carely.db` (SQLite file)
- **Tables**: User, Medication, Conversation, Reminder, MedicationLog, CaregiverAlert, CaregiverPatientAssignment, PersonalEvent
- **No external database service required** - self-contained SQLite file

### Environment Configuration
Required environment variables:
- `GROQ_API_KEY`: Groq API authentication for LLM responses
- `TELEGRAM_BOT_TOKEN`: Telegram bot authentication (optional for notifications)
- `TELEGRAM_CHAT_ID`: Telegram chat ID for caregiver notifications (optional)

### Local Storage and Persistence
- **ChromaDB Storage**: Vector embeddings stored in `./data/vectors/` directory
- **SQLite Database**: All structured data in `carely.db`
- **No External Vector Store**: All embeddings generated and stored locally using ChromaDB's default embedding function (all-MiniLM-L6-v2)

### Sample Data System
- Initialization logic in `data/sample_data.py`
- Auto-populates users, medications, caregivers, and relationships on first run
- Checks for existing data to prevent duplication
