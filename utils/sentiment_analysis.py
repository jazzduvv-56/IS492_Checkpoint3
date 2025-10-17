import os
import json
from groq import Groq
from typing import Dict, Any

class SentimentAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"  # Using Groq model
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using Groq LLM
        Returns: {
            "score": float (-1 to 1),
            "label": str ("positive", "negative", "neutral"),
            "confidence": float (0 to 1),
            "emotions": list of detected emotions
        }
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert sentiment analyzer specializing in elderly care conversations. 
                        Analyze the sentiment of the given text and provide:
                        1. A sentiment score from -1 (very negative) to 1 (very positive)
                        2. A label: "positive", "negative", or "neutral"
                        3. A confidence score from 0 to 1
                        4. A list of detected emotions (e.g., joy, sadness, anxiety, contentment, worry, etc.)
                        
                        Be especially sensitive to:
                        - Signs of pain, discomfort, or distress
                        - Loneliness or isolation
                        - Confusion or memory concerns
                        - Medication-related anxiety
                        - Family or social connections
                        
                        Respond with JSON only in this format:
                        {
                            "score": -0.5,
                            "label": "negative",
                            "confidence": 0.8,
                            "emotions": ["worry", "sadness"]
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the sentiment of this text: \"{text}\""
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and clean the response
            return {
                "score": max(-1, min(1, float(result.get("score", 0)))),
                "label": result.get("label", "neutral"),
                "confidence": max(0, min(1, float(result.get("confidence", 0.5)))),
                "emotions": result.get("emotions", [])
            }
            
        except Exception as e:
            # Fallback to simple rule-based analysis
            return self._fallback_analysis(text)
    
    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment analysis as fallback
        """
        text_lower = text.lower()
        
        # Define word lists
        positive_words = [
            "good", "great", "happy", "wonderful", "excellent", "love", "enjoy",
            "better", "fine", "well", "nice", "pleasant", "comfortable", "peaceful"
        ]
        
        negative_words = [
            "bad", "terrible", "awful", "hate", "horrible", "pain", "hurt", "sad",
            "worried", "anxious", "confused", "lost", "dizzy", "sick", "tired",
            "lonely", "scared", "frightened", "depressed", "upset"
        ]
        
        concern_words = [
            "pain", "hurt", "dizzy", "fall", "emergency", "help", "confused",
            "memory", "forgot", "lost", "scared", "can't", "unable", "difficult"
        ]
        
        # Count words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        concern_count = sum(1 for word in concern_words if word in text_lower)
        
        # Calculate score
        total_words = len(text_lower.split())
        if total_words == 0:
            return {"score": 0, "label": "neutral", "confidence": 0.5, "emotions": []}
        
        # Weight concerns more heavily
        score = (positive_count - negative_count - (concern_count * 1.5)) / total_words
        score = max(-1, min(1, score))
        
        # Determine label
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        # Basic emotion detection
        emotions = []
        if concern_count > 0:
            emotions.append("concern")
        if any(word in text_lower for word in ["pain", "hurt", "sick"]):
            emotions.append("discomfort")
        if any(word in text_lower for word in ["lonely", "alone", "miss"]):
            emotions.append("loneliness")
        if any(word in text_lower for word in ["happy", "good", "great"]):
            emotions.append("contentment")
        if any(word in text_lower for word in ["worried", "anxious", "scared"]):
            emotions.append("anxiety")
        
        return {
            "score": score,
            "label": label,
            "confidence": 0.6,  # Lower confidence for rule-based
            "emotions": emotions
        }

# Global instance for easy access
_analyzer = None

def get_analyzer():
    """Get singleton sentiment analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Convenience function to analyze sentiment
    """
    analyzer = get_analyzer()
    return analyzer.analyze(text)

# Additional utility functions
def get_sentiment_emoji(score: float) -> str:
    """Convert sentiment score to emoji"""
    if score > 0.5:
        return "😊"
    elif score > 0.2:
        return "🙂"
    elif score > -0.2:
        return "😐"
    elif score > -0.5:
        return "😟"
    else:
        return "😢"

def get_sentiment_color(score: float) -> str:
    """Convert sentiment score to color (for UI)"""
    if score > 0.3:
        return "green"
    elif score > -0.3:
        return "yellow"
    else:
        return "red"

def classify_concern_level(emotions: list, score: float) -> str:
    """Classify the level of concern based on emotions and score"""
    high_concern_emotions = ["anxiety", "fear", "distress", "confusion"]
    medium_concern_emotions = ["worry", "sadness", "discomfort", "loneliness"]
    
    if any(emotion in high_concern_emotions for emotion in emotions) or score < -0.7:
        return "high"
    elif any(emotion in medium_concern_emotions for emotion in emotions) or score < -0.3:
        return "medium"
    else:
        return "low"
