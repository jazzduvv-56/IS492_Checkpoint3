import os
from groq import Groq
from typing import Dict, Any
import json

class EmergencyDetector:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
        
    def detect_emergency(self, text: str) -> Dict[str, Any]:
        """
        Detect if a message contains emergency health concerns
        Returns: {
            "is_emergency": bool,
            "severity": str ("low", "medium", "high"),
            "concerns": list of detected health concerns,
            "confidence": float (0 to 1)
        }
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert health emergency detector for elderly care. 
                        Analyze the given text to determine if it contains emergency health concerns that require immediate attention.
                        
                        Look for these emergency indicators:
                        - Chest pain, pressure, or tightness
                        - Difficulty breathing, shortness of breath
                        - Severe dizziness, lightheadedness, or fainting
                        - Heart palpitations, irregular heartbeat, fast heartbeat
                        - Severe headache or confusion
                        - Weakness, numbness (especially on one side)
                        - Vision problems or speech difficulties
                        - Severe bleeding or injury
                        - Allergic reactions
                        - Severe abdominal pain
                        
                        Respond with JSON in this exact format:
                        {
                            "is_emergency": true/false,
                            "severity": "low/medium/high",
                            "concerns": ["list of specific health concerns detected"],
                            "confidence": 0.85
                        }
                        
                        - severity "high": immediate life-threatening symptoms (chest pain, difficulty breathing, stroke signs)
                        - severity "medium": concerning symptoms that need prompt attention (severe dizziness, fast heartbeat)
                        - severity "low": mild symptoms mentioned casually
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this message for emergency health concerns: \"{text}\""
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=200
            )
            
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            
            return {
                "is_emergency": result.get("is_emergency", False),
                "severity": result.get("severity", "low"),
                "concerns": result.get("concerns", []),
                "confidence": max(0, min(1, float(result.get("confidence", 0))))
            }
            
        except Exception as e:
            print(f"Error in emergency detection: {e}")
            return {
                "is_emergency": False,
                "severity": "low",
                "concerns": [],
                "confidence": 0
            }

def detect_emergency(text: str) -> Dict[str, Any]:
    """Helper function to detect emergency in text"""
    detector = EmergencyDetector()
    return detector.detect_emergency(text)
