import os
import csv
import json
import re
import requests
import redis
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.middleware.cors import CORSMiddleware
import openai
from openai import OpenAI

load_dotenv()

redis_conn = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0))
)


CSV_FILENAME = "travel_feedback.csv"
CSV_FIELDNAMES = ["user_id", "timestamp", "sentiment", "likes", "dislikes", "suggestions"]

class CSVFeedbackManager:
    def __init__(self):
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
    
    def save_feedback(self, user_id: str, feedback: dict):
        """Save feedback to CSV"""
        try:
            with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
                writer.writerow({
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "sentiment": feedback.get("sentiment", "neutral"),
                    "likes": json.dumps(feedback.get("likes", [])),
                    "dislikes": json.dumps(feedback.get("dislikes", [])),
                    "suggestions": json.dumps(feedback.get("suggestions", []))
                })
        except Exception as e:
            print(f"Error saving feedback: {str(e)}")

feedback_manager = CSVFeedbackManager()

class TravelAgentModel:
    def __init__(self):
        self.google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        
        # Check if the API key is properly formatted
        if self.openai_api_key:
            # Remove quotes if they exist in the API key
            if self.openai_api_key.startswith('"') and self.openai_api_key.endswith('"'):
                self.openai_api_key = self.openai_api_key[1:-1]
            
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            print(f"OpenAI API Key loaded: {self.openai_api_key[:8]}...{self.openai_api_key[-4:]}")
        else:
            print("WARNING: OpenAI API Key not found in environment variables")
            
        self.user_contexts = {}
        print("AI Travel Agent initialized")

    def process_user_input(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        self._init_user_context(user_id)
        self.user_contexts[user_id]["conversation_history"].append({"role": "user", "content": user_input})
        
        try:
            # Debugging checks
            print(f"\n=== Processing input for {user_id} ===")
            print(f"API Key: {'set' if self.openai_api_key else 'missing'}")
            print(f"Client initialized: {bool(self.openai_client)}")
            print(f"User input: {user_input[:100]}...")

            if not self.openai_client:
                raise ValueError("OpenAI client not initialized - check API key")

            # Construct the prompt
            prompt = f"""You are an expert AI Travel Assistant. Provide detailed, structured response about:
    {user_input}

    Include these elements if relevant:
    - Top 3 attractions with brief descriptions
    - Recommended local cuisine
    - Cultural norms/tips
    - Weather considerations
    - Transportation options
    - Safety advice

    Format with clear headings and bullet points. Keep paragraphs under 3 sentences."""
            
            print(f"\n=== Sending to OpenAI ===")
            print(f"Prompt: {prompt[:150]}...")
            print(f"Using model: gpt-3.5-turbo")

            # OpenAI API call
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=700,
                temperature=0.7
            )

            # Process response
            detailed_response = response.choices[0].message.content.strip()
            print(f"\n=== Received Response ===")
            print(f"Response length: {len(detailed_response)} characters")
            print(f"First 100 chars: {detailed_response[:100]}...")

            # Generate follow-up questions using OpenAI
            follow_up_questions = self._generate_follow_up_questions_with_openai(user_input)
            print(f"Generated follow-ups: {follow_up_questions}")

            ai_response = {
                "message": detailed_response,
                "next_questions": follow_up_questions[:3],
                "metadata": {
                    "model": "gpt-3.5-turbo",
                    "tokens_used": response.usage.total_tokens,
                    "response_time": datetime.now().isoformat()
                }
            }

            # Update conversation history
            self.user_contexts[user_id]["conversation_history"].append(
                {"role": "assistant", "content": ai_response["message"]}
            )

            return ai_response
            
        except Exception as e:
            print(f"\n!!! ERROR !!!\n{str(e)}\n")
            import traceback
            traceback.print_exc()

            # Fallback to simple response
            try:
                print("Attempting fallback response...")
                simple_response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Briefly answer: {user_input}"}],
                    max_tokens=150
                )
                return {
                    "message": simple_response.choices[0].message.content.strip(),
                    "next_questions": ["Could you clarify?", "What specifically interests you?"]
                }
            except Exception as fallback_error:
                print(f"Fallback failed: {str(fallback_error)}")
                return {
                    "message": "I'm experiencing technical difficulties. Please try again later.",
                    "next_questions": ["Try asking again", "Ask about another destination"]
                }
    
    def _generate_follow_up_questions_with_openai(self, user_input: str) -> List[str]:
        """Generate relevant follow-up questions using OpenAI"""
        try:
            prompt = f"""
            Based on this travel question: "{user_input}"
            
            Generate 3 natural follow-up questions that would be helpful for continuing the conversation.
            Return only the questions as a JSON array of strings.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=150
            )
            
            result = json.loads(response.choices[0].message.content)
            if "questions" in result and isinstance(result["questions"], list):
                return result["questions"][:3]  # Return up to 3 questions
            else:
                # If the format is unexpected, try to extract questions from the response
                content = response.choices[0].message.content
                if isinstance(content, str):
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            return parsed[:3]
                        for key in parsed:
                            if isinstance(parsed[key], list):
                                return parsed[key][:3]
                    except:
                        pass
                
                # Fallback questions
                return [
                    "Would you like to know more about this?",
                    "Do you have any specific questions?",
                    "What other travel information would be helpful?"
                ]
                
        except Exception as e:
            print(f"Error generating follow-up questions: {str(e)}")
            return [
                "Would you like to know more about this?",
                "Do you have any specific questions?",
                "What other travel information would be helpful?"
            ]

    # Helper methods
    def _init_user_context(self, user_id: str):
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                "preferences": {},
                "conversation_history": [],
                "feedback": []
            }

    def _get_map_data(self, destination: str) -> Dict[str, Any]:
        """Get map data for a destination using Google Maps API"""
        try:
            # If we have a Google Maps API key, use it to get geocoding data
            if self.google_maps_api_key:
                url = f"https://maps.googleapis.com/maps/api/geocode/json?address={destination}&key={self.google_maps_api_key}"
                response = requests.get(url)
                data = response.json()
                
                if data["status"] == "OK" and data["results"]:
                    location = data["results"][0]["geometry"]["location"]
                    return {
                        "name": destination,
                        "lat": location["lat"],
                        "lng": location["lng"],
                        "address": data["results"][0]["formatted_address"]
                    }
            
            # Fallback to hardcoded coordinates for common destinations
            coordinates = {
                "paris": {"lat": 48.8566, "lng": 2.3522},
                "tokyo": {"lat": 35.6762, "lng": 139.6503},
                "new york": {"lat": 40.7128, "lng": -74.0060},
                "london": {"lat": 51.5074, "lng": -0.1278},
                "rome": {"lat": 41.9028, "lng": 12.4964},
                "istanbul": {"lat": 41.0082, "lng": 28.9784},
                "bali": {"lat": -8.3405, "lng": 115.0920},
                "dubai": {"lat": 25.2048, "lng": 55.2708},
                "sydney": {"lat": -33.8688, "lng": 151.2093},
                "barcelona": {"lat": 41.3851, "lng": 2.1734},
                "turkey": {"lat": 38.9637, "lng": 35.2433}
            }
            
            destination_lower = destination.lower()
            for key, coords in coordinates.items():
                if key in destination_lower:
                    return {
                        "name": destination,
                        "lat": coords["lat"],
                        "lng": coords["lng"],
                        "address": f"{destination}"
                    }
            
            # Default coordinates (center of world map)
            return {
                "name": destination,
                "lat": 0,
                "lng": 0,
                "address": destination
            }
        except Exception as e:
            print(f"Error in _get_map_data: {str(e)}")
            # Default coordinates
            return {
                "name": destination,
                "lat": 0,
                "lng": 0,
                "address": destination
            }

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

travel_agent = TravelAgentModel()

class UserMessage(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, user_message: UserMessage):
    try:
        return travel_agent.process_user_input(user_message.user_id, user_message.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/itinerary/{user_id}")
async def get_itinerary(user_id: str):
    try:
        return {"itinerary": travel_agent.user_contexts[user_id].get("itinerary", "")}
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found")

@app.get("/preferences/{user_id}")
async def get_preferences(user_id: str):
    try:
        return {"preferences": travel_agent.user_contexts[user_id]["preferences"]}
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found")

@app.post("/feedback")
async def submit_feedback(request: Request, data: dict = Body(...)):
    try:
        user_id = data.get("user_id")
        item_id = data.get("item_id")
        rating = data.get("rating")
        
        if not all([user_id, item_id, rating]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        feedback = {
            "sentiment": "positive" if rating > 0 else "negative",
            "likes": [item_id] if rating > 0 else [],
            "dislikes": [item_id] if rating < 0 else [],
            "suggestions": []
        }
        
        feedback_manager.save_feedback(user_id, feedback)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset/{user_id}")
async def reset_user(user_id: str):
    if user_id in travel_agent.user_contexts:
        del travel_agent.user_contexts[user_id]
    return {"status": "success"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    print("Starting backend server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
