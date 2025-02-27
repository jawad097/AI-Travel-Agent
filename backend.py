import os
import csv
import json
import re
import requests
import redis
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Body, Request, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.middleware.cors import CORSMiddleware
import openai
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Redis connection
try:
    redis_conn = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )
    redis_conn.ping()
    print("Connected to Redis successfully!")
except redis.ConnectionError:
    print("Warning: Could not connect to Redis. Session persistence will be disabled.")
    redis_conn = None


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

# Redis helper functions
def save_to_redis(key: str, data: Any, expiration: int = 60*60*24*30) -> bool:
    """Save data to Redis with expiration (default 30 days)"""
    if redis_conn:
        try:
            redis_conn.set(key, json.dumps(data))
            redis_conn.expire(key, expiration)
            return True
        except Exception as e:
            print(f"Error saving to Redis: {str(e)}")
    return False

def get_from_redis(key: str) -> Any:
    """Get data from Redis"""
    if redis_conn:
        try:
            data = redis_conn.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Error getting from Redis: {str(e)}")
    return None

# Location and travel helper functions
def calculate_distance(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float) -> float:
    """Calculate distance between two points using Haversine formula (in km)"""
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(origin_lat)
    lon1 = math.radians(origin_lng)
    lat2 = math.radians(dest_lat)
    lon2 = math.radians(dest_lng)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def estimate_flight_cost(distance: float, price_level: str) -> float:
    """Estimate flight cost based on distance and price level"""
    # Base cost per km
    if price_level == "$":
        base_cost_per_km = 0.10  # Budget airlines
    elif price_level == "$$$":
        base_cost_per_km = 0.30  # Premium airlines
    else:  # $$
        base_cost_per_km = 0.20  # Standard airlines
    
    # Apply distance-based scaling (longer flights are cheaper per km)
    if distance < 1000:
        scaling_factor = 1.5  # Short flights are more expensive per km
    elif distance < 5000:
        scaling_factor = 1.0  # Medium flights
    else:
        scaling_factor = 0.8  # Long flights are cheaper per km
    
    # Calculate base cost
    base_cost = distance * base_cost_per_km * scaling_factor
    
    # Add random variation (±15%)
    variation = random.uniform(0.85, 1.15)
    
    # Round to nearest 10
    return round(base_cost * variation, -1)

def get_popular_places(destination: str) -> List[Dict[str, Any]]:
    """Get popular places for a destination"""
    popular_places = {
        "paris": [
            {
                "name": "Eiffel Tower",
                "description": "Iconic iron tower offering city views",
                "rating": 4.7,
                "lat": 48.8584, 
                "lng": 2.2945
            },
            {
                "name": "Louvre Museum",
                "description": "World's largest art museum & historic monument",
                "rating": 4.8,
                "lat": 48.8606, 
                "lng": 2.3376
            },
            {
                "name": "Notre-Dame Cathedral",
                "description": "Medieval Catholic cathedral with Gothic architecture",
                "rating": 4.7,
                "lat": 48.8530, 
                "lng": 2.3499
            },
            {
                "name": "Arc de Triomphe",
                "description": "Iconic triumphal arch honoring those who fought for France",
                "rating": 4.7,
                "lat": 48.8738, 
                "lng": 2.2950
            },
            {
                "name": "Montmartre",
                "description": "Artistic hilltop district with Sacré-Cœur Basilica",
                "rating": 4.6,
                "lat": 48.8867, 
                "lng": 2.3431
            }
        ],
        "london": [
            {
                "name": "Tower of London",
                "description": "Historic castle and former prison on the River Thames",
                "rating": 4.6,
                "lat": 51.5081, 
                "lng": -0.0759
            },
            {
                "name": "British Museum",
                "description": "Museum of human history, art and culture",
                "rating": 4.8,
                "lat": 51.5194, 
                "lng": -0.1269
            },
            {
                "name": "Buckingham Palace",
                "description": "The Queen's official London residence",
                "rating": 4.5,
                "lat": 51.5014, 
                "lng": -0.1419
            },
            {
                "name": "London Eye",
                "description": "Giant Ferris wheel on the South Bank of the Thames",
                "rating": 4.5,
                "lat": 51.5033, 
                "lng": -0.1195
            },
            {
                "name": "Westminster Abbey",
                "description": "Gothic abbey church and UNESCO World Heritage Site",
                "rating": 4.7,
                "lat": 51.4994, 
                "lng": -0.1276
            }
        ],
        "new york": [
            {
                "name": "Statue of Liberty",
                "description": "Iconic copper statue symbolizing freedom",
                "rating": 4.7,
                "lat": 40.6892, 
                "lng": -74.0445
            },
            {
                "name": "Empire State Building",
                "description": "Iconic 102-story skyscraper with observation decks",
                "rating": 4.7,
                "lat": 40.7484, 
                "lng": -73.9857
            },
            {
                "name": "Central Park",
                "description": "Urban park spanning 843 acres",
                "rating": 4.8,
                "lat": 40.7812, 
                "lng": -73.9665
            },
            {
                "name": "Times Square",
                "description": "Iconic commercial intersection and entertainment center",
                "rating": 4.7,
                "lat": 40.7580, 
                "lng": -73.9855
            },
            {
                "name": "Metropolitan Museum of Art",
                "description": "One of the world's largest and finest art museums",
                "rating": 4.8,
                "lat": 40.7794, 
                "lng": -73.9632
            }
        ],
        "tokyo": [
            {
                "name": "Tokyo Skytree",
                "description": "Tallest tower in Japan with observation decks",
                "rating": 4.6,
                "lat": 35.7101, 
                "lng": 139.8107
            },
            {
                "name": "Senso-ji Temple",
                "description": "Ancient Buddhist temple in Asakusa",
                "rating": 4.7,
                "lat": 35.7147, 
                "lng": 139.7966
            },
            {
                "name": "Meiji Shrine",
                "description": "Shinto shrine dedicated to Emperor Meiji",
                "rating": 4.7,
                "lat": 35.6763, 
                "lng": 139.6993
            },
            {
                "name": "Tokyo Imperial Palace",
                "description": "Primary residence of the Emperor of Japan",
                "rating": 4.5,
                "lat": 35.6852, 
                "lng": 139.7528
            },
            {
                "name": "Shibuya Crossing",
                "description": "Famous scramble crossing and shopping district",
                "rating": 4.7,
                "lat": 35.6595, 
                "lng": 139.7004
            }
        ],
        "rome": [
            {
                "name": "Colosseum",
                "description": "Iconic ancient Roman amphitheater",
                "rating": 4.8,
                "lat": 41.8902, 
                "lng": 12.4922
            },
            {
                "name": "Vatican Museums",
                "description": "Museums displaying works from the extensive collection of the Catholic Church",
                "rating": 4.7,
                "lat": 41.9060, 
                "lng": 12.4526
            },
            {
                "name": "Trevi Fountain",
                "description": "Iconic 18th-century fountain",
                "rating": 4.8,
                "lat": 41.9009, 
                "lng": 12.4833
            },
            {
                "name": "Pantheon",
                "description": "Former Roman temple, now a church",
                "rating": 4.8,
                "lat": 41.8986, 
                "lng": 12.4769
            },
            {
                "name": "Roman Forum",
                "description": "Ancient government buildings of Rome",
                "rating": 4.7,
                "lat": 41.8925, 
                "lng": 12.4853
            }
        ]
    }
    
    # Default to empty list if destination not found
    destination_lower = destination.lower()
    for key in popular_places:
        if key in destination_lower:
            return popular_places[key]
    
    # Return empty list if no match
    return []

class UserPreferences(BaseModel):
    """Model for user preferences"""
    destination: Optional[str] = None
    budget_range: List[int] = [800, 5000]
    price_preference: str = "$$ - Moderate"
    activities: List[str] = ["Cultural", "Adventure"]
    accommodation_type: str = "Hotel"
    trip_duration: int = 7
    travel_pace: str = "Moderate"
    cuisine_preferences: List[str] = ["Local", "International"]
    transportation: str = "Public Transport"
    accessibility_needs: bool = False
    travel_season: str = "Summer"
    language_preferences: List[str] = ["English"]

class UserLocation(BaseModel):
    """Model for user location"""
    lat: float
    lng: float
    address: str = "Unknown"

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
            
        # Initialize in-memory user contexts (will be backed by Redis)
        self.user_contexts = {}
        print("AI Travel Agent initialized")

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context with Redis fallback"""
        # Try to get from memory first
        if user_id in self.user_contexts:
            return self.user_contexts[user_id]
        
        # Try to get from Redis
        if redis_conn:
            context_key = f"user_context:{user_id}"
            redis_context = get_from_redis(context_key)
            if redis_context:
                self.user_contexts[user_id] = redis_context
                return redis_context
        
        # Initialize new context
        self._init_user_context(user_id)
        return self.user_contexts[user_id]
    
    def save_user_context(self, user_id: str) -> None:
        """Save user context to Redis"""
        if redis_conn and user_id in self.user_contexts:
            context_key = f"user_context:{user_id}"
            save_to_redis(context_key, self.user_contexts[user_id])

    def process_user_input(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """Main processing pipeline with personalization"""
        # Get user context (from memory or Redis)
        user_context = self.get_user_context(user_id)
        
        # Add message to conversation history
        user_context["conversation_history"].append({"role": "user", "content": user_input})
        
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

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences"""
        user_context = self.get_user_context(user_id)
        user_context["preferences"] = preferences
        
        # Save to Redis
        self.save_user_context(user_id)
        
        return user_context["preferences"]
    
    def update_user_location(self, user_id: str, location: UserLocation) -> Dict[str, Any]:
        """Update user location"""
        user_context = self.get_user_context(user_id)
        user_context["location"] = {
            "lat": location.lat,
            "lng": location.lng,
            "address": location.address
        }
        
        # Save to Redis
        self.save_user_context(user_id)
        
        return user_context["location"]

    # Helper methods
    def _init_user_context(self, user_id: str):
        """Initialize user context with default values"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                "preferences": {
                    "destination": None,
                    "budget_range": [800, 5000],
                    "price_preference": "$$ - Moderate",
                    "activities": ["Cultural", "Adventure"],
                    "accommodation_type": "Hotel",
                    "trip_duration": 7,
                    "travel_pace": "Moderate",
                    "cuisine_preferences": ["Local", "International"],
                    "transportation": "Public Transport",
                    "accessibility_needs": False,
                    "travel_season": "Summer",
                    "language_preferences": ["English"]
                },
                "location": {
                    "lat": 24.8607, # Default to Karachi
                    "lng": 67.0011,
                    "address": "Karachi, Pakistan"
                },
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

@app.post("/chat", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, user_message: UserMessage):
    """Process user chat message and return AI response"""
    try:
        response = travel_agent.process_user_input(user_message.user_id, user_message.message)
        
        # Add personalized recommendations based on user preferences
        if "message" in response:
            # Get user context
            user_context = travel_agent.get_user_context(user_message.user_id)
            preferences = user_context.get("preferences", {})
            user_location = user_context.get("location", {})
            
            # Extract destination from message
            destination = None
            for city in ["paris", "tokyo", "new york", "london", "rome"]:
                if city in user_message.message.lower():
                    destination = city.title()
                    break
            
            # If destination found, add travel information
            if destination:
                # Get destination coordinates
                dest_data = travel_agent._get_map_data(destination)
                
                # Calculate distance and estimate costs
                if user_location and "lat" in user_location and "lng" in user_location:
                    distance = calculate_distance(
                        user_location["lat"], 
                        user_location["lng"],
                        dest_data["lat"], 
                        dest_data["lng"]
                    )
                    
                    # Get price level from preferences
                    price_level = preferences.get("price_preference", "$$ - Moderate")[0]
                    
                    # Estimate flight cost
                    flight_cost = estimate_flight_cost(distance, price_level)
                    
                    # Add travel info to response
                    response["travel_info"] = {
                        "origin": {
                            "lat": user_location["lat"],
                            "lng": user_location["lng"],
                            "address": user_location.get("address", "Your location")
                        },
                        "destination": dest_data,
                        "distance_km": round(distance),
                        "estimated_flight_cost": flight_cost,
                        "currency": "USD"
                    }
                
                # Add popular places
                popular_places = get_popular_places(destination)
                if popular_places:
                    response["popular_places"] = popular_places
                
                # Set destination in response data
                if "data" not in response:
                    response["data"] = {}
                
                response["data"]["locations"] = [dest_data]
                
                # Add route for Google Maps
                if user_location and "lat" in user_location and "lng" in user_location:
                    response["route"] = {
                        "origin": {
                            "lat": user_location["lat"],
                            "lng": user_location["lng"]
                        },
                        "destination": {
                            "lat": dest_data["lat"],
                            "lng": dest_data["lng"]
                        }
                    }
            
            # Add sample activities with price levels matching user preferences
            if "activities" not in response and destination:
                price_level = preferences.get("price_preference", "$$ - Moderate")[0]
                budget_range = preferences.get("budget_range", [800, 5000])
                
                # Generate sample activities based on preferences
                response["activities"] = generate_sample_activities(
                    price_level, 
                    budget_range,
                    preferences.get("activities", [])
                )
            
            # Add sample accommodations
            if "accommodations" not in response and ("stay" in user_message.message.lower() or destination):
                price_level = preferences.get("price_preference", "$$ - Moderate")[0]
                accommodation_type = preferences.get("accommodation_type", "Hotel")
                
                # Generate sample accommodations based on preferences
                response["accommodations"] = generate_sample_accommodations(
                    price_level,
                    accommodation_type
                )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/location/{user_id}", response_model=Dict[str, Any])
async def update_location(user_id: str, location: UserLocation):
    """Update user location"""
    try:
        updated_location = travel_agent.update_user_location(user_id, location)
        return {"location": updated_location, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/itinerary/{user_id}", response_model=Dict[str, Any])
async def get_itinerary(user_id: str):
    """Get user's travel itinerary"""
    try:
        user_context = travel_agent.get_user_context(user_id)
        return {"itinerary": user_context.get("itinerary", "")}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User not found: {str(e)}")

@app.get("/preferences/{user_id}", response_model=Dict[str, Any])
async def get_preferences(user_id: str):
    """Get user preferences"""
    try:
        user_context = travel_agent.get_user_context(user_id)
        return {"preferences": user_context["preferences"]}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User not found: {str(e)}")

@app.post("/preferences/{user_id}", response_model=Dict[str, Any])
async def update_preferences(user_id: str, preferences: Dict[str, Any] = Body(...)):
    """Update user preferences"""
    try:
        updated_preferences = travel_agent.update_user_preferences(user_id, preferences)
        return {"preferences": updated_preferences, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: Request, data: dict = Body(...)):
    """Submit user feedback"""
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

@app.post("/reset/{user_id}", response_model=Dict[str, str])
async def reset_user(user_id: str):
    """Reset user conversation and context"""
    try:
        # Remove from memory
        if user_id in travel_agent.user_contexts:
            del travel_agent.user_contexts[user_id]
        
        # Remove from Redis
        if redis_conn:
            context_key = f"user_context:{user_id}"
            redis_conn.delete(context_key)
            
            # Also delete conversation and preferences
            redis_conn.delete(f"conversation:{user_id}")
            redis_conn.delete(f"preferences:{user_id}")
            
        return {"status": "success", "message": f"User {user_id} data has been reset"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Helper functions for generating sample data
def generate_sample_activities(price_level: str, budget_range: List[int], user_activities: List[str]) -> List[Dict[str, Any]]:
    """Generate sample activities based on user preferences"""
    activities = []
    
    # Base price ranges
    if price_level == "$":
        price_range = (20, 50)
    elif price_level == "$$$":
        price_range = (100, 300)
    else:  # $$
        price_range = (50, 100)
    
    # Sample activities
    activity_templates = [
        {
            "name": "Eiffel Tower Tour",
            "category": "Cultural",
            "description": "Visit the iconic Eiffel Tower and enjoy panoramic views of Paris.",
            "duration": "2 hours",
            "rating": 4.7,
            "price_level": "$$"
        },
        {
            "name": "Louvre Museum Visit",
            "category": "Cultural",
            "description": "Explore one of the world's largest art museums and see the Mona Lisa.",
            "duration": "3 hours",
            "rating": 4.8,
            "price_level": "$$"
        },
        {
            "name": "Seine River Cruise",
            "category": "Relaxation",
            "description": "Enjoy a relaxing cruise along the Seine River with views of Paris landmarks.",
            "duration": "1 hour",
            "rating": 4.5,
            "price_level": "$"
        },
        {
            "name": "Montmartre Walking Tour",
            "category": "Adventure",
            "description": "Explore the artistic neighborhood of Montmartre and visit Sacré-Cœur.",
            "duration": "2.5 hours",
            "rating": 4.6,
            "price_level": "$"
        },
        {
            "name": "Fine Dining Experience",
            "category": "Luxury",
            "description": "Enjoy a gourmet meal at a Michelin-starred restaurant in Paris.",
            "duration": "3 hours",
            "rating": 4.9,
            "price_level": "$$$"
        }
    ]
    
    # Filter and adjust based on user preferences
    for template in activity_templates:
        # Match price level
        if template["price_level"] == price_level:
            # Match activities if specified
            if not user_activities or any(activity.lower() in template["category"].lower() for activity in user_activities):
                # Generate a price within range
                import random
                price = random.randint(price_range[0], price_range[1])
                
                activity = template.copy()
                activity["price"] = price
                activities.append(activity)
    
    # Ensure we have at least 3 activities
    while len(activities) < 3:
        import random
        template = random.choice(activity_templates)
        price = random.randint(price_range[0], price_range[1])
        
        activity = template.copy()
        activity["price"] = price
        if activity not in activities:
            activities.append(activity)
    
    return activities

def generate_sample_accommodations(price_level: str, accommodation_type: str) -> List[Dict[str, Any]]:
    """Generate sample accommodations based on user preferences"""
    accommodations = []
    
    # Base price ranges per night
    if price_level == "$":
        price_range = (50, 100)
    elif price_level == "$$$":
        price_range = (300, 800)
    else:  # $$
        price_range = (100, 300)
    
    # Sample accommodations
    accommodation_templates = [
        {
            "name": "Luxury Palace Hotel",
            "type": "Luxury Resort",
            "description": "5-star luxury hotel with spa, pool, and gourmet restaurants.",
            "amenities": ["Pool", "Spa", "Gym", "Restaurant", "Room Service", "Free WiFi"],
            "rating": 4.9,
            "price_level": "$$$"
        },
        {
            "name": "Boutique City Center Hotel",
            "type": "Boutique Hotel",
            "description": "Charming boutique hotel in the heart of the city with unique rooms.",
            "amenities": ["Free WiFi", "Breakfast", "Bar", "Concierge"],
            "rating": 4.7,
            "price_level": "$$"
        },
        {
            "name": "Comfortable Budget Inn",
            "type": "Hotel",
            "description": "Clean and comfortable hotel with all the basic amenities.",
            "amenities": ["Free WiFi", "Breakfast", "24-hour Reception"],
            "rating": 4.2,
            "price_level": "$"
        },
        {
            "name": "Traveler's Hostel",
            "type": "Hostel",
            "description": "Social hostel with private and shared rooms, perfect for budget travelers.",
            "amenities": ["Free WiFi", "Shared Kitchen", "Lounge", "Laundry"],
            "rating": 4.0,
            "price_level": "$"
        },
        {
            "name": "Riverside Apartment",
            "type": "Apartment",
            "description": "Self-catering apartment with kitchen and living area, ideal for longer stays.",
            "amenities": ["Kitchen", "Washing Machine", "Free WiFi", "Living Room"],
            "rating": 4.5,
            "price_level": "$$"
        }
    ]
    
    # Filter and adjust based on user preferences
    for template in accommodation_templates:
        # Match price level
        if template["price_level"] == price_level:
            # Match accommodation type if specified
            if accommodation_type.lower() in template["type"].lower() or accommodation_type == "Any":
                # Generate a price within range
                import random
                price = random.randint(price_range[0], price_range[1])
                
                accommodation = template.copy()
                accommodation["price"] = price
                accommodations.append(accommodation)
    
    # Ensure we have at least 2 accommodations
    while len(accommodations) < 2:
        import random
        template = random.choice(accommodation_templates)
        price = random.randint(price_range[0], price_range[1])
        
        accommodation = template.copy()
        accommodation["price"] = price
        if accommodation not in accommodations:
            accommodations.append(accommodation)
    
    return accommodations

if __name__ == "__main__":
    import uvicorn
    print("Starting backend server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
