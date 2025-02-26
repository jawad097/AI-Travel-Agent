# frontend.py
import streamlit as st
import requests
from datetime import datetime
import json
import pandas as pd

# Backend API configuration
BACKEND_URL = "http://127.0.0.1:8000"  # Use 127.0.0.1 instead of localhost

# Print connection status for debugging
print(f"Connecting to backend at: {BACKEND_URL}")

def initialize_session():
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{datetime.now().timestamp()}"
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'destination': None,
            'budget': None,
            'activities': [],
            'accommodation': None
        }
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

def set_page_styling():
    """Apply custom styling to the app"""
    st.markdown("""
    <style>
    .stApp {
        background-color: #2c3e50;
    }
    .css-1d391kg {
        background-color: #34495e;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        border-radius: 20px;
        border: none;
        background-color: #3498db;
    }
    /* Keep text white to be visible on dark background */
    p, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

def display_map(locations):
    """Display locations on a map"""
    if locations:
        st.subheader("Explore Locations 🗺️")
        # Create a DataFrame with lat/long data
        map_data = pd.DataFrame(
            [[loc.get('lat', 0), loc.get('lng', 0)] for loc in locations],
            columns=['lat', 'lon']
        )
        st.map(map_data)
        
        # Display address information
        if len(locations) > 0 and 'address' in locations[0]:
            st.write(f"**Address:** {locations[0]['address']}")

def display_suggested_responses(next_questions):
    """Display suggested responses for the user to click"""
    if next_questions:
        st.write("Quick responses:")
        cols = st.columns(len(next_questions))
        # Generate a unique identifier for this set of quick responses
        # Use current timestamp to ensure uniqueness
        response_id = datetime.now().timestamp()
        for i, question in enumerate(next_questions):
            with cols[i]:
                if st.button(question, key=f"quick_{response_id}_{i}"):
                    st.session_state.user_input = question
                    handle_user_input(question)
                    st.rerun()

def display_recommendations(response):
    """Display recommendations with interactive elements"""
    # Display detailed destination information if available
    if 'data' in response and 'detailed_description' in response['data']:
        with st.expander("Detailed Information", expanded=True):
            st.markdown(f"### About {response['data'].get('name', 'this destination')}")
            st.markdown(response['data'].get('detailed_description', ''))
            
            if 'cultural_highlights' in response['data']:
                st.markdown("### Cultural Highlights")
                st.markdown(response['data'].get('cultural_highlights', ''))
            
            if 'nearby_places' in response['data'] and response['data']['nearby_places']:
                st.markdown("### Nearby Places Worth Visiting")
                nearby = response['data'].get('nearby_places', [])
                if isinstance(nearby, list):
                    for place in nearby:
                        st.markdown(f"- {place}")
                else:
                    st.markdown(nearby)
    
    # Display map if location data is available
    if 'data' in response and 'locations' in response['data']:
        display_map(response['data'].get('locations', []))
    
    if 'activities' in response:
        st.subheader("Recommended Activities 🎯")
        for activity in response.get('activities', []):
            with st.expander(f"{activity.get('name', 'Activity')} ⭐{activity.get('rating', 4.0)}"):
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image("https://via.placeholder.com/150x150?text=Activity", 
                            use_container_width=True)
                with cols[1]:
                    st.write(f"**Category:** {activity.get('category', 'General')}")
                    st.write(f"**Price:** {activity.get('price', 'N/A')}")
                    st.write(f"**Duration:** {activity.get('duration', 'Flexible')}")
                    st.write(activity.get('description', 'No description available'))
                    
                    # Feedback buttons
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("👍 Like", key=f"like_{activity.get('name', 'activity')}"):
                            send_feedback(activity.get('name', 'activity'), 1)
                    with c2:
                        if st.button("👎 Dislike", key=f"dislike_{activity.get('name', 'activity')}"):
                            send_feedback(activity.get('name', 'activity'), -1)

    if 'accommodations' in response:
        st.subheader("Recommended Accommodations 🏨")
        for hotel in response.get('accommodations', []):
            with st.expander(f"{hotel.get('name', 'Hotel')} ⭐{hotel.get('rating', 4.0)}"):
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image("https://via.placeholder.com/150x150?text=Hotel", 
                            use_container_width=True)
                with cols[1]:
                    st.write(f"**Type:** {hotel.get('type', 'Hotel')}")
                    st.write(f"**Price:** {hotel.get('price', 'N/A')}/night")
                    st.write(f"**Amenities:** {', '.join(hotel.get('amenities', []))}")
                    st.write(hotel.get('description', 'No description available'))

    if 'itinerary' in response:
        st.subheader("Your Travel Itinerary 📅")
        st.markdown(response['itinerary'])
    
    # Display suggested follow-up questions
    if 'next_questions' in response:
        display_suggested_responses(response.get('next_questions', []))

def send_feedback(item_name: str, rating: int):
    """Send feedback to backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "user_id": st.session_state.user_id,
                "item_id": item_name,
                "rating": rating
            }
        )
        if response.status_code == 200:
            st.success("Thanks for your feedback! We'll improve our recommendations.")
    except Exception as e:
        st.error(f"Error sending feedback: {str(e)}")

def handle_user_input(prompt):
    """Process user input and get AI response"""
    # Add user message to chat
    st.session_state.conversation.append({"role": "user", "content": prompt})
    
    # Get AI response
    try:
        print(f"Sending request to backend: {BACKEND_URL}/chat")
        
        # Make the request with increased timeout
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "user_id": st.session_state.user_id,
                "message": prompt
            }, 
            timeout=30  # Increased timeout for detailed responses
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Received successful response from backend")
            response_data = response.json()
            
            # Add assistant response to chat
            st.session_state.conversation.append({
                "role": "assistant",
                "content": response_data.get("message", "Sorry, I couldn't process your request"),
                "data": response_data
            })
        else:
            print(f"Backend returned error status code: {response.status_code}")
            error_message = f"Backend error: {response.status_code}"
            
            try:
                error_detail = response.json().get("detail", "No details provided")
                error_message += f" - {error_detail}"
            except:
                pass
                
            # Fallback response for error
            fallback_response = {
                "message": f"Error from backend: {error_message}. Please try again.",
                "next_questions": ["Try a different question", "Ask about another destination"]
            }
            
            st.session_state.conversation.append({
                "role": "assistant",
                "content": fallback_response["message"],
                "data": fallback_response
            })
            
            st.error(error_message)
        
    except requests.exceptions.Timeout:
        # Specific handling for timeout errors
        fallback_response = {
            "message": "The request to the backend timed out. This might be because the AI is generating a detailed response. Please try again or ask a simpler question.",
            "next_questions": ["Try a simpler question", "Ask about a specific aspect"]
        }
        
        st.session_state.conversation.append({
            "role": "assistant",
            "content": fallback_response["message"],
            "data": fallback_response
        })
        
        st.error("Request timed out. The backend might be busy.")
        
    except requests.exceptions.ConnectionError:
        # Specific handling for connection errors
        fallback_response = {
            "message": f"Could not connect to the backend at {BACKEND_URL}. Please make sure the backend server is running.",
            "next_questions": ["Try again later", "Ask a different question"]
        }
        
        st.session_state.conversation.append({
            "role": "assistant",
            "content": fallback_response["message"],
            "data": fallback_response
        })
        
        st.error(f"Connection error: Could not connect to {BACKEND_URL}")
        
    except Exception as e:
        # General fallback for other errors
        print(f"Error in handle_user_input: {str(e)}")
        
        fallback_response = {
            "message": f"Demo mode (backend not connected): You asked about '{prompt}'",
            "next_questions": ["Tell me about another destination", "What activities do you recommend?"]
        }
        
        st.session_state.conversation.append({
            "role": "assistant",
            "content": fallback_response["message"],
            "data": fallback_response
        })
        
        st.error(f"Error communicating with backend: {str(e)}")

def show_onboarding():
    """Display welcome screen for first-time users"""
    st.title("Welcome to AI Travel Assistant! 🌍✈️")
    st.write("I'll help you plan your perfect trip with personalized recommendations.")
    
    # Simple layout with direct text
    st.markdown("""
    ### How I can help you:
    * Suggest destinations based on your interests
    * Create personalized itineraries
    * Find accommodations that match your preferences
    * Recommend activities and attractions
    * Provide detailed information about any city or country
    * Show locations on interactive maps
    """)
    
    if st.button("Let's Start Planning! 🚀", key="start_planning"):
        st.session_state.onboarded = True
        st.rerun()  # Updated from experimental_rerun

def show_destination_explorer():
    """Display popular destinations section"""
    st.subheader("Explore Popular Destinations")
    
    cols = st.columns(3)
    destinations = [
        {"name": "Paris", "desc": "City of lights with iconic landmarks"},
        {"name": "Tokyo", "desc": "Blend of tradition and modernity"},
        {"name": "Bali", "desc": "Tropical paradise with rich culture"}
    ]
    
    for i, dest in enumerate(destinations):
        with cols[i % 3]:
            st.image(f"https://via.placeholder.com/300x200?text={dest['name']}", 
                    use_container_width=True)  # Updated from use_column_width
            st.write(dest['desc'])
            if st.button(f"Explore {dest['name']}", key=f"explore_{dest['name']}"):
                handle_user_input(f"Tell me about {dest['name']}")
                st.rerun()  # Updated from experimental_rerun

def chat_interface():
    """Main chat interface"""
    st.title("AI Travel Assistant 🌍")
    st.markdown("### Your Personalized Travel Planning Expert")

    # Sidebar with user preferences
    with st.sidebar:
        st.header("Your Preferences ⚙️")
        st.session_state.preferences['destination'] = st.text_input(
            "Destination",
            value=st.session_state.preferences['destination'] or ""
        )
        st.session_state.preferences['budget'] = st.selectbox(
            "Budget Range",
            options=["💰 Budget", "💵 Mid-Range", "💎 Luxury"],
            index=["💰 Budget", "💵 Mid-Range", "💎 Luxury"].index(
                st.session_state.preferences['budget'] or "💵 Mid-Range")
        )
        st.session_state.preferences['activities'] = st.multiselect(
            "Preferred Activities",
            options=["Adventure", "Cultural", "Relaxation", "Food", "Shopping"],
            default=st.session_state.preferences['activities']
        )
        
        st.session_state.preferences['duration'] = st.slider(
            "Trip Duration (days)",
            min_value=1,
            max_value=30,
            value=st.session_state.preferences.get('duration', 7)
        )
        
        if st.button("Apply Preferences"):
            # Send updated preferences to the backend
            preferences_prompt = f"I want to travel to {st.session_state.preferences['destination']} " + \
                                f"with a {st.session_state.preferences['budget']} budget for " + \
                                f"{st.session_state.preferences['duration']} days and I'm interested in " + \
                                f"{', '.join(st.session_state.preferences['activities'])}"
            handle_user_input(preferences_prompt)
            st.rerun()  # Updated from experimental_rerun
        
        if st.button("🔄 Reset Conversation"):
            try:
                requests.post(f"{BACKEND_URL}/reset/{st.session_state.user_id}")
            except Exception as e:
                st.error(f"Error resetting conversation: {str(e)}")
            st.session_state.conversation = []
            st.rerun()  # Updated from experimental_rerun

    # If no conversation yet, show destinations explorer
    if len(st.session_state.conversation) == 0:
        show_destination_explorer()

    # Chat history display
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "data" in msg and msg["role"] == "assistant":
                display_recommendations(msg["data"])

    # User input handling
    if prompt := st.chat_input("Ask about your trip...", key="chat_input"):
        handle_user_input(prompt)
        st.rerun()  # Updated from experimental_rerun

def main():
    """Main application function"""
    initialize_session()
    set_page_styling()
    
    if 'onboarded' not in st.session_state:
        show_onboarding()
    else:
        chat_interface()

if __name__ == "__main__":
    main()
