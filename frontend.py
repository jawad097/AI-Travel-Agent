import streamlit as st
import requests
from datetime import datetime
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import redis
import os
from dotenv import load_dotenv
import time
import uuid
import random

load_dotenv()

BACKEND_URL = "http://127.0.0.1:8000"

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True
)

try:
    redis_client.ping()
    print("Connected to Redis successfully!")
except redis.ConnectionError:
    print("Warning: Could not connect to Redis. Session persistence will be disabled.")
    redis_client = None

def initialize_session():
    if 'user_id' not in st.session_state:
        if 'travel_assistant_user_id' in st.session_state:
            st.session_state.user_id = st.session_state.travel_assistant_user_id
        else:
            st.session_state.user_id = f"user_{uuid.uuid4()}"
            st.session_state.travel_assistant_user_id = st.session_state.user_id
    
    if 'conversation' not in st.session_state:
        if redis_client:
            try:
                redis_key = f"conversation:{st.session_state.user_id}"
                saved_conversation = redis_client.get(redis_key)
                if saved_conversation:
                    st.session_state.conversation = json.loads(saved_conversation)
                    print(f"Loaded conversation history from Redis for {st.session_state.user_id}")
                else:
                    st.session_state.conversation = []
            except Exception as e:
                print(f"Error loading conversation from Redis: {str(e)}")
                st.session_state.conversation = []
        else:
            st.session_state.conversation = []

    if 'preferences' not in st.session_state:
        if redis_client:
            try:
                redis_key = f"preferences:{st.session_state.user_id}"
                saved_preferences = redis_client.get(redis_key)
                if saved_preferences:
                    st.session_state.preferences = json.loads(saved_preferences)
                    print(f"Loaded preferences from Redis for {st.session_state.user_id}")
                else:
                    st.session_state.preferences = get_default_preferences()
            except Exception as e:
                print(f"Error loading preferences from Redis: {str(e)}")
                st.session_state.preferences = get_default_preferences()
        else:
            st.session_state.preferences = get_default_preferences()

def get_default_preferences():
    return {
        'destination': None,
        'budget_range': [800, 5000],
        'price_preference': '$$ - Moderate',
        'activities': ['Cultural', 'Adventure'],
        'accommodation_type': 'Hotel',
        'trip_duration': 7,
        'travel_pace': 'Moderate',
        'cuisine_preferences': ['Local', 'International'],
        'transportation': 'Public Transport',
        'accessibility_needs': False,
        'travel_season': 'Summer',
        'language_preferences': ['English']
    }

def save_to_redis(key_prefix, data):
    if redis_client:
        try:
            redis_key = f"{key_prefix}:{st.session_state.user_id}"
            redis_client.set(redis_key, json.dumps(data))
            # Set expiration to 30 days
            redis_client.expire(redis_key, 60 * 60 * 24 * 30)
            return True
        except Exception as e:
            print(f"Error saving to Redis: {str(e)}")
            return False
    return False

def set_futuristic_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto:wght@300;400;500&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #00b4d8, #48cae4, #90e0ef);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 1.5rem !important;
    }
    
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stMultiselect>div>div>div {
        background: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 180, 216, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00b4d8, #0077b6) !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 10px 25px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.2);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 20px rgba(0, 180, 216, 0.5);
        background: linear-gradient(45deg, #0077b6, #00b4d8) !important;
    }
    
    /* Animated card */
    .card {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        padding: 20px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 180, 216, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 180, 216, 0.4);
    }
    
    .card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to bottom right,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.05) 50%,
            rgba(255, 255, 255, 0) 100%
        );
        transform: rotate(30deg);
        transition: transform 0.5s ease;
    }
    
    .card:hover::after {
        transform: rotate(30deg) translate(10%, 10%);
    }
    
    .price-tag {
        background: linear-gradient(45deg, #00b4d8, #0077b6);
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px 0;
        font-weight: bold;
        box-shadow: 0 2px 10px rgba(0, 180, 216, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: rgba(15, 12, 41, 0.7) !important;
        border-right: 1px solid rgba(0, 180, 216, 0.2);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 180, 216, 0.2);
        margin-bottom: 15px;
    }
    
    /* Loading animation */
    .stProgress .st-bo {
        background-color: #00b4d8 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00b4d8, #0077b6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #0077b6, #00b4d8);
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: rgba(0, 0, 0, 0.8);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: bold;
        border-radius: 15px;
        margin-right: 5px;
        background: rgba(0, 180, 216, 0.2);
        border: 1px solid rgba(0, 180, 216, 0.4);
    }
    
    /* Animated progress bar */
    @keyframes progress {
        0% { width: 0%; }
        100% { width: 100%; }
    }
    
    .progress-bar {
        height: 4px;
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        border-radius: 2px;
        margin: 10px 0;
        animation: progress 2s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

def display_budget_analysis(budget_range):
    with st.expander("💰 Budget Analysis & Optimization", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Smart Budget Allocation")
            price_level = st.session_state.preferences['price_preference']
            accommodation_type = st.session_state.preferences['accommodation_type']
            
            if price_level == '$ - Budget':
                accommodation = 0.20
                food = 0.20
                activities = 0.15
            elif price_level == '$$$ - Luxury':
                accommodation = 0.40
                food = 0.20
                activities = 0.15
            else:
                accommodation = 0.30
                food = 0.20
                activities = 0.20
                
            if accommodation_type == 'Hostel':
                accommodation -= 0.10
                activities += 0.05
                food += 0.05
            elif accommodation_type == 'Luxury Resort':
                accommodation += 0.10
                activities -= 0.05
                food -= 0.05
                
            transportation = 1.0 - (accommodation + food + activities + 0.10)
            
            categories = ['Accommodation', 'Food', 'Activities', 'Transportation', 'Miscellaneous']
            values = [accommodation, food, activities, transportation, 0.10]
            
            fig = go.Figure(data=[go.Pie(
                labels=categories,
                values=values,
                hole=.4,
                hoverinfo='label+percent',
                textinfo='label+percent',
                textfont_size=12,
                marker=dict(
                    colors=['#00b4d8', '#0077b6', '#023e8a', '#0096c7', '#48cae4'],
                    line=dict(color='rgba(0,0,0,0)', width=1)
                ),
                pull=[0.05, 0, 0, 0, 0]
            )])
            
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            total_budget = (budget_range[0] + budget_range[1]) / 2
            st.markdown("#### Estimated Costs")
            for cat, val in zip(categories, values):
                amount = total_budget * val
                st.markdown(f"**{cat}**: ${amount:.2f}")
        
        with col2:
            st.markdown("### Price Optimization")
            
            price_points = ['Budget', 'Value', 'Premium', 'Luxury']
            satisfaction = [65, 80, 90, 95]
            costs = [budget_range[0], (budget_range[0] + budget_range[1])/3, 
                    2*(budget_range[0] + budget_range[1])/3, budget_range[1]]
            
            value_score = [s/(c/100) for s, c in zip(satisfaction, costs)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=price_points,
                y=costs,
                name='Cost ($)',
                marker_color='#0077b6',
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=price_points,
                y=satisfaction,
                name='Satisfaction',
                marker=dict(size=10, color='#00b4d8', line=dict(width=2, color='white')),
                line=dict(width=3, color='#00b4d8'),
                yaxis='y2'
            ))
            
            max_value_index = value_score.index(max(value_score))
            
            fig.add_annotation(
                x=price_points[max_value_index],
                y=costs[max_value_index] + 500,
                text="BEST VALUE",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#00b4d8",
                font=dict(size=12, color="#ffffff"),
                bgcolor="#00b4d8",
                bordercolor="#00b4d8",
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            )
            
            fig.update_layout(
                yaxis=dict(
                    title=dict(text='Cost ($)', font=dict(color='#0077b6')),
                    tickfont=dict(color='#0077b6')
                ),
                yaxis2=dict(
                    title=dict(text='Satisfaction Score', font=dict(color='#00b4d8')),
                    tickfont=dict(color='#00b4d8'),
                    anchor="x",
                    overlaying="y",
                    side="right",
                    range=[0, 100]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 **Budget Tip**: The '{price_points[max_value_index]}' option offers the best value for your preferences.")

def display_recommendation_card(item, item_type):
    match_score = calculate_match_score(item, item_type)
    
    card_id = f"{item_type}_{item.get('name', 'item').replace(' ', '_').lower()}_{int(time.time())}"
    
    # Format price display
    price_display = format_price_display(item.get('price', 'N/A'), item_type)
    
    tags = generate_item_tags(item, item_type)
    
    personalization_indicators = []
    if match_score > 85:
        personalization_indicators.append("Perfect Match")
    if 'budget_friendly' in tags:
        personalization_indicators.append("Budget-Friendly")
    if any(activity in tags for activity in st.session_state.preferences['activities']):
        personalization_indicators.append("Matches Activities")
    
    # Format the indicators as badges
    badges_html = ""
    for indicator in personalization_indicators:
        badges_html += f'<span class="badge">{indicator}</span>'
    
    tags_str = ", ".join(tags)
    
    # Create the card content
    card_html = f"""
    <div class="card" id="{card_id}" style="height: auto; min-height: 250px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0;">{item.get('name', 'Item')} ⭐{item.get('rating', 4.0)}</h3>
            <div class="price-tag">{price_display}</div>
        </div>
        
        <div style="margin-bottom: 10px;">
            {badges_html}
        </div>
        
        <div style="margin: 15px 0;">
            <div class="progress-bar" style="width: {match_score}%;"></div>
            <div style="display: flex; justify-content: space-between; font-size: 12px;">
                <span>Match Score</span>
                <span>{match_score}%</span>
            </div>
        </div>
        
        <p>{item.get('description', 'No description available')}</p>
        
        <div style="margin-top: 10px; font-size: 12px; color: rgba(255,255,255,0.7);">
            <strong>Tags:</strong> {tags_str}
        </div>
        
        {create_feedback_buttons(item.get('name', 'item'), card_id)}
    </div>
    """
    
    with st.container():
        # Use markdown with unsafe_allow_html=True for proper rendering
        st.markdown(card_html, unsafe_allow_html=True)

def calculate_match_score(item, item_type):
    score = 70
    
    price_level = st.session_state.preferences['price_preference']
    item_price = item.get('price_level', '$$')
    if price_level[0] == item_price:
        score += 15
    
    if item_type == 'activity':
        user_activities = st.session_state.preferences['activities']
        item_category = item.get('category', '').lower()
        
        for activity in user_activities:
            if activity.lower() in item_category:
                score += 10
                break
    
    if item_type == 'hotel':
        user_accom_type = st.session_state.preferences['accommodation_type'].lower()
        item_type = item.get('type', '').lower()
        
        if user_accom_type in item_type:
            score += 15

    score += random.randint(-5, 5)
 
    return max(min(score, 98), 60)

def format_price_display(price, item_type):
    """Format price display based on item type"""
    if isinstance(price, (int, float)):
        if item_type == 'hotel':
            return f"${price}/night"
        elif item_type == 'activity':
            return f"${price}"
    return price

def generate_item_tags(item, item_type):
    tags = []
    
    if item.get('price', 0) < st.session_state.preferences['budget_range'][0]:
        tags.append("budget_friendly")
    
    if item_type == 'activity':
        category = item.get('category', '').lower()
        if 'adventure' in category:
            tags.append("Adventure")
        if 'cultural' in category:
            tags.append("Cultural")
        if 'family' in category:
            tags.append("Family-Friendly")
        
        # Duration-based tags
        duration = item.get('duration', '').lower()
        if 'hour' in duration:
            tags.append("Quick")
        if 'day' in duration:
            tags.append("Full-Day")
    
    elif item_type == 'hotel':
        # Hotel-specific tags
        amenities = item.get('amenities', [])
        if 'pool' in amenities:
            tags.append("Pool")
        if 'spa' in amenities:
            tags.append("Spa")
        if 'wifi' in amenities:
            tags.append("Free WiFi")
        
        # Type-based tags
        hotel_type = item.get('type', '').lower()
        if 'boutique' in hotel_type:
            tags.append("Boutique")
        if 'resort' in hotel_type:
            tags.append("Resort")
        if 'hostel' in hotel_type:
            tags.append("Hostel")
    
    return tags

def create_feedback_buttons(item_name, card_id):
    return f"""
    <div style="display: flex; gap: 10px; margin-top: 15px;">
        <button onclick="document.getElementById('{card_id}').style.borderColor='rgba(46, 204, 113, 0.5)'"
                style="background: linear-gradient(45deg, #27ae60, #2ecc71); color: white; border: none; 
                padding: 8px 20px; border-radius: 20px; cursor: pointer; transition: all 0.3s ease;
                box-shadow: 0 2px 10px rgba(46, 204, 113, 0.3);">
            👍 Like
        </button>
        <button onclick="document.getElementById('{card_id}').style.borderColor='rgba(231, 76, 60, 0.5)'"
                style="background: linear-gradient(45deg, #c0392b, #e74c3c); color: white; border: none; 
                padding: 8px 20px; border-radius: 20px; cursor: pointer; transition: all 0.3s ease;
                box-shadow: 0 2px 10px rgba(231, 76, 60, 0.3);">
            👎 Dislike
        </button>
    </div>
    """

def preferences_sidebar():
    with st.sidebar:
        st.header("⚙️ Travel Preferences")
        
        # Budget Controls
        st.session_state.preferences['budget_range'] = st.slider(
            "Total Trip Budget (USD)",
            min_value=500, max_value=10000, 
            value=st.session_state.preferences['budget_range'],
            step=500
        )
        
        st.session_state.preferences['price_preference'] = st.selectbox(
            "Price Level Preference",
            options=['$ - Budget', '$$ - Moderate', '$$$ - Luxury'],
            index=['$ - Budget', '$$ - Moderate', '$$$ - Luxury'].index(
                st.session_state.preferences['price_preference'])
        )
        
        # Activity Filters
        st.session_state.preferences['activities'] = st.multiselect(
            "Preferred Activities",
            options=["Adventure", "Cultural", "Luxury", "Family", "Nature", "Nightlife"],
            default=st.session_state.preferences['activities'],
            format_func=lambda x: f"🏔️ {x}" if x == "Adventure" else 
                                 f"🎭 {x}" if x == "Cultural" else 
                                 f"🌟 {x}" if x == "Luxury" else x
        )
        
        # Travel Style
        st.session_state.preferences['travel_pace'] = st.select_slider(
            "Travel Pace",
            options=['Relaxed', 'Moderate', 'Fast-paced'],
            value=st.session_state.preferences['travel_pace']
        )
        
        if st.button("🔄 Apply Preferences", use_container_width=True):
            update_preferences_backend()
        
        if 'onboarded' not in st.session_state:
            st.markdown("---")
            if st.button("⏩ Skip to Chat Interface", use_container_width=True):
                st.session_state.onboarded = True
                st.rerun()

def update_preferences_backend():
    try:
        response = requests.post(
            f"{BACKEND_URL}/preferences/{st.session_state.user_id}",
            json=st.session_state.preferences
        )
        if response.status_code == 200:
            st.success("Preferences updated successfully!")
        else:
            st.error("Failed to update preferences")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")

def chat_interface():
    st.title("🌍 AI Travel Assistant")
    st.markdown("### Your Smart Travel Planning Companion")
    
    st.info(f"User ID: {st.session_state.user_id}")
    st.info(f"Backend URL: {BACKEND_URL}")
    
    if len(st.session_state.conversation) > 0:
        for msg in st.session_state.conversation:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "data" in msg and msg["role"] == "assistant":
                    display_enhanced_recommendations(msg["data"])
    else:
        with st.chat_message("assistant"):
            st.markdown("👋 Hello! I'm your AI Travel Assistant. Ask me about any destination, and I'll help you plan your perfect trip!")
            st.markdown("Try asking something like: *'I want to go to Paris'* or *'Tell me about Tokyo'*")
    
    display_budget_analysis(st.session_state.preferences['budget_range'])
    
    prompt = st.chat_input("Ask about your trip...", key="chat_input")
    if prompt:
        handle_user_input(prompt)
        st.rerun()

def display_enhanced_recommendations(response):
    if 'activities' in response:
        st.subheader("🎯 Recommended Activities")
        for activity in response.get('activities', []):
            if activity.get('price', 0) <= st.session_state.preferences['budget_range'][1]:
                display_recommendation_card(activity, 'activity')
    
    if 'accommodations' in response:
        st.subheader("🏨 Recommended Stays")
        for hotel in response.get('accommodations', []):
            price_level = hotel.get('price_level', '$$')
            if price_level == st.session_state.preferences['price_preference'][0]:
                display_recommendation_card(hotel, 'hotel')

def handle_user_input(prompt):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    
    save_to_redis("conversation", st.session_state.conversation)
    
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.write("Thinking...")
            
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "user_id": st.session_state.user_id,
                    "message": prompt
                }
            )
            
            if response.status_code == 200:
                # Process the response
                ai_response = response.json()
                
                # Update the message placeholder with the AI response
                message_placeholder.markdown(ai_response.get("message", "Sorry, I couldn't process that."))
                
                # Add to conversation history
                st.session_state.conversation.append({
                    "role": "assistant", 
                    "content": ai_response.get("message", ""),
                    "data": ai_response
                })
                
                save_to_redis("conversation", st.session_state.conversation)
                
                if "activities" in ai_response or "accommodations" in ai_response:
                    display_enhanced_recommendations(ai_response)
                
                if "next_questions" in ai_response and ai_response["next_questions"]:
                    st.markdown("**You might also want to ask:**")
                    for question in ai_response["next_questions"]:
                        st.markdown(f"- *{question}*")
            else:
                message_placeholder.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error communicating with backend: {str(e)}")
        print(f"Error details: {str(e)}")

def main():
    # Set page config
    st.set_page_config(
        page_title="AI Travel Assistant",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    set_futuristic_style()
    
    # Initialize session state
    initialize_session()
    
    # Show sidebar with preferences
    preferences_sidebar()
    
    # Show main chat interface
    chat_interface()

# Run the app
if __name__ == "__main__":
    main()
