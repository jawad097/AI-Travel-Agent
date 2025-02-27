# AI-Travel-Agent
AI Travel Agent is a smart travel planning assistant that combines OpenAI's GPT technology with travel data to offer personalized recommendations. With a chat interface, it helps users discover destinations, optimize budgets, and find activities matching their preferences.
# 🌍 AI Travel Assistant

An intelligent travel planning assistant powered by AI that helps users discover destinations, get personalized recommendations, and plan trips within their budget.

## 🚀 Features

- **🤖 AI-Powered Recommendations**: Uses OpenAI's GPT-3.5-turbo to generate personalized travel advice
- **💬 Conversational Interface**: Natural language chat interface for an intuitive planning experience
- **💰 Smart Budget Analysis**: Visual budget allocation and optimization based on user preferences
- **🏨 Personalized Recommendations**: Activities and accommodations tailored to user preferences
- **📍 Location Awareness**: Distance calculation and cost estimations between origin and destination
- **📊 Visual Analytics**: Interactive charts for budget planning and price optimization
- **🔄 Persistent Sessions**: Redis-backed persistence for maintaining user context across sessions
- **📱 Modern UI**: Futuristic, responsive interface with animations and visual enhancements

## 🔧 Technical Architecture

### Backend (FastAPI)
- **API Integration**: OpenAI for natural language processing, Google Maps API for geocoding
- **Data Storage**: Redis for session persistence, CSV for feedback collection
- **Travel Intelligence**: Distance calculation, cost estimation, personalized recommendations

### Frontend (Streamlit)
- **Chat Interface**: Modern conversational UI with suggestion prompts
- **Visualization**: Interactive charts for budget analysis
- **Recommendation Cards**: Personalized cards with match scoring
- **Preference Management**: Comprehensive user preference settings

## 🏗️ Setup Instructions

### Prerequisites
- Python 3.8+
- Redis server
- OpenAI API key
- Google Maps API key (optional)

### Environment Variables
Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-travel-assistant.git
cd ai-travel-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
python backend.py
```

4. In a new terminal, start the frontend:
```bash
streamlit run frontend.py
```

5. Access the application at `http://localhost:8501`

## 📋 Key Components

### TravelAgentModel
The core backend class that handles:
- User context management
- OpenAI API interactions
- Personalized recommendation generation
- Location and travel data processing

### Frontend UI
- **Futuristic Design**: Custom CSS with gradients, animations, and modern card components
- **Budget Visualization**: Interactive charts for budget allocation and optimization
- **Recommendation Cards**: Dynamically generated cards with personalization indicators
- **Preference Controls**: Comprehensive sidebar for user preference management

## 🧩 How It Works

1. **User Input**: User enters a travel query through the chat interface
2. **AI Processing**: Backend processes the query using OpenAI's GPT-3.5-turbo model
3. **Personalization**: Recommendations are tailored based on user preferences
4. **Visualization**: Results are presented with interactive visualizations and recommendation cards
5. **Feedback Loop**: User can provide feedback to improve future recommendations

## 🔄 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Process user messages and return AI responses |
| `/location/{user_id}` | POST | Update user location data |
| `/preferences/{user_id}` | POST | Update user travel preferences |
| `/preferences/{user_id}` | GET | Retrieve user travel preferences |
| `/feedback` | POST | Submit user feedback on recommendations |
| `/reset/{user_id}` | POST | Reset user conversation and context |
| `/health` | GET | Health check endpoint |

## 💡 Sample Usage

Ask the AI Travel Assistant questions like:
- "I want to visit Paris for a week. What should I see?"
- "What's the best time to visit Tokyo?"
- "Tell me about budget-friendly activities in New York"
- "How much would a trip to London cost?"
- "What are the top restaurants in Rome?"

## 🛠️ Future Enhancements

- **Real-time Pricing**: Integration with flight and hotel booking APIs
- **Itinerary Builder**: Visual timeline for planning day-by-day activities
- **Weather Integration**: Real-time weather data for better planning
- **Language Support**: Multi-language capabilities for international travelers
- **Mobile App**: Native mobile application for on-the-go planning

## 📜 License

[MIT License](LICENSE)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
