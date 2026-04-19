"""
Emotion Recognition Web App with Recommendations (FIXED)
Run with: streamlit run app.py
"""

import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import re

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Emotion Recognition Assistant",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid;
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateX(5px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        transition: 0.2s;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_emotion_model():
    """Load the trained emotion recognition model"""
    model_path = "../model/emotion_model_final"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please train the model first using the training script.")
        return None, None
    
    try:
        # Try to load with return_all_scores
        classifier = pipeline("text-classification", model=model_path, return_all_scores=True)
        return classifier, True
    except:
        try:
            # Fallback without return_all_scores
            classifier = pipeline("text-classification", model=model_path)
            return classifier, False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None

# ============================================
# EMOTION EXTRACTION FUNCTION (FIXED)
# ============================================

def extract_emotion_scores(result, has_scores):
    """
    Extract emotion scores from model output
    Handles multiple output formats
    """
    scores = {}
    
    # Debug info (only in development)
    # st.write(f"Debug - has_scores: {has_scores}")
    # st.write(f"Debug - result type: {type(result)}")
    # st.write(f"Debug - result: {result}")
    
    # Define label mapping for LABEL_X format
    label_to_emotion = {
        'LABEL_0': 'sadness',
        'LABEL_1': 'joy',
        'LABEL_2': 'love',
        'LABEL_3': 'anger',
        'LABEL_4': 'fear',
        'LABEL_5': 'surprise'
    }
    
    # Case 1: Pipeline with return_all_scores=True
    if has_scores and isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        
        # Format: [[{'label': 'LABEL_0', 'score': 0.01}, ...]]
        if isinstance(first_item, list):
            for item in first_item:
                if isinstance(item, dict) and 'label' in item and 'score' in item:
                    label = item['label']
                    # Try to convert LABEL_X to emotion name
                    if label in label_to_emotion:
                        emotion = label_to_emotion[label]
                    else:
                        emotion = label.lower() if isinstance(label, str) else str(label)
                    scores[emotion] = item['score']
        
        # Format: [{'label': 'LABEL_0', 'score': 0.01}, ...]
        elif isinstance(first_item, dict):
            for item in result[0] if isinstance(result[0], list) else result:
                if isinstance(item, dict) and 'label' in item and 'score' in item:
                    label = item['label']
                    if label in label_to_emotion:
                        emotion = label_to_emotion[label]
                    else:
                        emotion = label.lower() if isinstance(label, str) else str(label)
                    scores[emotion] = item['score']
    
    # Case 2: Simple pipeline (no return_all_scores)
    elif not has_scores and isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        if isinstance(first_item, dict) and 'label' in first_item and 'score' in first_item:
            label = first_item['label']
            score = first_item['score']
            
            # Try to convert LABEL_X to emotion name
            if label in label_to_emotion:
                emotion = label_to_emotion[label]
            else:
                emotion = label.lower() if isinstance(label, str) else str(label)
            scores[emotion] = score
    
    # Case 3: Direct dictionary output
    elif isinstance(result, dict) and 'label' in result and 'score' in result:
        label = result['label']
        if label in label_to_emotion:
            emotion = label_to_emotion[label]
        else:
            emotion = label.lower() if isinstance(label, str) else str(label)
        scores[emotion] = result['score']
    
    # If still no scores, try to extract any numeric values
    if not scores:
        # Try to find any numbers in the result
        result_str = str(result)
        numbers = re.findall(r"score['\"]?:\s*([0-9.]+)", result_str)
        labels = re.findall(r"label['\"]?:\s*['\"]?([A-Za-z_0-9]+)['\"]?", result_str)
        
        if labels and numbers:
            for i, label in enumerate(labels[:6]):
                if i < len(numbers):
                    if label in label_to_emotion:
                        emotion = label_to_emotion[label]
                    else:
                        emotion = label.lower()
                    scores[emotion] = float(numbers[i])
    
    return scores

# ============================================
# EMOTION CONFIGURATION
# ============================================

EMOTION_CONFIG = {
    'sadness': {
        'emoji': '😢', 'color': '#4A90D9', 'bg_color': '#E8F4FD',
        'title': 'Feeling Sad?',
        'recommendations': [
            {"icon": "🎬", "title": "Watch a Comfort Movie", "description": "Try 'The Pursuit of Happyness' or 'Up'"},
            {"icon": "📖", "title": "Read Something Uplifting", "description": "Poetry or inspirational quotes can help"},
            {"icon": "🎵", "title": "Listen to Soothing Music", "description": "Classical or ambient music can calm your mind"},
            {"icon": "💬", "title": "Talk to Someone", "description": "Reach out to a friend or family member"},
        ],
        'activities': ["Meditate for 5 minutes", "Call a close friend", "Watch a comedy show", "Practice deep breathing"]
    },
    'joy': {
        'emoji': '😊', 'color': '#FFD93D', 'bg_color': '#FFF9E6',
        'title': 'Wonderful! You\'re Feeling Joyful!',
        'recommendations': [
            {"icon": "🎉", "title": "Share Your Happiness", "description": "Spread joy by complimenting someone"},
            {"icon": "📸", "title": "Capture the Moment", "description": "Take a photo to remember this feeling"},
            {"icon": "🎨", "title": "Channel Your Energy", "description": "Create something - art, music, or writing"},
            {"icon": "🏃", "title": "Celebrate with Movement", "description": "Dance or exercise to amplify the joy"},
        ],
        'activities': ["Share your happy moment", "Plan a fun activity", "Treat yourself", "Call someone to share"]
    },
    'love': {
        'emoji': '❤️', 'color': '#E74C3C', 'bg_color': '#FDE8E7',
        'title': 'Spreading Love!',
        'recommendations': [
            {"icon": "💝", "title": "Express Your Feelings", "description": "Tell someone you appreciate them"},
            {"icon": "📝", "title": "Write a Love Letter", "description": "Express your feelings in writing"},
            {"icon": "🎁", "title": "Surprise a Loved One", "description": "Small gestures mean a lot"},
            {"icon": "📞", "title": "Call Family", "description": "Reach out to loved ones"},
        ],
        'activities': ["Send a thoughtful message", "Plan quality time", "Volunteer", "Create a gratitude list"]
    },
    'anger': {
        'emoji': '😠', 'color': '#E67E22', 'bg_color': '#FEF3E8',
        'title': 'Feeling Angry? Let\'s Calm Down',
        'recommendations': [
            {"icon": "🧘", "title": "Take Deep Breaths", "description": "Inhale for 4, hold for 4, exhale for 4"},
            {"icon": "🚶", "title": "Walk Away", "description": "Remove yourself from the situation"},
            {"icon": "🎵", "title": "Listen to Calming Music", "description": "Soothing sounds can reduce tension"},
            {"icon": "🏋️", "title": "Exercise", "description": "Physical activity releases pent-up energy"},
        ],
        'activities': ["Take a 10-minute break", "Practice relaxation", "Go for a run", "Count backward from 100"]
    },
    'fear': {
        'emoji': '😨', 'color': '#9B59B6', 'bg_color': '#F4E8F9',
        'title': 'Feeling Anxious? You\'re Safe Here',
        'recommendations': [
            {"icon": "🧘", "title": "Grounding Exercise", "description": "5-4-3-2-1 technique"},
            {"icon": "🫂", "title": "Seek Support", "description": "Talk to someone you trust"},
            {"icon": "📝", "title": "Challenge Your Thoughts", "description": "Write down what you're afraid of"},
            {"icon": "🎵", "title": "Listen to Calm Music", "description": "Ambient music can soothe anxiety"},
        ],
        'activities': ["Practice mindful breathing", "List things you can control", "Visualize a peaceful place", "Drink warm tea"]
    },
    'surprise': {
        'emoji': '😲', 'color': '#1ABC9C', 'bg_color': '#E8F8F5',
        'title': 'What a Surprise!',
        'recommendations': [
            {"icon": "✨", "title": "Embrace the Unexpected", "description": "Surprises can be opportunities"},
            {"icon": "📝", "title": "Process the Moment", "description": "Write about what surprised you"},
            {"icon": "🤝", "title": "Share with Someone", "description": "Tell someone about your surprise"},
            {"icon": "🎯", "title": "Stay Curious", "description": "Explore what else is new around you"},
        ],
        'activities': ["Explore something new", "Try a new restaurant", "Watch a different genre movie", "Learn a random fact"]
    }
}

DEFAULT_CONFIG = {
    'emoji': '🎭', 'color': '#95A5A6', 'bg_color': '#F8F9FA',
    'title': 'Emotion Detected',
    'recommendations': [
        {"icon": "💭", "title": "Reflect", "description": "Take a moment to acknowledge your feeling"},
        {"icon": "📝", "title": "Journal", "description": "Write down what you're experiencing"},
    ],
    'activities': ["Take a deep breath", "Be present in the moment"]
}

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Emotion Recognition Assistant</h1>
        <p>Powered by RoBERTa - 94.45% Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading emotion recognition model..."):
        classifier, has_scores = load_emotion_model()
    
    if classifier is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 About")
        st.markdown("""
        This app uses a fine-tuned **RoBERTa model** to detect emotions in text.
        
        **Detected Emotions:**
        - 😢 Sadness
        - 😊 Joy  
        - ❤️ Love
        - 😠 Anger
        - 😨 Fear
        - 😲 Surprise
        
        **Model Performance:**
        - Accuracy: **94.45%**
        - F1 Score: **94.47%**
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Write complete sentences
        - Express your genuine feelings
        - Works best with English text
        """)
    
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 📝 Enter your text")
        user_text = st.text_area(
            "",
            placeholder="e.g., I'm so excited about my new job!",
            height=120,
            label_visibility="collapsed"
        )
        
        analyze_button = st.button("🔍 Analyze Emotion", use_container_width=True)
    
    if analyze_button and user_text.strip():
        with st.spinner("Analyzing your emotion..."):
            try:
                # Get prediction
                result = classifier(user_text)
                
                # Extract scores using the fixed function
                scores = extract_emotion_scores(result, has_scores)
                
                # Debug: Show what was extracted (remove in production)
                # st.write(f"Debug - Extracted scores: {scores}")
                
                if not scores:
                    st.error("Could not analyze emotion. Please try again with different text.")
                    st.info("Debug: Try simpler text like 'I am happy' or 'I feel sad'")
                    return
                
                # Find dominant emotion
                dominant_emotion = max(scores.items(), key=lambda x: x[1])
                emotion_name = dominant_emotion[0]
                confidence = dominant_emotion[1]
                
                # Get config
                config = EMOTION_CONFIG.get(emotion_name, DEFAULT_CONFIG)
                
                st.markdown("---")
                
                # Result display
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem;'>
                    <span style='font-size: 80px;'>{config['emoji']}</span>
                    <h2 style='color: {config["color"]};'>{config['title']}</h2>
                    <p style='font-size: 24px; font-weight: bold;'>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar chart of all emotions
                st.markdown("#### 📊 Emotion Breakdown")
                
                df = pd.DataFrame([
                    {"Emotion": e.capitalize(), "Confidence": s, "Emoji": EMOTION_CONFIG.get(e, DEFAULT_CONFIG)['emoji']}
                    for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
                ])
                
                fig = px.bar(
                    df,
                    x="Emotion",
                    y="Confidence",
                    color="Emotion",
                    text=[f"{s:.2%}" for s in df["Confidence"]],
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.markdown("## 💡 Recommendations for You")
                
                rec_col1, rec_col2 = st.columns(2)
                for i, rec in enumerate(config['recommendations'][:4]):
                    col = rec_col1 if i % 2 == 0 else rec_col2
                    with col:
                        st.markdown(f"""
                        <div class="recommendation-card" style="border-left-color: {config['color']}">
                            <h4>{rec['icon']} {rec['title']}</h4>
                            <p>{rec['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Activities
                st.markdown("### 🎯 Suggested Activities")
                activity_cols = st.columns(3)
                for i, activity in enumerate(config['activities'][:3]):
                    with activity_cols[i]:
                        st.markdown(f"""
                        <div style='background-color: {config["bg_color"]}; padding: 0.8rem; border-radius: 10px; text-align: center; margin: 0.2rem;'>
                            <span style='font-size: 1.2rem;'>✓</span>
                            <p style='margin: 0; font-size: 0.9rem;'>{activity}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please try again with different text.")
    
    elif analyze_button and not user_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>This AI model is for educational purposes only.</p>
        <p>If you're experiencing emotional distress, please reach out to a mental health professional.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()