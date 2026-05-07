import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Twitter Bot Detector", layout="wide")

# Helper function
def safe_div(a, b): 
    return a / b if b != 0 else 0

# Production-grade CSS to hide developer menus, headers, and footers
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #15202b; color: white; }
    
    /* Metrics Styling */
    .stMetric { background-color: #192734; padding: 15px; border-radius: 10px; border: 1px solid #38444d; }
    
    /* HIDE THE ENTIRE TOP MENU AND TOOLBAR */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Hide header anchor link icons */
    [data-testid="stHeaderActionElements"] { display: none !important; }
    .header-anchor { display: none !important; }
    
    /* Remove padding around headers */
    h1, h2, h3 { margin-bottom: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_model():
    with open('bot_detector_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('bot_detector_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("Account Metrics")

# Original Counts
followers = st.sidebar.number_input("Followers Count", min_value=0, value=1500)
friends = st.sidebar.number_input("Friends Count (Following)", min_value=0, value=800)
statuses = st.sidebar.number_input("Total Tweets (Statuses)", min_value=0, value=5000)
listed = st.sidebar.number_input("Listed Count", min_value=0, value=10)
favourites = st.sidebar.number_input("Likes Given (Favourites)", min_value=0, value=3000)
age = st.sidebar.slider("Account Age (Years)", 0, 15, 5)

st.sidebar.divider()
# Profile Details
st.sidebar.subheader("Profile Details")
handle_len = st.sidebar.slider("Username Length", 1, 30, 12)
has_pic = st.sidebar.checkbox("Has Profile Picture?", value=True)
has_desc = st.sidebar.checkbox("Has Bio/Description?", value=True)
desc_len = st.sidebar.slider("Bio Length", 0, 160, 80) if has_desc else 0
has_loc = st.sidebar.checkbox("Has Location?", value=True)
has_ext = st.sidebar.checkbox("Has Extended Profile?", value=True)
geo_en = st.sidebar.checkbox("Geo Enabled (GPS)?", value=True)
trans_en = st.sidebar.checkbox("Translation Enabled?", value=True)

# Dataset Specific
st.sidebar.divider()
tweets_in_set = st.sidebar.number_input("Recent Tweets count", 0, 200, 100)

# --- Calculation Logic ---
def get_prediction():
    data = {
        'followers_count': followers,
        'friends_count': friends,
        'statuses_count': statuses,
        'listed_count': listed,
        'favourites_count': favourites,
        'tweet_count': tweets_in_set, 
        'account_age': age,
        'has_profile_pic': 1 if has_pic else 0,
        'has_extended_profile': 1 if has_ext else 0,
        'geo_enabled': 1 if geo_en else 0,
        'is_translation_enabled': 1 if trans_en else 0,
        'has_description': 1 if has_desc else 0,
        'description_length': desc_len,
        'has_location': 1 if has_loc else 0,
        'screen_name_length': handle_len, 
        'followers_friends_ratio': safe_div(followers, friends),
        'statuses_friends_ratio': safe_div(statuses, friends),
        'listed_followers_ratio': safe_div(listed, followers),
        'favourites_statuses_ratio': safe_div(favourites, statuses),
        'reputation_score': safe_div(followers, (followers + friends)),
        'engagement_ratio': safe_div(favourites, statuses),
        'listing_rate': safe_div(listed, followers),
        'excessive_following': 1 if friends > 5000 else 0,
        'few_followers': 1 if followers < 50 else 0,
        'low_tweet_count': 1 if statuses < 10 else 0,
        'follower_tweet_ratio': safe_div(followers, statuses),
        'friend_tweet_ratio': safe_div(friends, statuses),
    }
    df_input = pd.DataFrame([data])
    scaled = scaler.transform(df_input)
    prob = model.predict_proba(scaled)[0]
    return prob, data

prob, calculated_data = get_prediction()
bot_prob = prob[1] * 100
rep_score = calculated_data['reputation_score']

# --- Main UI ---
st.title("Twitter Bot Detection Dashboard")

col1, col2 = st.columns([1, 1])

with col1:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = bot_prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Bot Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1da1f2"},
            'steps': [
                {'range': [0, 40], 'color': '#2ecc71'},
                {'range': [40, 70], 'color': '#f1c40f'},
                {'range': [70, 100], 'color': '#e74c3c'}],
        }
    ))
    fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)', font = {'color': "white"})
    st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

with col2:
    st.subheader("Verdict")
    if bot_prob < 40:
        st.success("Likely HUMAN")
    elif bot_prob < 70:
        st.warning("SUSPICIOUS")
    else:
        st.error("Likely BOT")
        
    st.divider()
    st.subheader("Key Indicator: Reputation Score")
    st.metric("Score", f"{rep_score:.3f}")
    if rep_score < 0.2:
        st.error("Bad: Low score indicates mass-following behavior.")
    elif rep_score < 0.5:
        st.warning("Average: Balanced following but could be a bot.")
    else:
        st.info("Good: High score indicates an organic user.")
st.divider()
