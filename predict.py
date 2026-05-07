import pandas as pd
import pickle
import os

# Load the model and scaler
MODEL_PATH = 'bot_detector_model.pkl'
SCALER_PATH = 'bot_detector_scaler.pkl'
DATA_PATH = 'twibot_full_features.csv'

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler not found. Please run train_model.py first.")
        return None, None
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_user(user_id):
    model, scaler = load_artifacts()
    if model is None: return

    # Load the feature data
    df = pd.read_csv(DATA_PATH)
    
    # Find the user
    user_data = df[df['id'] == user_id]
    
    if user_data.empty:
        print(f"User {user_id} not found in the dataset.")
        return

    # Extract features (remove id and label)
    # We also need to make sure we don't include 'verified' since we removed it from training
    actual_label = user_data['label'].values[0]
    features = user_data.drop(['id', 'label'], axis=1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    print("\n" + "="*40)
    print(f"PREDICTION FOR USER: {user_id}")
    print("="*40)
    print(f"AI Prediction : {'🤖 BOT' if prediction == 1 else '👤 HUMAN'}")
    print(f"Confidence    : {max(probability)*100:.2f}%")
    print(f"Actual Label  : {'BOT' if actual_label == 1 else 'HUMAN'}")
    print("="*40)
    
    if prediction == actual_label:
        print("✅ The AI got it RIGHT!")
    else:
        print("❌ The AI was WRONG this time.")

def predict_manual():
    model, scaler = load_artifacts()
    if model is None: return

    print("\n--- Manual Entry Mode ---")
    try:
        followers = int(input("Followers Count: "))
        friends = int(input("Friends Count: "))
        statuses = int(input("Total Tweets (Statuses): "))
        listed = int(input("Listed Count: "))
        favourites = int(input("Favourites (Likes) Count: "))
        age = int(input("Account Age (in years): "))
        
        # Calculate the engineered features
        def safe_div(a, b): return a / b if b != 0 else 0
        
        data = {
            'followers_count': followers,
            'friends_count': friends,
            'statuses_count': statuses,
            'listed_count': listed,
            'favourites_count': favourites,
            'tweet_count': 100,
            'account_age': age,
            'has_profile_pic': 1,
            'has_extended_profile': 0,
            'geo_enabled': 0,
            'is_translation_enabled': 0,
            'has_description': 1,
            'description_length': 50,
            'has_location': 1,
            'screen_name_length': 10,
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
        
        df_manual = pd.DataFrame([data])
        
        # Scale and Predict
        features_scaled = scaler.transform(df_manual)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        print("\n" + "="*40)
        print("MANUAL PREDICTION RESULT")
        print("="*40)
        print(f"AI Prediction : {'BOT' if prediction == 1 else 'HUMAN'}")
        print(f"Confidence    : {max(probability)*100:.2f}%")
        print("="*40)

    except Exception as e:
        print(f"Error during manual prediction: {e}")

if __name__ == "__main__":
    print("Twitter Bot Predictor")
    print("1. Predict using User ID (from dataset)")
    print("2. Predict by entering manual data (new user)")
    choice = input("Choice (1/2): ").strip()
    
    if choice == '1':
        uid = input("Enter User ID (e.g. u17461978): ").strip()
        predict_user(uid)
    elif choice == '2':
        predict_manual()
    else:
        print("Invalid choice.")
