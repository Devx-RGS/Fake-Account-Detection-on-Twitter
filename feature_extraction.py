import json
import pandas as pd
import os

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Twibot-20')
LABEL_CSV = os.path.join(BASE_PATH, 'label.csv')

def safe_div(a, b):
    return a / b if b != 0 else 0

def clean_int(val):
    try:
        return int(str(val).strip())
    except:
        return 0

def clean_bool(val):
    return str(val).strip().lower() == 'true'

def process_json(filepath, label_dict):
    print(f"  Processing: {os.path.basename(filepath)}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Found {len(data)} users")

    rows = []
    for user in data:
        uid     = 'u' + str(user['ID']).strip()
        profile = user.get('profile', {})
        tweets  = user.get('tweet') or []

        followers  = clean_int(profile.get('followers_count', 0))
        friends    = clean_int(profile.get('friends_count', 0))
        statuses   = clean_int(profile.get('statuses_count', 0))
        listed     = clean_int(profile.get('listed_count', 0))
        favourites = clean_int(profile.get('favourites_count', 0))

        description  = str(profile.get('description', '') or '').strip()
        location     = str(profile.get('location', '') or '').strip()
        screen_name  = str(profile.get('screen_name', '') or '').strip()
        img_url      = str(profile.get('profile_image_url', '') or '').lower()

        created_at = str(profile.get('created_at', '2021')).strip()
        try:
            # Creation date usually ends with the year: "Wed Dec 01 17:08:01 +0000 2010"
            creation_year = int(created_at.split()[-1])
        except:
            creation_year = 2021
        
        account_age = 2021 - creation_year
        has_description = 1 if description and description.lower() != 'none' else 0
        has_location    = 1 if location and location.lower() != 'none' else 0
        has_profile_pic = 0 if ('default_profile_normal' in img_url or not img_url) else 1

        row = {
            'id'                       : uid,
            'followers_count'          : followers,
            'friends_count'            : friends,
            'statuses_count'           : statuses,
            'listed_count'             : listed,
            'favourites_count'         : favourites,
            'tweet_count'              : len(tweets),
            'account_age'              : account_age,
            'has_profile_pic'          : has_profile_pic,
            'has_extended_profile'     : 1 if clean_bool(profile.get('has_extended_profile', False)) else 0,
            'geo_enabled'              : 1 if clean_bool(profile.get('geo_enabled', False)) else 0,
            'is_translation_enabled'   : 1 if clean_bool(profile.get('is_translation_enabled', False)) else 0,
            'has_description'          : has_description,
            'description_length'       : len(description),
            'has_location'             : has_location,
            'screen_name_length'       : len(screen_name),
            'followers_friends_ratio'  : safe_div(followers, friends),
            'statuses_friends_ratio'   : safe_div(statuses, friends),
            'listed_followers_ratio'   : safe_div(listed, followers),
            'favourites_statuses_ratio': safe_div(favourites, statuses),
            'reputation_score'         : safe_div(followers, (followers + friends)),
            'engagement_ratio'         : safe_div(favourites, statuses),
            'listing_rate'             : safe_div(listed, followers),
            'excessive_following'      : 1 if friends > 5000 else 0,
            'few_followers'            : 1 if followers < 50 else 0,
            'low_tweet_count'          : 1 if statuses < 10 else 0,
            'follower_tweet_ratio'     : safe_div(followers, statuses),
            'friend_tweet_ratio'       : safe_div(friends, statuses),
            'label'                    : label_dict.get(uid, -1),
        }
        rows.append(row)
    return rows

def extract_all():
    print("="*50)
    print("FULL FEATURE EXTRACTION (Train + Dev + Test)")
    print("="*50)

    # Load labels
    print("Loading label.csv...")
    labels_df = pd.read_csv(LABEL_CSV)
    labels_df['label_num'] = labels_df['label'].map({'bot': 1, 'human': 0})
    label_dict = dict(zip(labels_df['id'], labels_df['label_num']))
    print(f"Total labels: {len(label_dict)}")

    # Process all 3 splits
    all_rows = []
    for split in ['train.json', 'dev.json', 'test.json']:
        path = os.path.join(BASE_PATH, split)
        rows = process_json(path, label_dict)
        all_rows.extend(rows)

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    df = df[df['label'] != -1]   # drop users without a label

    # Save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'twibot_full_features.csv')
    df.to_csv(out, index=False)

    print()
    print(f"Total users processed : {len(df)}")
    print(f"Bots  : {(df['label']==1).sum()}  ({(df['label']==1).mean()*100:.1f}%)")
    print(f"Humans: {(df['label']==0).sum()}  ({(df['label']==0).mean()*100:.1f}%)")
    print(f"Saved to: {out}")
    print("="*50)

if __name__ == "__main__":
    extract_all()
