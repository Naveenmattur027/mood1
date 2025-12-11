# app.py - Mongo removed; file-backed JSON storage
import os
import json
import threading
from uuid import uuid4
from flask import Flask, render_template, request, jsonify, make_response
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import datetime
from datetime import timedelta
import re
import jwt
import bcrypt
from functools import wraps

nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_secret_key_here")
app.config['JWT_SECRET_KEY'] = os.environ.get("JWT_SECRET_KEY", "jwt_secret_key_here")

# Sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Storage configuration
DATA_FILE = os.environ.get("DATA_FILE", "data.json")
_data_lock = threading.Lock()
_data = None  # lazy-loaded dict with keys: 'users' and 'entries'

# -------------------------
# Storage helpers (file-backed)
# -------------------------
def _init_data_structure():
    return {"users": [], "entries": []}

def load_data():
    """Load data.json into memory. Creates file if missing."""
    global _data
    with _data_lock:
        if _data is not None:
            return _data
        if not os.path.exists(DATA_FILE):
            _data = _init_data_structure()
            save_data()
            return _data
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                _data = json.load(f)
                # Ensure keys exist
                if "users" not in _data:
                    _data["users"] = []
                if "entries" not in _data:
                    _data["entries"] = []
        except Exception:
            # If file corrupted, reinitialize to avoid crash
            _data = _init_data_structure()
            save_data()
        return _data

def save_data():
    """Persist _data to DATA_FILE."""
    global _data
    with _data_lock:
        if _data is None:
            return
        tmp_path = DATA_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, DATA_FILE)

# Convenience wrappers to work like collection operations
def find_users(filter_fn=None):
    data = load_data()
    users = data["users"]
    return [u for u in users if (filter_fn(u) if filter_fn else True)]

def insert_user(user_obj):
    data = load_data()
    with _data_lock:
        data["users"].append(user_obj)
        save_data()
    return user_obj

def find_entries(filter_fn=None):
    data = load_data()
    entries = data["entries"]
    return [e for e in entries if (filter_fn(e) if filter_fn else True)]

def insert_entry(entry_obj):
    data = load_data()
    with _data_lock:
        data["entries"].append(entry_obj)
        save_data()
    return entry_obj

def delete_all_entries():
    data = load_data()
    with _data_lock:
        data["entries"] = []
        save_data()

# -------------------------
# JWT helpers (unchanged)
# -------------------------
def generate_token(user_id):
    payload = {
        'user_id': str(user_id),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        'iat': datetime.datetime.utcnow()
    }
    return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        if token.startswith('Bearer '):
            token = token[7:]
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(user_id, *args, **kwargs)
    return decorated

# -------------------------
# Routes (adapted to file storage)
# -------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400

    # find user by username or email
    users = find_users(lambda u: u.get('username') == username or u.get('email') == username)
    user = users[0] if users else None

    if not user:
        return jsonify({'message': 'Invalid username or password!'}), 401

    # NOTE: currently stored as plain text in this demo; better to hash
    if 'password' in user and user['password'] != password:
        return jsonify({'message': 'Invalid username or password!'}), 401

    token = generate_token(user['id'])

    return jsonify({
        'message': 'Login successful!',
        'token': token,
        'user': {
            'id': str(user['id']),
            'username': user.get('username'),
            'email': user.get('email', '')
        }
    })

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'message': 'Username, email, and password are required!'}), 400

    # check existing user
    existing = find_users(lambda u: u.get('username') == username or u.get('email') == email)
    if existing:
        return jsonify({'message': 'Username or email already exists!'}), 409

    new_user = {
        'id': str(uuid4()),
        'username': username,
        'email': email,
        'password': password,  # in prod: store bcrypt.hashpw(...)
        'created_at': datetime.datetime.utcnow().isoformat()
    }
    insert_user(new_user)

    return jsonify({
        'message': 'User registered successfully!',
        'user': {
            'id': new_user['id'],
            'username': username,
            'email': email
        }
    }), 201

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/get_entries')
def get_entries():
    entries = find_entries()
    # sort by created_at descending (if present) else insertion order reversed
    try:
        entries_sorted = sorted(entries, key=lambda e: e.get('created_at', ''), reverse=True)
    except Exception:
        entries_sorted = list(reversed(entries))
    # remove internal fields if any and return
    out = [{"date": e.get("date"), "entry": e.get("entry"), "id": e.get("id")} for e in entries_sorted]
    return jsonify({"entries": out})

@app.route('/add_entry', methods=['POST'])
def add_entry():
    entry_text = request.form.get('entry') or request.get_json(silent=True, force=False) and request.get_json().get('entry')
    if not entry_text:
        return jsonify({"message": "No entry provided"}), 400

    date = datetime.date.today()
    date_str = date.strftime("%A, %Y-%m-%d")
    new_entry = {
        "id": str(uuid4()),
        "date": date_str,
        "entry": entry_text,
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    insert_entry(new_entry)

    entries = find_entries()
    # return sorted entries (descending)
    try:
        entries_sorted = sorted(entries, key=lambda e: e.get('created_at', ''), reverse=True)
    except Exception:
        entries_sorted = list(reversed(entries))
    out = [{"date": e.get("date"), "entry": e.get("entry"), "id": e.get("id")} for e in entries_sorted]
    return jsonify({"message": "Entry added successfully!", "entries": out})

@app.route('/get_sentiment', methods=['GET'])
def get_sentiment():
    entries = find_entries()
    if not entries:
        return jsonify({"message": "No entries available for sentiment analysis."})
    all_entries = " ".join([entry.get('entry', '') for entry in entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({"sentiment": sentiment_score})

@app.route('/get_comprehensive_analysis', methods=['POST'])
def get_comprehensive_analysis():
    entry_text = request.form.get('entry') or (request.get_json(silent=True) or {}).get('entry')
    if not entry_text:
        return jsonify({"message": "No entry provided for analysis."})
    try:
        sentiment_score = sia.polarity_scores(entry_text)
        overall_sentiment = get_sentiment_category(sentiment_score['compound'])
        intensity = get_sentiment_intensity(sentiment_score['compound'])
        try:
            emotions = get_custom_emotion(entry_text)
            emotion_keys = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
            for key in emotion_keys:
                if key not in emotions:
                    emotions[key] = 0
        except Exception:
            emotions = {"Happy": 0, "Angry": 0, "Surprise": 0, "Sad": 0, "Fear": 0}
        highlights = extract_highlights(entry_text)
        mental_patterns = identify_mental_patterns(entry_text)
        strengths = identify_strengths(entry_text)
        suggestions = generate_suggestions(sentiment_score['compound'], emotions, mental_patterns)
        summary = generate_summary(sentiment_score['compound'], emotions)
        trend_insight = "Not enough data for trend analysis yet."
        return jsonify({
            "overall_sentiment": {"category": overall_sentiment, "intensity": intensity},
            "emotions": emotions,
            "highlights": highlights,
            "mental_patterns": mental_patterns,
            "strengths": strengths,
            "suggestions": suggestions,
            "summary": summary,
            "trend_insight": trend_insight
        })
    except Exception as e:
        return jsonify({"message": f"Error in comprehensive analysis: {str(e)}"})

@app.route('/get_current_entry_sentiment', methods=['POST'])
def get_current_entry_sentiment():
    entry_text = request.form.get('entry') or (request.get_json(silent=True) or {}).get('entry')
    if not entry_text:
        return jsonify({"message": "No entry provided for sentiment analysis."})
    sentiment_score = sia.polarity_scores(entry_text)
    return jsonify({"sentiment": sentiment_score})

@app.route('/get_daily_sentiment', methods=['GET'])
def get_daily_sentiment():
    today = datetime.date.today()
    today_str = today.strftime("%A, %Y-%m-%d")
    entries = find_entries(lambda e: e.get('date') == today_str)
    if not entries:
        return jsonify({"message": "No entries available for today."})
    all_entries = " ".join([entry.get('entry', '') for entry in entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({"sentiment": sentiment_score, "period": "daily"})

@app.route('/get_weekly_sentiment', methods=['GET'])
def get_weekly_sentiment():
    today = datetime.date.today()
    week_ago = today - timedelta(days=7)
    entries = find_entries()
    filtered_entries = []
    for entry in entries:
        try:
            entry_date_str = entry.get('date', '').split(", ")[1]
            entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            if week_ago <= entry_date <= today:
                filtered_entries.append(entry)
        except Exception:
            pass
    if not filtered_entries:
        return jsonify({"message": "No entries available for the past week."})
    daily_sentiments = {}
    for entry in filtered_entries:
        try:
            entry_date_str = entry.get('date', '').split(", ")[1]
            sentiment_score = sia.polarity_scores(entry.get('entry', ''))
            daily_sentiments.setdefault(entry_date_str, []).append(sentiment_score['compound'])
        except Exception:
            pass
    daily_averages = {date: (sum(scores) / len(scores) if scores else 0) for date, scores in daily_sentiments.items()}
    all_entries = " ".join([entry.get('entry', '') for entry in filtered_entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({"sentiment": sentiment_score, "period": "weekly", "daily_data": daily_averages})

@app.route('/get_monthly_sentiment', methods=['GET'])
def get_monthly_sentiment():
    today = datetime.date.today()
    month_ago = today - timedelta(days=30)
    entries = find_entries()
    filtered_entries = []
    for entry in entries:
        try:
            entry_date_str = entry.get('date', '').split(", ")[1]
            entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            if month_ago <= entry_date <= today:
                filtered_entries.append(entry)
        except Exception:
            pass
    if not filtered_entries:
        return jsonify({"message": "No entries available for the past month."})
    daily_sentiments = {}
    for entry in filtered_entries:
        try:
            entry_date_str = entry.get('date', '').split(", ")[1]
            sentiment_score = sia.polarity_scores(entry.get('entry', ''))
            daily_sentiments.setdefault(entry_date_str, []).append(sentiment_score['compound'])
        except Exception:
            pass
    daily_averages = {date: (sum(scores) / len(scores) if scores else 0) for date, scores in daily_sentiments.items()}
    all_entries = " ".join([entry.get('entry', '') for entry in filtered_entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({"sentiment": sentiment_score, "period": "monthly", "daily_data": daily_averages})

@app.route('/get_sentiment_counts', methods=['GET'])
def get_sentiment_counts():
    entries = find_entries()
    if not entries:
        return jsonify({"counts": {"happy": 0, "neutral": 0, "sad": 0}})
    happy_count = 0
    neutral_count = 0
    sad_count = 0
    for entry in entries:
        try:
            sentiment_score = sia.polarity_scores(entry.get('entry', ''))
            compound = sentiment_score['compound']
            if compound >= 0.5:
                happy_count += 1
            elif compound > 0:
                happy_count += 1
            elif compound == 0:
                neutral_count += 1
            elif compound > -0.5:
                sad_count += 1
            else:
                sad_count += 1
        except Exception:
            pass
    return jsonify({"counts": {"happy": happy_count, "neutral": neutral_count, "sad": sad_count}})

@app.route('/clear_entries', methods=['POST'])
def clear_entries():
    delete_all_entries()
    return jsonify({"message": "Diary entries cleared!"})

# -------------------------
# Helper functions (same logic)
# -------------------------
def get_sentiment_category(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_sentiment_intensity(compound):
    abs_compound = abs(compound)
    if abs_compound >= 0.5:
        return "Strong"
    elif abs_compound >= 0.1:
        return "Moderate"
    else:
        return "Mild"

def extract_highlights(text):
    sentences = re.split(r'[.!?]+', text)
    highlights = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:
            highlights.append(sentence)
    return highlights[:7] if len(highlights) > 7 else highlights[3:] if len(highlights) < 3 else highlights

def identify_mental_patterns(text):
    patterns = []
    text_lower = text.lower()
    stress_keywords = ['stress', 'stressed', 'pressure', 'overwhelm', 'anxious', 'anxiety']
    if any(keyword in text_lower for keyword in stress_keywords):
        patterns.append('Stress')
    overthinking_keywords = ['think', 'thinking', 'thought', 'wonder', 'wondering', 'contemplate']
    if text_lower.count('think') > 3 or any(keyword in text_lower for keyword in overthinking_keywords):
        patterns.append('Overthinking')
    motivation_keywords = ['motivat', 'inspir', 'excit', 'enthusias', 'eager', 'drive']
    if any(keyword in text_lower for keyword in motivation_keywords):
        patterns.append('Motivation')
    avoidance_keywords = ['avoid', 'procrastin', 'delay', 'postpone']
    if any(keyword in text_lower for keyword in avoidance_keywords):
        patterns.append('Avoidance')
    self_criticism_keywords = ['should have', 'could have', 'would have', 'mistake', 'wrong', 'fail']
    if any(keyword in text_lower for keyword in self_criticism_keywords):
        patterns.append('Self-criticism')
    confidence_keywords = ['confident', 'proud', 'accomplish', 'success', 'achiev']
    if any(keyword in text_lower for keyword in confidence_keywords):
        patterns.append('Confidence')
    relationship_keywords = ['friend', 'family', 'relationship', 'partner', 'love', 'alone']
    if any(keyword in text_lower for keyword in relationship_keywords):
        patterns.append('Relationship concerns')
    fatigue_keywords = ['tired', 'exhaust', 'fatigue', 'sleepy', 'drain']
    if any(keyword in text_lower for keyword in fatigue_keywords):
        patterns.append('Fatigue')
    productivity_keywords = ['productiv', 'efficien', 'focus', 'concentrat', 'work']
    if any(keyword in text_lower for keyword in productivity_keywords):
        patterns.append('Productivity')
    return patterns

def identify_strengths(text):
    strengths = []
    text_lower = text.lower()
    effort_keywords = ['try', 'attempt', 'work', 'effort', 'strive']
    if any(keyword in text_lower for keyword in effort_keywords):
        strengths.append('Effort')
    honesty_keywords = ['honest', 'truth', 'admit', 'confess']
    if any(keyword in text_lower for keyword in honesty_keywords):
        strengths.append('Honesty')
    resilience_keywords = ['persever', 'persist', 'resilien', 'bounc', 'recover']
    if any(keyword in text_lower for keyword in resilience_keywords):
        strengths.append('Resilience')
    discipline_keywords = ['disciplin', 'routine', 'habit', 'schedule', 'plan']
    if any(keyword in text_lower for keyword in discipline_keywords):
        strengths.append('Discipline')
    responsibility_keywords = ['responsib', 'duty', 'obligat', 'accountab']
    if any(keyword in text_lower for keyword in responsibility_keywords):
        strengths.append('Responsibility')
    empathy_keywords = ['empath', 'understand', 'feel for', 'compassion']
    if any(keyword in text_lower for keyword in empathy_keywords):
        strengths.append('Empathy')
    self_awareness_keywords = ['realize', 'recognize', 'aware', 'understand myself']
    if any(keyword in text_lower for keyword in self_awareness_keywords):
        strengths.append('Self-awareness')
    return strengths

def generate_suggestions(compound, emotions, mental_patterns):
    suggestions = []
    if compound < -0.3:
        suggestions.append("It seems like you're going through a tough time. Remember that difficult moments are temporary, and you've overcome challenges before.")
    elif compound > 0.3:
        suggestions.append("You're in a positive headspace right now - that's wonderful! Consider what's contributing to this positivity and how you can cultivate more of it.")
    else:
        suggestions.append("You're in a balanced state of mind. This stability can be a great foundation for growth and self-reflection.")
    dominant_emotion = max(emotions, key=emotions.get) if emotions else "Happy"
    if emotions and emotions.get(dominant_emotion, 0) > 30:
        if dominant_emotion == 'Happy':
            suggestions.append("Your joy is contagious! Share this positive energy with someone you care about today.")
        elif dominant_emotion == 'Sad':
            suggestions.append("It's okay to feel sad sometimes. Be gentle with yourself and engage in activities that bring you comfort.")
        elif dominant_emotion == 'Anger':
            suggestions.append("Anger is a natural emotion. Try channeling this energy into something constructive, like exercise or creative expression.")
        elif dominant_emotion == 'Fear':
            suggestions.append("Fear can be protective, but don't let it hold you back. Take small steps toward what scares you.")
        elif dominant_emotion == 'Surprise':
            suggestions.append("Life's surprises can be challenging to process. Give yourself time to adjust to new information or changes.")
    if 'Stress' in mental_patterns:
        suggestions.append("You're experiencing stress. Try some deep breathing exercises or a short walk to help center yourself.")
    if 'Overthinking' in mental_patterns:
        suggestions.append("It seems like your mind is busy. Consider journaling your thoughts to help sort through them, or try mindfulness to stay present.")
    if 'Motivation' in mental_patterns:
        suggestions.append("Your motivation is a powerful force. Channel it toward a goal you've been putting off.")
    if 'Avoidance' in mental_patterns:
        suggestions.append("Avoidance is a common coping mechanism. Try breaking overwhelming tasks into smaller, manageable steps.")
    return suggestions if suggestions else ["You're doing great. Keep taking care of yourself and your emotional well-being."]

def generate_summary(compound, emotions):
    sentiment_word = get_sentiment_category(compound)
    dominant_emotion = max(emotions, key=emotions.get) if emotions else "Happy"
    emotion_percentage = emotions.get(dominant_emotion, 0) if emotions else 0
    if emotion_percentage > 50:
        return f"A {sentiment_word.lower()} entry dominated by {dominant_emotion.lower()} emotions."
    else:
        return f"A generally {sentiment_word.lower()} entry with mixed emotional tones."

def get_custom_emotion(text):
    emotions = {"Happy": 0, "Angry": 0, "Surprise": 0, "Sad": 0, "Fear": 0}
    text_lower = text.lower()
    happy_keywords = ['happy','joy','glad','pleased','delighted','cheerful','content','satisfied','excited','love','enjoy','fun','celebrate']
    angry_keywords = ['angry','mad','furious','annoyed','frustrated','irritated','hate','dislike','rage']
    surprise_keywords = ['surprise','amazed','astonished','shocked','stunned','startled','unexpected','incredible']
    sad_keywords = ['sad','unhappy','depressed','gloomy','melancholy','down','blue','upset','disappointed','lonely']
    fear_keywords = ['fear','afraid','scared','frightened','anxious','worried','nervous','panic','dread']
    happy_count = sum(1 for word in happy_keywords if word in text_lower)
    angry_count = sum(1 for word in angry_keywords if word in text_lower)
    surprise_count = sum(1 for word in surprise_keywords if word in text_lower)
    sad_count = sum(1 for word in sad_keywords if word in text_lower)
    fear_count = sum(1 for word in fear_keywords if word in text_lower)
    total_emotion_words = happy_count + angry_count + surprise_count + sad_count + fear_count
    if total_emotion_words > 0:
        emotions["Happy"] = round((happy_count / total_emotion_words) * 100)
        emotions["Angry"] = round((angry_count / total_emotion_words) * 100)
        emotions["Surprise"] = round((surprise_count / total_emotion_words) * 100)
        emotions["Sad"] = round((sad_count / total_emotion_words) * 100)
        emotions["Fear"] = round((fear_count / total_emotion_words) * 100)
    return emotions

# -------------------------
# Run
# -------------------------
if __name__ == '__main__':
    # Ensure data file exists on startup
    load_data()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(os.environ.get("FLASK_DEBUG", "false").lower() == "true"))
