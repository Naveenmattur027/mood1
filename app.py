import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, make_response
from pymongo import MongoClient, errors
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import datetime
from datetime import timedelta
import re
import jwt
import bcrypt
from functools import wraps

# Only download vader_lexicon once; safe to keep but skip if running in restrictive envs
nltk.download('vader_lexicon', quiet=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_secret_key_here")  # change for prod
app.config['JWT_SECRET_KEY'] = os.environ.get("JWT_SECRET_KEY", "jwt_secret_key_here")

# Mongo configuration via env
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "personal_diary")

# Globals for lazy client
client = None

# Sentiment analyzer
sia = SentimentIntensityAnalyzer()

# -------------------------
# Mongo helper functions
# -------------------------
def ensure_client():
    """Ensure `client` is connected; try to (re)connect if not."""
    global client
    if client:
        return True
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        client.admin.command("ping")
        app.logger.info("Connected to MongoDB")
        return True
    except errors.PyMongoError as e:
        app.logger.error("MongoDB connect failed: %s", e)
        client = None
        return False

def get_db():
    """Return the database object or None if DB unavailable."""
    if not ensure_client():
        return None
    return client[DB_NAME]

def get_collection(name):
    db = get_db()
    if db is None:
        return None
    return db[name]

# -------------------------
# JWT Helper functions
# -------------------------
def generate_token(user_id):
    """Generate JWT token for user"""
    payload = {
        'user_id': str(user_id),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        'iat': datetime.datetime.utcnow()
    }
    return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """Verify JWT token and return user ID"""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def token_required(f):
    """Decorator to require valid JWT token for routes"""
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
# Routes
# -------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400

    users_collection = get_collection('users')
    if users_collection is None:
        return make_response(jsonify({'message': 'Database unavailable'}), 503)

    user = users_collection.find_one({'$or': [{'username': username}, {'email': username}]})

    if not user:
        return jsonify({'message': 'Invalid username or password!'}), 401

    # NOTE: demo uses plain text; replace with bcrypt.checkpw in production
    if password != user['password']:
        return jsonify({'message': 'Invalid username or password!'}), 401

    token = generate_token(user['_id'])

    return jsonify({
        'message': 'Login successful!',
        'token': token,
        'user': {
            'id': str(user['_0'] if '_0' in user else user['_id']),
            'username': user['username'],
            'email': user.get('email', '')
        }
    })

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'message': 'Username, email, and password are required!'}), 400

    users_collection = get_collection('users')
    if users_collection is None:
        return make_response(jsonify({'message': 'Database unavailable'}), 503)

    if users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
        return jsonify({'message': 'Username or email already exists!'}), 409

    new_user = {
        'username': username,
        'email': email,
        'password': password,  # in prod: use bcrypt.hashpw(...)
        'created_at': datetime.datetime.utcnow()
    }

    result = users_collection.insert_one(new_user)

    return jsonify({
        'message': 'User registered successfully!',
        'user': {
            'id': str(result.inserted_id),
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
    entries_collection = get_collection('entries')
    if entries_collection is None:
        return make_response(jsonify({"error": "Database unavailable"}), 503)
    try:
        entries = list(entries_collection.find({}, {'_id': 0}).sort('_id', -1))
        return jsonify({"entries": entries})
    except errors.PyMongoError as e:
        app.logger.error("DB error in /get_entries: %s", e)
        return make_response(jsonify({"error": "Database error"}), 503)

@app.route('/add_entry', methods=['POST'])
def add_entry():
    entries_collection = get_collection('entries')
    if entries_collection is None:
        return make_response(jsonify({"error": "Database unavailable"}), 503)

    entry_text = request.form.get('entry')
    date = datetime.date.today()
    date_str = date.strftime("%A, %Y-%m-%d")

    new_entry = {"date": date_str, "entry": entry_text}
    try:
        entries_collection.insert_one(new_entry)
        entries = list(entries_collection.find({}, {'_id': 0}).sort('_id', -1))
        return jsonify({"message": "Entry added successfully!", "entries": entries})
    except errors.PyMongoError as e:
        app.logger.error("DB error in /add_entry: %s", e)
        return make_response(jsonify({"error": "Database error"}), 503)

@app.route('/get_sentiment', methods=['GET'])
def get_sentiment():
    entries_collection = get_collection('entries')
    if entries_collection is None:
        return make_response(jsonify({"error": "Database unavailable"}), 503)
    try:
        entries = list(entries_collection.find({}, {'_id': 0}))
    except errors.PyMongoError as e:
        app.logger.error("DB error in /get_sentiment: %s", e)
        return make_response(jsonify({"error": "Database error"}), 503)

    if not entries:
        return jsonify({"message": "No entries available for sentiment analysis."})

    all_entries = " ".join([entry['entry'] for entry in entries])
    sentiment_score = sia.polarity_scores(all_entries)

    return jsonify({"sentiment": sentiment_score})

@app.route('/get_comprehensive_analysis', methods=['POST'])
def get_comprehensive_analysis():
    entry_text = request.form.get('entry')
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
            "overall_sentiment": {
                "category": overall_sentiment,
                "intensity": intensity
            },
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
    entry_text = request.form.get('entry')
    if not entry_text:
        return jsonify({"message": "No entry provided for sentiment analysis."})

    sentiment_score = sia.polarity_scores(entry_text)
    return jsonify({"sentiment": sentiment_score})

@app.route('/get_daily_sentiment', methods=['GET'])
def get_daily_sentiment():
    entries_collection = get_collection('entries')
    if entries_collection is None:
        return make_response(jsonify({"error": "Database unavailable"}), 503)

    today = datetime.date.today()
    today_str = today.strftime("%A, %Y-%m-%d")
    try:
        entries = list(entries_collection.find({"date": today_str}, {'_id': 0}))
    except errors.PyMongoError as e:
        app.logger.error("DB error in /get_daily_sentiment: %s", e)
        return make_response(jsonify({"error": "Database error"}), 503)

    if not entries:
        return jsonify({"message": "No entries available for today."})

    all_entries = " ".join([entry['entry'] for entry in entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({"sentiment": sentiment_score, "period": "daily"})

@app.route('/get_weekly_sentiment', methods=['GET'])
def get_weekly_sentiment():
    entries_collection = get_collection('entries')
    if entries_collection is None:
        return make_response(jsonify({"error": "Database unavailable"}), 503)

    today = datetime.date.today()
    week_ago = today - timedelta(days=7)

    try:
        entries = list(entries_collection.find({}, {'_id': 0}))
    except errors.PyMongoError as e:
        app.logger.error("DB error in /get_weekly_sentiment: %s", e)
        return make_response(jsonify({"error": "Database error"}), 503)

    filtered_entries = []
    for entry in entries:
        try:
            entry_date_str = entry['date'].split(", ")[1]
            entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            if entry_date >= week_ago and entry_date <= today:
                filtered_entries.append(entry)
        except:
            pass

    if not filtered_entries:
        return jsonify({"message": "No entries available for the past week."})

    daily_sentiments = {}
    for entry in filtered_entries:
        try:
            entry_date_str = entry['date'].split(", ")[1]
            sentiment_score = sia.polarity_scores(entry['entry'])
            daily_sentiments.setdefault(entry_date_str, []).append(sentiment_score['compound'])
        except:
            pass

    daily_averages = {date: (sum(scores) / len(scores) if scores else 0) for date, scores in daily_sentiments.items()}

    all_entries = " ".join([entry['entry'] for entry in filtered_entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({
        "sentiment": sentiment_score,
        "period": "weekly",
        "daily_data": daily_averages
    })

@app.route('/get_monthly_sentiment', methods=['GET'])
def get_monthly_sentiment():
    entries_collection = get_collection('entries')
    if entries_collection is None:
        return make_response(jsonify({"error": "Database unavailable"}), 503)

    today = datetime.date.today()
    month_ago = today - timedelta(days=30)

    try:
        entries = list(entries_collection.find({}, {'_id': 0}))
    except errors.PyMongoError as e:
        app.logger.error("DB error in /get_monthly_sentiment: %s", e)
        return make_response(jsonify({"error": "Database error"}), 503)

    filtered_entries = []
    for entry in entries:
        try:
            entry_date_str = entry['date'].split(", ")[1]
            entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            if entry_date >= month_ago and entry_date <= today:
                filtered_entries.append(entry)
        except:
            pass

    if not filtered_entries:
        return jsonify({"message": "No entries available for the past month."})

    daily_sentiments = {}
    for entry in filtered_entries:
        try:
            entry_date_str = entry['date'].split(", ")[1]
            sentiment_score = sia.polarity_scores(entry['entry'])
            daily_sentiments.setdefault(entry_date_str, []).append(sentiment_score['compound'])
        except:
            pass

    daily_averages = {date: (sum(scores) / len(scores) if scores else 0) for date, scores in daily_sentiments.items()}

    all_entries = " ".join([entry['entry'] for entry in filtered_entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({
        "sentiment": sentiment_score,
        "period": "monthly",
        "daily_data": daily_averages
    })

@app.route('/get_sentiment_counts', methods=['GET'])
def get_sentiment_counts():
    entries_collection = get_collection('entries')
    if entries_collection is None:
        return make_response(jsonify({"error": "Database unavailable"}), 503)

    try:
        entries = list(entries_collection.find({}, {'_id': 0}))
    except errors.PyMongoError as e:
        app.logger.error("DB error in /get_sentiment_counts: %s", e)
        return make_response(jsonify({"error": "Database error"}), 503)

    if not entries:
        return jsonify({"counts": {"happy": 0, "neutral": 0, "sad": 0}})

    happy_count = 0
    neutral_count = 0
    sad_count = 0

    for entry in entries:
        sentiment_score = sia.polarity_scores(entry['entry'])
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

    return jsonify({"counts": {"happy": happy_count, "neutral": neutral_count, "sad": sad_count}})

@app.route('/clear_entries', methods=['POST'])
def clear_entries():
    entries_collection = get_collection('entries')
    if entries_collection is None:
        return make_response(jsonify({"error": "Database unavailable"}), 503)
    try:
        entries_collection.delete_many({})
        return jsonify({"message": "Diary entries cleared!"})
    except errors.PyMongoError as e:
        app.logger.error("DB error in /clear_entries: %s", e)
        return make_response(jsonify({"error": "Database error"}), 503)

# -------------------------
# Helper functions (unchanged)
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
    dominant_emotion = max(emotions, key=emotions.get)
    if emotions[dominant_emotion] > 30:
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
    dominant_emotion = max(emotions, key=emotions.get)
    emotion_percentage = emotions[dominant_emotion]
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
    # In Render, the host/port and debug are controlled by the service; for local testing:
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(os.environ.get("FLASK_DEBUG", "false").lower() == "true"))
