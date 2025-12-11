from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from pymongo import MongoClient
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import datetime
from datetime import timedelta
import re
import jwt
import bcrypt
from functools import wraps

nltk.download('vader_lexicon')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key
app.config['JWT_SECRET_KEY'] = 'jwt_secret_key_here'  # Change this to a random secret key for JWT

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['personal_diary']
entries_collection = db['entries']
users_collection = db['users']

# User class
sia = SentimentIntensityAnalyzer()

# JWT Helper functions
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
        
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
            
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(user_id, *args, **kwargs)
    return decorated

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    # Handle POST request for login
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'message': 'Username and password are required!'}), 400
    
    # Find user in database
    user = users_collection.find_one({'$or': [{'username': username}, {'email': username}]})
    
    if not user:
        return jsonify({'message': 'Invalid username or password!'}), 401
    
    # Check password
    # For demo purposes, we're checking the plain password
    # In a real application, you would use:
    # if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
    if password != user['password']:  # This is just for demo, in real app use bcrypt
        return jsonify({'message': 'Invalid username or password!'}), 401
    
    # Generate JWT token
    token = generate_token(user['_id'])
    
    return jsonify({
        'message': 'Login successful!',
        'token': token,
        'user': {
            'id': str(user['_id']),
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
    
    # Check if user already exists
    if users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
        return jsonify({'message': 'Username or email already exists!'}), 409
    
    # Hash password (in a real app)
    # For demo purposes, we'll store the plain password
    # In a real application, you would use:
    # hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Create new user (for demo, we're storing plain password)
    new_user = {
        'username': username,
        'email': email,
        'password': password,  # In real app, store hashed password
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
    # For demo purposes, we'll allow access without authentication
    # In a real application, you would uncomment the following lines:
    # token = request.headers.get('Authorization')
    # if not token:
    #     return redirect(url_for('login'))
    # 
    # if token.startswith('Bearer '):
    #     token = token[7:]
    # 
    # user_id = verify_token(token)
    # if not user_id:
    #     return redirect(url_for('login'))
    
    return render_template('index.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/get_entries')
def get_entries():
    entries = list(entries_collection.find({}, {'_id': 0}).sort('_id', -1))
    return jsonify({"entries": entries})

@app.route('/add_entry', methods=['POST'])
def add_entry():
    entry_text = request.form.get('entry')
    date = datetime.date.today()
    # Format date to include day of week
    date_str = date.strftime("%A, %Y-%m-%d")
    
    new_entry = {"date": date_str, "entry": entry_text}
    entries_collection.insert_one(new_entry)
    
    entries = list(entries_collection.find({}, {'_id': 0}).sort('_id', -1))
    return jsonify({"message": "Entry added successfully!", "entries": entries})

@app.route('/get_sentiment', methods=['GET'])
def get_sentiment():
    entries = list(entries_collection.find({}, {'_id': 0}))
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
        # 1. Overall Sentiment with intensity
        sentiment_score = sia.polarity_scores(entry_text)
        overall_sentiment = get_sentiment_category(sentiment_score['compound'])
        intensity = get_sentiment_intensity(sentiment_score['compound'])
        
        # 2. Emotion Detection
        try:
            emotions = get_custom_emotion(entry_text)
            # Ensure we have all required emotions with default values
            emotion_keys = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
            for key in emotion_keys:
                if key not in emotions:
                    emotions[key] = 0
        except Exception as e:
            # Handle any error with emotion detection
            emotions = {"Happy": 0, "Angry": 0, "Surprise": 0, "Sad": 0, "Fear": 0}
        
        # 3. Highlights / Important Events
        highlights = extract_highlights(entry_text)
        
        # 4. Mental State Patterns
        mental_patterns = identify_mental_patterns(entry_text)
        
        # 5. Personal Strengths Shown
        strengths = identify_strengths(entry_text)
        
        # 6. Supportive Suggestions
        suggestions = generate_suggestions(sentiment_score['compound'], emotions, mental_patterns)
        
        # 7. One-Line Summary
        summary = generate_summary(sentiment_score['compound'], emotions)
        
        # 8. Trend Insight (for current entry, not enough data)
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
    today = datetime.date.today()
    # Format date to include day of week
    today_str = today.strftime("%A, %Y-%m-%d")
    entries = list(entries_collection.find({"date": today_str}, {'_id': 0}))
    if not entries:
        return jsonify({"message": "No entries available for today."})
    
    all_entries = " ".join([entry['entry'] for entry in entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({"sentiment": sentiment_score, "period": "daily"})

@app.route('/get_weekly_sentiment', methods=['GET'])
def get_weekly_sentiment():
    today = datetime.date.today()
    week_ago = today - timedelta(days=7)
    
    # Get entries for the past week
    entries = list(entries_collection.find({}, {'_id': 0}))
    filtered_entries = []
    for entry in entries:
        try:
            entry_date_str = entry['date'].split(", ")[1]  # Get "YYYY-MM-DD" part
            entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            if entry_date >= week_ago and entry_date <= today:
                filtered_entries.append(entry)
        except:
            pass
    
    if not filtered_entries:
        return jsonify({"message": "No entries available for the past week."})
    
    # Group entries by date for line graph
    daily_sentiments = {}
    for entry in filtered_entries:
        try:
            entry_date_str = entry['date'].split(", ")[1]  # Get "YYYY-MM-DD" part
            sentiment_score = sia.polarity_scores(entry['entry'])
            if entry_date_str not in daily_sentiments:
                daily_sentiments[entry_date_str] = []
            daily_sentiments[entry_date_str].append(sentiment_score['compound'])
        except:
            pass
    
    # Calculate average sentiment per day
    daily_averages = {}
    for date, scores in daily_sentiments.items():
        daily_averages[date] = sum(scores) / len(scores) if scores else 0
    
    all_entries = " ".join([entry['entry'] for entry in filtered_entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({
        "sentiment": sentiment_score, 
        "period": "weekly",
        "daily_data": daily_averages
    })

@app.route('/get_monthly_sentiment', methods=['GET'])
def get_monthly_sentiment():
    today = datetime.date.today()
    month_ago = today - timedelta(days=30)
    
    # Get entries for the past month
    entries = list(entries_collection.find({}, {'_id': 0}))
    filtered_entries = []
    for entry in entries:
        try:
            entry_date_str = entry['date'].split(", ")[1]  # Get "YYYY-MM-DD" part
            entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            if entry_date >= month_ago and entry_date <= today:
                filtered_entries.append(entry)
        except:
            pass
    
    if not filtered_entries:
        return jsonify({"message": "No entries available for the past month."})
    
    # Group entries by date for line graph
    daily_sentiments = {}
    for entry in filtered_entries:
        try:
            entry_date_str = entry['date'].split(", ")[1]  # Get "YYYY-MM-DD" part
            sentiment_score = sia.polarity_scores(entry['entry'])
            if entry_date_str not in daily_sentiments:
                daily_sentiments[entry_date_str] = []
            daily_sentiments[entry_date_str].append(sentiment_score['compound'])
        except:
            pass
    
    # Calculate average sentiment per day
    daily_averages = {}
    for date, scores in daily_sentiments.items():
        daily_averages[date] = sum(scores) / len(scores) if scores else 0
    
    all_entries = " ".join([entry['entry'] for entry in filtered_entries])
    sentiment_score = sia.polarity_scores(all_entries)
    return jsonify({
        "sentiment": sentiment_score, 
        "period": "monthly",
        "daily_data": daily_averages
    })

@app.route('/get_sentiment_counts', methods=['GET'])
def get_sentiment_counts():
    entries = list(entries_collection.find({}, {'_id': 0}))
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
    entries_collection.delete_many({})
    return jsonify({"message": "Diary entries cleared!"})

# Helper functions
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
    # Simple highlight extraction based on sentence structure
    sentences = re.split(r'[.!?]+', text)
    highlights = []
    
    # Filter out empty sentences and short ones
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Only consider sentences with more than 10 characters
            highlights.append(sentence)
    
    # Return 3-7 highlights
    return highlights[:7] if len(highlights) > 7 else highlights[3:] if len(highlights) < 3 else highlights

def identify_mental_patterns(text):
    patterns = []
    text_lower = text.lower()
    
    # Stress indicators
    stress_keywords = ['stress', 'stressed', 'pressure', 'overwhelm', 'anxious', 'anxiety']
    if any(keyword in text_lower for keyword in stress_keywords):
        patterns.append('Stress')
    
    # Overthinking indicators
    overthinking_keywords = ['think', 'thinking', 'thought', 'wonder', 'wondering', 'contemplate']
    if text_lower.count('think') > 3 or any(keyword in text_lower for keyword in overthinking_keywords):
        patterns.append('Overthinking')
    
    # Motivation indicators
    motivation_keywords = ['motivat', 'inspir', 'excit', 'enthusias', 'eager', 'drive']
    if any(keyword in text_lower for keyword in motivation_keywords):
        patterns.append('Motivation')
    
    # Avoidance indicators
    avoidance_keywords = ['avoid', 'procrastin', 'delay', 'postpone']
    if any(keyword in text_lower for keyword in avoidance_keywords):
        patterns.append('Avoidance')
    
    # Self-criticism indicators
    self_criticism_keywords = ['should have', 'could have', 'would have', 'mistake', 'wrong', 'fail']
    if any(keyword in text_lower for keyword in self_criticism_keywords):
        patterns.append('Self-criticism')
    
    # Confidence indicators
    confidence_keywords = ['confident', 'proud', 'accomplish', 'success', 'achiev']
    if any(keyword in text_lower for keyword in confidence_keywords):
        patterns.append('Confidence')
    
    # Relationship concerns
    relationship_keywords = ['friend', 'family', 'relationship', 'partner', 'love', 'alone']
    if any(keyword in text_lower for keyword in relationship_keywords):
        patterns.append('Relationship concerns')
    
    # Fatigue indicators
    fatigue_keywords = ['tired', 'exhaust', 'fatigue', 'sleepy', 'drain']
    if any(keyword in text_lower for keyword in fatigue_keywords):
        patterns.append('Fatigue')
    
    # Productivity indicators
    productivity_keywords = ['productiv', 'efficien', 'focus', 'concentrat', 'work']
    if any(keyword in text_lower for keyword in productivity_keywords):
        patterns.append('Productivity')
    
    return patterns

def identify_strengths(text):
    strengths = []
    text_lower = text.lower()
    
    # Effort
    effort_keywords = ['try', 'attempt', 'work', 'effort', 'strive']
    if any(keyword in text_lower for keyword in effort_keywords):
        strengths.append('Effort')
    
    # Honesty
    honesty_keywords = ['honest', 'truth', 'admit', 'confess']
    if any(keyword in text_lower for keyword in honesty_keywords):
        strengths.append('Honesty')
    
    # Resilience
    resilience_keywords = ['persever', 'persist', 'resilien', 'bounc', 'recover']
    if any(keyword in text_lower for keyword in resilience_keywords):
        strengths.append('Resilience')
    
    # Discipline
    discipline_keywords = ['disciplin', 'routine', 'habit', 'schedule', 'plan']
    if any(keyword in text_lower for keyword in discipline_keywords):
        strengths.append('Discipline')
    
    # Responsibility
    responsibility_keywords = ['responsib', 'duty', 'obligat', 'accountab']
    if any(keyword in text_lower for keyword in responsibility_keywords):
        strengths.append('Responsibility')
    
    # Empathy
    empathy_keywords = ['empath', 'understand', 'feel for', 'compassion']
    if any(keyword in text_lower for keyword in empathy_keywords):
        strengths.append('Empathy')
    
    # Self-awareness
    self_awareness_keywords = ['realize', 'recognize', 'aware', 'understand myself']
    if any(keyword in text_lower for keyword in self_awareness_keywords):
        strengths.append('Self-awareness')
    
    return strengths

def generate_suggestions(compound, emotions, mental_patterns):
    suggestions = []
    
    # Based on overall sentiment
    if compound < -0.3:
        suggestions.append("It seems like you're going through a tough time. Remember that difficult moments are temporary, and you've overcome challenges before.")
    elif compound > 0.3:
        suggestions.append("You're in a positive headspace right now - that's wonderful! Consider what's contributing to this positivity and how you can cultivate more of it.")
    else:
        suggestions.append("You're in a balanced state of mind. This stability can be a great foundation for growth and self-reflection.")
    
    # Based on dominant emotion
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
    
    # Based on mental patterns
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
    """
    Custom emotion detection function to replace text2emotion
    which has compatibility issues with newer emoji library versions
    """
    # Initialize emotion scores
    emotions = {
        "Happy": 0,
        "Angry": 0,
        "Surprise": 0,
        "Sad": 0,
        "Fear": 0
    }
    
    # Convert to lowercase for easier matching
    text_lower = text.lower()
    
    # Define keywords for each emotion with more comprehensive lists
    happy_keywords = ['happy', 'joy', 'joyful', 'glad', 'pleased', 'delighted', 'cheerful', 'content', 'satisfied', 'excited', 'thrilled', 'great', 'wonderful', 'fantastic', 'amazing', 'awesome', 'love', 'like', 'pleasure', 'enjoy', 'fun', 'celebrate']
    angry_keywords = ['angry', 'mad', 'furious', 'irate', 'annoyed', 'frustrated', 'irritated', 'outraged', 'livid', 'enraged', 'hate', 'dislike', 'rage', 'fume', 'resent', 'bitter']
    surprise_keywords = ['surprise', 'surprising', 'surprisingly', 'amazed', 'amazing', 'astonished', 'shocked', 'stunned', 'startled', 'unexpected', 'incredible', 'unbelievable', 'astounding', 'remarkable']
    sad_keywords = ['sad', 'unhappy', 'depressed', 'sorrowful', 'gloomy', 'melancholy', 'down', 'blue', 'upset', 'disappointed', 'disheartened', 'discouraged', 'miserable', 'grief', 'sorrow', 'heartbroken', 'lonely']
    fear_keywords = ['fear', 'afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous', 'panicked', 'dread', 'panic', 'fearful', 'apprehensive', 'concerned']
    
    # Count occurrences of each emotion keyword
    happy_count = sum(1 for word in happy_keywords if word in text_lower)
    angry_count = sum(1 for word in angry_keywords if word in text_lower)
    surprise_count = sum(1 for word in surprise_keywords if word in text_lower)
    sad_count = sum(1 for word in sad_keywords if word in text_lower)
    fear_count = sum(1 for word in fear_keywords if word in text_lower)
    
    # Calculate percentages (simple approach)
    total_emotion_words = happy_count + angry_count + surprise_count + sad_count + fear_count
    
    if total_emotion_words > 0:
        emotions["Happy"] = round((happy_count / total_emotion_words) * 100)
        emotions["Angry"] = round((angry_count / total_emotion_words) * 100)
        emotions["Surprise"] = round((surprise_count / total_emotion_words) * 100)
        emotions["Sad"] = round((sad_count / total_emotion_words) * 100)
        emotions["Fear"] = round((fear_count / total_emotion_words) * 100)
    
    return emotions

if __name__ == '__main__':
    app.run(debug=True)