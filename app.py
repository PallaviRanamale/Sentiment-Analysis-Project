from flask import Flask, render_template, request, jsonify
import tweepy
import pickle
import os
from dotenv import load_dotenv
from tweepy.errors import TweepyException

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Twitter API credentials (Use Bearer Token for v2 authentication)
BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

if not BEARER_TOKEN:
    raise Exception("Missing Bearer Token in environment variables.")

# Tweepy v2 authentication using Bearer Token
try:
    client = tweepy.Client(bearer_token=BEARER_TOKEN)
    print("Authentication successful.")
except Exception as e:
    raise Exception(f"Error authenticating with Twitter API: {str(e)}") from e

# Load the sentiment analysis model
MODEL_PATH = 'model/sentiment_analysis_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to process sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'Twitter username is required'}), 400

    tweets = []
    try:
        # Fetch recent tweets using the user_timeline endpoint (for Essential Access)
        print(f"Attempting to fetch tweets from user: {user_id}")
        response = client.get_users_tweets(
            id=user_id,
            max_results=10,  # Limit to the number of tweets returned per request (10 for free tier)
            tweet_fields=["text"]  # Fetch only the tweet text
        )

        if response.data:
            tweets.extend([tweet.text for tweet in response.data])  # Collect tweet text
        else:
            print(f"No tweets found for user {user_id}.")

        print(f"Successfully fetched {len(tweets)} tweets.")

    except Exception as e:
        return jsonify({'error': f"Error fetching tweets: {str(e)}"}), 500

    # Check if no tweets were fetched
    if not tweets:
        return jsonify({'error': f"No tweets found for user {user_id}."}), 404

    # Perform sentiment analysis
    try:
        sentiments = [model.predict([tweet])[0] for tweet in tweets]
    except Exception as e:
        return jsonify({'error': f"Error during sentiment analysis: {str(e)}"}), 500

    # Count positive, negative, neutral
    positive = sentiments.count('Positive')
    negative = sentiments.count('Negative')
    neutral = sentiments.count('Neutral')

    return render_template(
        'result.html',
        user_id=user_id,
        tweets=tweets,
        positive=positive,
        negative=negative,
        neutral=neutral,
    )

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5001)
    print("Flask server started.")
