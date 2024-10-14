from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import requests
from datetime import datetime, timedelta
import time
import joblib  # For saving and loading models
from apscheduler.schedulers.background import BackgroundScheduler  # To automate the process

app = Flask(__name__)

# Global variables for the model and data
model = None
scheduler = BackgroundScheduler()

# Function to load data from the NBA API
def load_data():
    url = "https://free-nba.p.rapidapi.com/games"
    headers = {
        "X-RapidAPI-Host": "free-nba.p.rapidapi.com",
        "X-RapidAPI-Key": "35d59d5501msh3a3171dd81abf0dp1a6e5ajsn50a2d7eb4519"
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Fetch data for the last year
    
    all_games = []
    page = 0
    
    while True:
        querystring = {
            "page": str(page),
            "per_page": "100",
            "dates": [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
        }
        
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()
            
            if not data['data']:
                break  # No more data to fetch
            
            all_games.extend(data['data'])
            page += 1
            time.sleep(1)  # Avoid rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break
    
    if not all_games:
        raise Exception("No data fetched from the API")
    
    processed_data = []
    for game in all_games:
        home_team = game['home_team']
        visitor_team = game['visitor_team']
        
        processed_data.append({
            'home_team_id': home_team['id'],
            'visitor_team_id': visitor_team['id'],
            'home_team_score': game['home_team_score'],
            'visitor_team_score': game['visitor_team_score'],
            'result': 'Home' if game['home_team_score'] > game['visitor_team_score'] else 'Away'
        })
    
    return pd.DataFrame(processed_data)

# Function to prepare data for model training
def prepare_data(data):
    X = data[['home_team_id', 'visitor_team_id']]
    y = data['result']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train the RandomForest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to load or retrain the model periodically
def retrain_model():
    global model
    try:
        print("Fetching data...")
        data = load_data()
        print(f"Fetched {len(data)} games")
        
        if len(data) < 100:
            print("Limited data available. Model may not be accurate.")
            return
        
        X_train, X_test, y_train, y_test = prepare_data(data)
        
        print("Training model...")
        model = train_model(X_train, y_train)
        
        # Save the trained model to disk
        joblib.dump(model, 'nba_model.pkl')
        print("Model saved to disk.")

        # Evaluate model accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy:.2f}")
    
    except Exception as e:
        print(f"An error occurred during model retraining: {e}")

# Function to load the saved model
def load_saved_model():
    global model
    try:
        model = joblib.load('nba_model.pkl')
        print("Model loaded from disk.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None


def train():
    try:
        retrain_model()
        return jsonify({"message": "Model retrained and saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to predict the outcome of a game
@app.route('/predict', methods=['POST'])
def predict_game():
    train()
    try:
        data = request.json
        home_team_id = data.get('home_team_id')
        visitor_team_id = data.get('visitor_team_id')

        if not home_team_id or not visitor_team_id:
            return jsonify({"error": "home_team_id and visitor_team_id are required"}), 400
        
        if model is None:
            load_saved_model()
        
        if model is None:
            return jsonify({"error": "Model is not available. Try retraining first."}), 500
        
        new_match = np.array([[home_team_id, visitor_team_id]])
        result = model.predict(new_match)

        return jsonify({
            "predicted_result": result[0]
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Scheduler to retrain the model every day at midnight
scheduler.add_job(retrain_model, 'interval', days=1)
scheduler.start()

if __name__ == "__main__":
    # Load the saved model at startup
    load_saved_model()
    
    # Start Flask app
    app.run(debug=True)
