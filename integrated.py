import os
import wave
import pyaudio
import speech_recognition as sr
from twilio.rest import Client
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from geocoder import ip
import joblib
import numpy as np
import librosa
from librosa.feature import spectral_contrast
import logging
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for
import webbrowser
import threading
from threading import Event, Thread, Lock
from langchain_ollama import ChatOllama
import re
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from Main import upload_to_drive
import geocoder


# Initialize Flask app
app = Flask(__name__)

# Initialize the ChatOllama model
try:
    model_chat = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434/")
except Exception as e:
    print(f"Warning: Could not initialize ChatOllama model: {e}")
    model_chat = None

# Twilio account credentials
account_sid = "AC428132525b161a85f44b619be525ec23"
auth_token = "e39fae28c1f850c1f6dbe5d06df718d6"
client = Client(account_sid, auth_token)

# Load ML model for audio detection
model_audio = joblib.load('arpit_random_forest_model.pkl')

# Modified crime data loading function
def load_crime_data():
    try:
        df = pd.read_csv('C:/Users/abhis/Downloads/SCREAM_DETECTION (2)/local_crime_data_20241026_020944.csv')
        crime_data = []
        
        intensity_colors = {
            'High': 'red',
            'Medium': 'yellow',
            'Low': 'green'
        }
        
        for _, row in df.iterrows():
            crime_data.append({
                'lat': row['Latitude'],
                'lng': row['Longitude'],
                'type': row['Incident_Type'],
                'date': row['Date'],
                'time': row['Time'],
                'intensity': row['Intensity'],
                'color': intensity_colors[row['Intensity']]
            })
        return crime_data
    except Exception as e:
        print(f"Warning: Could not load crime data: {e}")
        return []

# Load initial crime data
crime_data = load_crime_data()

# PyAudio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = RATE * 3
SILENCE_THRESHOLD = 0.1
FEATURES_LENGTH = 77

# Directory to save audio chunks
OUTPUT_DIR = 'audio_chunks1'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create events for alert handling
alert_cancelled = Event()
alert_active = False

# Helper Functions
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)*2 + cos(lat1) * cos(lat2) * sin(dlon/2)*2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def check_nearby_crimes(user_lat, user_lon, radius_km=2):
    """
    Check if there are any crimes within the specified radius
    Returns tuple (is_safe, nearby_crimes)
    """
    if crime_data.empty:
        return True, []
        
    nearby_crimes = []
    
    for _, crime in crime_data.iterrows():
        distance = calculate_distance(
            user_lat, user_lon,
            crime['Latitude'], crime['Longitude']
        )
        
        if distance <= radius_km:
            nearby_crimes.append({
                'type': crime['Incident_Type'],
                'distance': round(distance, 2),
                'intensity': crime['Intensity']
            })
    
    return len(nearby_crimes) == 0, nearby_crimes
def record_audio(file_path="output.wav", record_seconds=8):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print(f"Recording for {record_seconds} seconds...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("Recording finished.")
    return file_path

def extract_features(file_path):
    try:
        audio_np, _ = librosa.load(file_path, sr=RATE, mono=True)
        
        if np.max(np.abs(audio_np)) < SILENCE_THRESHOLD:
            logging.info("Silence detected, skipping feature extraction.")
            return None
        
        # Extract features
        rms = librosa.feature.rms(y=audio_np)
        mfccs = librosa.feature.mfcc(y=audio_np, sr=RATE, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=RATE)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_np, sr=RATE)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_np)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_np)
        chroma = librosa.feature.chroma_stft(y=audio_np, sr=RATE)
        spectral_contrasts = spectral_contrast(y=audio_np, sr=RATE)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_np, sr=RATE, n_mels=40)

        # Calculate averages
        features = np.concatenate((
            [np.mean(rms), np.mean(spectral_centroid), np.mean(spectral_bandwidth),
             np.mean(spectral_flatness), np.mean(zero_crossing_rate)],
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrasts, axis=1),
            np.mean(mel_spectrogram, axis=1)
        ))

        if len(features) != FEATURES_LENGTH:
            logging.error(f"Feature length mismatch: Expected {FEATURES_LENGTH}, got {len(features)}")
            return None
        
        return features
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def predict_audio(file_path):
    features = extract_features(file_path)
    if features is None:
        return False
    prediction = model_audio.predict([features])
    logging.info(f"Prediction: {prediction[0]}")
    return prediction[0] == 1

# Sleep mode variables and functions

sleep_timer = None
keyword = None
keyword_lock = Lock()
# Sleep mode variables and functions
sleep_until = 0
sleep_lock = Lock()

def announce_sleep_mode():
    wake_time = time.strftime('%H:%M:%S', time.localtime(sleep_until))
    announcement = f"""
    ============================
    SYSTEM ENTERING SLEEP MODE
    Time: {time.strftime('%H:%M:%S')}
    Will wake at: {wake_time}
    Duration: 30 minutes
    ============================
    """
    logging.info(announcement)
    return announcement

def is_system_sleeping():
    """Check if the system is currently in sleep mode."""
    global sleep_until
    with sleep_lock:
        return time.time() < sleep_until

def toggle_sleep_mode(is_sleeping):
    """Toggle sleep mode on/off with a 30-minute default timer."""
    global sleep_until
    with sleep_lock:
        if is_sleeping:
            # Set sleep mode for 30 minutes
            sleep_duration = 30 * 60  # 30 minutes in seconds
            sleep_until = time.time() + sleep_duration
            announce_sleep_mode()
        else:
            # Deactivate sleep mode immediately
            sleep_until = 0
            logging.info("""
            ============================
            SLEEP MODE DEACTIVATED
            System resuming normal operation
            ============================
            """)

@app.route('/toggle_sleep', methods=['POST'])
def toggle_sleep():
    data = request.json
    is_sleeping = data.get('sleeping', False)
    toggle_sleep_mode(is_sleeping)
    return jsonify({"success": True, "sleeping": is_system_sleeping()})

# @app.route('/toggle_sleep', methods=['POST'])
# def toggle_sleep():
#     """Toggle sleep mode on/off with a 30-minute default timer"""
#     global sleep_until
#     data = request.json
#     is_sleeping = data.get('sleeping', False)
    
#     with sleep_lock:
#         if is_sleeping:
#             # Set sleep mode for 30 minutes
#             sleep_until = time.time() + 30 * 60
#             logging.info("Sleep mode activated for 30 minutes")
#         else:
#             # Deactivate sleep mode immediately
#             sleep_until = 0
#             logging.info("Sleep mode deactivated")
    
#     return jsonify({"success": True, "sleeping": is_sleeping})

def verify_keyword(spoken_text):
    global keyword
    with keyword_lock:
        if not keyword:
            return True
        return keyword.lower() in spoken_text.lower()

def two_stage_verification(recognizer, source):
    if is_system_sleeping():
        return False
        
    trigger_detected = False
    
    try:
        audio = recognizer.listen(source, timeout=3)
        text = recognizer.recognize_google(audio)
        print(f"Stage 1: You said: {text}")

        trigger_word_detected = "help" in text.lower() and verify_keyword(text)
        audio_file_path = record_audio(record_seconds=8)
        scream_detected = predict_audio(audio_file_path)

        if trigger_word_detected or scream_detected:
            trigger_detected = True
            print("Stage 1: Trigger detected!")
    except Exception as e:
        print(f"Stage 1 error: Could not understand audio, continuing")

    if trigger_detected:
        print("Proceeding to Stage 2...")
        end_time = time.time() + 60

        while time.time() < end_time:
            try:
                audio = recognizer.listen(source, timeout=3)
                text = recognizer.recognize_google(audio)
                print(f"Stage 2: You said: {text}")

                trigger_word_detected = "help" in text.lower() and verify_keyword(text)
                audio_file_path = record_audio(record_seconds=8)
                scream_detected = predict_audio(audio_file_path)

                if trigger_word_detected or scream_detected:
                    return True
            except Exception as e:
                print(f"Stage 2 error: {e}")

    return False

def send_alert(location, map_link, shareable_link, client):
    try:
        message = client.messages.create(
            body=f"EMERGENCY ALERT: User in danger! Location: {location}. Map: {map_link}. Audio: {shareable_link}",
            from_="+12097530237",
            to="+919511972070"
        )
        logging.info("SMS alert sent successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")
        return False

def handle_alert_process(client, location, map_link, shareable_link):
    global alert_active
    
    alert_cancelled.clear()
    webbrowser.open('http://127.0.0.1:5000/alert')
    
    cancelled = alert_cancelled.wait(timeout=10)
    
    if cancelled:
        logging.info("Alert was cancelled by user")
        alert_active = False
        return False
    else:
        if alert_active:
            logging.info("No cancellation received, sending alert")
            return send_alert(location, map_link, shareable_link, client)
        return False

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map_view():
    return render_template('map.html')

@app.route('/get_crime_data')
def get_crime_data():
    return jsonify(crime_data)

@app.route('/next_page')
def next_page_route():
    return render_template('chatbot.html')

@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/alert')
def alert():
    global alert_active
    alert_active = True
    return render_template('alert.html')

@app.route('/cancel', methods=['POST'])
def cancel_alert():
    global alert_active
    alert_active = False
    alert_cancelled.set()
    return '', 204

@app.route('/set_keyword', methods=['POST'])
def set_keyword():
    global keyword
    data = request.json
    with keyword_lock:
        keyword = data.get('keyword')
    return jsonify({"success": True, "message": "Keyword set successfully"})

@app.route('/sleep_status', methods=['GET'])
def get_sleep_status():
    with sleep_lock:
        is_sleeping = time.time() < sleep_until
        remaining_time = max(0, sleep_until - time.time()) if is_sleeping else 0
        
        status_info = {
            "sleeping": is_sleeping,
            "remaining_minutes": round(remaining_time / 60, 1),
            "wake_time": time.strftime('%H:%M:%S', time.localtime(sleep_until)) if is_sleeping else None,
            "status_message": "System is in sleep mode" if is_sleeping else "System is active"
        }
        
        return jsonify(status_info)

@app.route("/generate_response", methods=["POST"])
def generate_response():
    input_text = request.json.get("input_text")
    
    # List of emergency trigger phrases
    emergency_phrases = [
        "i am in danger",
        "help me",
        "emergency",
        "save me",
        "sos",
        "urgent help",
        "being followed",
        "someone is following",
        "need police",
        "call police",
        "need help urgently",
        "stalker"
    ]
    
    # Check for exact emergency phrases
    if any(phrase in input_text.lower().strip() for phrase in emergency_phrases):
        try:
            # Get user's location immediately
            g = geocoder.ip('me')
            if g.latlng:
                location = f"{g.latlng[0]},{g.latlng[1]}"
                print(f"Emergency detected! User location: {location}")
            
            crisis_response = (
                "EMERGENCY ALERT ACTIVATED. Stay calm. "
                "Your location has been tracked and help is being dispatched. "
                "Please stay on this chat. Redirecting to emergency interface..."
            )
            
            return jsonify({
                "response": crisis_response,
                "redirect_url": url_for("alert")
            })
            
        except Exception as e:
            print(f"Error processing emergency: {str(e)}")
            return jsonify({
                "response": "Emergency services are being notified. Please stay safe.",
                "redirect_url": url_for("alert")
            })
    
    # Handle safety check command
    if input_text.lower().strip() == "safe":
      if input_text.lower().strip() == "safe":
        try:
            # Get user's location
            g = geocoder.ip('me')
            if not g.latlng:
                return jsonify({"response": "Unable to get your location. Please ensure location services are enabled."})
            
            user_lat, user_lon = g.latlng
            is_safe, nearby_crimes = check_nearby_crimes(user_lat, user_lon)
            
            if is_safe:
                response = "Your current location appears to be safe. No reported incidents within 2km radius."
            else:
                response = f"⚠ CAUTION: There are {len(nearby_crimes)} reported incidents within 2km of your location./n/n"
                # response += "Nearby incidents:/n"
                # for crime in nearby_crimes:
                #     response += f"- {crime['type']} ({crime['distance']}km away, {crime['intensity']} intensity)/n"
                response += "/nPlease stay vigilant and avoid walking alone."
            
            return jsonify({"response": response})
            
        except Exception as e:
            return jsonify({"response": f"Error checking location safety: {str(e)}"})
    
    # Crisis keywords check
    crisis_keywords = r"/b(danger|help|emergency|urgent|followed|suspicious|assist|need help)/b"
    if re.search(crisis_keywords, input_text, re.IGNORECASE):
        crisis_response = (
            "It sounds like you may be in a crisis or emergency situation. "
            "Please stay calm. We are sending an alert to the authorities to assist you."
        )
        return jsonify({"response": crisis_response, "redirect_url": url_for("alert")})
    
    # Regular chatbot response for non-emergency queries
    detailed_prompt = (
        "You are a highly intelligent and helpful assistant designed to provide support and information to users. "
        "If you detect any mention of danger, emergency, or need for assistance, respond empathetically and clearly, "
        "informing the user that an alert will be sent. For regular questions, respond as usual./n"
        f"User Question: {input_text}"
    )
    
    response = model_chat.invoke(detailed_prompt)
    return jsonify({"response": response.content})

def main_audio_monitoring():
    global alert_active
    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                
                alert_active = False
                alert_cancelled.clear()

                verified = two_stage_verification(recognizer, source)

                if verified:
                    logging.info("Verification passed. Starting alert process...")
                    audio_file_path = record_audio(record_seconds=8)
                    shareable_link = upload_to_drive(audio_file_path)

                    g = ip('me')
                    location = f"{g.latlng[0]},{g.latlng[1]}"
                    map_link = f"https://www.google.com/maps/place/{location}"

                    alert_sent = handle_alert_process(client, location, map_link, shareable_link)
                    
                    if alert_sent:
                        logging.info("Alert sent successfully")
                    else:
                        logging.info("Alert process completed, no SMS sent")
                    
                    time.sleep(2)

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(1)
            
@app.route("/confirm_alert", methods=["POST"])
def confirm_alert():
    try:
        # Get device location
        g = geocoder.ip('me')
        location = f"{g.latlng[0]},{g.latlng[1]}"
        print(f"Alert confirmed! Device location: {location}")

        # Create a Google Maps link
        map_link = f"https://www.google.com/maps/place/{location}"

        # Send SMS with location
        message = client.messages.create(
            body=f"EMERGENCY ALERT: User in danger! Location: {location}. Map: {map_link}",
            from_="+12097530237",
            to="+919511972070"
        )
        print("Emergency SMS sent!")

        return jsonify({
            "message": "Alert confirmed and sent to emergency services.",
            "location": location
        })
        
    except Exception as e:
        print(f"Error in confirm_alert: {str(e)}")
        return jsonify({
            "message": "Error processing alert. Emergency services have been notified.",
            "error": str(e)
        }), 500
        


if __name__ == '__main__':
    # Start audio monitoring in a separate thread
    audio_thread = Thread(target=main_audio_monitoring)
    audio_thread.daemon = True
    audio_thread.start()
    
    # Run Flask app
    app.run(debug=True, use_reloader=False)