import numpy as np
from utils_voice import extract_voice_features
import joblib
import os
import json
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify, send_from_directory
from model_loader import load_model, models  # models dict auto-loaded
from werkzeug.utils import secure_filename
from extract_features import get_features_for_video
import pickle

# -------------------
# Flask App Setup
# -------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Upload config for gait module
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------
# User Auth Utilities
# -------------------
USERS_FILE = 'auth/users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# -------------------
# Models Loading
# -------------------
voice_model = joblib.load('models/voice_model.pkl')
typing_model = joblib.load('models/typing_model.pkl')
spiral_model = load_model('models/trained_model_spiral.pth')
wave_model = load_model('models/trained_model_wave.pth')

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------
# Routes
# -------------------
@app.route('/')
def home():
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = load_users()
        email = request.form['email']
        if email in users:
            return "User already exists"

        users[email] = {
            "name": request.form['name'],
            "password": request.form['password'],
            "age": int(request.form['age']),
            "gender": request.form['gender'],
            "hand": request.form['hand'],
            "onset_years": request.form.get('onset_years', 0),
            "role": request.form['role']
        }

        save_users(users)
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = load_users()
        email = request.form['email']
        password = request.form['password']
        
        if email in users:
            user = users[email]
            if user.get('password') == password:
                session['username'] = user.get('name', 'Guest')
                session['role'] = user.get('role', 'user')
                return redirect('/dashboard')
            else:
                return "Incorrect password"
        else:
            return "User not found"
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect('/login')
    return render_template('dashboard.html', user=session['username'], role=session['role'])

# -------------------
# Voice Module
# -------------------
@app.route('/voice-module', methods=['GET', 'POST'])
def voice_module():
    if request.method == 'POST':
        if 'audioFile' not in request.files:
            flash('No audio file part')
            return redirect(request.url)

        file = request.files['audioFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            try:
                filepath = os.path.join('uploads', file.filename)
                file.save(filepath)

                features = extract_voice_features(filepath)
                features = np.array(features).reshape(1, -1)

                prediction = voice_model.predict(features)[0]
                result = 'Parkinson\'s Detected' if prediction == 1 else 'No Parkinson\'s Detected'
                os.remove(filepath)

                return render_template('voice.html', prediction=result)

            except Exception as e:
                return render_template('voice.html', prediction=f"Error processing audio: {e}")

    return render_template('voice.html')

# -------------------
# Typing Module
# -------------------
@app.route('/typing-module', methods=['GET'])
def show_typing_module():
    return render_template('typing.html')

@app.route('/typing-module', methods=['POST'])
def typing_module():
    try:
        data = request.get_json()

        hold_times = data.get("hold_times", [])
        flight1 = data.get("flight1", [])
        flight2 = data.get("flight2", [])
        pct_L = data.get("pct_L", 0)
        pct_R = data.get("pct_R", 0)
        pct_S = data.get("pct_S", 0)
        keys_per_sec = data.get("keys_per_sec", 0)

        if not hold_times or not flight1 or not flight2:
            return jsonify({'error': 'Incomplete typing data'}), 400

        def extract_stats(values):
            if len(values) == 0:
                return [0, 0, 0, 0, 0]
            arr = np.array(values)
            return [
                np.mean(arr),
                np.median(arr),
                np.std(arr),
                np.min(arr),
                np.max(arr)
            ]

        features = []
        features.extend(extract_stats(hold_times))
        features.extend(extract_stats(flight1))
        features.extend(extract_stats(flight2))
        features.extend([pct_L, pct_R, pct_S])
        features.append(keys_per_sec)

        features = np.array(features).reshape(1, -1)

        prediction = typing_model.predict(features)[0]
        label = "Parkinson's" if prediction == 1 else "Healthy"

        confidence = None
        if hasattr(typing_model, "predict_proba"):
            proba = typing_model.predict_proba(features)
            confidence = float(np.max(proba))

        return jsonify({'prediction': label, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------
# Handwriting Module
# -------------------
@app.route('/handwriting-module', methods=['GET', 'POST'])
def handwriting_module():
    spiral_result = wave_result = None

    if request.method == 'POST':
        spiral_img = request.files.get('spiral_img')
        wave_img = request.files.get('wave_img')

        if spiral_img:
            img = Image.open(spiral_img).convert("RGB")
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = spiral_model(img)
                pred = torch.argmax(output, dim=1).item()
                spiral_result = "Parkinson" if pred == 1 else "Healthy"

        if wave_img:
            img = Image.open(wave_img).convert("RGB")
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = wave_model(img)
                pred = torch.argmax(output, dim=1).item()
                wave_result = "Parkinson" if pred == 1 else "Healthy"

    return render_template('handwriting.html',
                           spiral_result=spiral_result,
                           wave_result=wave_result)

# -------------------
# Gait Module
# -------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/gait", methods=["GET", "POST"])
def gait_page():
    result = None
    if request.method == "POST":
        if "video" not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files["video"]
        if file.filename == "":
            flash("No selected file", "danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            feats = get_features_for_video(save_path)
            if feats is None:
                flash("Could not extract landmarks/features. Use a clear side-view walking video.", "warning")
                return render_template("gait.html", result=None)

            gait_model = models.get("gait")
            if gait_model is None:
                flash("Gait model not found. Run train_dummy_model.py to create models/gait_parkinsons_model.pkl", "danger")
                return render_template("gait.html", result=None)

            pred = gait_model.predict(feats.reshape(1, -1))[0]
            prob = gait_model.predict_proba(feats.reshape(1, -1))[0]
            label = "Parkinson'sNormal" if pred == 1 else "Parkinson's"
            confidence = float(prob[1]) if pred == 1 else float(prob[0])

            result = {
                "label": label,
                "confidence": round(confidence * 100, 2),
                "filename": filename
            }
            return render_template("gait.html", result=result)
        else:
            flash("File type not allowed. Use mp4/avi/mov/mkv.", "danger")
            return redirect(request.url)

    return render_template("gait.html", result=result)

@app.route("/gait-predict-dashboard", methods=["POST"])
def gait_predict_dashboard():
    if "video" not in request.files:
        return redirect(url_for("dashboard"))
    
    video = request.files["video"]
    if video.filename == "":
        return redirect(url_for("dashboard"))

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
    video.save(video_path)

    # Dummy prediction - replace with real gait processing
    features = np.random.rand(10).reshape(1, -1)
    prediction = gait_model.predict(features)[0]
    gait_result = "Parkinson's Detected" if prediction == 1 else "No Parkinson's"

    return render_template("dashboard.html", gait_result=gait_result)

# -------------------
# Logout
# -------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# -------------------
# Main Entry
# -------------------
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
