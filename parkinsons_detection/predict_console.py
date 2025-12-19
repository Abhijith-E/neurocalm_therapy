# predict_console.py
import sys
import os
import joblib
from extract_features import get_features_for_video

MODEL_PATH = os.path.join("models", "gait_parkinsons_model.pkl")

def main(video_path):
    if not os.path.exists(video_path):
        print("Video file not found:", video_path); return
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_dummy_model.py to create it."); return

    features = get_features_for_video(video_path)
    if features is None:
        print("Could not extract features. Check video (side view walking recommended).")
        return

    model = joblib.load(MODEL_PATH)
    pred = model.predict(features.reshape(1, -1))[0]
    proba = model.predict_proba(features.reshape(1, -1))[0]

    label = "Parkinson's" if pred == 1 else "Normal"
    confidence = proba[1] if pred == 1 else proba[0]
    print(f"Prediction: {label} (confidence {confidence*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_console.py path/to/video.mp4")
    else:
        main(sys.argv[1])
