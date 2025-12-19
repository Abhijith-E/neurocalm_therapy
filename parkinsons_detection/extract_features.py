# extract_features.py
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_pose_landmarks(video_path, resize=(640, 480), max_frames=None):
    """
    Returns an array of shape (num_frames, 33, 3) of landmarks (x,y,z) in normalized coords.
    If no landmarks found returns None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    landmarks = []
    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                joints = []
                for lm in results.pose_landmarks.landmark:
                    joints.append([lm.x, lm.y, lm.z])
                landmarks.append(joints)
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
    cap.release()
    if len(landmarks) == 0:
        return None
    return np.array(landmarks)  # shape: (T, 33, 3)

def extract_gait_features(landmarks):
    """
    Compute a simple set of gait features from pose landmarks.
    landmarks: np.array (T, 33, 3)
    Returns: 1D numpy array of length 10 (match expected by model)
    """
    if landmarks is None or landmarks.shape[0] < 2:
        return None

    # MediaPipe indices: left ankle=27, right ankle=28
    try:
        left_ankle_y = landmarks[:, 27, 1]
        right_ankle_y = landmarks[:, 28, 1]
    except IndexError:
        return None

    # Step difference (proxy for step length asymmetry / rhythm)
    step_diff = np.abs(left_ankle_y - right_ankle_y)

    # Simple features — expand later if needed
    feats = [
        np.mean(step_diff),
        np.std(step_diff),
        np.max(step_diff),
        np.min(step_diff),
        np.mean(left_ankle_y),
        np.mean(right_ankle_y),
        np.std(left_ankle_y),
        np.std(right_ankle_y),
        np.percentile(step_diff, 25),
        np.percentile(step_diff, 75),
    ]

    return np.array(feats, dtype=np.float32)

def get_features_for_video(video_path):
    """
    High-level helper: given a video path, returns the 10-dim feature vector or None.
    """
    landmarks = extract_pose_landmarks(video_path)
    if landmarks is None:
        return None
    return extract_gait_features(landmarks)
