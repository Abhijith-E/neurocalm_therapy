import os
import torch
import torch.nn as nn
from torchvision import models as tv_models  # ✅ renamed to avoid overwrite
import joblib  # ✅ for .pkl models

# =========================
# Existing ResNet Loader
# =========================
def build_resnet152_model(num_classes=2):
    """Builds a ResNet152 model with custom classifier head."""
    model = tv_models.resnet152(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

def load_resnet_model(path):
    """Loads a .pth model file into a ResNet152 architecture."""
    model = build_resnet152_model()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# =========================
# .pkl Model Loader
# =========================
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

def _load_pkl_if_exists(filename):
    """Loads a .pkl model if file exists, else returns None."""
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"[model_loader] Failed to load {filename}: {e}")
            return None
    else:
        print(f"[model_loader] File not found: {filename}")
    return None

# =========================
# Main Loader for All Models
# =========================
def load_all_models():
    """
    Loads all available models:
    - gait (.pkl)
    - typing (.pkl)
    - voice (.pkl)
    - spiral (.pth)
    - wave (.pth)
    """
    loaded_models = {}
    
    # ✅ Load .pkl models
    loaded_models['gait'] = _load_pkl_if_exists("gait_parkinsons_model.pkl")
    loaded_models['typing'] = _load_pkl_if_exists("typing_model.pkl")
    loaded_models['voice'] = _load_pkl_if_exists("voice_model.pkl")
    
    # ✅ Load .pth models
    spiral_path = os.path.join(MODELS_DIR, "trained_model_spiral.pth")
    wave_path = os.path.join(MODELS_DIR, "trained_model_wave.pth")

    if os.path.exists(spiral_path):
        loaded_models['spiral'] = load_resnet_model(spiral_path)
    if os.path.exists(wave_path):
        loaded_models['wave'] = load_resnet_model(wave_path)

    return loaded_models

# =========================
# Compatibility Function
# =========================
def load_model(path):
    """
    Backward-compatible function:
    - If path ends with .pth → load ResNet model
    - If path ends with .pkl → load sklearn model
    """
    if path.endswith(".pth"):
        return load_resnet_model(path)
    elif path.endswith(".pkl"):
        return joblib.load(path)
    else:
        raise ValueError(f"Unsupported model format: {path}")

# =========================
# Auto-load when imported
# =========================
models = load_all_models()
