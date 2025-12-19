import numpy as np
import librosa

def extract_voice_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Extract 13 MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Extract 9 Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Combine MFCC + Chroma: 13 + 9 = 22 features
    features = np.concatenate((mfccs_mean, chroma_mean[:9]))

    return features
