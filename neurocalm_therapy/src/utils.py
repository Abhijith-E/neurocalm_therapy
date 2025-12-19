"""
Utility functions for NeuroCalm Voice Therapy System
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Tuple, List
import os


def create_directories():
    """Create necessary project directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/models',
        'output/therapeutic_audio',
        'output/reports'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ Directory structure created")


def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target RMS level in dB
    
    Args:
        audio: Input audio signal
        target_level: Target RMS level in dB
    
    Returns:
        Normalized audio
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    
    # Convert target dB to linear scale
    target_linear = 10 ** (target_level / 20.0)
    gain = target_linear / rms
    
    return audio * gain


def time_align_signals(reference: np.ndarray, signal_to_align: np.ndarray) -> np.ndarray:
    """
    Align signal to reference using cross-correlation
    
    Args:
        reference: Reference signal
        signal_to_align: Signal to be aligned
    
    Returns:
        Time-aligned signal
    """
    correlation = signal.correlate(reference, signal_to_align, mode='full')
    lag = correlation.argmax() - (len(signal_to_align) - 1)
    
    if lag > 0:
        aligned = np.pad(signal_to_align, (lag, 0), mode='constant')[:-lag]
    elif lag < 0:
        aligned = np.pad(signal_to_align, (0, -lag), mode='constant')[-lag:]
    else:
        aligned = signal_to_align
    
    # Ensure same length
    min_len = min(len(reference), len(aligned))
    return aligned[:min_len]


def apply_smoothing(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing
    
    Args:
        data: Input data
        window_size: Size of smoothing window
    
    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return data
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')


def extract_envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract amplitude envelope of audio signal
    
    Args:
        audio: Input audio signal
        sr: Sample rate
    
    Returns:
        Amplitude envelope
    """
    # Hilbert transform for analytic signal
    analytic = signal.hilbert(audio)
    envelope = np.abs(analytic)
    
    # Smooth the envelope
    return apply_smoothing(envelope, window_size=int(sr * 0.01))


def calculate_snr(signal_audio: np.ndarray, noise_audio: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB
    
    Args:
        signal_audio: Clean signal
        noise_audio: Noise signal
    
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal_audio**2)
    noise_power = np.mean(noise_audio**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def segment_audio(audio: np.ndarray, sr: int, segment_duration: float = 3.0) -> List[np.ndarray]:
    """
    Segment audio into fixed-duration chunks
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        segment_duration: Duration of each segment in seconds
    
    Returns:
        List of audio segments
    """
    segment_length = int(segment_duration * sr)
    segments = []
    
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    
    return segments


def detect_voice_activity(audio: np.ndarray, sr: int, 
                         threshold: float = 0.02) -> np.ndarray:
    """
    Simple voice activity detection
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        threshold: Energy threshold
    
    Returns:
        Boolean array indicating voice activity
    """
    # Calculate frame energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    frames = librosa.util.frame(audio, frame_length=frame_length, 
                               hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    
    # Normalize energy
    energy = energy / np.max(energy) if np.max(energy) > 0 else energy
    
    # Threshold
    vad = energy > threshold
    
    return vad


def generate_synthetic_data(num_samples: int = 100, duration: float = 5.0, 
                          sr: int = 22050) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate synthetic voice data with stress labels for testing
    
    Args:
        num_samples: Number of samples to generate
        duration: Duration of each sample
        sr: Sample rate
    
    Returns:
        Tuple of (audio_samples, labels)
    """
    audio_samples = []
    labels = []
    
    t = np.linspace(0, duration, int(duration * sr))
    
    for i in range(num_samples):
        # Randomly choose stress level
        stress_level = np.random.choice(['low', 'medium', 'high'])
        
        if stress_level == 'low':
            # Low stress: low pitch, slow rate
            freq = 100 + np.random.randn() * 10
            noise_level = 0.02
        elif stress_level == 'medium':
            # Medium stress: medium pitch, medium rate
            freq = 150 + np.random.randn() * 15
            noise_level = 0.05
        else:
            # High stress: high pitch, fast rate, more variance
            freq = 220 + np.random.randn() * 20
            noise_level = 0.1
        
        # Generate signal
        audio = np.sin(2 * np.pi * freq * t)
        
        # Add harmonics
        audio += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        audio += 0.2 * np.sin(2 * np.pi * 3 * freq * t)
        
        # Add noise
        audio += np.random.randn(len(audio)) * noise_level
        
        # Add envelope
        envelope = np.exp(-t / (duration * 0.3))
        audio *= envelope
        
        # Normalize
        audio = normalize_audio(audio)
        
        audio_samples.append(audio)
        labels.append(stress_level)
    
    return audio_samples, labels


def save_audio(audio: np.ndarray, sr: int, filepath: str):
    """
    Save audio to file
    
    Args:
        audio: Audio signal
        sr: Sample rate
        filepath: Output file path
    """
    sf.write(filepath, audio, sr)
    print(f"✓ Audio saved to {filepath}")


def load_audio(filepath: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio from file
    
    Args:
        filepath: Input file path
        sr: Target sample rate
    
    Returns:
        Tuple of (audio, sample_rate)
    """
    audio, original_sr = librosa.load(filepath, sr=sr)
    return audio, sr