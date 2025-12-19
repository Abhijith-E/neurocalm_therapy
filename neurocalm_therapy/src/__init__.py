"""
NeuroCalm Voice Therapy System
Speech-Based Therapeutic System for Stress Relief and Headache Management

Core modules:
- audio_processing: FFT, LPC, feature extraction
- stress_detector: HMM, Viterbi, DTW for stress detection
- frequency_generator: Healing frequencies and binaural beats
- meditation_guide: Voice recognition and meditation guidance
- utils: Helper functions and data generation
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .audio_processing import AudioProcessor
from .stress_detector import StressDetector, TemporalStressAnalyzer
from .frequency_generator import TherapeuticAudioGenerator
from .meditation_guide import MeditationGuide, BreathingCoach
from .utils import (
    create_directories,
    normalize_audio,
    generate_synthetic_data,
    save_audio,
    load_audio
)

__all__ = [
    'AudioProcessor',
    'StressDetector',
    'TemporalStressAnalyzer',
    'TherapeuticAudioGenerator',
    'MeditationGuide',
    'BreathingCoach',
    'create_directories',
    'normalize_audio',
    'generate_synthetic_data',
    'save_audio',
    'load_audio'
]