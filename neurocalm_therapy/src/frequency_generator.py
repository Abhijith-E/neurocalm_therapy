"""
Therapeutic Frequency Generator
Creates healing frequencies and binaural beats using FFT manipulation
"""

import numpy as np
from scipy import signal
from typing import Tuple, List
import librosa


class TherapeuticAudioGenerator:
    """
    Generate therapeutic audio with healing frequencies
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        
        # Healing frequencies (Hz)
        self.healing_freqs = {
            'solfeggio_174': 174,   # Pain reduction
            'solfeggio_285': 285,   # Tissue healing
            'solfeggio_396': 396,   # Liberation from fear
            'solfeggio_417': 417,   # Transformation
            'solfeggio_528': 528,   # DNA repair, love
            'solfeggio_639': 639,   # Relationships
            'solfeggio_741': 741,   # Awakening intuition
            'solfeggio_852': 852,   # Spiritual order
            'solfeggio_963': 963,   # Divine consciousness
            'schumann': 7.83,       # Earth's resonance
        }
        
        # Brainwave frequencies for binaural beats
        self.brainwave_freqs = {
            'delta': (0.5, 4),      # Deep sleep, pain relief
            'theta': (4, 8),        # Deep relaxation, meditation
            'alpha': (8, 13),       # Relaxed focus, stress reduction
            'beta': (13, 30),       # Active thinking
            'gamma': (30, 100)      # Higher consciousness
        }
    
    def generate_healing_tone(self, frequency: float, duration: float,
                             amplitude: float = 0.3) -> np.ndarray:
        """
        Generate pure healing tone
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            amplitude: Amplitude (0-1)
        
        Returns:
            Audio signal
        """
        t = np.linspace(0, duration, int(duration * self.sr))
        tone = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Apply fade in/out
        fade_samples = int(0.1 * self.sr)  # 100ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
        
        return tone
    
    def generate_binaural_beat(self, base_freq: float, beat_freq: float,
                              duration: float, amplitude: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate binaural beat (stereo)
        
        Args:
            base_freq: Base frequency (e.g., 200 Hz)
            beat_freq: Beat frequency (e.g., 7 Hz for theta)
            duration: Duration in seconds
            amplitude: Amplitude (0-1)
        
        Returns:
            Tuple of (left_channel, right_channel)
        """
        t = np.linspace(0, duration, int(duration * self.sr))
        
        # Left ear: base frequency
        left = amplitude * np.sin(2 * np.pi * base_freq * t)
        
        # Right ear: base + beat frequency
        right = amplitude * np.sin(2 * np.pi * (base_freq + beat_freq) * t)
        
        # Apply envelope
        fade_samples = int(0.5 * self.sr)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        left[:fade_samples] *= fade_in
        left[-fade_samples:] *= fade_out
        right[:fade_samples] *= fade_in
        right[-fade_samples:] *= fade_out
        
        return left, right
    
    def transform_voice_to_healing_freq(self, audio: np.ndarray, 
                                        target_freq: float = 528) -> np.ndarray:
        """
        Transform voice to healing frequency using FFT
        
        Args:
            audio: Input audio
            target_freq: Target healing frequency
        
        Returns:
            Transformed audio
        """
        # Apply FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        # Get magnitude and phase
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Find fundamental frequency
        fundamental_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
        fundamental_freq = freqs[fundamental_idx]
        
        if fundamental_freq == 0:
            return audio
        
        # Calculate frequency shift ratio
        shift_ratio = target_freq / fundamental_freq
        
        # Shift frequencies
        new_freqs = freqs * shift_ratio
        
        # Interpolate magnitude to new frequencies
        new_magnitude = np.interp(freqs, new_freqs, magnitude)
        
        # Reconstruct signal
        new_fft = new_magnitude * np.exp(1j * phase)
        transformed = np.fft.irfft(new_fft, n=len(audio))
        
        # Normalize
        transformed = transformed / np.max(np.abs(transformed)) * 0.7
        
        return transformed
    
    def create_harmonic_series(self, fundamental: float, num_harmonics: int,
                              duration: float) -> np.ndarray:
        """
        Create harmonic series based on fundamental frequency
        
        Args:
            fundamental: Fundamental frequency
            num_harmonics: Number of harmonics
            duration: Duration in seconds
        
        Returns:
            Audio with harmonic series
        """
        t = np.linspace(0, duration, int(duration * self.sr))
        audio = np.zeros(len(t))
        
        for n in range(1, num_harmonics + 1):
            freq = fundamental * n
            amplitude = 1.0 / n  # Decreasing amplitude
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.5
        
        return audio
    
    def add_pink_noise(self, audio: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        Add pink noise (1/f noise) for natural sound
        
        Args:
            audio: Input audio
            noise_level: Noise amplitude
        
        Returns:
            Audio with pink noise
        """
        # Generate white noise
        white_noise = np.random.randn(len(audio))
        
        # Apply 1/f filter in frequency domain
        fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(len(white_noise), 1/self.sr)
        
        # Create 1/f^0.5 filter (pink noise)
        with np.errstate(divide='ignore', invalid='ignore'):
            filter_response = 1.0 / np.sqrt(freqs)
            filter_response[0] = 0  # Remove DC
        
        # Apply filter
        pink_fft = fft * filter_response
        pink_noise = np.fft.irfft(pink_fft, n=len(audio))
        
        # Normalize and scale
        pink_noise = pink_noise / np.max(np.abs(pink_noise)) * noise_level
        
        return audio + pink_noise
    
    def create_breathing_guide(self, bpm: int, duration: float,
                               inhale_ratio: float = 0.4,
                               hold_ratio: float = 0.1) -> np.ndarray:
        """
        Create audio breathing guide
        
        Args:
            bpm: Breaths per minute
            duration: Duration in seconds
            inhale_ratio: Ratio of breath for inhale
            hold_ratio: Ratio of breath for hold
        
        Returns:
            Audio breathing guide
        """
        breath_duration = 60.0 / bpm
        num_breaths = int(duration / breath_duration)
        
        samples_per_breath = int(breath_duration * self.sr)
        audio = np.zeros(samples_per_breath * num_breaths)
        
        # Phase durations
        inhale_samples = int(inhale_ratio * samples_per_breath)
        hold_samples = int(hold_ratio * samples_per_breath)
        exhale_samples = samples_per_breath - inhale_samples - hold_samples
        
        for i in range(num_breaths):
            start = i * samples_per_breath
            
            # Inhale (rising tone)
            t_inhale = np.linspace(0, 1, inhale_samples)
            freq_inhale = 200 + 100 * t_inhale
            audio[start:start+inhale_samples] = 0.2 * np.sin(
                2 * np.pi * freq_inhale * t_inhale
            )
            
            # Hold (constant tone)
            t_hold = np.linspace(0, hold_ratio * breath_duration, hold_samples)
            audio[start+inhale_samples:start+inhale_samples+hold_samples] = \
                0.1 * np.sin(2 * np.pi * 300 * t_hold)
            
            # Exhale (falling tone)
            t_exhale = np.linspace(0, 1, exhale_samples)
            freq_exhale = 300 - 100 * t_exhale
            audio[start+inhale_samples+hold_samples:start+samples_per_breath] = \
                0.2 * np.sin(2 * np.pi * freq_exhale * t_exhale)
        
        return audio
    
    def generate_isochronic_tones(self, frequency: float, pulse_freq: float,
                                  duration: float) -> np.ndarray:
        """
        Generate isochronic tones (pulsed tones for brainwave entrainment)
        
        Args:
            frequency: Carrier frequency
            pulse_freq: Pulse frequency (target brainwave)
            duration: Duration in seconds
        
        Returns:
            Isochronic tone audio
        """
        t = np.linspace(0, duration, int(duration * self.sr))
        
        # Carrier tone
        carrier = np.sin(2 * np.pi * frequency * t)
        
        # Pulse envelope
        pulse = 0.5 * (1 + np.sin(2 * np.pi * pulse_freq * t))
        
        # Combine
        audio = 0.3 * carrier * pulse
        
        return audio
    
    def create_therapeutic_session(self, base_audio: np.ndarray,
                                   session_type: str = 'stress_relief') -> np.ndarray:
        """
        Create complete therapeutic audio session
        
        Args:
            base_audio: Base audio (e.g., transformed voice)
            session_type: Type of session
        
        Returns:
            Complete therapeutic audio
        """
        duration = len(base_audio) / self.sr
        
        if session_type == 'stress_relief':
            # Alpha waves for relaxation
            left, right = self.generate_binaural_beat(200, 10, duration)
            binaural = (left + right) / 2
            
            # Add 528 Hz healing frequency
            healing = self.generate_healing_tone(528, duration, amplitude=0.2)
            
            # Combine
            therapeutic = 0.5 * base_audio + 0.3 * binaural + 0.2 * healing
            
        elif session_type == 'deep_relaxation':
            # Theta waves
            left, right = self.generate_binaural_beat(150, 6, duration)
            binaural = (left + right) / 2
            
            # Schumann resonance
            schumann = self.generate_healing_tone(7.83, duration, amplitude=0.15)
            
            therapeutic = 0.4 * base_audio + 0.4 * binaural + 0.2 * schumann
            
        elif session_type == 'headache_relief':
            # Delta waves for pain relief
            left, right = self.generate_binaural_beat(100, 2, duration)
            binaural = (left + right) / 2
            
            # 174 Hz for pain
            pain_relief = self.generate_healing_tone(174, duration, amplitude=0.2)
            
            therapeutic = 0.4 * base_audio + 0.4 * binaural + 0.2 * pain_relief
            
        else:
            therapeutic = base_audio
        
        # Add subtle pink noise for naturalness
        therapeutic = self.add_pink_noise(therapeutic, noise_level=0.02)
        
        # Normalize
        therapeutic = therapeutic / np.max(np.abs(therapeutic)) * 0.7
        
        return therapeutic