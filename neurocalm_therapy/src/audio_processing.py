"""
Audio Processing Module - Feature Extraction using DSP Techniques
Implements: FFT, LPC, Time-Domain Features
"""

import numpy as np
import librosa
from scipy import signal
from scipy.linalg import toeplitz
from typing import Dict, Tuple
import python_speech_features as psf


class AudioProcessor:
    """Comprehensive audio processing and feature extraction"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = int(0.025 * sr)  # 25ms
        self.hop_length = int(0.010 * sr)     # 10ms
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all features for stress detection
        
        Args:
            audio: Input audio signal
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # FFT-based features
        features['spectral_centroid'] = self.extract_spectral_centroid(audio)
        features['spectral_rolloff'] = self.extract_spectral_rolloff(audio)
        features['spectral_flux'] = self.extract_spectral_flux(audio)
        features['mfcc'] = self.extract_mfcc(audio)
        
        # LPC features
        features['lpc_coeffs'] = self.extract_lpc_coefficients(audio)
        features['formants'] = self.estimate_formants(audio)
        
        # Time-domain features
        features['zcr'] = self.extract_zero_crossing_rate(audio)
        features['energy'] = self.extract_energy(audio)
        features['pitch'] = self.extract_pitch(audio)
        
        # Voice quality features
        features['jitter'] = self.calculate_jitter(audio)
        features['shimmer'] = self.calculate_shimmer(audio)
        
        return features
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral centroid using FFT
        Indicates the "center of mass" of the spectrum
        """
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        return centroid
    
    def extract_spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral rolloff - frequency below which 85% of energy is contained
        """
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]
        return rolloff
    
    def extract_spectral_flux(self, audio: np.ndarray) -> np.ndarray:
        """
        Calculate spectral flux - measure of spectral change
        """
        # Compute spectrogram
        S = np.abs(librosa.stft(audio, n_fft=self.frame_length, 
                                hop_length=self.hop_length))
        
        # Calculate flux
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        # Pad to match frame count
        flux = np.pad(flux, (1, 0), mode='edge')
        
        return flux
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract Mel-Frequency Cepstral Coefficients
        """
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )
        return mfcc
    
    def extract_lpc_coefficients(self, audio: np.ndarray, order: int = 12) -> np.ndarray:
        """
        Extract Linear Predictive Coding coefficients
        Models vocal tract resonance
        
        Args:
            audio: Input audio
            order: LPC order (typically 12-16 for speech)
        
        Returns:
            LPC coefficients for each frame
        """
        # Frame the signal
        frames = librosa.util.frame(audio, frame_length=self.frame_length, 
                                    hop_length=self.hop_length)
        
        lpc_coeffs = []
        
        for frame in frames.T:
            # Apply window
            windowed = frame * signal.windows.hamming(len(frame))
            
            # Calculate autocorrelation
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Solve Yule-Walker equations using Levinson-Durbin
            if len(autocorr) > order:
                r = autocorr[:order+1]
                # Levinson-Durbin recursion
                lpc = self._levinson_durbin(r, order)
                lpc_coeffs.append(lpc)
            else:
                lpc_coeffs.append(np.zeros(order))
        
        return np.array(lpc_coeffs).T
    
    def _levinson_durbin(self, r: np.ndarray, order: int) -> np.ndarray:
        """
        Levinson-Durbin recursion for LPC coefficient calculation
        """
        a = np.zeros(order + 1)
        e = np.zeros(order + 1)
        
        a[0] = 1.0
        e[0] = r[0]
        
        for i in range(1, order + 1):
            lambda_i = -np.sum(a[:i] * r[i:0:-1]) / e[i-1]
            a[1:i+1] = a[:i][::-1] * lambda_i
            a[i] = lambda_i
            e[i] = (1 - lambda_i**2) * e[i-1]
        
        return a[1:]  # Return coefficients without a[0]
    
    def estimate_formants(self, audio: np.ndarray, num_formants: int = 3) -> np.ndarray:
        """
        Estimate formant frequencies from LPC coefficients
        """
        lpc_coeffs = self.extract_lpc_coefficients(audio, order=12)
        
        formants_list = []
        
        for coeffs in lpc_coeffs.T:
            # Find roots of LPC polynomial
            roots = np.roots(np.concatenate([[1], -coeffs]))
            
            # Keep roots inside unit circle
            roots = roots[np.abs(roots) < 1]
            
            # Convert to frequencies
            angles = np.angle(roots)
            freqs = angles * (self.sr / (2 * np.pi))
            
            # Keep positive frequencies and sort
            freqs = np.sort(freqs[freqs > 0])
            
            # Pad if not enough formants
            if len(freqs) < num_formants:
                freqs = np.pad(freqs, (0, num_formants - len(freqs)), 
                              mode='constant', constant_values=0)
            
            formants_list.append(freqs[:num_formants])
        
        return np.array(formants_list).T
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract zero crossing rate - indicates noisiness
        """
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.frame_length, 
            hop_length=self.hop_length
        )[0]
        return zcr
    
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract frame energy (RMS)
        """
        energy = librosa.feature.rms(
            y=audio, frame_length=self.frame_length, 
            hop_length=self.hop_length
        )[0]
        return energy
    
    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch contour using autocorrelation
        """
        # Use librosa's pyin for robust pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # Replace NaN with 0
        f0 = np.nan_to_num(f0)
        
        return f0
    
    def calculate_jitter(self, audio: np.ndarray) -> float:
        """
        Calculate jitter (pitch period variation)
        Indicator of voice quality and stress
        """
        f0 = self.extract_pitch(audio)
        f0 = f0[f0 > 0]  # Remove unvoiced frames
        
        if len(f0) < 2:
            return 0.0
        
        # Period jitter
        periods = 1.0 / f0
        period_diff = np.abs(np.diff(periods))
        jitter = np.mean(period_diff) / np.mean(periods) * 100
        
        return jitter
    
    def calculate_shimmer(self, audio: np.ndarray) -> float:
        """
        Calculate shimmer (amplitude variation)
        Indicator of voice quality and stress
        """
        energy = self.extract_energy(audio)
        energy = energy[energy > 0]
        
        if len(energy) < 2:
            return 0.0
        
        # Amplitude shimmer
        energy_diff = np.abs(np.diff(energy))
        shimmer = np.mean(energy_diff) / np.mean(energy) * 100
        
        return shimmer
    
    def compute_feature_vector(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute comprehensive feature vector for classification
        
        Returns:
            Feature vector with statistics of all features
        """
        features = self.extract_all_features(audio)
        
        feature_vector = []
        
        # Statistical features for each
        for key, value in features.items():
            if key in ['mfcc', 'lpc_coeffs', 'formants']:
                # For multi-dimensional features, compute stats per dimension
                for dim in value:
                    feature_vector.extend([
                        np.mean(dim),
                        np.std(dim),
                        np.max(dim),
                        np.min(dim)
                    ])
            elif key in ['jitter', 'shimmer']:
                # Scalar values
                feature_vector.append(value)
            else:
                # For 1D time series
                feature_vector.extend([
                    np.mean(value),
                    np.std(value),
                    np.max(value),
                    np.min(value),
                    np.percentile(value, 25),
                    np.percentile(value, 75)
                ])
        
        return np.array(feature_vector)
    
    def apply_fft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Fast Fourier Transform
        
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        frequencies = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        return frequencies, magnitude
    
    def apply_inverse_fft(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Apply Inverse FFT to reconstruct signal
        
        Args:
            magnitude: Magnitude spectrum
            phase: Phase spectrum
        
        Returns:
            Reconstructed time-domain signal
        """
        complex_spectrum = magnitude * np.exp(1j * phase)
        audio = np.fft.irfft(complex_spectrum)
        
        return audio