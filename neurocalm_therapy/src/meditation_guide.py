"""
Speech-Guided Meditation Module (FIXED)
Uses Viterbi algorithm for command recognition
"""

import numpy as np
from hmmlearn import hmm
from typing import List, Dict, Tuple
import python_speech_features as psf
import librosa


class MeditationGuide:
    """
    Interactive meditation guide with voice command recognition
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.commands = [
            'start',
            'pause',
            'resume',
            'deeper',
            'lighter',
            'stop'
        ]
        self.command_models = {}
        self.current_state = 'idle'
        
    def train_command_recognizer(self, training_data: Dict[str, List[np.ndarray]]):
        """
        Train HMM models for each command using Viterbi
        
        Args:
            training_data: Dict of command -> list of audio samples
        """
        print("Training meditation command recognizer...")
        
        for command in self.commands:
            if command not in training_data:
                print(f"Warning: No training data for command '{command}'")
                continue
            
            # Extract MFCC features for all samples
            features_list = []
            lengths = []
            
            for audio in training_data[command]:
                mfcc = psf.mfcc(audio, self.sr, numcep=13, 
                               nfilt=26, nfft=512, winstep=0.01)
                features_list.append(mfcc)
                lengths.append(len(mfcc))
            
            # Concatenate features
            X = np.vstack(features_list)
            
            # Train HMM for this command
            model = hmm.GaussianHMM(n_components=5, covariance_type='diag',
                                   n_iter=100, random_state=42)
            model.fit(X, lengths)
            
            self.command_models[command] = model
            
        print(f"✓ Trained recognizers for {len(self.command_models)} commands")
    
    def recognize_command(self, audio: np.ndarray) -> Tuple[str, float]:
        """
        Recognize spoken command using Viterbi decoding
        
        Args:
            audio: Audio sample
        
        Returns:
            Tuple of (recognized_command, confidence)
        """
        # Extract MFCC features
        mfcc = psf.mfcc(audio, self.sr, numcep=13, nfilt=26, 
                       nfft=512, winstep=0.01)
        
        # Score against each command model
        scores = {}
        for command, model in self.command_models.items():
            try:
                # Use Viterbi to get log likelihood
                log_prob = model.score(mfcc)
                scores[command] = log_prob
            except:
                scores[command] = -np.inf
        
        # Get best command
        best_command = max(scores, key=scores.get)
        best_score = scores[best_command]
        
        # Calculate confidence (normalize scores)
        score_values = np.array(list(scores.values()))
        score_values = score_values[np.isfinite(score_values)]
        
        if len(score_values) > 0:
            normalized_score = (best_score - score_values.min()) / \
                             (score_values.max() - score_values.min() + 1e-10)
            confidence = normalized_score
        else:
            confidence = 0.0
        
        return best_command, confidence
    
    def generate_meditation_script(self, duration: float, 
                                  focus_type: str = 'breath') -> List[Dict]:
        """
        Generate timed meditation guidance script
        
        Args:
            duration: Total duration in seconds
            focus_type: Type of meditation focus
        
        Returns:
            List of guidance prompts with timestamps
        """
        script = []
        
        if focus_type == 'breath':
            script = [
                {'time': 0, 'text': 'Find a comfortable position. Close your eyes.'},
                {'time': 10, 'text': 'Begin to notice your breath. Natural and easy.'},
                {'time': 30, 'text': 'Feel the air flowing in through your nose.'},
                {'time': 50, 'text': 'Notice the gentle rise of your chest and belly.'},
                {'time': 70, 'text': 'As you exhale, feel your body relax.'},
                {'time': 90, 'text': 'Let go of any tension with each out-breath.'},
                {'time': 120, 'text': 'If your mind wanders, gently return to the breath.'},
                {'time': 150, 'text': 'Continue breathing naturally and calmly.'},
                {'time': duration-30, 'text': 'Begin to deepen your breath slowly.'},
                {'time': duration-10, 'text': 'When ready, gently open your eyes.'}
            ]
            
        elif focus_type == 'body_scan':
            script = [
                {'time': 0, 'text': 'Lie down in a comfortable position.'},
                {'time': 15, 'text': 'Bring awareness to your feet. Notice any sensations.'},
                {'time': 45, 'text': 'Move attention to your legs. Allow them to relax.'},
                {'time': 75, 'text': 'Notice your hips and lower back. Release any tension.'},
                {'time': 105, 'text': 'Feel your chest and shoulders. Let them soften.'},
                {'time': 135, 'text': 'Bring awareness to your arms and hands.'},
                {'time': 165, 'text': 'Notice your neck and face. Relax your jaw.'},
                {'time': 195, 'text': 'Feel your whole body at once. Complete and at peace.'},
                {'time': duration-20, 'text': 'Slowly bring movement back to your fingers and toes.'},
                {'time': duration-5, 'text': 'Open your eyes when you are ready.'}
            ]
            
        elif focus_type == 'stress_relief':
            script = [
                {'time': 0, 'text': 'Settle into your space. Notice how you feel.'},
                {'time': 20, 'text': 'Acknowledge any stress without judgment.'},
                {'time': 45, 'text': 'Breathe in calm. Breathe out tension.'},
                {'time': 75, 'text': 'With each breath, feel stress melting away.'},
                {'time': 110, 'text': 'Your mind is becoming clearer and calmer.'},
                {'time': 145, 'text': 'You are safe. You are peaceful.'},
                {'time': 180, 'text': 'Carry this calm with you as you return.'},
                {'time': duration-10, 'text': 'Take three deep breaths and open your eyes.'}
            ]
        
        return script
    
    def calculate_meditation_metrics(self, audio_segments: List[np.ndarray]) -> Dict:
        """
        Calculate meditation quality metrics from breathing audio
        
        Args:
            audio_segments: List of audio segments during meditation
        
        Returns:
            Dictionary of metrics
        """
        breathing_rates = []
        calmness_scores = []
        
        for audio in audio_segments:
            # Extract features
            mfcc = psf.mfcc(audio, self.sr, numcep=13)
            
            # Estimate breathing rate from periodicity
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks (breaths)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(autocorr, distance=self.sr)
            
            if len(peaks) > 1:
                avg_peak_distance = np.mean(np.diff(peaks))
                breathing_rate = 60.0 * self.sr / avg_peak_distance  # BPM
                breathing_rates.append(breathing_rate)
            
            # Calmness from spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio, sr=self.sr
            ))
            # Lower centroid = calmer
            calmness = 1.0 - (spectral_centroid / 4000.0)
            calmness = np.clip(calmness, 0, 1)
            calmness_scores.append(calmness)
        
        metrics = {
            'avg_breathing_rate': np.mean(breathing_rates) if breathing_rates else 0,
            'breathing_rate_std': np.std(breathing_rates) if breathing_rates else 0,
            'avg_calmness': np.mean(calmness_scores) if calmness_scores else 0,
            'meditation_quality': np.mean(calmness_scores) * 100 if calmness_scores else 0
        }
        
        return metrics
    
    def adaptive_guidance(self, current_stress_level: float) -> str:
        """
        Provide adaptive guidance based on stress level
        
        Args:
            current_stress_level: Stress score (0-100)
        
        Returns:
            Guidance text
        """
        if current_stress_level > 70:
            return "You seem quite tense. Let's take some deep, slow breaths together."
        elif current_stress_level > 40:
            return "You're doing well. Continue with steady, calm breathing."
        else:
            return "Excellent. You've reached a deep state of relaxation."
    
    def generate_progress_report(self, session_data: List[Dict]) -> Dict:
        """
        Generate meditation session progress report
        
        Args:
            session_data: List of session dictionaries with timestamps and metrics
        
        Returns:
            Progress report
        """
        report = {
            'total_sessions': len(session_data),
            'total_duration': sum(s.get('duration', 0) for s in session_data),
            'avg_stress_reduction': 0,
            'consistency_score': 0,
            'improvement_trend': 0
        }
        
        if len(session_data) > 0:
            # Calculate stress reduction
            stress_reductions = []
            for session in session_data:
                if 'start_stress' in session and 'end_stress' in session:
                    reduction = session['start_stress'] - session['end_stress']
                    stress_reductions.append(reduction)
            
            if stress_reductions:
                report['avg_stress_reduction'] = np.mean(stress_reductions)
            
            # Calculate consistency (sessions per week)
            if 'timestamps' in session_data[0]:
                timestamps = [s['timestamp'] for s in session_data]
                time_span = max(timestamps) - min(timestamps)
                weeks = time_span / (7 * 24 * 3600)
                report['consistency_score'] = len(session_data) / max(weeks, 1)
            
            # Trend analysis
            if len(stress_reductions) > 2:
                trend = np.polyfit(range(len(stress_reductions)), 
                                 stress_reductions, 1)[0]
                report['improvement_trend'] = trend
        
        return report


class BreathingCoach:
    """
    Interactive breathing coach with audio feedback
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        
    def detect_breathing_pattern(self, audio: np.ndarray) -> Dict:
        """
        Detect breathing pattern from audio
        
        Args:
            audio: Audio recording of breathing
        
        Returns:
            Dictionary with breathing metrics
        """
        # Extract amplitude envelope
        from src.utils import extract_envelope
        envelope = extract_envelope(audio, self.sr)
        
        # Find inhalation and exhalation
        from scipy.signal import find_peaks
        
        # Peaks = inhalations
        peaks, _ = find_peaks(envelope, distance=self.sr, prominence=0.1)
        
        # Troughs = exhalations
        troughs, _ = find_peaks(-envelope, distance=self.sr, prominence=0.1)
        
        # Calculate metrics
        if len(peaks) > 1:
            inhale_times = np.diff(peaks) / self.sr
            avg_breath_duration = np.mean(inhale_times)
            breathing_rate = 60.0 / avg_breath_duration
        else:
            breathing_rate = 0
            avg_breath_duration = 0
        
        # Regularity
        if len(peaks) > 2:
            regularity = 1.0 - (np.std(np.diff(peaks)) / np.mean(np.diff(peaks)))
            regularity = np.clip(regularity, 0, 1)
        else:
            regularity = 0
        
        pattern = {
            'breathing_rate': breathing_rate,
            'avg_breath_duration': avg_breath_duration,
            'regularity': regularity,
            'num_breaths': len(peaks)
        }
        
        return pattern
    
    def provide_feedback(self, pattern: Dict) -> str:
        """
        Provide feedback on breathing pattern
        
        Args:
            pattern: Breathing pattern dictionary
        
        Returns:
            Feedback text
        """
        rate = pattern['breathing_rate']
        regularity = pattern['regularity']
        
        feedback = []
        
        # Rate feedback
        if rate < 8:
            feedback.append("Your breathing is very slow and deep. Excellent for relaxation.")
        elif rate < 12:
            feedback.append("Your breathing rate is ideal for meditation.")
        elif rate < 16:
            feedback.append("Your breathing is slightly fast. Try to slow it down.")
        else:
            feedback.append("Your breathing is rapid. Let's work on slowing it down together.")
        
        # Regularity feedback
        if regularity > 0.8:
            feedback.append("Your breathing rhythm is very consistent.")
        elif regularity > 0.6:
            feedback.append("Your breathing rhythm is good.")
        else:
            feedback.append("Try to maintain a more consistent rhythm.")
        
        return " ".join(feedback)