"""
Stress Detection Module using Hidden Markov Models (HMM) and Viterbi Algorithm
"""

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib
from typing import List, Tuple, Dict
from dtaidistance import dtw
import warnings
warnings.filterwarnings('ignore')


class StressDetector:
    """
    Stress detection using HMM with Viterbi decoding
    States: Low Stress, Medium Stress, High Stress
    """
    
    def __init__(self, n_states: int = 3):
        """
        Initialize stress detector
        
        Args:
            n_states: Number of stress states (default: 3 for low/med/high)
        """
        self.n_states = n_states
        self.state_names = ['Low Stress', 'Medium Stress', 'High Stress']
        self.model = None
        self.scaler = StandardScaler()
        self.feature_dim = None
        
    def train(self, features: List[np.ndarray], labels: List[str]):
        """
        Train HMM on feature sequences
        
        Args:
            features: List of feature sequences (each is T x D)
            labels: List of stress labels
        """
        print("Training Stress Detection HMM...")
        
        # Concatenate all features for scaling
        all_features = np.vstack(features)
        self.feature_dim = all_features.shape[1]
        
        # Fit scaler
        self.scaler.fit(all_features)
        
        # Normalize features
        normalized_features = [self.scaler.transform(f) for f in features]
        
        # Create lengths array
        lengths = [len(f) for f in normalized_features]
        
        # Concatenate for HMM training
        X = np.vstack(normalized_features)
        
        # Initialize HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag',
            n_iter=100,
            random_state=42
        )
        
        # Train
        self.model.fit(X, lengths)
        
        print(f"✓ HMM trained with {self.n_states} states")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Training samples: {len(features)}")
        
        # Print learned parameters
        print("\nLearned HMM Parameters:")
        print("Transition Matrix:")
        print(self.model.transmat_)
        print("\nInitial State Probabilities:")
        print(self.model.startprob_)
    
    def predict_stress_level(self, features: np.ndarray) -> Tuple[str, np.ndarray, float]:
        """
        Predict stress level using Viterbi algorithm
        
        Args:
            features: Feature sequence (T x D)
        
        Returns:
            Tuple of (stress_label, state_sequence, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Normalize features
        features_norm = self.scaler.transform(features)
        
        # Viterbi decoding to find most likely state sequence
        log_prob, state_sequence = self.model.decode(
            features_norm, 
            algorithm='viterbi'
        )
        
        # Get most common state
        unique, counts = np.unique(state_sequence, return_counts=True)
        dominant_state = unique[np.argmax(counts)]
        
        # Calculate confidence as proportion of dominant state
        confidence = np.max(counts) / len(state_sequence)
        
        # Map state to label
        stress_label = self.state_names[dominant_state]
        
        return stress_label, state_sequence, confidence
    
    def get_state_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Get posterior probabilities of each state over time
        
        Args:
            features: Feature sequence (T x D)
        
        Returns:
            State probabilities (T x n_states)
        """
        features_norm = self.scaler.transform(features)
        
        # Forward-backward algorithm
        log_prob, posteriors = self.model.score_samples(features_norm)
        
        return posteriors
    
    def calculate_stress_score(self, features: np.ndarray) -> float:
        """
        Calculate continuous stress score (0-100)
        
        Args:
            features: Feature sequence
        
        Returns:
            Stress score
        """
        posteriors = self.get_state_probabilities(features)
        
        # Weight states: 0 (low), 50 (medium), 100 (high)
        state_weights = np.array([0, 50, 100])
        
        # Calculate weighted average over time
        stress_scores = np.dot(posteriors, state_weights)
        avg_stress = np.mean(stress_scores)
        
        return avg_stress
    
    def detect_stress_transitions(self, features: np.ndarray) -> List[Tuple[int, str, str]]:
        """
        Detect transitions between stress states
        
        Args:
            features: Feature sequence
        
        Returns:
            List of (frame_index, from_state, to_state)
        """
        _, state_sequence, _ = self.predict_stress_level(features)
        
        transitions = []
        for i in range(1, len(state_sequence)):
            if state_sequence[i] != state_sequence[i-1]:
                from_state = self.state_names[state_sequence[i-1]]
                to_state = self.state_names[state_sequence[i]]
                transitions.append((i, from_state, to_state))
        
        return transitions
    
    def compare_stress_patterns(self, features1: np.ndarray, 
                               features2: np.ndarray) -> float:
        """
        Compare two stress patterns using Dynamic Time Warping (DTW)
        
        Args:
            features1: First feature sequence
            features2: Second feature sequence
        
        Returns:
            DTW distance (lower = more similar)
        """
        # Get state sequences
        _, seq1, _ = self.predict_stress_level(features1)
        _, seq2, _ = self.predict_stress_level(features2)
        
        # Convert to float for DTW
        seq1 = seq1.astype(float)
        seq2 = seq2.astype(float)
        
        # Calculate DTW distance
        distance = dtw.distance(seq1, seq2)
        
        return distance
    
    def analyze_stress_progression(self, features: np.ndarray, 
                                   window_size: int = 10) -> Dict:
        """
        Analyze how stress changes over time
        
        Args:
            features: Feature sequence
            window_size: Number of frames per window
        
        Returns:
            Dictionary with analysis results
        """
        posteriors = self.get_state_probabilities(features)
        
        # Calculate stress over windows
        n_windows = len(posteriors) // window_size
        window_stress = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_post = posteriors[start:end]
            
            # Average stress in window
            state_weights = np.array([0, 50, 100])
            avg_stress = np.mean(np.dot(window_post, state_weights))
            window_stress.append(avg_stress)
        
        # Calculate trend
        if len(window_stress) > 1:
            trend = np.polyfit(range(len(window_stress)), window_stress, 1)[0]
        else:
            trend = 0
        
        analysis = {
            'window_stress_scores': window_stress,
            'overall_mean': np.mean(window_stress),
            'overall_std': np.std(window_stress),
            'trend': trend,  # Positive = increasing stress
            'max_stress': np.max(window_stress),
            'min_stress': np.min(window_stress)
        }
        
        return analysis
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'hmm_model': self.model,
            'scaler': self.scaler,
            'feature_dim': self.feature_dim,
            'n_states': self.n_states,
            'state_names': self.state_names
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['hmm_model']
        self.scaler = model_data['scaler']
        self.feature_dim = model_data['feature_dim']
        self.n_states = model_data['n_states']
        self.state_names = model_data['state_names']
        print(f"✓ Model loaded from {filepath}")


class TemporalStressAnalyzer:
    """
    Advanced temporal analysis using DTW for stress pattern matching
    """
    
    def __init__(self):
        self.reference_patterns = {}
    
    def add_reference_pattern(self, name: str, features: np.ndarray, 
                             label: str):
        """
        Add a reference stress pattern
        
        Args:
            name: Pattern name
            features: Feature sequence
            label: Stress label
        """
        self.reference_patterns[name] = {
            'features': features,
            'label': label
        }
    
    def find_similar_patterns(self, query_features: np.ndarray, 
                             top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find most similar reference patterns using DTW
        
        Args:
            query_features: Query feature sequence
            top_k: Number of top matches to return
        
        Returns:
            List of (pattern_name, distance) sorted by similarity
        """
        distances = []
        
        for name, pattern_data in self.reference_patterns.items():
            ref_features = pattern_data['features']
            
            # Use DTW to compare feature sequences
            distance = self._compute_dtw_distance(query_features, ref_features)
            distances.append((name, distance))
        
        # Sort by distance (lower is better)
        distances.sort(key=lambda x: x[1])
        
        return distances[:top_k]
    
    def _compute_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Compute DTW distance between two sequences
        
        Args:
            seq1: First sequence
            seq2: Second sequence
        
        Returns:
            DTW distance
        """
        # If multidimensional, use first dimension or compute average
        if seq1.ndim > 1:
            seq1 = np.mean(seq1, axis=1)
        if seq2.ndim > 1:
            seq2 = np.mean(seq2, axis=1)
        
        distance = dtw.distance(seq1, seq2)
        return distance
    
    def align_sequences(self, seq1: np.ndarray, seq2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two sequences using DTW
        
        Args:
            seq1: First sequence
            seq2: Second sequence
        
        Returns:
            Tuple of aligned sequences
        """
        if seq1.ndim > 1:
            seq1_1d = np.mean(seq1, axis=1)
        else:
            seq1_1d = seq1
            
        if seq2.ndim > 1:
            seq2_1d = np.mean(seq2, axis=1)
        else:
            seq2_1d = seq2
        
        # Compute DTW path
        path = dtw.warping_path(seq1_1d, seq2_1d)
        
        # Create aligned sequences
        aligned_seq1 = seq1[np.array([p[0] for p in path])]
        aligned_seq2 = seq2[np.array([p[1] for p in path])]
        
        return aligned_seq1, aligned_seq2