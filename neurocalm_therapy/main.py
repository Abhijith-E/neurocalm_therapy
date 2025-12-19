"""
NeuroCalm Voice Therapy System - Main Application (FIXED)
Complete therapeutic system for stress relief and headache management
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.utils import (create_directories, normalize_audio, 
                       generate_synthetic_data, save_audio)
from src.audio_processing import AudioProcessor
from src.stress_detector import StressDetector, TemporalStressAnalyzer
from src.frequency_generator import TherapeuticAudioGenerator
from src.meditation_guide import MeditationGuide, BreathingCoach


class NeuroCalm:
    """
    Main NeuroCalm Voice Therapy System
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.audio_processor = AudioProcessor(sr)
        self.stress_detector = StressDetector(n_states=3)
        self.frequency_generator = TherapeuticAudioGenerator(sr)
        self.meditation_guide = MeditationGuide(sr)
        self.breathing_coach = BreathingCoach(sr)
        self.temporal_analyzer = TemporalStressAnalyzer()
        
        # Create directories
        create_directories()
        
        print("=" * 70)
        print("🧠 NeuroCalm Voice Therapy System Initialized")
        print("=" * 70)
    
    def train_system(self, num_samples: int = 100):
        """
        Train the system with synthetic data
        
        Args:
            num_samples: Number of training samples
        """
        print("\n📚 Training System...")
        print("-" * 70)
        
        # Generate synthetic training data
        print("\n1. Generating synthetic voice data...")
        audio_samples, labels = generate_synthetic_data(
            num_samples=num_samples, 
            duration=5.0, 
            sr=self.sr
        )
        print(f"   ✓ Generated {num_samples} samples")
        
        # Extract features
        print("\n2. Extracting audio features...")
        features_list = []
        for i, audio in enumerate(audio_samples):
            features = self.audio_processor.extract_all_features(audio)
            # Create feature matrix (time x features)
            feature_matrix = self.audio_processor.compute_feature_vector(audio)
            # Reshape to sequence
            seq_features = np.tile(feature_matrix, (10, 1))  # 10 frames
            features_list.append(seq_features)
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i+1}/{num_samples} samples")
        
        print(f"   ✓ Extracted features from all samples")
        
        # Train stress detector
        print("\n3. Training stress detection model (HMM + Viterbi)...")
        self.stress_detector.train(features_list, labels)
        
        # Save model
        model_path = 'data/models/stress_detector.pkl'
        self.stress_detector.save_model(model_path)
        
        print("\n✓ System training complete!")
        print("=" * 70)
    
    def analyze_voice(self, audio: np.ndarray, verbose: bool = True) -> dict:
        """
        Comprehensive voice analysis
        
        Args:
            audio: Input audio
            verbose: Print detailed results
        
        Returns:
            Analysis results dictionary
        """
        if verbose:
            print("\n🔍 Analyzing Voice...")
            print("-" * 70)
        
        # Extract features
        features = self.audio_processor.extract_all_features(audio)
        
        # Create feature sequence
        feature_vector = self.audio_processor.compute_feature_vector(audio)
        feature_seq = np.tile(feature_vector, (10, 1))
        
        # Detect stress level
        stress_label, state_seq, confidence = self.stress_detector.predict_stress_level(
            feature_seq
        )
        stress_score = self.stress_detector.calculate_stress_score(feature_seq)
        
        # Analyze progression
        progression = self.stress_detector.analyze_stress_progression(feature_seq)
        
        results = {
            'stress_level': stress_label,
            'stress_score': stress_score,
            'confidence': confidence,
            'features': features,
            'progression': progression
        }
        
        if verbose:
            print(f"\n📊 Results:")
            print(f"   Stress Level: {stress_label}")
            print(f"   Stress Score: {stress_score:.1f}/100")
            print(f"   Confidence: {confidence*100:.1f}%")
            print(f"   Trend: {'Increasing' if progression['trend'] > 0 else 'Decreasing'}")
            print("\n   Key Features:")
            print(f"   - Average Pitch: {np.mean(features['pitch']):.1f} Hz")
            print(f"   - Jitter: {features['jitter']:.3f}%")
            print(f"   - Shimmer: {features['shimmer']:.3f}%")
            print(f"   - Energy: {np.mean(features['energy']):.4f}")
        
        return results
    
    def generate_therapeutic_audio(self, input_audio: np.ndarray,
                                   session_type: str = 'stress_relief',
                                   save_path: str = None) -> np.ndarray:
        """
        Generate personalized therapeutic audio
        
        Args:
            input_audio: User's voice input
            session_type: Type of therapy session
            save_path: Path to save output
        
        Returns:
            Therapeutic audio
        """
        print(f"\n🎵 Generating Therapeutic Audio ({session_type})...")
        print("-" * 70)
        
        # Transform to healing frequency
        print("   1. Transforming voice to 528 Hz (healing frequency)...")
        healed_voice = self.frequency_generator.transform_voice_to_healing_freq(
            input_audio, target_freq=528
        )
        
        # Create therapeutic session
        print("   2. Adding binaural beats and therapeutic frequencies...")
        therapeutic = self.frequency_generator.create_therapeutic_session(
            healed_voice, session_type=session_type
        )
        
        # Normalize
        therapeutic = normalize_audio(therapeutic)
        
        # Save if path provided
        if save_path:
            save_audio(therapeutic, self.sr, save_path)
        
        print("   ✓ Therapeutic audio generated successfully!")
        
        return therapeutic
    
    def run_meditation_session(self, duration: float = 300,
                               focus_type: str = 'breath') -> dict:
        """
        Run guided meditation session
        
        Args:
            duration: Session duration in seconds
            focus_type: Type of meditation
        
        Returns:
            Session results
        """
        print(f"\n🧘 Starting {duration/60:.0f}-Minute Meditation Session")
        print(f"   Focus: {focus_type.replace('_', ' ').title()}")
        print("-" * 70)
        
        # Generate script
        script = self.meditation_guide.generate_meditation_script(
            duration, focus_type
        )
        
        print(f"\n   Session includes {len(script)} guidance prompts")
        print("\n   Preview:")
        for prompt in script[:3]:
            print(f"   - [{prompt['time']}s] {prompt['text']}")
        print("   ...")
        
        # Simulate session
        session_data = {
            'duration': duration,
            'focus_type': focus_type,
            'script': script,
            'timestamp': datetime.now().timestamp(),
            'completed': True
        }
        
        print(f"\n   ✓ Session structure prepared")
        
        return session_data
    
    def create_complete_therapy_program(self, user_audio: np.ndarray,
                                       output_dir: str = 'output/therapeutic_audio'):
        """
        Create complete therapeutic program
        
        Args:
            user_audio: User's voice sample
            output_dir: Output directory
        """
        print("\n" + "=" * 70)
        print("🌟 Creating Complete Therapeutic Program")
        print("=" * 70)
        
        # Step 1: Analyze current state
        print("\n[Step 1/4] Analyzing current stress state...")
        analysis = self.analyze_voice(user_audio, verbose=False)
        print(f"   Current stress level: {analysis['stress_level']}")
        print(f"   Stress score: {analysis['stress_score']:.1f}/100")
        
        # Step 2: Generate therapeutic sessions
        print("\n[Step 2/4] Generating therapeutic audio sessions...")
        
        sessions = {
            'stress_relief': 'Stress Relief & Relaxation',
            'deep_relaxation': 'Deep Relaxation & Meditation',
            'headache_relief': 'Headache & Pain Relief'
        }
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for session_type, description in sessions.items():
            print(f"\n   Creating: {description}")
            therapeutic = self.generate_therapeutic_audio(
                user_audio,
                session_type=session_type,
                save_path=f"{output_dir}/{session_type}.wav"
            )
        
        # Step 3: Create breathing guide
        print("\n[Step 3/4] Creating breathing guidance audio...")
        breathing_guide = self.frequency_generator.create_breathing_guide(
            bpm=6,  # 6 breaths per minute for deep relaxation
            duration=300  # 5 minutes
        )
        save_audio(breathing_guide, self.sr, 
                  f"{output_dir}/breathing_guide.wav")
        
        # Step 4: Generate meditation scripts
        print("\n[Step 4/4] Generating meditation session scripts...")
        meditation_types = ['breath', 'body_scan', 'stress_relief']
        
        for med_type in meditation_types:
            session = self.run_meditation_session(
                duration=600,  # 10 minutes
                focus_type=med_type
            )
            print(f"   ✓ {med_type.replace('_', ' ').title()} session ready")
        
        print("\n" + "=" * 70)
        print("✅ Complete therapeutic program created!")
        print(f"   All audio files saved to: {output_dir}/")
        print("=" * 70)
        
        return {
            'analysis': analysis,
            'output_dir': output_dir,
            'sessions_created': len(sessions) + 1  # +1 for breathing guide
        }
    
    def visualize_analysis(self, audio: np.ndarray, analysis: dict,
                          save_path: str = 'output/reports/analysis.png'):
        """
        Create visualization of voice analysis
        
        Args:
            audio: Input audio
            analysis: Analysis results
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('NeuroCalm Voice Analysis Report', fontsize=16, fontweight='bold')
        
        # Waveform
        axes[0, 0].plot(audio, linewidth=0.5)
        axes[0, 0].set_title('Audio Waveform')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=self.sr, x_axis='time', 
                                       y_axis='hz', ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram')
        plt.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
        
        # Pitch contour
        axes[1, 0].plot(analysis['features']['pitch'])
        axes[1, 0].set_title('Pitch Contour')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Energy
        axes[1, 1].plot(analysis['features']['energy'])
        axes[1, 1].set_title('Energy Profile')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('RMS Energy')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Stress metrics
        metrics = ['stress_score', 'confidence', 'jitter', 'shimmer']
        values = [
            analysis['stress_score'],
            analysis['confidence'] * 100,
            analysis['features']['jitter'],
            analysis['features']['shimmer']
        ]
        axes[2, 0].bar(metrics, values, color=['red', 'blue', 'orange', 'green'])
        axes[2, 0].set_title('Stress Indicators')
        axes[2, 0].set_ylabel('Value')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # Stress level gauge - FIXED VERSION
        stress_levels = ['Low Stress', 'Medium Stress', 'High Stress']
        colors = ['green', 'yellow', 'red']
        
        # Find matching stress level
        current_level = stress_levels.index(analysis['stress_level'])
        
        axes[2, 1].barh(stress_levels, [1, 1, 1], color=colors, alpha=0.3)
        axes[2, 1].barh(stress_levels[current_level], 1, color=colors[current_level])
        axes[2, 1].set_title('Detected Stress Level')
        axes[2, 1].set_xlim(0, 1)
        axes[2, 1].set_xlabel('Probability')
        
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n   ✓ Visualization saved to {save_path}")
        
        plt.close()


def demo_complete_system():
    """
    Complete system demonstration
    """
    print("\n" + "=" * 70)
    print("🎯 NeuroCalm Voice Therapy System - Complete Demo")
    print("=" * 70)
    
    # Initialize system
    neurocalm = NeuroCalm(sr=22050)
    
    # Train system
    neurocalm.train_system(num_samples=50)
    
    # Generate test audio
    print("\n📢 Generating test voice sample...")
    t = np.linspace(0, 5, 5 * 22050)
    test_audio = 0.5 * np.sin(2 * np.pi * 180 * t)  # Medium stress voice
    test_audio += 0.2 * np.sin(2 * np.pi * 360 * t)
    test_audio += np.random.randn(len(test_audio)) * 0.05
    test_audio = normalize_audio(test_audio)
    
    # Save test audio
    save_audio(test_audio, 22050, 'data/raw/test_voice.wav')
    
    # Run complete analysis
    analysis = neurocalm.analyze_voice(test_audio)
    
    # Create therapeutic program
    results = neurocalm.create_complete_therapy_program(test_audio)
    
    # Create visualization
    print("\n📊 Creating visualization...")
    neurocalm.visualize_analysis(test_audio, analysis)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 Session Summary")
    print("=" * 70)
    print(f"   Initial Stress Level: {analysis['stress_level']}")
    print(f"   Stress Score: {analysis['stress_score']:.1f}/100")
    print(f"   Confidence: {analysis['confidence']*100:.1f}%")
    print(f"   Therapeutic Sessions Created: {results['sessions_created']}")
    print(f"   Output Directory: {results['output_dir']}/")
    print("\n   Recommended Actions:")
    if analysis['stress_score'] > 70:
        print("   - Start with Headache Relief session")
        print("   - Follow with Deep Relaxation")
        print("   - Practice breathing exercises daily")
    elif analysis['stress_score'] > 40:
        print("   - Begin with Stress Relief session")
        print("   - Use breathing guide regularly")
        print("   - Try meditation sessions")
    else:
        print("   - Maintain current practice")
        print("   - Use sessions for preventive care")
    
    print("\n" + "=" * 70)
    print("✨ Demo Complete! Check output/ directory for all files.")
    print("=" * 70)


if __name__ == "__main__":
    demo_complete_system()