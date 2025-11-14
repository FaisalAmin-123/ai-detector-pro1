"""
Train ML Detector with Real Samples
Uses your actual test data to improve accuracy

Author: Faisal
Date: November 6, 2024
"""

import os
from detector_ml_proper import ProperMLDetector

def load_samples(folder):
    """Load all text samples from folder"""
    samples = []
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                    samples.append(f.read())
    return samples

if __name__ == "__main__":
    print("=" * 70)
    print("🎓 TRAINING ML DETECTOR WITH YOUR SAMPLES")
    print("=" * 70)
    
    # Load detector
    print("\nInitializing detector...")
    detector = ProperMLDetector()
    
    # Load your test samples
    print("\nLoading test samples...")
    ai_samples = load_samples('tests/ai_samples')
    human_samples = load_samples('tests/human_samples')
    
    print(f"Found {len(ai_samples)} AI samples")
    print(f"Found {len(human_samples)} human samples")
    
    if not ai_samples or not human_samples:
        print("\n⚠️  No samples found! Make sure tests/ai_samples and tests/human_samples exist.")
        exit(1)
    
    # Train
    print("\n🎓 Training classifier with your samples...")
    detector.improve_with_samples(ai_samples, human_samples)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nThe detector is now trained on your specific data.")
    print("Run test_suite_hybrid.py to see improved accuracy!")