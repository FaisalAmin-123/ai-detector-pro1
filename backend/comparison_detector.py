"""
Comparison Module: Your Hybrid System vs Qwen3
Runs both detectors and provides side-by-side comparison

Author: Faisal + Assistant
Date: November 2024
"""

from combined_detector_hybrid import HybridDetector
from qwen_alternative_detector import AlternativeAIDetector as Qwen3Detector
import time


class ComparisonDetector:
    """
    Runs both detection systems and compares results
    """
    
    def __init__(self):
        """Initialize both detectors"""
        print("\n" + "=" * 70)
        print("🚀 INITIALIZING COMPARISON SYSTEM")
        print("=" * 70)
        
        # Initialize your hybrid detector
        print("\n[1/2] Loading Your Hybrid System...")
        self.hybrid_detector = HybridDetector()
        
        # Initialize Qwen3 detector
        print("\n[2/2] Loading Qwen3 Detector...")
        self.qwen3_detector = Qwen3Detector()
        
        print("\n" + "=" * 70)
        print("✅ BOTH DETECTORS READY FOR COMPARISON!")
        print("=" * 70 + "\n")
    
    def compare(self, text):
        """
        Run both detectors and compare results
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Comparison results from both systems
        """
        print(f"\n🔍 Running comparison analysis...")
        print(f"   Text length: {len(text)} characters")
        print("-" * 70)
        
        results = {
            'text_info': {
                'length': len(text),
                'word_count': len(text.split()),
                'preview': text[:100] + '...' if len(text) > 100 else text
            },
            'your_system': {},
            'qwen3': {},
            'comparison': {}
        }
        
        # Run your hybrid system
        print("   [1/2] Running Your Hybrid System (13 methods)...")
        start_time = time.time()
        try:
            hybrid_result = self.hybrid_detector.analyze(text)
            hybrid_time = time.time() - start_time
            
            results['your_system'] = {
                'label': hybrid_result['final_decision']['label'],
                'confidence': hybrid_result['final_decision']['confidence'],
                'is_ai_generated': hybrid_result['final_decision']['is_ai_generated'],
                'agreement': hybrid_result['agreement']['agreement_level'],
                'votes_ai': hybrid_result['agreement']['votes_for_ai'],
                'votes_human': hybrid_result['agreement']['votes_for_human'],
                'processing_time': round(hybrid_time, 2),
                'methods': {
                    'ml_detection': hybrid_result.get('ml_detection', {}),
                    'perplexity': hybrid_result.get('perplexity_analysis', {}),
                    'burstiness': hybrid_result.get('burstiness_analysis', {}),
                    'advanced_features': hybrid_result.get('advanced_features', {})
                }
            }
            
            print(f"        → {hybrid_result['final_decision']['label']} "
                  f"({hybrid_result['final_decision']['confidence']:.1f}%) "
                  f"in {hybrid_time:.2f}s")
            
        except Exception as e:
            print(f"        ❌ Error: {e}")
            results['your_system'] = {'error': str(e)}
        
        # Run Qwen3
        print("   [2/2] Running Qwen3 Model...")
        start_time = time.time()
        try:
            qwen3_result = self.qwen3_detector.detect(text)
            qwen3_time = time.time() - start_time
            
            results['qwen3'] = {
                'label': qwen3_result['label'],
                'confidence': qwen3_result['confidence'],
                'is_ai_generated': qwen3_result.get('is_ai_generated', False),
                'processing_time': round(qwen3_time, 2),
                'raw_response': qwen3_result.get('raw_response', 'N/A'),
                'model': 'Qwen2.5-0.5B-Instruct'
            }
            
            print(f"        → {qwen3_result['label']} "
                  f"({qwen3_result['confidence']:.1f}%) "
                  f"in {qwen3_time:.2f}s")
            
        except Exception as e:
            print(f"        ❌ Error: {e}")
            results['qwen3'] = {'error': str(e)}
        
        # Compare results
        print("\n   📊 Generating comparison...")
        results['comparison'] = self._compare_results(
            results['your_system'],
            results['qwen3']
        )
        
        print("-" * 70)
        print(f"✅ Comparison complete!")
        
        return results
    
    def _compare_results(self, hybrid, qwen3):
        """
        Compare results from both systems
        
        Args:
            hybrid: Results from your system
            qwen3: Results from Qwen3
            
        Returns:
            dict: Comparison analysis
        """
        comparison = {}
        
        # Check if both succeeded
        if 'error' in hybrid or 'error' in qwen3:
            comparison['status'] = 'error'
            comparison['agreement'] = 'Unable to compare due to errors'
            return comparison
        
        # Check agreement
        both_ai = hybrid['is_ai_generated'] and qwen3['is_ai_generated']
        both_human = not hybrid['is_ai_generated'] and not qwen3['is_ai_generated']
        
        if both_ai or both_human:
            comparison['agreement_status'] = 'agree'
            comparison['agreement_message'] = f"✅ Both systems agree: {hybrid['label']}"
        else:
            comparison['agreement_status'] = 'disagree'
            comparison['agreement_message'] = f"⚠️ Systems disagree - Your system: {hybrid['label']}, Qwen3: {qwen3['label']}"
        
        # Confidence comparison
        conf_diff = abs(hybrid['confidence'] - qwen3['confidence'])
        comparison['confidence_difference'] = round(conf_diff, 2)
        
        if conf_diff < 10:
            comparison['confidence_analysis'] = "Very similar confidence levels"
        elif conf_diff < 25:
            comparison['confidence_analysis'] = "Moderate difference in confidence"
        else:
            comparison['confidence_analysis'] = "Significant difference in confidence"
        
        # Speed comparison
        comparison['speed_comparison'] = {
            'your_system': hybrid['processing_time'],
            'qwen3': qwen3['processing_time'],
            'faster': 'Your System' if hybrid['processing_time'] < qwen3['processing_time'] else 'Qwen3'
        }
        
        # Overall assessment
        if both_ai or both_human:
            if conf_diff < 15:
                comparison['overall'] = "Strong agreement with similar confidence"
            else:
                comparison['overall'] = "Agreement on label, but different confidence levels"
        else:
            comparison['overall'] = "Disagreement - manual review recommended"
        
        return comparison
    
    def display_comparison(self, results):
        """
        Display comparison results in a formatted way
        
        Args:
            results: Results from compare()
        """
        print("\n" + "=" * 70)
        print("📊 SIDE-BY-SIDE COMPARISON RESULTS")
        print("=" * 70)
        
        # Text info
        print(f"\n📝 Text Information:")
        print(f"   Length: {results['text_info']['length']} characters")
        print(f"   Words: {results['text_info']['word_count']}")
        print(f"   Preview: {results['text_info']['preview']}")
        
        # Your System
        print(f"\n🔬 YOUR HYBRID SYSTEM (13 Methods):")
        if 'error' not in results['your_system']:
            hs = results['your_system']
            print(f"   Prediction: {hs['label']}")
            print(f"   Confidence: {hs['confidence']:.1f}%")
            print(f"   Agreement: {hs['agreement']}")
            print(f"   Votes: {hs['votes_ai']} AI, {hs['votes_human']} Human")
            print(f"   Processing Time: {hs['processing_time']}s")
        else:
            print(f"   ❌ Error: {results['your_system']['error']}")
        
        # Qwen3
        print(f"\n🤖 QWEN3 MODEL:")
        if 'error' not in results['qwen3']:
            qw = results['qwen3']
            print(f"   Prediction: {qw['label']}")
            print(f"   Confidence: {qw['confidence']:.1f}%")
            print(f"   Model: {qw['model']}")
            print(f"   Processing Time: {qw['processing_time']}s")
        else:
            print(f"   ❌ Error: {results['qwen3']['error']}")
        
        # Comparison
        print(f"\n⚖️  COMPARISON ANALYSIS:")
        comp = results['comparison']
        if 'error' not in comp:
            print(f"   {comp['agreement_message']}")
            print(f"   Confidence Difference: {comp['confidence_difference']:.1f}%")
            print(f"   {comp['confidence_analysis']}")
            print(f"   Faster System: {comp['speed_comparison']['faster']}")
            print(f"\n   📌 Overall: {comp['overall']}")
        
        print("\n" + "=" * 70 + "\n")


# Test function
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING COMPARISON SYSTEM")
    print("=" * 70)
    
    comparator = ComparisonDetector()
    
    # Test 1: AI text
    ai_text = """
    Machine learning algorithms have revolutionized the field of computational analysis.
    These sophisticated systems demonstrate remarkable capabilities in data processing.
    The implementation of neural networks enables advanced pattern recognition functionality.
    Modern artificial intelligence continues to evolve with unprecedented capabilities.
    """
    
    print("\n" + "=" * 70)
    print("TEST 1: AI-GENERATED TEXT")
    print("=" * 70)
    result1 = comparator.compare(ai_text)
    comparator.display_comparison(result1)
    
    # Test 2: Human text
    human_text = """
    omg you won't believe what happened today!! so i was just walking
    to class right? and i see this guy from my chemistry lab. we started talking
    and turns out he's actually super cool! like we have so much in common lol.
    anyway how's your day been? mine's been pretty crazy not gonna lie haha
    """
    
    print("\n" + "=" * 70)
    print("TEST 2: HUMAN-WRITTEN TEXT")
    print("=" * 70)
    result2 = comparator.compare(human_text)
    comparator.display_comparison(result2)
    
    print("=" * 70)
    print("✅ COMPARISON TESTING COMPLETE!")
    print("=" * 70 + "\n")