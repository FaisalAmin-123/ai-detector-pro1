




"""
Professional Hybrid AI Detector - WITH ADVANCED FEATURES
Combines: ML + Perplexity + Burstiness + 10 Advanced Statistical Features

Author: Faisal + Assistant
Date: November 9, 2025
"""

from detector_ml_proper import ProperMLDetector
from perplexity import PerplexityCalculator
from burstiness import BurstinessAnalyzer
from advanced_feature_analyzer import AdvancedFeatureAnalyzer
import os


class HybridDetector:
    """
    Professional-grade hybrid detector with advanced features
    Now includes 13 different detection methods!
    """
    
    def __init__(self):
        """Initialize all detection methods"""
        print("Initializing Professional Hybrid Detector (Advanced Edition)...")
        print("-" * 70)
        
        print("\n[1/4] Loading ML Detector...")
        self.ml_detector = ProperMLDetector()
        
        print("\n[2/4] Loading Perplexity Calculator...")
        self.perplexity_calc = PerplexityCalculator()
        
        print("\n[3/4] Loading Burstiness Analyzer...")
        self.burstiness_analyzer = BurstinessAnalyzer()
        
        print("\n[4/4] Loading Advanced Feature Analyzer...")
        self.advanced_analyzer = AdvancedFeatureAnalyzer()
        
        # Updated weights for all methods
        self.ml_weight = 0.40         # 40% - ML detector
        self.perplexity_weight = 0.25  # 25% - Perplexity
        self.burstiness_weight = 0.10  # 10% - Burstiness
        self.advanced_weight = 0.25    # 25% - Advanced features combined
        
        print("\n" + "-" * 70)
        print("✅ Professional Hybrid Detector Ready (13 Methods)!")
        print("=" * 70 + "\n")
    
    def analyze(self, text):
        """
        Comprehensive analysis using all methods
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Complete analysis results with advanced features
        """
        print("🔍 Analyzing text with 13 methods...")
        
        # Method 1: ML Detection
        print("  [1/4] Running ML detection...")
        ml_result = self.ml_detector.detect(text)
        if 'error' not in ml_result:
            print(f"        → {ml_result['label']} ({ml_result['confidence']:.1f}%)")
        
        # Method 2: Perplexity
        print("  [2/4] Calculating perplexity...")
        perp_result = self.perplexity_calc.calculate_perplexity(text)
        if 'error' not in perp_result:
            print(f"        → Score: {perp_result['perplexity']:.1f}")
        
        # Method 3: Burstiness
        print("  [3/4] Analyzing burstiness...")
        burst_result = self.burstiness_analyzer.analyze(text)
        if 'error' not in burst_result:
            print(f"        → Score: {burst_result['burstiness_score']:.1f}")
        
        # Method 4: Advanced Features (10 sub-methods)
        print("  [4/4] Running advanced feature analysis...")
        advanced_result = self.advanced_analyzer.analyze_all(text)
        if 'error' not in advanced_result:
            print(f"        → Overall AI Score: {advanced_result['overall_ai_score']:.1f}%")
            print(f"        → {advanced_result['interpretation']}")
        
        # Calculate combined score
        combined_score, votes, agreement = self._calculate_combined_score(
            ml_result, perp_result, burst_result, advanced_result
        )
        
        final_decision = combined_score > 50
        
        # Build comprehensive result
        result = {
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'text_length': len(text),
            'word_count': len(text.split()),
            
            # ML Results
            'ml_detection': ({
                'label': ml_result.get('label', 'Error'),
                'confidence': ml_result.get('confidence', 0),
                'is_ai': ml_result.get('is_ai_generated'),
                'probabilities': ml_result.get('probabilities', {})
            } if 'error' not in ml_result else {'error': ml_result.get('error')}),
            
            # Perplexity Results
            'perplexity_analysis': ({
                'score': perp_result.get('perplexity'),
                'interpretation': perp_result.get('interpretation'),
                'is_ai': perp_result.get('is_ai_likely'),
                'confidence': perp_result.get('confidence')
            } if 'error' not in perp_result else {'error': perp_result.get('error')}),
            
            # Burstiness Results
            'burstiness_analysis': ({
                'burstiness_score': burst_result.get('burstiness_score'),
                'score': burst_result.get('burstiness_score'),
                'interpretation': burst_result.get('interpretation'),
                'is_ai': burst_result.get('is_ai_likely'),
                'confidence': burst_result.get('confidence'),
                'sentence_count': burst_result.get('sentence_count'),
                'mean_sentence_length': burst_result.get('mean_sentence_length'),
                'std_deviation': burst_result.get('std_deviation')
            } if burst_result.get('error') is None else {'error': burst_result.get('error')}),
            
            # NEW: Advanced Features Results
            'advanced_features': ({
                'overall_score': advanced_result.get('overall_ai_score'),
                'interpretation': advanced_result.get('interpretation'),
                'features': {
                    'token_entropy': advanced_result.get('token_entropy'),
                    'ttr': advanced_result.get('ttr'),
                    'hapax_ratio': advanced_result.get('hapax_ratio'),
                    'function_word_freq': advanced_result.get('function_word_freq'),
                    'lexical_diversity': advanced_result.get('lexical_diversity'),
                    'avg_word_length': advanced_result.get('avg_word_length'),
                    'punctuation_density': advanced_result.get('punctuation_density'),
                    'char_ngram_features': advanced_result.get('char_ngram_features'),
                    'capitalization_ratio': advanced_result.get('capitalization_ratio'),
                    'repetition_score': advanced_result.get('repetition_score')
                }
            } if 'error' not in advanced_result else {'error': advanced_result.get('error')}),
            
            # Final Decision
            'final_decision': {
                'is_ai_generated': final_decision,
                'confidence': combined_score,
                'label': 'AI-Generated' if final_decision else 'Human-Written'
            },
            
            # Agreement Analysis
            'agreement': {
                'votes_for_ai': votes['ai'],
                'votes_for_human': votes['human'],
                'total_methods': votes['total'],
                'agreement_level': agreement,
                'all_agree': votes['ai'] == votes['total'] or votes['human'] == votes['total']
            }
        }
        
        return result
    
    def _calculate_combined_score(self, ml_result, perp_result, burst_result, advanced_result):
        """
        Calculate weighted combined score from all methods including advanced features
        
        Returns:
            tuple: (combined_score, votes_dict, agreement_level)
        """
        scores = []
        weights = []
        votes = {'ai': 0, 'human': 0, 'total': 0}
        
        # ML Score
        if 'error' not in ml_result:
            ml_ai_score = ml_result['probabilities']['ai']
            scores.append(ml_ai_score)
            weights.append(self.ml_weight)
            votes['total'] += 1
            if ml_result['is_ai_generated']:
                votes['ai'] += 1
            else:
                votes['human'] += 1
        
        # Perplexity Score
        if 'error' not in perp_result:
            if perp_result['is_ai_likely']:
                perp_ai_score = perp_result['confidence']
            else:
                perp_ai_score = 100 - perp_result['confidence']
            scores.append(perp_ai_score)
            weights.append(self.perplexity_weight)
            votes['total'] += 1
            if perp_result['is_ai_likely']:
                votes['ai'] += 1
            else:
                votes['human'] += 1
        
        # Burstiness Score
        if 'error' not in burst_result and burst_result.get('burstiness_score') is not None:
            if burst_result['is_ai_likely']:
                burst_ai_score = burst_result['confidence']
            else:
                burst_ai_score = 100 - burst_result['confidence']
            scores.append(burst_ai_score)
            weights.append(self.burstiness_weight)
            votes['total'] += 1
            if burst_result['is_ai_likely']:
                votes['ai'] += 1
            else:
                votes['human'] += 1
        
        # Advanced Features Score
        if 'error' not in advanced_result:
            advanced_ai_score = advanced_result['overall_ai_score']
            scores.append(advanced_ai_score)
            weights.append(self.advanced_weight)
            votes['total'] += 1
            if advanced_ai_score > 50:
                votes['ai'] += 1
            else:
                votes['human'] += 1
        
        # Normalize weights
        if weights:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted average
            combined_score = sum(s * w for s, w in zip(scores, normalized_weights))
        else:
            combined_score = 50  # Neutral if all failed
        
        # Determine agreement level
        if votes['total'] == 0:
            agreement = "No data"
        elif votes['ai'] == votes['total'] or votes['human'] == votes['total']:
            agreement = "All methods agree"
        elif votes['ai'] >= votes['total'] * 0.75 or votes['human'] >= votes['total'] * 0.75:
            agreement = "Strong majority"
        elif votes['ai'] >= votes['total'] * 0.6 or votes['human'] >= votes['total'] * 0.6:
            agreement = "Majority agrees"
        else:
            agreement = "Methods split"
        
        return round(combined_score, 2), votes, agreement
    
    def analyze_file(self, file_path):
        """
        Analyze file (PDF, Image, or Text)
        
        Args:
            file_path: Path to file
            
        Returns:
            dict: Analysis results
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return self.analyze(text)
        elif ext == '.pdf':
            return {'error': 'PDF support not yet implemented'}
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return {'error': 'Image OCR not yet implemented'}
        else:
            return {'error': f'Unsupported file type: {ext}'}
    
    def display_results(self, result):
        """Display comprehensive results including advanced features"""
        print("\n" + "=" * 70)
        print("📊 PROFESSIONAL HYBRID DETECTION RESULTS (ADVANCED)")
        print("=" * 70)
        
        print(f"\n📝 Text Information:")
        print(f"   Preview: {result['text_preview']}")
        print(f"   Length: {result['text_length']} characters")
        print(f"   Words: {result['word_count']}")
        
        # ML Results
        if 'error' not in result['ml_detection']:
            print(f"\n🤖 Method 1: Machine Learning")
            print(f"   Prediction: {result['ml_detection']['label']}")
            print(f"   Confidence: {result['ml_detection']['confidence']:.1f}%")
        
        # Perplexity Results
        if 'error' not in result['perplexity_analysis']:
            print(f"\n📈 Method 2: Perplexity Analysis")
            print(f"   Score: {result['perplexity_analysis']['score']:.1f}")
            print(f"   Interpretation: {result['perplexity_analysis']['interpretation']}")
        
        # Burstiness Results
        if 'error' not in result['burstiness_analysis']:
            print(f"\n📊 Method 3: Burstiness Analysis")
            print(f"   Score: {result['burstiness_analysis']['burstiness_score']:.1f}")
            print(f"   Interpretation: {result['burstiness_analysis']['interpretation']}")
        
        # Advanced Features Results
        if 'error' not in result['advanced_features']:
            print(f"\n🔬 Method 4: Advanced Statistical Features")
            print(f"   Overall AI Score: {result['advanced_features']['overall_score']:.1f}%")
            print(f"   Interpretation: {result['advanced_features']['interpretation']}")
            
            features = result['advanced_features']['features']
            print(f"\n   Feature Breakdown:")
            if features.get('token_entropy'):
                print(f"      • Token Entropy: {features['token_entropy']['score']:.2f} - {features['token_entropy']['interpretation']}")
            if features.get('ttr'):
                print(f"      • Type-Token Ratio: {features['ttr']['percentage']:.1f}% - {features['ttr']['interpretation']}")
            if features.get('hapax_ratio'):
                print(f"      • Hapax Legomena: {features['hapax_ratio']['percentage']:.1f}% - {features['hapax_ratio']['interpretation']}")
            if features.get('lexical_diversity'):
                lex_div = features['lexical_diversity']
                lex_value = lex_div.get('percentage', lex_div.get('mattr', 0) * 100)
                print(f"      • Lexical Diversity: {lex_value:.1f}% - {lex_div.get('interpretation', 'N/A')}")
        # Final Decision
        print(f"\n🎯 FINAL DECISION:")
        print(f"   Result: {result['final_decision']['label']}")
        print(f"   Confidence: {result['final_decision']['confidence']:.1f}%")
        
        # Agreement
        print(f"\n🤝 Agreement Analysis:")
        print(f"   AI Votes: {result['agreement']['votes_for_ai']}/{result['agreement']['total_methods']}")
        print(f"   Human Votes: {result['agreement']['votes_for_human']}/{result['agreement']['total_methods']}")
        print(f"   Agreement Level: {result['agreement']['agreement_level']}")
        print(f"   Status: {'✅ High Confidence' if result['agreement']['all_agree'] else '⚠️ Mixed Signals'}")
        
        print("\n" + "=" * 70 + "\n")


# Test
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 PROFESSIONAL HYBRID DETECTOR - ADVANCED EDITION TEST")
    print("=" * 70 + "\n")
    
    detector = HybridDetector()
    
    # Test 1: AI text
    ai_text = """
    Machine learning algorithms have revolutionized the field of computational analysis.
    These sophisticated systems demonstrate remarkable capabilities in data processing.
    The implementation of neural networks enables advanced pattern recognition functionality.
    Modern artificial intelligence continues to evolve with unprecedented capabilities.
    """
    
    print("=" * 70)
    print("TEST 1: AI-GENERATED TEXT (Expected: AI)")
    print("=" * 70)
    result1 = detector.analyze(ai_text)
    detector.display_results(result1)
    
    is_correct_1 = result1['final_decision']['is_ai_generated'] == True
    print(f"{'✅ CORRECT!' if is_correct_1 else '❌ INCORRECT - Should be AI'}\n")
    
    # Test 2: Human text
    human_text = """
    omg you literally won't believe what happened today!! so i was just walking
    to class right? and i see this guy from my chemistry lab. we started talking
    and turns out he's actually super cool! like we have so much in common lol.
    anyway how's your day been? mine's been pretty crazy not gonna lie haha
    """
    
    print("=" * 70)
    print("TEST 2: HUMAN-WRITTEN TEXT (Expected: Human)")
    print("=" * 70)
    result2 = detector.analyze(human_text)
    detector.display_results(result2)
    
    is_correct_2 = result2['final_decision']['is_ai_generated'] == False
    print(f"{'✅ CORRECT!' if is_correct_2 else '❌ INCORRECT - Should be Human'}\n")
    
    # Summary
    print("=" * 70)
    print(f"TESTS PASSED: {sum([is_correct_1, is_correct_2])}/2")
    if is_correct_1 and is_correct_2:
        print("🎉 ALL TESTS PASSED! Advanced system is working!")
    print("=" * 70 + "\n")