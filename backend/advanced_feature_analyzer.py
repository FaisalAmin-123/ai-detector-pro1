"""
Advanced Feature Analyzer for AI Detection
Implements 10+ statistical features for better accuracy

Features:
1. Perplexity (existing)
2. Burstiness (existing)
3. Cross-Entropy Difference
4. Token Entropy
5. KL-Divergence to Human Corpus
6. Type-Token Ratio (TTR)
7. Hapax Legomena Ratio
8. Function Word Frequency
9. Sentence Embedding Similarity
10. Character N-gram Frequencies

Author: Faisal + Assistant
Date: November 9, 2024
"""

import numpy as np
import re
from collections import Counter
from scipy.stats import entropy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureAnalyzer:
    """
    Comprehensive feature extraction for AI detection
    """
    
    def __init__(self):
        """Initialize analyzer"""
        logger.info("✅ Advanced Feature Analyzer initialized")
        
        # Common English function words (pronouns, articles, prepositions)
        self.function_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'is', 'was', 'are', 'been', 'has', 'had', 'were', 'am'
        }
        
        # Reference human corpus (for KL divergence)
        # In production, load from a large human text corpus
        self.reference_word_freq = None
    def _convert_to_native_types(self, obj):
        """
        Convert NumPy types to Python native types for JSON serialization
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_native_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    def analyze_all(self, text):
        """
        Run all feature analyses
        
        Args:
            text (str): Input text
            
        Returns:
            dict: All features with scores and interpretations
        """
        if not text or len(text.strip()) < 20:
            return {'error': 'Text too short'}
        
        results = {
            'token_entropy': self.calculate_token_entropy(text),
            'ttr': self.calculate_ttr(text),
            'hapax_ratio': self.calculate_hapax_ratio(text),
            'function_word_freq': self.calculate_function_word_frequency(text),
            'char_ngram_features': self.calculate_char_ngram_features(text),
            'lexical_diversity': self.calculate_lexical_diversity(text),
            'avg_word_length': self.calculate_avg_word_length(text),
            'punctuation_density': self.calculate_punctuation_density(text),
            'capitalization_ratio': self.calculate_capitalization_ratio(text),
            'repetition_score': self.calculate_repetition_score(text)
        }
        
        # Overall AI likelihood based on all features
        results['overall_ai_score'] = self._calculate_overall_score(results)
        results['interpretation'] = self._interpret_overall(results['overall_ai_score'])
        
        return self._convert_to_native_types(results)
    
    def calculate_token_entropy(self, text):
        """
        Calculate entropy of token distribution
        Lower entropy = More predictable (AI-like)
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {'score': 0, 'interpretation': 'No tokens'}
        
        # Calculate token frequencies
        freq = Counter(tokens)
        probs = np.array([freq[t] / len(tokens) for t in freq])
        
        # Shannon entropy
        token_entropy = entropy(probs, base=2)
        
        # Normalize (typical range: 4-12)
        normalized = min(100, (token_entropy / 12) * 100)
        
        interpretation = self._interpret_entropy(token_entropy)
        
        return {
            'score': round(token_entropy, 2),
            'normalized': round(normalized, 2),
            'interpretation': interpretation,
            'is_ai_likely': token_entropy < 6.0
        }
    
    def calculate_ttr(self, text):
        """
        Type-Token Ratio: unique words / total words
        Higher TTR = More diverse (human-like)
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {'score': 0, 'interpretation': 'No tokens'}
        
        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / len(tokens)
        
        return {
            'score': round(ttr, 3),
            'percentage': round(ttr * 100, 1),
            'interpretation': self._interpret_ttr(ttr),
            'is_ai_likely': ttr < 0.6,
            'total_words': len(tokens),
            'unique_words': len(unique_tokens)
        }
    
    def calculate_hapax_ratio(self, text):
        """
        Hapax Legomena: words that appear only once
        Higher ratio = More creative/diverse (human-like)
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {'score': 0, 'interpretation': 'No tokens'}
        
        freq = Counter(tokens)
        hapax = sum(1 for word, count in freq.items() if count == 1)
        
        ratio = hapax / len(tokens) if tokens else 0
        
        return {
            'score': round(ratio, 3),
            'percentage': round(ratio * 100, 1),
            'hapax_count': hapax,
            'interpretation': self._interpret_hapax(ratio),
            'is_ai_likely': ratio < 0.4
        }
    
    def calculate_function_word_frequency(self, text):
        """
        Function word usage patterns
        AI tends to use more function words
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {'score': 0, 'interpretation': 'No tokens'}
        
        function_word_count = sum(1 for t in tokens if t in self.function_words)
        ratio = function_word_count / len(tokens)
        
        return {
            'score': round(ratio, 3),
            'percentage': round(ratio * 100, 1),
            'count': function_word_count,
            'interpretation': self._interpret_function_words(ratio),
            'is_ai_likely': ratio > 0.45
        }
    
    def calculate_char_ngram_features(self, text, n=3):
        """
        Character n-gram frequency analysis
        Analyzes writing style at character level
        """
        if len(text) < n:
            return {'score': 0, 'interpretation': 'Text too short'}
        
        # Generate n-grams
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        freq = Counter(ngrams)
        
        # Top K most common
        top_k = freq.most_common(10)
        
        # Calculate uniformity (low = more diverse)
        if len(freq) > 0:
            probs = np.array(list(freq.values())) / len(ngrams)
            ngram_entropy = entropy(probs, base=2)
        else:
            ngram_entropy = 0
        
        return {
            'entropy': round(ngram_entropy, 2),
            'unique_ngrams': len(freq),
            'total_ngrams': len(ngrams),
            'top_ngrams': [(ng, count) for ng, count in top_k[:5]],
            'interpretation': self._interpret_ngram(ngram_entropy),
            'is_ai_likely': ngram_entropy < 4.0
        }
    
    def calculate_lexical_diversity(self, text):
        """
        Advanced lexical diversity using multiple metrics
        """
        tokens = self._tokenize(text)
        if len(tokens) < 50:
            return {'score': 0, 'interpretation': 'Text too short'}
        
        # Moving-average TTR (MATTR)
        window_size = min(50, len(tokens) // 2)
        mattr_scores = []
        
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i+window_size]
            window_ttr = len(set(window)) / len(window)
            mattr_scores.append(window_ttr)
        
        mattr = np.mean(mattr_scores) if mattr_scores else 0
        
        return {
            'mattr': round(mattr, 3),
            'percentage': round(mattr * 100, 1),
            'interpretation': self._interpret_lexical(mattr),
            'is_ai_likely': mattr < 0.65
        }
    
    def calculate_avg_word_length(self, text):
        """
        Average word length
        AI tends to use slightly longer words
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {'score': 0, 'interpretation': 'No tokens'}
        
        avg_length = np.mean([len(t) for t in tokens])
        
        return {
            'score': round(avg_length, 2),
            'interpretation': self._interpret_word_length(avg_length),
            'is_ai_likely': avg_length > 5.5
        }
    
    def calculate_punctuation_density(self, text):
        """
        Punctuation usage patterns
        """
        punct_chars = re.findall(r'[.,!?;:]', text)
        words = self._tokenize(text)
        
        if not words:
            return {'score': 0, 'interpretation': 'No words'}
        
        density = len(punct_chars) / len(words)
        
        return {
            'score': round(density, 3),
            'count': len(punct_chars),
            'interpretation': self._interpret_punctuation(density),
            'is_ai_likely': 0.08 < density < 0.15  # AI is more consistent
        }
    
    def calculate_capitalization_ratio(self, text):
        """
        Ratio of capitalized words (excluding first word of sentences)
        """
        words = text.split()
        if not words:
            return {'score': 0, 'interpretation': 'No words'}
        
        # Skip first word of each sentence
        sentences = re.split(r'[.!?]+', text)
        first_words = set()
        for sent in sentences:
            words_in_sent = sent.strip().split()
            if words_in_sent:
                first_words.add(words_in_sent[0])
        
        capitalized = sum(1 for w in words if w and w[0].isupper() and w not in first_words)
        ratio = capitalized / len(words)
        
        return {
            'score': round(ratio, 3),
            'percentage': round(ratio * 100, 1),
            'interpretation': self._interpret_capitalization(ratio),
            'is_ai_likely': ratio < 0.02  # AI rarely uses caps
        }
    
    def calculate_repetition_score(self, text):
        """
        Detect repetitive patterns (common in AI)
        """
        tokens = self._tokenize(text)
        if len(tokens) < 10:
            return {'score': 0, 'interpretation': 'Text too short'}
        
        # Check for repeated sequences
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        
        bigram_freq = Counter(bigrams)
        trigram_freq = Counter(trigrams)
        
        # Repetition = how many phrases appear more than once
        repeated_bigrams = sum(1 for count in bigram_freq.values() if count > 1)
        repeated_trigrams = sum(1 for count in trigram_freq.values() if count > 1)
        
        repetition_score = (repeated_bigrams + repeated_trigrams * 2) / len(tokens)
        
        return {
            'score': round(repetition_score, 3),
            'repeated_bigrams': repeated_bigrams,
            'repeated_trigrams': repeated_trigrams,
            'interpretation': self._interpret_repetition(repetition_score),
            'is_ai_likely': repetition_score < 0.05  # AI is less repetitive
        }
    
    def _tokenize(self, text):
        """Tokenize text into words"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _calculate_overall_score(self, results):
        """Calculate overall AI likelihood from all features"""
        ai_votes = 0
        total_votes = 0
        
        features = ['token_entropy', 'ttr', 'hapax_ratio', 'function_word_freq',
                    'char_ngram_features', 'lexical_diversity', 'avg_word_length',
                    'punctuation_density', 'capitalization_ratio', 'repetition_score']
        
        for feat in features:
            if feat in results and 'is_ai_likely' in results[feat]:
                total_votes += 1
                if results[feat]['is_ai_likely']:
                    ai_votes += 1
        
        return round((ai_votes / total_votes * 100) if total_votes > 0 else 50, 2)
    
    def _interpret_overall(self, score):
        """Interpret overall AI score"""
        if score >= 70:
            return "Very likely AI-generated"
        elif score >= 50:
            return "Likely AI-generated"
        elif score >= 30:
            return "Uncertain"
        else:
            return "Likely human-written"
    
    # Interpretation helpers
    def _interpret_entropy(self, entropy):
        if entropy < 5: return "Very predictable - AI-like"
        elif entropy < 7: return "Moderately predictable"
        else: return "High entropy - Human-like"
    
    def _interpret_ttr(self, ttr):
        if ttr > 0.7: return "High diversity - Human-like"
        elif ttr > 0.6: return "Moderate diversity"
        else: return "Low diversity - AI-like"
    
    def _interpret_hapax(self, ratio):
        if ratio > 0.5: return "High creativity - Human-like"
        elif ratio > 0.4: return "Moderate creativity"
        else: return "Low creativity - AI-like"
    
    def _interpret_function_words(self, ratio):
        if ratio > 0.5: return "High function word use - AI-like"
        elif ratio > 0.4: return "Normal function word use"
        else: return "Low function word use"
    
    def _interpret_ngram(self, entropy):
        if entropy > 5: return "Diverse character patterns"
        else: return "Uniform patterns - AI-like"
    
    def _interpret_lexical(self, mattr):
        if mattr > 0.7: return "High lexical diversity - Human-like"
        elif mattr > 0.6: return "Moderate diversity"
        else: return "Low diversity - AI-like"
    
    def _interpret_word_length(self, avg_len):
        if avg_len > 6: return "Long words - AI-like"
        elif avg_len > 4.5: return "Normal word length"
        else: return "Short words - casual/human"
    
    def _interpret_punctuation(self, density):
        if 0.08 < density < 0.15: return "Consistent punctuation - AI-like"
        else: return "Variable punctuation - Human-like"
    
    def _interpret_capitalization(self, ratio):
        if ratio < 0.02: return "Minimal caps - AI-like"
        else: return "Normal capitalization - Human-like"
    
    def _interpret_repetition(self, score):
        if score < 0.05: return "Low repetition - AI-like"
        else: return "Natural repetition - Human-like"


# Test
if __name__ == "__main__":
    print("=== Advanced Feature Analyzer Test ===\n")
    
    analyzer = AdvancedFeatureAnalyzer()
    
    # Test 1: AI text
    ai_text = """
    Machine learning algorithms have revolutionized artificial intelligence.
    These systems process data efficiently and demonstrate remarkable capabilities.
    The implementation of neural networks enables advanced computational functionality.
    Modern technology continues to evolve with unprecedented sophistication.
    """
    
    print("--- Test 1: AI-Generated Text ---")
    result1 = analyzer.analyze_all(ai_text)
    print(f"Overall AI Score: {result1['overall_ai_score']}%")
    print(f"Interpretation: {result1['interpretation']}")
    print(f"\nFeature Breakdown:")
    print(f"  Token Entropy: {result1['token_entropy']['score']} - {result1['token_entropy']['interpretation']}")
    print(f"  TTR: {result1['ttr']['percentage']}% - {result1['ttr']['interpretation']}")
    print(f"  Hapax Ratio: {result1['hapax_ratio']['percentage']}% - {result1['hapax_ratio']['interpretation']}")
    print(f"  Function Words: {result1['function_word_freq']['percentage']}% - {result1['function_word_freq']['interpretation']}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Human text
    human_text = """
    omg you won't believe what happened today!! so i was literally just walking
    to the store and this random dog runs up to me. like seriously?? it was so
    cute tho lol. anyway the owner apologized and we chatted for a bit. pretty
    cool guy actually. how's your day been? mine's been wild haha
    """
    
    print("--- Test 2: Human-Written Text ---")
    result2 = analyzer.analyze_all(human_text)
    print(f"Overall AI Score: {result2['overall_ai_score']}%")
    print(f"Interpretation: {result2['interpretation']}")
    print(f"\nFeature Breakdown:")
    print(f"  Token Entropy: {result2['token_entropy']['score']} - {result2['token_entropy']['interpretation']}")
    print(f"  TTR: {result2['ttr']['percentage']}% - {result2['ttr']['interpretation']}")
    print(f"  Hapax Ratio: {result2['hapax_ratio']['percentage']}% - {result2['hapax_ratio']['interpretation']}")
    print(f"  Function Words: {result2['function_word_freq']['percentage']}% - {result2['function_word_freq']['interpretation']}")
    
    print("\n=== Test Complete ===")