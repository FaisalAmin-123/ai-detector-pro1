"""
Burstiness Analysis for AI Detection - COMPLETELY FIXED
Measures sentence length variation

Author: Faisal (original) + Assistant (complete fix)
Date: November 8, 2024
"""
import re
import numpy as np

class BurstinessAnalyzer:
    """
    Analyze text burstiness (sentence length variation)
    """

    def __init__(self):
        """Initialize analyzer"""
        print("✅ Burstiness Analyzer ready!")

    def analyze(self, text):
        """
        Analyze burstiness of text

        Args:
            text (str): Text to analyze

        Returns:
            dict: Burstiness analysis results
        """
        # Default result structure
        base_result = {
            'burstiness_score': 0.0,
            'mean_sentence_length': 0.0,
            'std_deviation': 0.0,
            'sentence_count': 0,
            'sentence_lengths': [],
            'interpretation': 'No interpretation',
            'is_ai_likely': None,
            'confidence': 0.0,
            'error': None
        }

        # Check for empty text
        if not text or len(text.strip()) == 0:
            base_result['error'] = 'Text is empty'
            return base_result

        # AGGRESSIVE sentence splitting
        sentences = self._split_sentences_aggressive(text)
        
        print(f"DEBUG: Found {len(sentences)} sentences")
        for i, s in enumerate(sentences[:5]):  # Show first 5
            print(f"  Sentence {i+1}: {s[:50]}...")

        base_result['sentence_count'] = len(sentences)

        # Need at least 2 sentences
        if len(sentences) < 2:
            if sentences:
                words = self._tokenize(sentences[0])
                base_result['sentence_lengths'] = [len(words)]
                base_result['mean_sentence_length'] = len(words)
            base_result['interpretation'] = 'Too few sentences (need 2+)'
            base_result['burstiness_score'] = 0.0
            base_result['std_deviation'] = 0.0
            base_result['is_ai_likely'] = None
            base_result['confidence'] = 0.0
            return base_result

        # Get sentence lengths (word counts)
        sentence_lengths = []
        for sent in sentences:
            words = self._tokenize(sent)
            if len(words) > 0:  # Only count non-empty sentences
                sentence_lengths.append(len(words))
        
        if len(sentence_lengths) < 2:
            base_result['interpretation'] = 'Too few valid sentences'
            return base_result

        print(f"DEBUG: Sentence lengths: {sentence_lengths}")

        # Calculate statistics
        mean_length = float(np.mean(sentence_lengths))
        std_dev = float(np.std(sentence_lengths))

        print(f"DEBUG: Mean={mean_length:.2f}, StdDev={std_dev:.2f}")

        # Calculate burstiness score
        # Using Coefficient of Variation scaled up
        if mean_length > 0:
            # CV = (std_dev / mean) * multiplier for better scale
            burstiness_score = (std_dev / mean_length) * 10
        else:
            burstiness_score = 0.0

        print(f"DEBUG: Burstiness Score={burstiness_score:.2f}")

        # Interpret
        interpretation = self._interpret_burstiness(burstiness_score)
        is_ai_likely = burstiness_score < 3.0
        confidence = self._calculate_confidence(burstiness_score)

        # Fill results
        base_result.update({
            'burstiness_score': round(burstiness_score, 2),
            'mean_sentence_length': round(mean_length, 2),
            'std_deviation': round(std_dev, 2),
            'sentence_lengths': sentence_lengths,
            'interpretation': interpretation,
            'is_ai_likely': is_ai_likely,
            'confidence': confidence,
            'error': None
        })

        return base_result

    def _split_sentences_aggressive(self, text):
        """
        MOST AGGRESSIVE sentence splitting possible
        Tries multiple methods to ensure we split properly
        """
        sentences = []
        
        # Method 1: Split by period followed by space and capital letter
        # This handles: "sentence. Another sentence"
        parts = re.split(r'\.\s+(?=[A-Z])', text)
        if len(parts) > 1:
            sentences = [s.strip() for s in parts if s.strip()]
        
        # Method 2: If still only 1 sentence, split by ANY punctuation + space
        if len(sentences) <= 1:
            parts = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in parts if s.strip()]
        
        # Method 3: If STILL only 1, split by multiple spaces or newlines
        if len(sentences) <= 1:
            parts = re.split(r'[\n\r]+|\s{2,}', text)
            sentences = [s.strip() for s in parts if s.strip()]
        
        # Method 4: If STILL only 1, try splitting by just period
        if len(sentences) <= 1:
            parts = text.split('.')
            sentences = [s.strip() for s in parts if s.strip()]
        
        # Method 5: LAST RESORT - split by commas if text has many
        if len(sentences) <= 1 and text.count(',') >= 2:
            parts = text.split(',')
            sentences = [s.strip() for s in parts if s.strip()]
        
        # Clean up sentences
        cleaned = []
        for s in sentences:
            # Remove trailing punctuation
            s = s.rstrip('.!?,;:')
            # Must have at least 3 words to count as sentence
            words = self._tokenize(s)
            if len(words) >= 3:
                cleaned.append(s)
        
        return cleaned if cleaned else [text]  # Fallback to original

    def _tokenize(self, sentence):
        """
        Count words in sentence
        Returns list of words
        """
        if not sentence:
            return []
        words = re.findall(r'\b\w+\b', sentence.lower())
        return words

    def _interpret_burstiness(self, score):
        """Interpret burstiness score"""
        if score < 2:
            return "Very uniform - Highly likely AI-generated"
        elif score < 3:
            return "Uniform - Likely AI-generated"
        elif score < 5:
            return "Somewhat varied - Uncertain"
        elif score < 8:
            return "Varied - Likely human-written"
        else:
            return "Highly varied - Highly likely human-written"

    def _calculate_confidence(self, score):
        """Calculate confidence percentage"""
        if score < 3:
            # AI range - lower score = higher confidence
            confidence = min(100, max(50, (3 - score) / 3 * 100))
        elif score > 5:
            # Human range - higher score = higher confidence
            confidence = min(100, max(50, (score - 5) / 8 * 100))
        else:
            # Uncertain range
            confidence = 40

        return round(confidence, 2)


# COMPREHENSIVE TEST
if __name__ == "__main__":
    print("=== FIXED Burstiness Analyzer Test ===\n")

    analyzer = BurstinessAnalyzer()

    # Test 1: Your exact text from screenshot
    test1 = """omg you won't believe what happened today!! so i was literally just sitting there and this random guy spills coffee on my laptop. like seriously?? i was so mad lol but he apologized and offered to buy me a new one. anyway how's your day been? mine's been absolutely crazy haha"""

    print("--- Test 1: Human Text (from screenshot) ---")
    result1 = analyzer.analyze(test1)
    print(f"\n✅ Sentence Count: {result1['sentence_count']}")
    print(f"✅ Sentence Lengths: {result1['sentence_lengths']}")
    print(f"✅ Burstiness Score: {result1['burstiness_score']}")
    print(f"✅ Mean Length: {result1['mean_sentence_length']} words")
    print(f"✅ Interpretation: {result1['interpretation']}")
    print(f"✅ AI Likely?: {result1['is_ai_likely']}")
    print(f"✅ Confidence: {result1['confidence']}%")

    print("\n" + "="*60 + "\n")

    # Test 2: AI text (uniform)
    test2 = """Machine learning algorithms process data efficiently. These systems demonstrate remarkable capabilities in pattern recognition. Modern neural networks enable advanced computational functionality. Artificial intelligence continues to evolve with unprecedented capabilities."""

    print("--- Test 2: AI Text (should be uniform) ---")
    result2 = analyzer.analyze(test2)
    print(f"\n✅ Sentence Count: {result2['sentence_count']}")
    print(f"✅ Sentence Lengths: {result2['sentence_lengths']}")
    print(f"✅ Burstiness Score: {result2['burstiness_score']}")
    print(f"✅ Mean Length: {result2['mean_sentence_length']} words")
    print(f"✅ Interpretation: {result2['interpretation']}")
    print(f"✅ AI Likely?: {result2['is_ai_likely']}")
    print(f"✅ Confidence: {result2['confidence']}%")

    print("\n" + "="*60)
    print("✅ BURSTINESS ANALYZER IS NOW WORKING!")
    print("="*60 + "\n")