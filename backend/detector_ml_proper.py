"""
Professional ML-Based AI Detector
Uses text features + simple classifier for reliable detection

Author: Faisal
Date: November 6, 2024
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProperMLDetector:
    """
    ML detector using text embeddings + feature engineering
    """
    
    def __init__(self):
        """Initialize detector with embeddings model"""
        logger.info("Initializing Proper ML Detector...")
        
        # Use a reliable embedding model (not a classifier)
        logger.info("Loading embedding model...")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.eval()
        
        logger.info("✅ Embedding model loaded!")
        
        # Initialize classifier (will be trained on first use)
        self.classifier = None
        self.trained = False
        
        # Try to load pre-trained classifier
        self._load_or_create_classifier()
    
    def _load_or_create_classifier(self):
        """Load existing classifier or create with default training"""
        classifier_path = 'models/ml_classifier.pkl'
        
        if os.path.exists(classifier_path):
            try:
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                    self.trained = True
                logger.info("✅ Loaded pre-trained classifier")
                return
            except:
                pass
        
        # Create and train with synthetic data
        logger.info("Creating new classifier with bootstrap training...")
        self._bootstrap_train()
    
    def _extract_features(self, text):
        """
        Extract features from text
        Combines embeddings + statistical features
        """
        # Get embeddings
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Statistical features
        words = text.split()
        sentences = text.split('.')
        
        stats = [
            len(text),  # Text length
            len(words),  # Word count
            len(sentences),  # Sentence count
            np.mean([len(w) for w in words]) if words else 0,  # Avg word length
            len(set(words)) / len(words) if words else 0,  # Vocabulary diversity
            np.std([len(s) for s in sentences]) if len(sentences) > 1 else 0,  # Sentence length variance
            text.count(',') / len(words) if words else 0,  # Comma density
            text.count('!') + text.count('?'),  # Exclamation/question marks
        ]
        
        # Combine embeddings and stats
        features = np.concatenate([embeddings, stats])
        
        return features
    
    def _bootstrap_train(self):
        """
        Bootstrap training with synthetic examples
        This creates a basic classifier
        """
        logger.info("Bootstrap training with synthetic data...")
        
        # AI-like texts (formal, structured, consistent)
        ai_texts = [
            "Machine learning algorithms process data efficiently. These systems demonstrate remarkable capabilities.",
            "Artificial intelligence has revolutionized technology. Implementation of neural networks enables advanced functionality.",
            "The implementation of computational systems requires extensive analysis. These methodologies demonstrate significant improvements.",
            "Advanced algorithms facilitate data processing. Modern systems demonstrate enhanced performance metrics.",
            "Technological advancements enable sophisticated analysis. Implementation strategies demonstrate optimal results.",
        ]
        
        # Human-like texts (casual, varied, emotional)
        human_texts = [
            "omg you won't believe what happened today!! so crazy lol",
            "hey how are you? i'm doing pretty good, just chilling you know",
            "seriously?? that's insane! i can't believe it haha",
            "idk maybe we should just go with the first option? what do you think",
            "tbh i'm really tired today. work was exhausting but whatever",
        ]
        
        # Extract features
        X = []
        y = []
        
        for text in ai_texts:
            X.append(self._extract_features(text))
            y.append(1)  # 1 = AI
        
        for text in human_texts:
            X.append(self._extract_features(text))
            y.append(0)  # 0 = Human
        
        X = np.array(X)
        y = np.array(y)
        
        # Train simple classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)
        self.trained = True
        
        # Save classifier
        os.makedirs('models', exist_ok=True)
        with open('models/ml_classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        logger.info("✅ Bootstrap training complete!")
    
    def detect(self, text):
        """
        Detect if text is AI-generated
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Detection results
        """
        if not text or len(text.strip()) < 20:
            return {
                'error': 'Text too short',
                'is_ai_generated': None,
                'confidence': 0,
                'label': 'Error'
            }
        
        try:
            # Extract features
            features = self._extract_features(text).reshape(1, -1)
            
            # Predict
            prediction = self.classifier.predict(features)[0]
            probabilities = self.classifier.predict_proba(features)[0]
            
            is_ai = bool(prediction == 1)
            confidence = float(probabilities[prediction] * 100)
            
            return {
                'is_ai_generated': is_ai,
                'confidence': round(confidence, 2),
                'label': 'AI-Generated' if is_ai else 'Human-Written',
                'method': 'ML Embeddings + Random Forest',
                'probabilities': {
                    'human': round(float(probabilities[0]) * 100, 2),
                    'ai': round(float(probabilities[1]) * 100, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                'error': str(e),
                'is_ai_generated': None,
                'confidence': 0,
                'label': 'Error'
            }
    
    def improve_with_samples(self, ai_texts, human_texts):
        """
        Improve classifier with real samples
        
        Args:
            ai_texts (list): List of known AI texts
            human_texts (list): List of known human texts
        """
        logger.info(f"Improving classifier with {len(ai_texts)} AI and {len(human_texts)} human samples...")
        
        X = []
        y = []
        
        for text in ai_texts:
            X.append(self._extract_features(text))
            y.append(1)
        
        for text in human_texts:
            X.append(self._extract_features(text))
            y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Retrain
        self.classifier.fit(X, y)
        
        # Save
        with open('models/ml_classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
        
        logger.info("✅ Classifier improved and saved!")


# Test
if __name__ == "__main__":
    print("=== Proper ML Detector Test ===\n")
    
    detector = ProperMLDetector()
    
    # Test 1: AI text
    ai_text = """
    Machine learning algorithms have revolutionized the field of artificial intelligence.
    These sophisticated systems process vast amounts of data efficiently and systematically.
    The implementation of neural networks enables unprecedented capabilities for automation.
    """
    
    print("--- Test 1: AI Text ---")
    result1 = detector.detect(ai_text)
    print(f"Result: {result1['label']}")
    print(f"Confidence: {result1['confidence']}%")
    print(f"Probabilities: AI={result1['probabilities']['ai']}%, Human={result1['probabilities']['human']}%")
    print(f"Correct?: {'✅ YES' if result1['is_ai_generated'] else '❌ NO - Should be AI'}\n")
    
    # Test 2: Human text  
    human_text = """
    omg you won't believe what happened today!! so i was at the mall right?
    and this random guy literally spills his entire coffee on my shoes lol
    i was so mad but he apologized like crazy so whatever. how's your day going?
    """
    
    print("--- Test 2: Human Text ---")
    result2 = detector.detect(human_text)
    print(f"Result: {result2['label']}")
    print(f"Confidence: {result2['confidence']}%")
    print(f"Probabilities: AI={result2['probabilities']['ai']}%, Human={result2['probabilities']['human']}%")
    print(f"Correct?: {'✅ YES' if not result2['is_ai_generated'] else '❌ NO - Should be Human'}\n")
    
    print("=== Test Complete ===")