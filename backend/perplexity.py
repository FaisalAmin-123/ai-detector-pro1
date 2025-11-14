# """
# Perplexity-based AI Text Detection using GPT-2
# Author: Faisal
# """

# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
# import torch
# import math

# class PerplexityCalculator:
#     def __init__(self):
#         print("Loading GPT-2 model for perplexity calculation...")
#         self.model = GPT2LMHeadModel.from_pretrained("gpt2")
#         self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#         self.model.eval()
#         print("✅ GPT-2 loaded successfully!\n")

#     def calculate_perplexity(self, text):
#         encodings = self.tokenizer(text, return_tensors="pt")
#         with torch.no_grad():
#             outputs = self.model(**encodings, labels=encodings["input_ids"])
#             loss = outputs.loss
#             perplexity = math.exp(loss.item())

#         # Interpret results
#         if perplexity < 20:
#             interpretation = "Very predictable - Highly likely AI-generated"
#             is_ai = True
#             confidence = 90
#         elif perplexity < 60:
#             interpretation = "Moderately predictable - Possibly AI-generated"
#             is_ai = True
#             confidence = 70
#         else:
#             interpretation = "Highly variable - Likely human-written"
#             is_ai = False
#             confidence = 85

#         return {
#             "perplexity": round(perplexity, 2),
#             "interpretation": interpretation,
#             "is_ai_likely": is_ai,
#             "confidence": confidence
#         }

# if __name__ == "__main__":
#     print("=== Perplexity Calculator Test ===\n")
#     calc = PerplexityCalculator()

#     ai_text = "Quantum computing is an advanced field of computer science that leverages the principles of quantum mechanics to process information in ways that classical computers cannot. Unlike traditional computers, which use bits that represent either 0 or 1, quantum computers use quantum bits, or qubits, which can exist in multiple states simultaneously through a property called superposition. Additionally, qubits can become entangled, meaning the state of one qubit can depend on the state of another, allowing quantum computers to perform complex calculations at exponentially faster rates for certain problems. This technology has the potential to revolutionize fields such as cryptography, drug discovery, artificial intelligence, and optimization, although it is still in its early stages of development and faces significant technical challenges."
#     human_text = "hey kamran today i was very busy with my internship project.This project is about toanalysis Nifty500 and suggest weather a person should buy or sell.For this project i have to create an automation in n8n that checks the Nifty500 and their history price changes.then it gives a result for a person to buy or sell .so i created this project ,it was very intresting i learnt very things"

#     print("--- Test 1: AI-Generated Text ---")
#     result1 = calc.calculate_perplexity(ai_text)
#     print(f"Text:\n{ai_text[:100]}...")
#     print(f"Perplexity: {result1['perplexity']}")
#     print(f"Interpretation: {result1['interpretation']}\n")

#     print("--- Test 2: Human-Written Text ---")
#     result2 = calc.calculate_perplexity(human_text)
#     print(f"Text:\n{human_text[:100]}...")
#     print(f"Perplexity: {result2['perplexity']}")
#     print(f"Interpretation: {result2['interpretation']}\n")

#     print("=== Test Complete ===")



# """
# Perplexity Calculator for AI Detection
# Measures how predictable text is

# Lower perplexity (< 50) = Likely AI-generated
# Higher perplexity (> 100) = Likely Human-written

# Author: Faisal
# Date: November 4, 2024
# """

# import torch
# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
# import numpy as np

# class PerplexityCalculator:
#     """
#     Calculate perplexity of text using GPT-2 model
#     Perplexity = measure of how "surprised" the model is by the text
#     """
    
#     def __init__(self):
#         """
#         Initialize with GPT-2 model
#         """
#         print("Loading GPT-2 model for perplexity calculation...")
        
#         # Load GPT-2 model and tokenizer
#         self.model = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        
#         # Set model to evaluation mode
#         self.model.eval()
        
#         print("✅ GPT-2 loaded successfully!")
    
#     def calculate_perplexity(self, text):
#         """
#         Calculate perplexity score for given text
        
#         Args:
#             text (str): Text to analyze
            
#         Returns:
#             dict: Perplexity score and interpretation
#         """
#         # Check for empty text
#         if not text or len(text.strip()) == 0:
#             return {
#                 'error': 'Text is empty',
#                 'perplexity': None,
#                 'interpretation': 'Error',
#                 'is_ai_likely': None,
#                 'confidence': 0
#             }
        
#         # Tokenize text
#         encodings = self.tokenizer(text, return_tensors='pt')
        
#         # Get input IDs
#         input_ids = encodings.input_ids
        
#         # Maximum sequence length
#         max_length = self.model.config.n_positions
#         stride = 512
        
#         # Calculate negative log-likelihoods
#         nlls = []
        
#         # Process text in chunks
#         for i in range(0, input_ids.size(1), stride):
#             begin_loc = max(i + stride - max_length, 0)
#             end_loc = min(i + stride, input_ids.size(1))
#             trg_len = end_loc - i
            
#             input_ids_chunk = input_ids[:, begin_loc:end_loc]
#             target_ids = input_ids_chunk.clone()
#             target_ids[:, :-trg_len] = -100
            
#             with torch.no_grad():
#                 outputs = self.model(input_ids_chunk, labels=target_ids)
#                 neg_log_likelihood = outputs.loss * trg_len
            
#             nlls.append(neg_log_likelihood)
        
#         # Calculate perplexity
#         perplexity = torch.exp(torch.stack(nlls).sum() / end_loc).item()
        
#         # Interpret result
#         interpretation = self._interpret_perplexity(perplexity)
        
#         return {
#             'perplexity': round(perplexity, 2),
#             'interpretation': interpretation,
#             'is_ai_likely': perplexity < 50,
#             'confidence': self._calculate_confidence(perplexity)
#         }
    
#     def _interpret_perplexity(self, perplexity):
#         """
#         Interpret perplexity score
        
#         Args:
#             perplexity (float): Calculated perplexity
            
#         Returns:
#             str: Human-readable interpretation
#         """
#         if perplexity < 30:
#             return "Very predictable - Highly likely AI-generated"
#         elif perplexity < 50:
#             return "Predictable - Likely AI-generated"
#         elif perplexity < 100:
#             return "Moderately predictable - Uncertain"
#         elif perplexity < 200:
#             return "Unpredictable - Likely human-written"
#         else:
#             return "Very unpredictable - Highly likely human-written"
    
#     def _calculate_confidence(self, perplexity):
#         """
#         Calculate confidence percentage
        
#         Args:
#             perplexity (float): Calculated perplexity
            
#         Returns:
#             float: Confidence percentage
#         """
#         if perplexity < 50:
#             # AI range
#             confidence = min(100, 100 - perplexity)
#         elif perplexity > 100:
#             # Human range
#             confidence = min(100, (perplexity - 100) / 2)
#         else:
#             # Uncertain range
#             confidence = 50
        
#         return round(confidence, 2)


# # Test the calculator
# if __name__ == "__main__":
#     print("=== Perplexity Calculator Test ===\n")
    
#     # Create calculator instance
#     calculator = PerplexityCalculator()
    
#     # Test 1: AI-generated text
#     ai_text = "Quantum computing is an advanced field of computer science that leverages the principles of quantum mechanics to process information in ways that classical computers cannot. Unlike traditional computers, which use bits that represent either 0 or 1, quantum computers use quantum bits, or qubits, which can exist in multiple states simultaneously through a property called superposition. Additionally, qubits can become entangled, meaning the state of one qubit can depend on the state of another, allowing quantum computers to perform complex calculations at exponentially faster rates for certain problems. This technology has the potential to revolutionize fields such as cryptography, drug discovery, artificial intelligence, and optimization, although it is still in its early stages of development and faces significant technical challenges."
#     print("\n--- Test 1: AI-Generated Text ---")
#     print(f"Text: {ai_text[:80]}...")
#     result1 = calculator.calculate_perplexity(ai_text)
#     print(f"Perplexity: {result1['perplexity']}")
#     print(f"Interpretation: {result1['interpretation']}")
#     print(f"AI Likely?: {result1['is_ai_likely']}")
#     print(f"Confidence: {result1['confidence']}%")
    
#     # Test 2: Human-written text
#     human_text =  "hey kamran today i was very busy with my internship project.This project is about toanalysis Nifty500 and suggest weather a person should buy or sell.For this project i have to create an automation in n8n that checks the Nifty500 and their history price changes.then it gives a result for a person to buy or sell .so i created this project ,it was very intresting i learnt very things"
    
#     print("\n--- Test 2: Human-Written Text ---")
#     print(f"Text: {human_text[:80]}...")
#     result2 = calculator.calculate_perplexity(human_text)
#     print(f"Perplexity: {result2['perplexity']}")
#     print(f"Interpretation: {result2['interpretation']}")
#     print(f"AI Likely?: {result2['is_ai_likely']}")
#     print(f"Confidence: {result2['confidence']}%")
    
#     print("\n=== Test Complete ===")



"""
Perplexity Calculator for AI Detection (corrected)

Measures how predictable text is using GPT-2.
Lower perplexity (< 50) = Likely AI-generated
Higher perplexity (> 100) = Likely Human-written

Author: Faisal (original)
Fixed and improved: Assistant
Date: 2025-11-05
"""

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math


class PerplexityCalculator:
    """
    Calculate perplexity of text using GPT-2 model
    Perplexity = exp(total_negative_log_likelihood / total_predicted_tokens)
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        """
        model_name: HF model id or local path
        device: 'cuda' or 'cpu' or None to auto-select
        """
        # device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading GPT-2 model for perplexity calculation on device={self.device}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

        # move model to device and set eval
        self.model.to(self.device)
        self.model.eval()

        # model's max positions (context length)
        self.max_length = getattr(self.model.config, "n_positions", 1024)

        print("✅ GPT-2 loaded successfully!")

    def calculate_perplexity(self, text: str) -> dict:
        """
        Calculate perplexity for given text.

        Returns dictionary with:
          - perplexity (float)
          - interpretation (str)
          - is_ai_likely (bool)
          - confidence (float)
        """
        if not text or len(text.strip()) == 0:
            return {
                "error": "Text is empty",
                "perplexity": None,
                "interpretation": "Error",
                "is_ai_likely": None,
                "confidence": 0.0,
            }

        # Tokenize and move to device
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)  # shape (1, seq_len)
        seq_len = input_ids.size(1)

        # window/stride parameters
        # We'll step by 'stride' tokens each window to cover the whole sequence.
        # stride should be <= max_length.
        stride = min(512, self.max_length)  # reasonable default; ensure <= max_length

        nlls = []
        total_predicted_tokens = 0

        # Process in sliding windows so very long text is handled
        for i in range(0, seq_len, stride):
            begin_loc = max(i + stride - self.max_length, 0)
            end_loc = min(i + stride, seq_len)
            trg_len = end_loc - i  # number of tokens we will predict in this step

            if trg_len <= 0:
                continue

            input_ids_chunk = input_ids[:, begin_loc:end_loc].to(self.device)  # (1, chunk_len)
            target_ids = input_ids_chunk.clone()
            # mask out tokens we don't want to predict (so loss computes only for last trg_len tokens)
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids_chunk, labels=target_ids)
                # outputs.loss is average negative log-likelihood over non -100 tokens
                # multiply by trg_len to get sum NLL for this chunk
                neg_log_likelihood = outputs.loss * trg_len

            # accumulate
            nlls.append(neg_log_likelihood)
            total_predicted_tokens += trg_len

        if total_predicted_tokens == 0:
            # safety: shouldn't happen, but guard against division by zero
            return {
                "error": "Could not compute perplexity (no tokens processed)",
                "perplexity": None,
                "interpretation": "Error",
                "is_ai_likely": None,
                "confidence": 0.0,
            }

        # sum NLLs (ensure on CPU for final math if needed)
        sum_nll = torch.stack(nlls).sum()
        # Perplexity = exp(sum_nll / total_predicted_tokens)
        # move to CPU/float for exp to avoid device surprises
        avg_nll = (sum_nll / total_predicted_tokens).to("cpu").item()
        perplexity = math.exp(avg_nll)

        interpretation = self._interpret_perplexity(perplexity)
        confidence = self._calculate_confidence(perplexity)

        return {
            "perplexity": round(perplexity, 2),
            "interpretation": interpretation,
            "is_ai_likely": perplexity < 50,
            "confidence": confidence,
        }

    def _interpret_perplexity(self, perplexity: float) -> str:
        if perplexity < 30:
            return "Very predictable - Highly likely AI-generated"
        elif perplexity < 50:
            return "Predictable - Likely AI-generated"
        elif perplexity < 100:
            return "Moderately predictable - Uncertain"
        elif perplexity < 200:
            return "Unpredictable - Likely human-written"
        else:
            return "Very unpredictable - Highly likely human-written"

    def _calculate_confidence(self, perplexity: float) -> float:
        """
        Keep your original heuristic mapping (you can tune this later).
        """
        if perplexity < 50:
            confidence = min(100.0, 100.0 - perplexity)
        elif perplexity > 100:
            confidence = min(100.0, (perplexity - 100.0) / 2.0)
        else:
            confidence = 50.0
        return round(confidence, 2)


if __name__ == "__main__":
    print("=== Perplexity Calculator Test ===\n")

    calc = PerplexityCalculator()

    ai_text = (
        "Quantum computing is an advanced field of computer science that leverages the "
        "principles of quantum mechanics to process information in ways that classical computers cannot. "
        "Unlike traditional computers, which use bits that represent either 0 or 1, quantum computers use quantum bits, "
        "or qubits, which can exist in multiple states simultaneously through a property called superposition. "
        "Additionally, qubits can become entangled, meaning the state of one qubit can depend on the state of another, "
        "allowing quantum computers to perform complex calculations at exponentially faster rates for certain problems. "
        "This technology has the potential to revolutionize fields such as cryptography, drug discovery, artificial intelligence, "
        "and optimization, although it is still in its early stages of development and faces significant technical challenges."
    )
    print("\n--- Test 1: AI-Generated Text ---")
    print(f"Text: {ai_text[:80]}...")
    r1 = calc.calculate_perplexity(ai_text)
    print(f"Perplexity: {r1['perplexity']}")
    print(f"Interpretation: {r1['interpretation']}")
    print(f"AI Likely?: {r1['is_ai_likely']}")
    print(f"Confidence: {r1['confidence']}%")

    human_text = (
        "hey kamran today i was very busy with my internship project.This project is about toanalysis Nifty500 "
        "and suggest weather a person should buy or sell.For this project i have to create an automation in n8n that checks "
        "the Nifty500 and their history price changes.then it gives a result for a person to buy or sell .so i created this project ,"
        "it was very intresting i learnt very things"
    )
    print("\n--- Test 2: Human-Written Text ---")
    print(f"Text: {human_text[:80]}...")
    r2 = calc.calculate_perplexity(human_text)
    print(f"Perplexity: {r2['perplexity']}")
    print(f"Interpretation: {r2['interpretation']}")
    print(f"AI Likely?: {r2['is_ai_likely']}")
    print(f"Confidence: {r2['confidence']}%")

    print("\n=== Test Complete ===")
