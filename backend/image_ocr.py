"""
Image OCR for AI Detection
Extract text from images using Tesseract

Author: Faisal
Date: November 8, 2024
"""

import logging

# Try importing OCR libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL not installed. Run: pip install pillow")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not installed. Run: pip install pytesseract")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageOCR:
    """
    Extract text from images using OCR
    """
    
    def __init__(self):
        """Initialize OCR processor"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available. Install: pip install pillow")
        
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract not available. Install: pip install pytesseract")
        
        # Test if Tesseract is installed on system
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Image OCR initialized")
        except:
            raise RuntimeError(
                "Tesseract OCR not installed on system!\n"
                "Install instructions:\n"
                "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  Mac: brew install tesseract\n"
                "  Linux: sudo apt-get install tesseract-ocr"
            )
    
    def extract_text(self, image_path):
        """
        Extract text from image using OCR
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: {
                'text': extracted text,
                'confidence': OCR confidence,
                'dimensions': image dimensions,
                'error': error message if failed
            }
        """
        result = {
            'text': '',
            'confidence': 0,
            'dimensions': None,
            'format': None,
            'error': None
        }
        
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            # Open image
            image = Image.open(image_path)
            result['dimensions'] = image.size
            result['format'] = image.format
            
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Get detailed OCR data for confidence
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if not text.strip():
                result['error'] = 'No text detected in image'
                logger.warning("No text found in image")
                return result
            
            result['text'] = text.strip()
            result['confidence'] = round(avg_confidence, 2)
            
            logger.info(f"✅ Extracted {len(text)} characters")
            logger.info(f"   Confidence: {avg_confidence:.1f}%")
            
            return result
            
        except Exception as e:
            result['error'] = f'OCR failed: {str(e)}'
            logger.error(f"❌ OCR error: {e}")
            return result
    
    def preprocess_image(self, image_path, output_path=None):
        """
        Preprocess image for better OCR results
        (contrast enhancement, grayscale, etc.)
        
        Args:
            image_path (str): Input image path
            output_path (str): Output path (optional)
            
        Returns:
            str: Path to preprocessed image
        """
        try:
            from PIL import ImageEnhance, ImageFilter
            
            image = Image.open(image_path)
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2)
            
            # Sharpen
            image = image.filter(ImageFilter.SHARPEN)
            
            # Save
            if not output_path:
                output_path = image_path.replace('.', '_processed.')
            
            image.save(output_path)
            logger.info(f"✅ Image preprocessed: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image_path  # Return original


# Test
if __name__ == "__main__":
    print("=== Image OCR Test ===\n")
    
    if not PIL_AVAILABLE:
        print("❌ PIL not installed!")
        print("Install: pip install pillow")
        exit(1)
    
    if not TESSERACT_AVAILABLE:
        print("❌ pytesseract not installed!")
        print("Install: pip install pytesseract")
        exit(1)
    
    try:
        ocr = ImageOCR()
        print("✅ Image OCR ready!")
        print("\nTo test with a real image:")
        print("  1. Place an image with text in the same directory")
        print("  2. Uncomment the code below and add your filename")
        print()
        
        # UNCOMMENT TO TEST:
        result = ocr.extract_text('your_image.png')
        if result['error']:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success!")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Format: {result['format']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Text length: {len(result['text'])} characters")
            print(f"   Preview: {result['text'][:200]}...")
        
    except RuntimeError as e:
        print(f"❌ {e}")
        print("\n⚠️  You need to install Tesseract OCR on your system!")