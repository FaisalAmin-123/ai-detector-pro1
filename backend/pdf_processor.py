"""
PDF Text Extraction for AI Detection
Simple and reliable PDF processing

Author: Faisal
Date: November 8, 2024
"""

import logging

# Try importing PDF libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("⚠️  PyPDF2 not installed. Run: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️  pdfplumber not installed. Run: pip install pdfplumber")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Extract text from PDF files
    """
    
    def __init__(self):
        """Initialize PDF processor"""
        if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "No PDF libraries available. Install with:\n"
                "pip install PyPDF2 pdfplumber"
            )
        
        logger.info("✅ PDF Processor initialized")
        logger.info(f"   PyPDF2: {'✅' if PYPDF2_AVAILABLE else '❌'}")
        logger.info(f"   pdfplumber: {'✅' if PDFPLUMBER_AVAILABLE else '❌'}")
    
    def extract_text(self, pdf_path):
        """
        Extract text from PDF file
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            dict: {
                'text': extracted text,
                'pages': number of pages,
                'method': extraction method used,
                'error': error message if failed
            }
        """
        result = {
            'text': '',
            'pages': 0,
            'method': None,
            'error': None
        }
        
        # Try pdfplumber first (better for complex PDFs)
        if PDFPLUMBER_AVAILABLE:
            try:
                logger.info(f"Extracting text from PDF using pdfplumber: {pdf_path}")
                text, pages = self._extract_with_pdfplumber(pdf_path)
                
                if text and len(text.strip()) > 0:
                    result['text'] = text
                    result['pages'] = pages
                    result['method'] = 'pdfplumber'
                    logger.info(f"✅ Extracted {len(text)} characters from {pages} pages")
                    return result
                else:
                    logger.warning("pdfplumber returned empty text, trying PyPDF2...")
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}, trying PyPDF2...")
        
        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                logger.info(f"Extracting text from PDF using PyPDF2: {pdf_path}")
                text, pages = self._extract_with_pypdf2(pdf_path)
                
                if text and len(text.strip()) > 0:
                    result['text'] = text
                    result['pages'] = pages
                    result['method'] = 'PyPDF2'
                    logger.info(f"✅ Extracted {len(text)} characters from {pages} pages")
                    return result
                else:
                    result['error'] = 'PDF appears to be empty or scanned (image-based)'
            except Exception as e:
                result['error'] = f'PDF extraction failed: {str(e)}'
                logger.error(f"PyPDF2 failed: {e}")
        
        # If both failed
        if not result['text']:
            result['error'] = result['error'] or 'Could not extract text from PDF'
            logger.error(f"❌ Failed to extract text from {pdf_path}")
        
        return result
    
    def _extract_with_pdfplumber(self, pdf_path):
        """Extract using pdfplumber"""
        text = ""
        page_count = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text.strip(), page_count
    
    def _extract_with_pypdf2(self, pdf_path):
        """Extract using PyPDF2"""
        text = ""
        page_count = 0
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text.strip(), page_count


# Test
if __name__ == "__main__":
    print("=== PDF Processor Test ===\n")
    
    if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
        print("❌ No PDF libraries installed!")
        print("\nInstall with:")
        print("  pip install PyPDF2 pdfplumber")
        exit(1)
    
    processor = PDFProcessor()
    
    print("\n✅ PDF Processor ready!")
    print("\nTo test with a real PDF:")
    print("  1. Place a PDF file in the same directory")
    print("  2. Uncomment the code below and add your filename")
    print()
    
    # UNCOMMENT TO TEST:
    result = processor.extract_text('Life_in_Bad_Situations_Essay.pdf')
    if result['error']:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Success!")
        print(f"   Method: {result['method']}")
        print(f"   Pages: {result['pages']}")
        print(f"   Text length: {len(result['text'])} characters")
        print(f"   Preview: {result['text'][:200]}...")