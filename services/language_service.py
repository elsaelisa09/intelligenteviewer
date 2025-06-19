# language_service.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import langid
from deep_translator import GoogleTranslator
import logging
from typing import Tuple, Dict, Optional, List
from functools import lru_cache
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Indonesian language model to langid
# This helps improve detection for Indonesian specifically
langid.set_languages(['en', 'id', 'ms', 'jv'])  # English, Indonesian, Malay, Javanese

class LanguageService:
    """Service to handle language detection and translation."""
    
    def __init__(self):
        """Initialize the language service."""
        self._translation_cache = {}
        
        # Indonesian language patterns (common words and patterns)
        self.indonesian_patterns = [
            r'\b(yang|dengan|untuk|dari|kepada|adalah|tersebut|ini|itu)\b',
            r'\b(apa|siapa|mengapa|bagaimana|kapan|dimana|kemana)\b',
            r'\b(dan|atau|tetapi|namun|karena|sebab|jika|maka)\b',
            r'\b(di|ke|pada|dalam|tentang|mengenai|oleh)\b',
            r'\bmeng[a-z]+\b',  # Words starting with "meng"
            r'\bber[a-z]+\b',   # Words starting with "ber"
            r'\bpe[a-z]+an\b'   # Words with "pe-an" pattern
        ]
        
        logger.info("Language_service.py - LanguageService - init : Language service initialized with enhanced Indonesian detection")
        
    def detect_language(self, text: str, log_detection: bool = False) -> Tuple[str, float]:
        """
        Detect the language of the given text with improved Indonesian detection.
        
        Args:
            text (str): Text to detect language from
            log_detection (bool): Whether to log the language detection (default: False)
            
        Returns:
            tuple: (language_code, confidence)
        """
        try:
            # Ensure input is a string and has meaningful content
            if not text or not isinstance(text, str):
                if log_detection:
                    logger.warning("Language_service.py - LanguageService - detect_language : Invalid input for language detection")
                return "en", 0.0
            
            # Remove extra whitespace
            text = text.strip()
            
            # Check text length
            if len(text) < 5:
                if log_detection:
                    logger.warning("Language_service.py - LanguageService - detect_language : Text too short for reliable language detection")
                return "en", 0.0
            
            # Enhanced Indonesian detection
            # Check for common Indonesian patterns before using langid
            indonesian_matches = sum(1 for pattern in self.indonesian_patterns if re.search(pattern, text.lower()))
            total_words = len(text.split())
            
            # If we have a significant number of Indonesian pattern matches
            if total_words > 3 and indonesian_matches >= 2:
                # This text has strong Indonesian indicators
                if log_detection:
                    logger.info(f"Language_service.py - LanguageService - detect_language : Detected Indonesian via patterns ({indonesian_matches}/{total_words} words matched)")
                return "id", 0.9
            
            # Use langid for language detection
            lang, confidence = langid.classify(text)
            
            # Normalize confidence to be between 0 and 1
            confidence = min(1.0, abs(confidence) / 10.0)
            
            # Additional heuristics for Indonesian texts that might be misclassified
            if lang == "en" and confidence < 0.8:
                # For ambiguous cases, check for Indonesian patterns
                if indonesian_matches > 0:
                    # Adjust detection to Indonesian with moderate confidence
                    lang = "id"
                    confidence = 0.7 + (indonesian_matches / (total_words * 2))
                    if log_detection:
                        logger.info(f"Language_service.py - LanguageService - detect_language : Corrected to Indonesian ({indonesian_matches}/{total_words} words matched)")
            
            if log_detection:
                logger.info(f"Language_service.py - LanguageService - detect_language : Detected language: {lang} with confidence: {confidence}")
            
            return lang, confidence
        except Exception as e:
            if log_detection:
                logger.error(f"Language_service.py - LanguageService - detect_language : Error detecting language: {str(e)}")
        return "en", 0.0
    
    @lru_cache(maxsize=100)
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source language to target language using deep_translator.
        Uses caching to avoid redundant translations.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            str: Translated text
        """
        if not text or source_lang == target_lang:
            return text
        source_lang = source_lang.replace('"', '')
        target_lang = target_lang.replace('"', '')
        # Create cache key
        cache_key = f"{source_lang}|{target_lang}|{text[:50]}"
        
        # Check cache first
        if cache_key in self._translation_cache:
            logger.info(f"Language_service.py - LanguageService - translate_text : Translation cache hit for: {cache_key[:30]}...")
            return self._translation_cache[cache_key]
        
        try:
            # Initialize the translator
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            
            # Perform translation
            translated_text = translator.translate(text)
            
            # Cache the result
            self._translation_cache[cache_key] = translated_text
            logger.info(f"Language_service.py - LanguageService - translate_text : Translated from {source_lang} to {target_lang}: {text[:30]}... -> {translated_text[:30]}...")
            
            return translated_text
        except Exception as e:
            logger.error(f"Language_service.py - LanguageService - translate_text : Translation error: {str(e)}")
            return text
        
    def process_query_for_chunks(self, query: str, chunks: List[Dict], log_detection: bool = True) -> Tuple[str, str, Dict]:
        """
        Process query for a set of chunks, ensuring language compatibility.
        Enhanced to better handle Indonesian.
        
        Args:
            query (str): User query
            chunks (List[Dict]): List of chunks to search
            log_detection (bool): Whether to log language detection details
            
        Returns:
            Tuple[str, str, Dict]: (processed_query, query_lang, translations)
                - processed_query: The query to use (original or translated)
                - query_lang: Detected query language
                - translations: Dictionary of translations for caching
        """
        # Detect query language with optional logging
        query_lang, query_confidence = self.detect_language(query, log_detection)
        
        # Determine main language of chunks
        chunks_langs = []
        for chunk in chunks:
            # Get text from chunk
            text = chunk.get('text', '') if isinstance(chunk, dict) else ''
            if not text:
                continue
                
            # First check for Indonesian patterns
            indonesian_matches = sum(1 for pattern in self.indonesian_patterns if re.search(pattern, text.lower()))
            if indonesian_matches >= 2:
                chunks_langs.append("id")
                if log_detection:
                    logger.info(f"Language_service.py - LanguageService - process_query_for_chunks : Detected Indonesian content in chunk via patterns")
            else:
                # Otherwise use standard detection
                lang, _ = self.detect_language(text, log_detection=False)
                chunks_langs.append(lang)
        
        # Default to English if no languages detected
        if not chunks_langs:
            chunk_lang = 'en'
        else:
            # Use most common language in chunks
            from collections import Counter
            chunk_lang = Counter(chunks_langs).most_common(1)[0][0]
        
        if log_detection:
            logger.info(f"Language_service.py - LanguageService - process_query_for_chunks : Query language: {query_lang} (confidence: {query_confidence}), Chunks language: {chunk_lang}")
        
        # Initialize translation cache for this query
        translations = {
            "query": {
                "original": query,
                "lang": query_lang
            }
        }
        
        # More aggressive translation threshold - translate if languages differ
        should_translate = (query_lang != chunk_lang)
        
        # Perform translation if needed
        if should_translate:
            translated_query = self.translate_text(query, query_lang, chunk_lang)
            
            # Update translations cache
            translations["query"]["translated"] = translated_query
            translations["query"]["translated_lang"] = chunk_lang
            
            if log_detection:
                logger.info(f"Language_service.py - LanguageService - process_query_for_chunks : Translated query from {query_lang} to {chunk_lang}: {query} -> {translated_query}")
            
            return translated_query, query_lang, translations
        
        # No translation needed
        if log_detection:
            logger.info("Language_service.py - LanguageService - process_query_for_chunks : No translation needed. Using original query.")
        return query, query_lang, translations
            
    def get_main_language(self, chunks: List[Dict], log_detection: bool = False) -> str:
        """
        Determine the main language of a collection of chunks.
        
        Args:
            chunks (List[Dict]): List of chunk dictionaries containing text
            log_detection (bool): Whether to log language detection details
            
        Returns:
            str: Most common language code
        """
        if not chunks:
            return "en"
            
        # Count language occurrences with improved Indonesian detection
        lang_counts = {}
        for chunk in chunks:
            # Safely get text from the chunk
            text = chunk.get('text', '') if isinstance(chunk, dict) else ''
            
            # Only process non-empty strings
            if not text or not isinstance(text, str):
                continue
                
            # Check for Indonesian patterns first
            indonesian_matches = sum(1 for pattern in self.indonesian_patterns if re.search(pattern, text.lower()))
            if indonesian_matches >= 2:
                # Strong indicator of Indonesian
                lang_counts["id"] = lang_counts.get("id", 0) + 1
                continue
                
            # Detect language
            lang, confidence = self.detect_language(text, log_detection=False)
            
            # Only count languages with reasonable confidence
            if confidence > 0.5:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
        # Return most common language
        if not lang_counts:
            return "en"
            
        return max(lang_counts.items(), key=lambda x: x[1])[0]