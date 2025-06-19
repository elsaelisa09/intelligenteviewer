# services/text_analysis.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import re
import numpy as np
from typing import Set, List, Tuple, Dict
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

class TextAnalyzer:
    def __init__(self):
        """
        Initialize TextAnalyzer with an embeddings model.
        
        Args:
            embeddings_model: Model for creating text embeddings
        """
        self.embeddings = embeddings

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using multiple metrics.
        
        Args:
            text1: First text for comparison
            text2: Second text for comparison
            
        Returns:
            float: Combined similarity score between 0 and 1
        """
        try:
            # Get embeddings
            embed1 = self.embeddings.embed_query(text1)
            embed2 = self.embeddings.embed_query(text2)
            
            # Cosine similarity
            vec1 = np.array(embed1)
            vec2 = np.array(embed2)
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # N-gram similarity
            ngram_sim = self._calculate_ngram_similarity(text1, text2)
            
            # Key phrase overlap
            phrases1 = self._find_key_phrases(text1)
            phrases2 = self._find_key_phrases(text2)
            phrase_sim = len(phrases1.intersection(phrases2)) / max(len(phrases1.union(phrases2)), 1)
            
            # Word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            word_sim = len(words1.intersection(words2)) / max(len(words1.union(words2)), 1)
            
            # Combined weighted similarity
            weights = {
                'cosine': 0.4,
                'ngram': 0.2,
                'phrase': 0.25,
                'word': 0.15
            }
            
            similarity = (
                weights['cosine'] * cosine_sim +
                weights['ngram'] * ngram_sim +
                weights['phrase'] * phrase_sim +
                weights['word'] * word_sim
            )
            
            return similarity
            
        except Exception as e:
            print(f"Error in semantic similarity calculation: {e}")
            return 0.0

    def has_significant_overlap(self, text1: str, text2: str, threshold: float = 0.75) -> bool:
        """
        Check for significant content overlap between two texts.
        
        Args:
            text1: First text for comparison
            text2: Second text for comparison
            threshold: Similarity threshold for considering significant overlap
            
        Returns:
            bool: True if significant overlap is found
        """
        sentences1 = re.split('[.!?]+', self._normalize_text(text1))
        sentences2 = re.split('[.!?]+', self._normalize_text(text2))
        
        for s1 in sentences1:
            s1 = s1.strip()
            if len(s1.split()) < 4:  # Skip very short sentences
                continue
            for s2 in sentences2:
                s2 = s2.strip()
                if len(s2.split()) < 4:
                    continue
                    
                sim = self.calculate_semantic_similarity(s1, s2)
                if sim > threshold:
                    return True
        return False

    def _get_ngrams(self, text: str, n: int) -> Tuple[Set[str], Set[str]]:
        """
        Get character and word n-grams from text.
        
        Args:
            text: Input text
            n: Size of n-grams
            
        Returns:
            Tuple containing sets of character and word n-grams
        """
        words = text.split()
        char_ngrams = set()
        word_ngrams = set()
        
        # Character n-grams
        for i in range(len(text) - n + 1):
            char_ngrams.add(text[i:i+n].lower())
            
        # Word n-grams
        for i in range(len(words) - n + 1):
            word_ngrams.add(' '.join(words[i:i+n]).lower())
            
        return char_ngrams, word_ngrams

    def _calculate_ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Calculate n-gram based similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            n: Size of n-grams
            
        Returns:
            float: Combined n-gram similarity score
        """
        char_ngrams1, word_ngrams1 = self._get_ngrams(text1, n)
        char_ngrams2, word_ngrams2 = self._get_ngrams(text2, n)
        
        char_similarity = len(char_ngrams1.intersection(char_ngrams2)) / max(len(char_ngrams1.union(char_ngrams2)), 1)
        word_similarity = len(word_ngrams1.intersection(word_ngrams2)) / max(len(word_ngrams1.union(word_ngrams2)), 1)
        
        return 0.3 * char_similarity + 0.7 * word_similarity

    def _find_key_phrases(self, text: str) -> Set[str]:
        """
        Extract key phrases and important information from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of extracted key phrases
        """
        named_entities = set(re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text))
        numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?(?:\s*(?:%|percent|kg|km|miles|dollars|euros))?\b', text.lower()))
        quotes = set(re.findall(r'"([^"]+)"', text))
        technical_terms = set(re.findall(r'\b(?:[A-Z][a-z]+(?:\d+)?|[A-Z]{2,})\b', text))
        
        return named_entities.union(numbers).union(quotes).union(technical_terms)

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving key information.
        
        Args:
            text: Input text
            
        Returns:
            str: Normalized text
        """
        text = re.sub(r'[^\w\s.,!?"\']', ' ', text)
        sentences = re.split('[.!?]+', text)
        normalized_sentences = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            normalized_words = []
            for word in words:
                if not any(c.isupper() for c in word[1:]):
                    word = word.lower()
                normalized_words.append(word)
            normalized_sentences.append(' '.join(normalized_words))
            
        return ' '.join(normalized_sentences)