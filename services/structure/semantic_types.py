import re
import spacy
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

class SemanticType(Enum):
    """Enumeration of semantic types for document chunks."""
    DEFINITION = auto()
    PROCEDURE = auto()
    EXAMPLE = auto()
    SUMMARY = auto()
    TABLE_REFERENCE = auto()
    CODE = auto()
    EQUATION = auto()
    LIST = auto()
    CITATION = auto()
    QUESTION = auto()
    GENERAL = auto()

@dataclass
class TypePattern:
    """Dataclass for semantic type patterns."""
    patterns: List[str]
    keywords: Set[str]
    indicators: Set[str]
    min_confidence: float = 0.5

class SemanticTypeDetector:
    """
    Detector for semantic types in text content.
    """
    
    def __init__(self):
        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize pattern registry
        self._initialize_patterns()
        
        # Cache for parsed documents
        self.doc_cache = {}
        
    def _initialize_patterns(self):
        """Initialize semantic type patterns and rules."""
        self.type_patterns = {
            SemanticType.DEFINITION: TypePattern(
                patterns=[
                    r'(?:is|are) defined as\s+[^.]+\.',
                    r'refers to\s+[^.]+\.',
                    r'means\s+[^.]+\.',
                    r'^definition:\s+[^.]+\.',
                    r'(?:is|are) (?:a|an|the)\s+[^.]+\.'
                ],
                keywords={
                    'define', 'definition', 'means', 'refers', 'called',
                    'known as', 'understood as', 'described as'
                },
                indicators={
                    'is', 'are', 'refers', 'means', 'defines', 'represents'
                }
            ),
            
            SemanticType.PROCEDURE: TypePattern(
                patterns=[
                    r'(?:steps?|procedure|process|method)\s*:\s*(?:\d+\.|\-|\*)',
                    r'(?:first|second|third|finally|lastly)(?:\s*,)?\s+[^.]+\.',
                    r'^\d+\.\s+[^.]+\.',
                    r'(?:follow|following) (?:these|the) steps'
                ],
                keywords={
                    'step', 'procedure', 'process', 'method', 'instruction',
                    'guide', 'how to', 'workflow', 'sequence'
                },
                indicators={
                    'first', 'then', 'next', 'finally', 'lastly', 'following'
                }
            ),
            
            SemanticType.EXAMPLE: TypePattern(
                patterns=[
                    r'(?:example|instance|illustration)[\s:]+',
                    r'(?:e\.g\.|i\.e\.|for example|for instance)',
                    r'such as\s+[^.]+\.',
                    r'(?:consider|take|let\'s look at)\s+[^.]+\.'
                ],
                keywords={
                    'example', 'instance', 'illustration', 'case', 'scenario',
                    'sample', 'demonstration'
                },
                indicators={
                    'e.g.', 'i.e.', 'such as', 'like', 'consider'
                }
            ),
            
            SemanticType.SUMMARY: TypePattern(
                patterns=[
                    r'(?:in summary|to summarize|in conclusion)',
                    r'^summary:\s+[^.]+\.',
                    r'(?:key points|main points)\s*:',
                    r'(?:to conclude|in closing|finally)'
                ],
                keywords={
                    'summary', 'conclusion', 'overview', 'recap',
                    'key points', 'main points', 'takeaway'
                },
                indicators={
                    'summarize', 'conclude', 'summary', 'finally', 'overall'
                }
            ),
            
            SemanticType.TABLE_REFERENCE: TypePattern(
                patterns=[
                    r'(?:table|tbl\.|fig\.|figure)\s+\d+',
                    r'as shown in\s+[^.]+\.',
                    r'illustrated in\s+[^.]+\.',
                    r'(?:refer to|see)\s+(?:table|figure)'
                ],
                keywords={
                    'table', 'figure', 'diagram', 'chart', 'graph',
                    'illustration', 'visualization'
                },
                indicators={
                    'shown', 'illustrated', 'depicted', 'displayed'
                }
            ),
            
            SemanticType.CODE: TypePattern(
                patterns=[
                    r'```[\s\S]*?```',
                    r'(?:code|function|class|method)\s*{',
                    r'(?:def|class|function)\s+\w+\s*[\(\{]'
                ],
                keywords={
                    'code', 'function', 'class', 'method', 'implementation',
                    'algorithm', 'snippet'
                },
                indicators={
                    'implement', 'code', 'function', 'class', 'return'
                }
            ),
            
            SemanticType.EQUATION: TypePattern(
                patterns=[
                    r'\$.*?\$',
                    r'\\begin{equation}.*?\\end{equation}',
                    r'(?:formula|equation):\s*[^.]+\.'
                ],
                keywords={
                    'equation', 'formula', 'calculation', 'expression',
                    'mathematical', 'computation'
                },
                indicators={
                    'equals', 'calculated', 'computed', 'derived'
                }
            ),
            
            SemanticType.LIST: TypePattern(
                patterns=[
                    r'(?:following|these|here are|:)\s*(?:\d+\.|\-|\*)',
                    r'^\s*[•\-\*]\s+[^.]+\.',
                    r'^\s*\d+\.\s+[^.]+\.'
                ],
                keywords={
                    'list', 'items', 'elements', 'points', 'components',
                    'factors', 'aspects'
                },
                indicators={
                    'include', 'consist', 'comprise', 'contain'
                }
            ),
            
            SemanticType.CITATION: TypePattern(
                patterns=[
                    r'\[\d+\]',
                    r'\(\w+\s*(?:et al\.?)?(?:,|\s)\s*\d{4}\)',
                    r'according to\s+[^.]+\.'
                ],
                keywords={
                    'reference', 'citation', 'source', 'cited', 'according',
                    'stated', 'reported'
                },
                indicators={
                    'cited', 'referenced', 'reported', 'stated'
                }
            ),
            
            SemanticType.QUESTION: TypePattern(
                patterns=[
                    r'\?\s*$',
                    r'^(?:what|who|when|where|why|how)\s+[^.]+\?',
                    r'(?:question|query|problem):\s+[^.]+\?'
                ],
                keywords={
                    'question', 'query', 'problem', 'ask', 'inquire',
                    'wonder', 'explain'
                },
                indicators={
                    'what', 'who', 'when', 'where', 'why', 'how'
                }
            )
        }

    def detect_type(self, text: str) -> Tuple[SemanticType, float]:
        """
        Detect the semantic type of text content with confidence score.
        
        Args:
            text (str): Text content to analyze
            
        Returns:
            Tuple[SemanticType, float]: Detected type and confidence score
        """
        try:
            # Initialize scores for each type
            type_scores = defaultdict(float)
            
            # Get spaCy doc (use cache if available)
            doc = self.doc_cache.get(text)
            if not doc:
                doc = self.nlp(text)
                self.doc_cache[text] = doc
            
            # Calculate scores for each type
            for sem_type, type_pattern in self.type_patterns.items():
                score = self._calculate_type_score(text, doc, type_pattern)
                if score >= type_pattern.min_confidence:
                    type_scores[sem_type] = score
            
            # Select type with highest score
            if type_scores:
                best_type = max(type_scores.items(), key=lambda x: x[1])
                return best_type[0], best_type[1]
            
            return SemanticType.GENERAL, 1.0
            
        except Exception as e:
            self.logger.error(f"Error in type detection: {str(e)}")
            return SemanticType.GENERAL, 1.0

    def _calculate_type_score(self, text: str, doc: spacy.tokens.Doc, 
                            type_pattern: TypePattern) -> float:
        """
        Calculate confidence score for a semantic type.
        
        Args:
            text (str): Raw text content
            doc (spacy.tokens.Doc): Parsed spaCy document
            type_pattern (TypePattern): Pattern to match against
            
        Returns:
            float: Confidence score between 0 and 1
        """
        score_components = []
        
        # Pattern matching score
        pattern_matches = 0
        for pattern in type_pattern.patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                pattern_matches += 1
        pattern_score = pattern_matches / max(1, len(type_pattern.patterns))
        score_components.append(pattern_score)
        
        # Keyword presence score
        text_lower = text.lower()
        keyword_matches = sum(1 for kw in type_pattern.keywords 
                            if kw.lower() in text_lower)
        keyword_score = keyword_matches / max(1, len(type_pattern.keywords))
        score_components.append(keyword_score)
        
        # Indicator word score
        indicator_matches = sum(1 for ind in type_pattern.indicators 
                              if ind.lower() in text_lower)
        indicator_score = indicator_matches / max(1, len(type_pattern.indicators))
        score_components.append(indicator_score)
        
        # Linguistic features score
        linguistic_score = self._calculate_linguistic_score(doc, type_pattern)
        score_components.append(linguistic_score)
        
        # Calculate weighted average
        weights = [0.4, 0.3, 0.2, 0.1]  # Adjust weights as needed
        final_score = sum(s * w for s, w in zip(score_components, weights))
        
        return final_score

    def _calculate_linguistic_score(self, doc: spacy.tokens.Doc, 
                                  type_pattern: TypePattern) -> float:
        """
        Calculate score based on linguistic features.
        
        Args:
            doc (spacy.tokens.Doc): Parsed spaCy document
            type_pattern (TypePattern): Pattern to match against
            
        Returns:
            float: Linguistic feature score between 0 and 1
        """
        score = 0.0
        total_features = 0
        
        # Check sentence structure
        for sent in doc.sents:
            total_features += 1
            
            # Definition pattern
            if type_pattern == self.type_patterns[SemanticType.DEFINITION]:
                if any(token.dep_ == "attr" for token in sent):
                    score += 1
            
            # Procedure pattern
            elif type_pattern == self.type_patterns[SemanticType.PROCEDURE]:
                if any(token.dep_ == "nummod" for token in sent):
                    score += 1
            
            # Example pattern
            elif type_pattern == self.type_patterns[SemanticType.EXAMPLE]:
                if any(token.text.lower() in {"like", "such"} for token in sent):
                    score += 1
            
            # Add more linguistic patterns for other types...
        
        return score / max(1, total_features)

    def clear_cache(self):
        """Clear the document cache."""
        self.doc_cache.clear()