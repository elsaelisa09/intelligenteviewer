import re
import spacy
import uuid
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime

from ..models.structured_chunk import StructuredChunk

class StructureAwareChunkProcessor:
    """Processor for creating structure-aware document chunks."""

    def __init__(self):
        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Section detection patterns
        self.section_patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown'),            # Markdown headers
            (r'^(\d+\.)+\s+(.+)$', 'numbered'),          # Numbered sections
            (r'^[A-Z][^a-z]+(?:\s+[A-Z][^a-z]+)*$', 'uppercase'),  # ALL CAPS
            (r'^\s*([A-Z][a-z]+(?:\s+[a-z]+)*)\s*$', 'title'),     # Title case
            (r'^(.*?):\s*$', 'colon'),                   # Colon headers
        ]
        
        # Semantic type patterns
        self.semantic_patterns = {
            'definition': [
                r'(?:is|are) defined as\s+[^.]+\.',
                r'refers to\s+[^.]+\.',
                r'means\s+[^.]+\.',
                r'^definition:\s+[^.]+\.'
            ],
            'procedure': [
                r'(?:steps?|procedure|process|method)\s*:\s*(?:\d+\.|\-|\*)',
                r'(?:first|second|third|finally|lastly)(?:\s*,)?\s+[^.]+\.',
                r'^\d+\.\s+[^.]+\.'
            ],
            'example': [
                r'(?:example|instance|illustration)[\s:]+',
                r'(?:e\.g\.|i\.e\.|for example|for instance)',
                r'such as\s+[^.]+\.'
            ],
            'summary': [
                r'(?:in summary|to summarize|in conclusion)',
                r'^summary:\s+[^.]+\.',
                r'(?:key points|main points)\s*:'
            ],
            'table_reference': [
                r'(?:table|tbl\.|fig\.|figure)\s+\d+',
                r'as shown in\s+[^.]+\.',
                r'illustrated in\s+[^.]+\.'
            ]
        }
        
        # Chunking parameters
        self.chunk_params = {
            'min_chunk_size': 100,
            'max_chunk_size': 500,
            'optimal_chunk_size': 300,
            'overlap_size': 50
        }

    def process_document(self, content: str, filename: str) -> List[StructuredChunk]:
        """
        Process a document into structure-aware chunks.
        
        Args:
            content (str): Document content
            filename (str): Name of the document file
            
        Returns:
            List[StructuredChunk]: List of processed chunks with structural information
        """
        try:
            self.logger.info(f"Starting document processing for {filename}")
            
            # Extract document structure
            doc_structure = self.extract_document_structure(content)
            
            # Create chunks while preserving structure
            chunks = []
            chunk_index = 0
            
            # Process each section recursively
            for section in doc_structure['sections']:
                new_chunks = self._process_section(
                    section=section,
                    parent_path=[],
                    filename=filename,
                    start_index=chunk_index
                )
                chunks.extend(new_chunks)
                chunk_index += len(new_chunks)
            
            # Add relationships between chunks
            self._add_chunk_relationships(chunks)
            
            # Final processing and validation
            chunks = self._post_process_chunks(chunks)
            
            self.logger.info(f"Successfully processed {len(chunks)} chunks from {filename}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing document {filename}: {str(e)}")
            raise

    def extract_document_structure(self, content: str) -> Dict:
        """
        Extract hierarchical document structure.
        
        Args:
            content (str): Document content
            
        Returns:
            Dict: Hierarchical structure of the document , document is stack of section
        """
        doc_structure = {
            'sections': [],
            'hierarchy': defaultdict(list),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_sections': 0
            }
        }
        
        current_section = None
        section_stack = []
        current_content = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Handle empty lines
            if not line:
                if current_content:
                    current_content.append('')
                continue
            
            # Check for section headers
            section_match = self._match_section_header(line) ##for each line get the type and  string portion of the title
            
            if section_match: #if header part
                # Process previous section content : joining and cleaning
                if current_section:
                    current_section['content'] = self._clean_content(current_content)
                
                # Create new section
                section_type, section_title = section_match
                new_section = {
                    'title': section_title,
                    'type': section_type,
                    'level': len(section_stack),
                    'content': [],
                    'subsections': [],
                    'metadata': {
                        'position': doc_structure['metadata']['total_sections'],
                        'parent': current_section['title'] if current_section else None
                    }
                }
                
                # Update section hierarchy
                while section_stack and section_stack[-1]['level'] >= new_section['level']:
                    section_stack.pop()
                
                if section_stack:
                    section_stack[-1]['subsections'].append(new_section)
                else:
                    doc_structure['sections'].append(new_section)
                
                section_stack.append(new_section)
                current_section = new_section
                current_content = []
                
                doc_structure['metadata']['total_sections'] += 1
                
            else:
                # Add line to current content
                current_content.append(line)
        
        # Process final section
        if current_section and current_content:
            current_section['content'] = self._clean_content(current_content)
        
        return doc_structure

    def _process_section(self, section: Dict, parent_path: List[str], 
                        filename: str, start_index: int) -> List[StructuredChunk]:
        """
        Process a section and its subsections into chunks.
        
        Args:
            section (Dict): Section data
            parent_path (List[str]): Path of parent sections
            filename (str): Document filename
            start_index (int): Starting index for chunks
            
        Returns:
            List[StructuredChunk]: List of processed chunks for this section
        """
        chunks = []
        current_path = parent_path + [section['title']]
        
        # Process section content
        if section['content']:
            # Create chunks from section content
            section_chunks = self._create_section_chunks(
                content=section['content'],
                section_title=section['title'],
                hierarchical_path=current_path,
                filename=filename,
                start_index=start_index + len(chunks)
            )
            chunks.extend(section_chunks)
        
        # Process subsections recursively
        for subsection in section.get('subsections', []):
            subsection_chunks = self._process_section(
                section=subsection,
                parent_path=current_path,
                filename=filename,
                start_index=start_index + len(chunks)
            )
            chunks.extend(subsection_chunks)
        
        return chunks

    def _create_section_chunks(self, content: str, section_title: str,
                             hierarchical_path: List[str], filename: str,
                             start_index: int) -> List[StructuredChunk]:
        """
        Create chunks from section content while preserving semantic units.
        
        Args:
            content (str): Section content
            section_title (str): Title of the section
            hierarchical_path (List[str]): Full path in document hierarchy
            filename (str): Document filename
            start_index (int): Starting index for chunks
            
        Returns:
            List[StructuredChunk]: List of chunks for this section
        """
        chunks = []
        doc = self.nlp(content)
        
        current_chunk = []
        current_length = 0
        chunk_count = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)
            
            # Check if this sentence would exceed max chunk size
            if (current_length + sent_length > self.chunk_params['max_chunk_size'] 
                and current_chunk):
                # Create chunk from accumulated sentences
                chunk = self._create_chunk(
                    text=' '.join(current_chunk),
                    section_title=section_title,
                    hierarchical_path=hierarchical_path,
                    filename=filename,
                    chunk_index=start_index + chunk_count
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - 2)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk)
                chunk_count += 1
            
            current_chunk.append(sent_text)
            current_length += sent_length
        
        # Handle remaining content
        if current_chunk:
            chunk = self._create_chunk(
                text=' '.join(current_chunk),
                section_title=section_title,
                hierarchical_path=hierarchical_path,
                filename=filename,
                chunk_index=start_index + chunk_count
            )
            chunks.append(chunk)
        
        return chunks

    def _create_chunk(self, text: str, section_title: str,
                     hierarchical_path: List[str], filename: str,
                     chunk_index: int) -> StructuredChunk:
        """
        Create a single chunk with metadata and semantic information.
        
        Args:
            text (str): Chunk text content
            section_title (str): Title of the section
            hierarchical_path (List[str]): Full path in document hierarchy
            filename (str): Document filename
            chunk_index (int): Index of the chunk
            
        Returns:
            StructuredChunk: Processed chunk with metadata
        """
        # Generate unique ID
        chunk_id = f"chunk_{str(uuid.uuid4())}"
        
        # Detect semantic type
        semantic_type = self._detect_semantic_type(text)
        
        # Extract keywords and topics
        keywords, topics = self._extract_semantic_info(text)
        
        # Create the chunk
        chunk = StructuredChunk(
            id=chunk_id,
            text=text,
            metadata={
                'filename': filename,
                'section': section_title,
                'keywords': keywords,
                'topics': topics,
                'creation_time': datetime.now().isoformat()
            },
            section_title=section_title,
            parent_section=hierarchical_path[-2] if len(hierarchical_path) > 1 else None,
            hierarchical_path=hierarchical_path,
            semantic_type=semantic_type,
            topics=topics,
            keywords=keywords,
            chunk_index=chunk_index
        )
        
        return chunk

    def _add_chunk_relationships(self, chunks: List[StructuredChunk]) -> None:
        """
        Add relationships between chunks based on references and content similarity.
        
        Args:
            chunks (List[StructuredChunk]): List of chunks to process
        """
        # Build indices for efficient lookup
        section_chunks = defaultdict(list)
        keyword_chunks = defaultdict(set)
        
        for chunk in chunks:
            section_chunks[chunk.section_title].append(chunk)
            for keyword in chunk.keywords:
                keyword_chunks[keyword].add(chunk.id)
        
        # Process each chunk
        for chunk in chunks:
            # Find explicit references
            references = self._find_references(chunk.text)
            
            # Add reference relationships
            for ref in references:
                referenced_chunks = self._find_referenced_chunks(ref, chunks)
                for ref_chunk in referenced_chunks:
                    chunk.add_reference(ref_chunk.id)
                    ref_chunk.add_referenced_by(chunk.id)
            
            # Add relationships with nearby chunks in same section
            same_section_chunks = section_chunks[chunk.section_title]
            chunk_idx = same_section_chunks.index(chunk)
            
            # Add adjacent chunks as related
            if chunk_idx > 0:
                chunk.add_related_chunk(same_section_chunks[chunk_idx - 1].id)
            if chunk_idx < len(same_section_chunks) - 1:
                chunk.add_related_chunk(same_section_chunks[chunk_idx + 1].id)
            
            # Add relationships based on shared keywords
            chunk_keywords = set(chunk.keywords)
            for keyword in chunk_keywords:
                related_ids = keyword_chunks[keyword]
                for related_id in related_ids:
                    if related_id != chunk.id:
                        chunk.add_related_chunk(related_id)

    def _match_section_header(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Match line against section header patterns.
        
        Args:
            line (str): Line to check for section header
            
        Returns:
            Optional[Tuple[str, str]]: Tuple of (header_type, title) if matched, None otherwise
        """
        for pattern, header_type in self.section_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                if header_type == 'numbered':
                    return (header_type, match.group(2)) #match the string after digit
                return (header_type, match.group(1))
        return None

    def _detect_semantic_type(self, text: str) -> str:
        """
        Detect the semantic type of text content.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected semantic type
        """
        for semantic_type, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return semantic_type
        
        # Use spaCy for additional semantic analysis
        doc = self.nlp(text)
        
        # Check for specific linguistic patterns
        if any(ent.label_ == "DEFINITION" for ent in doc.ents):
            return 'definition'
        if any(token.dep_ == "enumerate" for token in doc):
            return 'list'
        
        return 'general'

def _extract_semantic_info(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract keywords and topics from text.
        
        Args:
            text (str): Text content to analyze
            
        Returns:
            Tuple[List[str], List[str]]: Lists of keywords and topics
        """
        doc = self.nlp(text)
        
        # Extract keywords (important nouns and named entities)
        keywords = []
        seen_phrases = set()
        
        # Add named entities
        for ent in doc.ents:
            if ent.text.lower() not in seen_phrases:
                keywords.append(ent.text)
                seen_phrases.add(ent.text.lower())
        
        # Add important noun phrases
        for chunk in doc.noun_chunks:
            # Filter for relevant noun phrases
            if (not chunk.root.is_stop and
                len(chunk.text) > 3 and
                chunk.text.lower() not in seen_phrases):
                keywords.append(chunk.text)
                seen_phrases.add(chunk.text.lower())
        
        # Extract topics (subject-verb-object patterns)
        topics = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    # Get the subject-verb pair
                    subject = token.text
                    verb = token.head.text
                    
                    # Look for object
                    obj = None
                    for child in token.head.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            obj = child.text
                            break
                    
                    if obj:
                        topic = f"{subject} {verb} {obj}"
                        if topic not in topics:
                            topics.append(topic)
        
        return keywords[:10], topics[:5]  # Limit to top keywords and topics

def _find_references(self, text: str) -> List[str]:
    """
    Find references to other parts of the document in text.
    
    Args:
        text (str): Text to analyze for references
        
    Returns:
        List[str]: List of found references
    """
    references = []
    
    # Patterns for different types of references
    reference_patterns = [
        r'(?:see|refer to|as shown in|as described in)\s+([^.]+)',
        r'(?:table|figure|section|chapter)\s+(\d+(?:\.\d+)*)',
        r'\(([^)]+)\)',  # Parenthetical references
        r'(?:above|below|following|preceding)\s+([^.]+)',
        r'(?:according to|based on)\s+([^.]+)'
    ]
    
    for pattern in reference_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            ref_text = match.group(1).strip()
            if ref_text and len(ref_text) > 3:  # Minimum length to be considered
                references.append(ref_text)
    
    return list(set(references))  # Remove duplicates

def _find_referenced_chunks(self, reference: str, 
                            chunks: List[StructuredChunk]) -> List[StructuredChunk]:
    """
    Find chunks that match a reference.
    
    Args:
        reference (str): Reference text to match
        chunks (List[StructuredChunk]): List of chunks to search
        
    Returns:
        List[StructuredChunk]: List of chunks that match the reference
    """
    referenced_chunks = []
    ref_clean = reference.lower().strip()
    
    # Helper function to calculate text similarity
    def text_similarity(text1: str, text2: str) -> float:
        text1_words = set(text1.lower().split())
        text2_words = set(text2.lower().split())
        intersection = text1_words & text2_words
        union = text1_words | text2_words
        return len(intersection) / len(union) if union else 0
    
    for chunk in chunks:
        # Check section title match
        if chunk.section_title and text_similarity(ref_clean, chunk.section_title.lower()) > 0.5:
            referenced_chunks.append(chunk)
            continue
        
        # Check content match
        content_preview = chunk.text[:100].lower()
        if text_similarity(ref_clean, content_preview) > 0.3:
            referenced_chunks.append(chunk)
            continue
        
        # Check keyword match
        for keyword in chunk.keywords:
            if text_similarity(ref_clean, keyword.lower()) > 0.7:
                referenced_chunks.append(chunk)
                break
    
    return referenced_chunks

def _post_process_chunks(self, chunks: List[StructuredChunk]) -> List[StructuredChunk]:
    """
    Perform final processing and validation of chunks.
    
    Args:
        chunks (List[StructuredChunk]): List of chunks to process
        
    Returns:
        List[StructuredChunk]: Processed and validated chunks
    """
    processed_chunks = []
    seen_texts = set()
    
    for chunk in chunks:
        # Skip empty or too short chunks
        if not chunk.text or len(chunk.text.strip()) < self.chunk_params['min_chunk_size']:
            continue
        
        # Skip duplicate content
        text_fingerprint = ' '.join(chunk.text.lower().split())
        if text_fingerprint in seen_texts:
            continue
        seen_texts.add(text_fingerprint)
        
        # Update importance score
        chunk.update_importance_score()
        
        # Validate and clean chunk data
        if self._validate_chunk(chunk):
            processed_chunks.append(chunk)
    
    return processed_chunks

def _validate_chunk(self, chunk: StructuredChunk) -> bool:
    """
    Validate a chunk's data integrity.
    
    Args:
        chunk (StructuredChunk): Chunk to validate
        
    Returns:
        bool: True if chunk is valid, False otherwise
    """
    try:
        # Check required fields
        if not chunk.id or not chunk.text:
            return False
        
        # Validate text length
        if not (self.chunk_params['min_chunk_size'] <= 
                len(chunk.text) <= 
                self.chunk_params['max_chunk_size']):
            return False
        
        # Validate hierarchical path
        if not chunk.hierarchical_path:
            return False
        
        # Validate metadata
        if not chunk.metadata or 'filename' not in chunk.metadata:
            return False
        
        # Validate relationships (no self-references)
        if chunk.id in chunk.references or chunk.id in chunk.related_chunks:
            return False
        
        return True
        
    except Exception as e:
        self.logger.error(f"Chunk validation error: {str(e)}")
        return False

def _clean_content(self, content_lines: List[str]) -> str:
    """
    Clean and normalize content text.
    
    Args:
        content_lines (List[str]): Lines of content to clean
        
    Returns:
        str: Cleaned and normalized content
    """
    # Join lines
    text = ' '.join(content_lines)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])(?!["\')])', r'\1 ', text)
    
    # Normalize quotes
    text = re.sub(r'[''´`]', "'", text)
    text = re.sub(r'["""]', '"', text)
    
    return text.strip()