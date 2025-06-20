#services/strucutre_aware_processor.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import fitz
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class StructuredChunk:
    id: str
    text: str
    metadata: Dict
    section_title: Optional[str] = None
    parent_section: Optional[str] = None
    subsections: List[str] = None
    references: List[str] = None
    entities: List[Dict] = None
    embedding: Optional[np.ndarray] = None
    semantic_units: List[str] = None  # Store semantic units for better matching

class StructureAwareProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not found. Using basic text processing.")
            self.nlp = None
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.logger = logging.getLogger(__name__)
        self.min_chunk_size = 100  # Minimum characters per chunk
        self.max_chunk_size = 500  # Maximum characters per chunk
        self.overlap_size = 50  # Overlap between chunks

    def _extract_semantic_units(self, text: str) -> List[str]:
        """Extract meaningful semantic units from text."""
        units = []
        if self.nlp:
            doc = self.nlp(text)
            for sent in doc.sents:
                # Only consider sentences with substantial content
                if len(sent.text.split()) >= 5:
                    units.append(sent.text.strip())
        else:
            # Fallback to regex-based splitting
            for sent in re.split(r'[.!?]+', text):
                if len(sent.strip().split()) >= 5:
                    units.append(sent.strip())
        return units

    def _analyze_section_hierarchy(self, doc) -> Dict:
        """Analyze document structure with improved header detection."""
        hierarchy = {
            'headers': [],
            'relationships': defaultdict(list)
        }
        
        current_font_sizes = set()
        header_candidates = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            spans = line["spans"]
                            if spans:
                                text = ' '.join(span["text"] for span in spans)
                                max_font_size = max(span["size"] for span in spans)
                                is_bold = any("bold" in span["font"].lower() 
                                            for span in spans)
                                
                                # Record font sizes for analysis
                                current_font_sizes.add(max_font_size)
                                
                                # Header candidate criteria
                                if (len(text.split()) <= 10 and  # Not too long
                                    not text.endswith('.') and   # Not ending with period
                                    (is_bold or max_font_size > 11)):  # Bold or larger font
                                    
                                    header_candidates.append({
                                        'text': text,
                                        'font_size': max_font_size,
                                        'is_bold': is_bold,
                                        'page': page_num,
                                        'bbox': block["bbox"]
                                    })
        
        # Analyze font size distribution
        if current_font_sizes:
            font_sizes = sorted(current_font_sizes, reverse=True)
            
            # Determine header levels based on font size clusters
            size_clusters = self._cluster_font_sizes(font_sizes)
            
            # Assign levels to headers
            for candidate in header_candidates:
                level = self._determine_header_level(candidate, size_clusters)
                if level <= 3:  # Only consider up to 3 levels of headers
                    hierarchy['headers'].append({
                        'text': candidate['text'],
                        'level': level,
                        'page': candidate['page'],
                        'bbox': candidate['bbox']
                    })
        
        return hierarchy

    def _cluster_font_sizes(self, font_sizes: List[float]) -> List[float]:
        """Cluster font sizes to determine distinct header levels."""
        clusters = []
        current_cluster = [font_sizes[0]]
        
        for size in font_sizes[1:]:
            if abs(size - current_cluster[-1]) <= 0.5:  # Font size threshold
                current_cluster.append(size)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [size]
        
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return sorted(clusters, reverse=True)[:3]  # Return top 3 sizes

    def _determine_header_level(self, candidate: Dict, size_clusters: List[float]) -> int:
        """Determine header level based on font size and formatting."""
        font_size = candidate['font_size']
        
        for i, cluster_size in enumerate(size_clusters):
            if abs(font_size - cluster_size) <= 0.5:
                return i + 1
            
        return 4  # Not a header

    def _create_semantic_chunks2(self, text: str, section_info: Dict) -> List[StructuredChunk]:
        """Create chunks with semantic awareness and proper boundaries."""
        chunks = []
        semantic_units = self._extract_semantic_units(text)
        
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for unit in semantic_units:
            unit_length = len(unit)
            
            if current_length + unit_length > self.max_chunk_size:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, section_info, chunk_id))
                    chunk_id += 1
                    
                    # Create overlap with previous chunk
                    if len(current_chunk) > 1:
                        current_chunk = current_chunk[-1:]
                        current_length = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_length = 0
            
            current_chunk.append(unit)
            current_length += unit_length
        
        # Add remaining content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, section_info, chunk_id))
        
        return chunks

    def _create_chunk(self, text: str, section_info: Dict, chunk_id: int) -> StructuredChunk:
        """Create a single chunk with enhanced metadata."""
        semantic_units = self._extract_semantic_units(text)
        embedding = self.model.encode(text)
        
        return StructuredChunk(
            id=f"{section_info['title']}_{chunk_id}",
            text=text,
            metadata={
                'page': section_info.get('page', 0),
                'bbox': section_info.get('bbox'),
                'level': section_info.get('level', 1),
                'position': chunk_id
            },
            section_title=section_info['title'],
            parent_section=section_info.get('parent'),
            semantic_units=semantic_units,
            embedding=embedding
        )

    def process_document(self, doc, filename: str) -> List[StructuredChunk]:
        """Process document with improved error handling and fallback mechanisms"""
        print("using structure aware chunking")
        try:
            self.logger.info(f"Processing document: {filename}")
            chunks = []

            # Handle different input types
            if isinstance(doc, dict):
                # Handle pre-processed text structure
                chunks = self._process_text_document(doc)
            elif isinstance(doc, fitz.Document):
                # Handle PDF document
                chunks = self._process_pdf_document(doc)
            elif isinstance(doc, str):
                # Handle plain text
                chunks = self._process_plain_text(doc)
            else:
                # Try to convert to string
                try:
                    text = str(doc)
                    chunks = self._process_plain_text(text)
                except:
                    self.logger.error(f"Unable to process document of type {type(doc)}")
                    return []

            if chunks:
                self._add_semantic_references(chunks)
                self.logger.info(f"Created {len(chunks)} chunks")
                return chunks

            # If no chunks were created, try fallback chunking
            self.logger.warning("No chunks created, attempting fallback chunking")
            return self._create_fallback_chunks(doc)

        except Exception as e:
            self.logger.error(f"Error processing document {filename}: {str(e)}")
            self.logger.exception(e)
            return self._create_fallback_chunks(doc)

    def _process_text_document(self, doc_dict: Dict) -> List[StructuredChunk]:
        """Process pre-processed text document"""
        try:
            content = doc_dict.get('content', '').strip()
            if not content:
                return []

            return self._create_semantic_chunks(
                content,
                {
                    'title': doc_dict.get('title', 'Document'),
                    'page': doc_dict.get('page', 0),
                    'bbox': doc_dict.get('bbox'),
                    'level': doc_dict.get('level', 1),
                    'parent': doc_dict.get('parent')
                }
            )
        except Exception as e:
            self.logger.error(f"Error in _process_text_document: {str(e)}")
            return []

    def _process_pdf_document(self, doc: fitz.Document) -> List[StructuredChunk]:
        """Process PDF document"""
        try:
            hierarchy = self._analyze_section_hierarchy(doc)
            sections = self._extract_hierarchical_sections(doc, hierarchy)
            
            chunks = []
            for section in sections:
                section_chunks = self._create_semantic_chunks(
                    section['content'],
                    {
                        'title': section['title'],
                        'page': section['page'],
                        'bbox': section['bbox'],
                        'level': section['level'],
                        'parent': section.get('parent')
                    }
                )
                chunks.extend(section_chunks)
            
            return chunks
        except Exception as e:
            self.logger.error(f"Error in _process_pdf_document: {str(e)}")
            return []

    def _process_plain_text(self, text: str) -> List[StructuredChunk]:
        """Process plain text document"""
        try:
            if not text.strip():
                return []

            semantic_units = self._extract_semantic_units(text)
            if not semantic_units:
                # If no semantic units found, create a single chunk
                return [self._create_chunk(text, {'title': 'Document'}, 0)]

            chunks = []
            current_text = ''
            chunk_id = 0

            for unit in semantic_units:
                if len(current_text) + len(unit) > self.max_chunk_size:
                    if current_text:
                        chunk = self._create_chunk(current_text, {'title': 'Document'}, chunk_id)
                        if chunk:
                            chunks.append(chunk)
                            chunk_id += 1
                        current_text = unit
                else:
                    current_text = (current_text + ' ' + unit).strip()

            # Add remaining text
            if current_text:
                chunk = self._create_chunk(current_text, {'title': 'Document'}, chunk_id)
                if chunk:
                    chunks.append(chunk)

            return chunks
        except Exception as e:
            self.logger.error(f"Error in _process_plain_text: {str(e)}")
            return []

    def _create_fallback_chunks(self, doc) -> List[StructuredChunk]:
        """Create basic chunks as a fallback mechanism"""
        try:
            # Extract text from document
            if isinstance(doc, fitz.Document):
                text = ""
                for page_num in range(len(doc)):
                    text += doc[page_num].get_text() + "\n"
            elif isinstance(doc, dict):
                text = doc.get('content', '')
            else:
                text = str(doc)

            text = text.strip()
            if not text:
                return []

            # Create single chunk if text is small enough
            if len(text) <= self.max_chunk_size:
                chunk = StructuredChunk(
                    id="chunk_0",
                    text=text,
                    metadata={'page': 0},
                    section_title="Document",
                    embedding=self.model.encode(text)
                )
                return [chunk]

            # Split into chunks
            chunks = []
            words = text.split()
            current_chunk = []
            current_length = 0
            chunk_id = 0

            for word in words:
                word_len = len(word)
                if current_length + word_len > self.max_chunk_size:
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunk = StructuredChunk(
                            id=f"chunk_{chunk_id}",
                            text=chunk_text,
                            metadata={'page': 0},
                            section_title="Document",
                            embedding=self.model.encode(chunk_text)
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                        current_chunk = [word]
                        current_length = word_len
                else:
                    current_chunk.append(word)
                    current_length += word_len + 1

            # Add remaining text
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk = StructuredChunk(
                    id=f"chunk_{chunk_id}",
                    text=chunk_text,
                    metadata={'page': 0},
                    section_title="Document",
                    embedding=self.model.encode(chunk_text)
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            self.logger.error(f"Error in fallback chunking: {str(e)}")
            return []

    def _create_semantic_chunks(self, text: str, section_info: Dict) -> List[StructuredChunk]:
        """Create chunks with improved handling of shorter texts."""
        if not text.strip():
            return []
            
        chunks = []
        semantic_units = self._extract_semantic_units(text)
        
        # Handle short texts
        if len(text) < self.min_chunk_size:
            chunk = self._create_chunk(text, section_info, 0)
            return [chunk] if chunk else []
        
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for unit in semantic_units:
            unit_length = len(unit)
            
            # Handle long semantic units
            if unit_length > self.max_chunk_size:
                # Process any accumulated content first
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk = self._create_chunk(chunk_text, section_info, chunk_id)
                    if chunk:
                        chunks.append(chunk)
                        chunk_id += 1
                    current_chunk = []
                    current_length = 0
                
                # Split long unit into smaller pieces
                words = unit.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    if temp_length + len(word) > self.max_chunk_size:
                        chunk_text = ' '.join(temp_chunk)
                        chunk = self._create_chunk(chunk_text, section_info, chunk_id)
                        if chunk:
                            chunks.append(chunk)
                            chunk_id += 1
                        temp_chunk = [word]
                        temp_length = len(word)
                    else:
                        temp_chunk.append(word)
                        temp_length += len(word) + 1
                
                if temp_chunk:
                    chunk_text = ' '.join(temp_chunk)
                    chunk = self._create_chunk(chunk_text, section_info, chunk_id)
                    if chunk:
                        chunks.append(chunk)
                        chunk_id += 1
                
            elif current_length + unit_length > self.max_chunk_size:
                # Create chunk from accumulated content
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk = self._create_chunk(chunk_text, section_info, chunk_id)
                    if chunk:
                        chunks.append(chunk)
                        chunk_id += 1
                    
                    # Create overlap with previous chunk
                    if len(current_chunk) > 1:
                        current_chunk = current_chunk[-1:]
                        current_length = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(unit)
                current_length = unit_length
            else:
                current_chunk.append(unit)
                current_length += unit_length + 1
        
        # Add remaining content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = self._create_chunk(chunk_text, section_info, chunk_id)
            if chunk:
                chunks.append(chunk)
        
        return chunks

    def _extract_hierarchical_sections(self, doc, hierarchy: Dict) -> List[Dict]:
        """Extract sections while maintaining proper hierarchy."""
        sections = []
        headers = sorted(hierarchy['headers'], 
                        key=lambda x: (x['page'], x['bbox'][1]))
        
        current_section = None
        section_stack = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    text = ' '.join(
                        span["text"] 
                        for line in block["lines"] 
                        for span in line["spans"]
                    )
                    
                    # Check if this is a header
                    header_match = None
                    for header in headers:
                        if (header['page'] == page_num and 
                            header['text'].strip() == text.strip()):
                            header_match = header
                            break
                    
                    if header_match:
                        # Close current section if exists
                        if current_section:
                            sections.append(current_section)
                        
                        # Update section stack based on header level
                        while (section_stack and 
                               section_stack[-1]['level'] >= header_match['level']):
                            section_stack.pop()
                        
                        # Create new section
                        current_section = {
                            'title': header_match['text'],
                            'content': '',
                            'page': page_num,
                            'bbox': header_match['bbox'],
                            'level': header_match['level'],
                            'parent': section_stack[-1]['title'] if section_stack else None
                        }
                        
                        section_stack.append(current_section)
                    elif current_section:
                        current_section['content'] += text + '\n'
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections

    def _add_semantic_references(self, chunks: List[StructuredChunk]):
        """Add cross-references based on semantic similarity."""
        for i, chunk in enumerate(chunks):
            chunk.references = []
            chunk_embedding = chunk.embedding
            
            for j, other_chunk in enumerate(chunks):
                if i != j:
                    similarity = np.dot(chunk_embedding, other_chunk.embedding)
                    if similarity > 0.8:  # High similarity threshold
                        # Check if they share semantic units
                        common_units = set(chunk.semantic_units) & set(other_chunk.semantic_units)
                        if common_units:
                            chunk.references.append(other_chunk.id)