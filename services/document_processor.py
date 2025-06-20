#document_processor.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from typing import Tuple, List, Optional, Dict, Any
import base64
import io
import fitz
import docx2txt
import PyPDF2
import pandas as pd
import camelot
from pathlib import Path
from PIL import Image
import io
import fitz
import os
import json
import logging
import re
import traceback
from services.structure_aware_processor import StructureAwareProcessor, StructuredChunk

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

class DocumentProcessor:
    def __init__(self, max_doc_size: int = 50 * 1024 * 1024):
        self.max_doc_size = max_doc_size
        self.logger = logging.getLogger(__name__)
        
    def process_document(self, contents: str, filename: str) -> Tuple[Any, List[dict], List[pd.DataFrame], str, str]:
        if not contents:
            return None, [], [], "", None

        try:
            content_type, content_string = contents.split(',', 1)
            decoded = base64.b64decode(content_string)

            if len(decoded) > self.max_doc_size:
                raise ValueError("File too large (max 50MB)")

            if filename.lower().endswith('.pdf'):
                # Process PDF content
                doc = fitz.open(stream=io.BytesIO(decoded), filetype="pdf")
                content = self._extract_text_with_layout(doc)
                images = self._extract_images(doc)
                tables = self._extract_tables(doc)
                plain_text = "\n".join(
                    " ".join(span["text"] for spans in page for span in spans)
                    for page in content
                )
                pdf_blob = f"data:application/pdf;base64,{content_string}"
                doc.close()
                return content, images, tables, plain_text, pdf_blob
            else:
                # Handle non-PDF documents
                if filename.lower().endswith('.docx'):
                    content = docx2txt.process(io.BytesIO(decoded))
                else:  # Handle .txt and other text files
                    content = decoded.decode('utf-8')
                    
                return content, [], [], content, None

        except Exception as e:
            self.logger.error(f"document_processor.py - DocumentProcessor - process_document : Document processing error: {str(e)}")
            raise Exception(f"document_processor.py - DocumentProcessor - process_document : Error processing file: {str(e)}")

    def process_pdf(self, pdf_bytes: bytes) -> Tuple[Any, List[dict], List[pd.DataFrame], str]:
        """Process PDF content and return structured data"""
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            #content = self._extract_text_with_layout(doc)
            content = self.structure_processor.process_document(doc)
            images = self._extract_images(doc)
            tables = self._extract_tables(doc)
            
            # Create plain text version
            plain_text = "\n".join(
            " ".join(span["text"] for spans in page for span in spans)
            for page in content)
            doc.close()
            return content, images, tables, plain_text
            
        except Exception as e:
            logger.error(f"document_processor.py - DocumentProcessor - process_pdf : PDF processing error: {str(e)}")
            raise Exception(f"Error processing PDF: {str(e)}")

    def get_pdf_highlights(self, pdf_content: str, highlight_chunks: List[str], chunk_mapping: Dict, llm_answer: str = None) -> List[Dict]:
        """Get highlights with improved handling for on-the-fly documents."""
        if not highlight_chunks or not chunk_mapping:
            logger.info("No chunks or mapping provided")
            return []

        try:
            logger.info(f"Processing highlights for {len(highlight_chunks)} chunks")
            logger.info(f"Chunk mapping keys: {list(chunk_mapping.keys())}")
            
            if not pdf_content.startswith('data:application/pdf;base64,'):
                logger.info("Invalid PDF content format")
                return []

            content_type, content_string = pdf_content.split(',', 1)
            pdf_bytes = base64.b64decode(content_string)
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            highlights = []
            seen_highlights = set()
            
            # For on-the-fly documents, we may not have source_id in chunk mapping
            # We'll create an internal document identifier
            doc_source_id = None
            for chunk_id in highlight_chunks:
                if chunk_id in chunk_mapping:
                    chunk_data = chunk_mapping[chunk_id]
                    if 'source_id' in chunk_data:
                        doc_source_id = chunk_data['source_id']
                        logger.info(f"Found source_id in chunk mapping: {doc_source_id}")
                        break
            
            # If we still don't have a source_id, generate one from the content
            if not doc_source_id:
                import hashlib
                # Create a hash from the first page content as a fallback source_id
                first_page_text = ""
                if doc.page_count > 0:
                    first_page_text = doc[0].get_text()
                doc_source_id = hashlib.md5(first_page_text.encode()).hexdigest()
                logger.info(f"Generated source_id from content: {doc_source_id}")
            
            # Process each chunk
            for chunk_id in highlight_chunks:
                if chunk_id in chunk_mapping:
                    chunk_data = chunk_mapping[chunk_id]
                    
                    # Ensure source_id is set
                    if 'source_id' not in chunk_data:
                        chunk_data['source_id'] = doc_source_id
                    
                    chunk_text = chunk_data.get('text', '').strip()
                    if not chunk_text:
                        logger.info(f"No text found for chunk {chunk_id}")
                        continue
                    
                    logger.info(f"Processing chunk: {chunk_text[:100]}...")
                    
                    # Try to find text in PDF
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        
                        # Split chunk text into sentences for better matching
                        sentences = self._break_into_semantic_units(chunk_text)
                        
                        for sentence in sentences:
                            if len(sentence) < 20:  # Skip very short sentences
                                continue
                                
                            # Clean up whitespace
                            clean_sentence = ' '.join(sentence.split())
                            
                            # Search in the page
                            text_instances = page.search_for(clean_sentence)
                            
                            if not text_instances:
                                # Try more flexible matching by using a shorter substring
                                words = clean_sentence.split()
                                if len(words) > 5:
                                    shorter_text = ' '.join(words[:5])
                                    text_instances = page.search_for(shorter_text)
                            
                            # Create highlight for each instance found
                            for rect in text_instances:
                                highlight_rect = {
                                    'x1': rect.x0,
                                    'y1': rect.y0,
                                    'x2': rect.x1,
                                    'y2': rect.y1
                                }
                                
                                # Create a unique key to avoid duplicate highlights
                                rect_key = f"{page_num}_{rect.x0:.1f}_{rect.y0:.1f}"
                                
                                if rect_key not in seen_highlights:
                                    highlights.append({
                                        'page': page_num,
                                        'coords': highlight_rect,
                                        'text': clean_sentence[:100],
                                        'chunk_id': chunk_id,
                                        'source_id': doc_source_id
                                    })
                                    seen_highlights.add(rect_key)
                                    logger.info(f"Added highlight on page {page_num} at {highlight_rect}")
                    
                    # If no highlights found for this chunk, create a fallback
                    if not any(h['chunk_id'] == chunk_id for h in highlights):
                        # Try to get page metadata if available
                        page_num = 0
                        if 'metadata' in chunk_data and 'page' in chunk_data['metadata']:
                            page_num = chunk_data['metadata']['page']
                        elif 'page' in chunk_data:
                            page_num = chunk_data['page']
                        
                        # Ensure page is valid
                        page_num = min(max(0, page_num), len(doc) - 1)
                        
                        # Create a default highlight in the upper part of the page
                        fallback_rect = {
                            'x1': 50,
                            'y1': 100,
                            'x2': 500,
                            'y2': 150
                        }
                        
                        highlights.append({
                            'page': page_num,
                            'coords': fallback_rect,
                            'text': chunk_text[:100],
                            'chunk_id': chunk_id,
                            'source_id': doc_source_id,
                            'is_fallback': True
                        })
                        logger.info(f"Added fallback highlight for chunk {chunk_id} on page {page_num}")
            
            # Merge overlapping highlights
            if highlights:
                highlights = self._merge_overlapping_highlights(highlights)
            
            logger.info(f"document_processor.py - DocumentProcessor - get_pdf_highlights : Created {len(highlights)} total highlights")
            doc.close()
            return highlights

        except Exception as e:
            logger.error(f"document_processor.py - DocumentProcessor - get_pdf_highlights :Error in get_pdf_highlights: {str(e)}")
            traceback.print_exc()
            return []

    def _break_into_semantic_units(self, text: str) -> List[str]:
        """Break text into meaningful semantic units."""
        units = []
        
        # First try sentence-level splitting
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            # Split long sentences at conjunctions and relative pronouns
            if len(sentence.split()) > 15:
                sub_units = re.split(r',\s*(?:and|or|but|which|that|who|where|when)\s+', sentence)
                units.extend(u.strip() for u in sub_units if u.strip())
            else:
                units.append(sentence)
        
        # Filter out units that are too short or lack meaning
        return [unit for unit in units if len(unit.split()) >= 3]

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text segments."""
        # Simple word overlap ratio for basic similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def _merge_overlapping_highlights(self, highlights: List[Dict]) -> List[Dict]:
        """Merge overlapping highlights while preserving core highlights."""
        if not highlights:
            return []
        
        def overlap(rect1: Dict, rect2: Dict) -> bool:
            """Check if two rectangles overlap."""
            return not (rect1['x2'] < rect2['x1'] or
                    rect1['x1'] > rect2['x2'] or
                    rect1['y2'] < rect2['y1'] or
                    rect1['y1'] > rect2['y2'])
        
        def merge_rects(rect1: Dict, rect2: Dict) -> Dict:
            """Merge two rectangles into one."""
            return {
                'x1': min(rect1['x1'], rect2['x1']),
                'y1': min(rect1['y1'], rect2['y1']),
                'x2': max(rect1['x2'], rect2['x2']),
                'y2': max(rect1['y2'], rect2['y2'])
            }
        
        # Sort highlights by page and position
        sorted_highlights = sorted(highlights, 
                                key=lambda h: (h['page'], h['coords']['y1'], h['coords']['x1']))
        merged = []
        
        while sorted_highlights:
            current = sorted_highlights.pop(0)
            i = 0
            while i < len(sorted_highlights):
                if (overlap(current['coords'], sorted_highlights[i]['coords']) and
                    not (current.get('is_core') and sorted_highlights[i].get('is_core'))):
                    # Merge while preserving core status
                    current['coords'] = merge_rects(current['coords'], sorted_highlights[i]['coords'])
                    current['is_core'] = current.get('is_core') or sorted_highlights[i].get('is_core')
                    current['text'] = f"{current['text']} {sorted_highlights[i]['text']}"
                    sorted_highlights.pop(i)
                else:
                    i += 1
            merged.append(current)
        
        return merged

    def validate_highlight_position(self, coords: Dict) -> bool:
        """Validate highlight coordinates."""
        try:
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Basic sanity checks
            if x1 >= x2 or y1 >= y2:
                return False
                
            # Check reasonable bounds (adjust if needed)
            if any(coord < 0 or coord > 1000 for coord in [x1, y1, x2, y2]):
                return False
                
            # Check minimum size
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                return False
            
            return True
            
        except (KeyError, TypeError):
            return False
        
    def _get_highlight_rect(self, page: fitz.Page, coords: Dict) -> Optional[Dict]:
        """Calculate correct highlight rectangle coordinates"""
        try:
            # Get page dimensions
            page_rect = page.rect
            
            # Convert coordinates from PDF space
            x1 = float(coords['x1'])
            y1 = float(coords['y1'])
            x2 = float(coords['x2'])
            y2 = float(coords['y2'])
            
            # PDF coordinates start from bottom-left, we need top-left
            # Convert y-coordinates
            y1 = page_rect.height - y1
            y2 = page_rect.height - y2
            
            # Swap y1 and y2 since we flipped the coordinates
            y1, y2 = y2, y1
            
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                return None
                
            # Ensure coordinates are within page bounds
            x1 = max(0, min(x1, page_rect.width))
            x2 = max(0, min(x2, page_rect.width))
            y1 = max(0, min(y1, page_rect.height))
            y2 = max(0, min(y2, page_rect.height))
            
            # Add small padding for better visibility
            padding = 2
            x1 = max(0, x1 - padding)
            x2 = min(page_rect.width, x2 + padding)
            y1 = max(0, y1 - padding)
            y2 = min(page_rect.height, y2 + padding)
            
            return {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            }
            
        except Exception as e:
            logger.error(f"document_processor.py - DocumentProcessor - _get_highlight_rect : Error calculating highlight rectangle: {str(e)}")
            return None

    def _is_chunk_from_document(self, chunk_data: Dict, pdf_content: str) -> bool:
        """Check if a chunk belongs to the current document"""
        try:
            # Extract document identifier from PDF content
            content_hash = hashlib.md5(pdf_content.encode()).hexdigest()
            
            # Compare with chunk's source document
            chunk_source = chunk_data.get('source_id')
            chunk_content = chunk_data.get('text', '')
            
            if not chunk_source or not chunk_content:
                return False
                
            # Check if content appears in the document
            content_type, content_string = pdf_content.split(',', 1)
            pdf_bytes = base64.b64decode(content_string)
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            
            # Search for chunk content in document
            found = False
            for page in doc:
                if chunk_content in page.get_text():
                    found = True
                    break
                    
            doc.close()
            return found
            
        except Exception as e:
            logger.error(f"document_processor.py - DocumentProcessor - _is_chunk_from_document : Error checking chunk document: {str(e)}")
            return False

    def _normalize_coordinates(self, coords: Dict, page: fitz.Page) -> Optional[Dict]:
        """Normalize and validate coordinates for a page"""
        try:
            # Get page dimensions
            page_rect = page.rect
            
            # Extract coordinates
            x1 = float(coords['x1'])
            y1 = float(coords['y1'])
            x2 = float(coords['x2'])
            y2 = float(coords['y2'])
            
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                return None
                
            # Ensure coordinates are within page bounds
            x1 = max(0, min(x1, page_rect.width))
            x2 = max(0, min(x2, page_rect.width))
            y1 = max(0, min(y1, page_rect.height))
            y2 = max(0, min(y2, page_rect.height))
            
            # Add padding for visibility
            padding = 2
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(page_rect.width, x2 + padding)
            y2 = min(page_rect.height, y2 + padding)
            
            return {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            }
            
        except Exception as e:
            logger.error(f"document_processor.py - DocumentProcessor - normalize_coordinates : Error normalizing coordinates: {str(e)}")
            return None

    def _find_chunk_in_pdf(self, doc: fitz.Document, chunk_text: str) -> List[Dict]:
        """Find chunk text in PDF with improved accuracy"""
        highlights = []
        
        # Clean and normalize text
        chunk_text = ' '.join(chunk_text.split())
        
        # Split into sentences for better matching
        sentences = self._split_into_sentences(chunk_text)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_rect = page.rect
            
            for sentence in sentences:
                if len(sentence) < 20:
                    continue
                    
                # Try exact matching first
                instances = page.search_for(sentence)
                
                if not instances:
                    # Try fuzzy matching with word sequences
                    words = sentence.split()
                    if len(words) > 3:
                        partial_text = ' '.join(words[:4])  # Try first 4 words
                        instances = page.search_for(partial_text)
                
                for rect in instances:
                    # Normalize coordinates
                    coords = self._normalize_coordinates({
                        'x1': rect.x0,
                        'y1': rect.y0,
                        'x2': rect.x1,
                        'y2': rect.y1
                    }, page)
                    
                    if coords:
                        highlights.append({
                            'page': page_num,
                            'coords': coords
                        })
        
        return highlights

    def highlight_pdf(self, pdf_content: str, highlights: List[Dict]) -> Optional[str]:
        """Apply highlights to PDF with correct positioning"""
        if not highlights:
            return pdf_content

        try:
            content_type, content_string = pdf_content.split(',', 1)
            pdf_bytes = base64.b64decode(content_string)
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            
            for highlight in highlights:
                page_num = highlight['page']
                coords = highlight['coords']
                
                if page_num < len(doc):
                    page = doc[page_num]
                    # Create highlight rectangle with proper coordinates
                    rect = fitz.Rect(
                        coords['x1'],
                        coords['y1'],
                        coords['x2'],
                        coords['y2']
                    )
                    
                    # Apply highlight with improved visibility
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0.8, 0))  # Bright yellow
                    annot.set_opacity(0.3)  # More visible
                    annot.update()

            # Save highlighted PDF
            output_buffer = io.BytesIO()
            doc.save(output_buffer)
            doc.close()
            output_buffer.seek(0)
            
            highlighted_content = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            return f"data:application/pdf;base64,{highlighted_content}"

        except Exception as e:
            logger.info(f"Error in highlight_pdf: {str(e)}")
            return pdf_content
    
    def get_pdf_highlights2(self, pdf_content: str, highlight_chunks: List[str], chunk_mapping: Dict) -> List[Dict]:
        if not highlight_chunks or not chunk_mapping:
            return []

        try:
            content_type, content_string = pdf_content.split(',', 1)
            pdf_bytes = base64.b64decode(content_string)
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            highlights = []
            
            # Preprocess chunk texts for better matching
            def clean_text(text: str) -> str:
                return ' '.join(text.split())  # Normalize whitespace
                
            def extract_phrases(text: str, min_length: int = 20) -> List[str]:
                # Split by multiple delimiters and clean
                delimiters = ['. ', '.\n', '\n', '? ', '! ', '; ']
                phrases = []
                temp = text
                for delim in delimiters:
                    parts = temp.split(delim)
                    temp = parts[0]
                    phrases.extend(parts[1:])
                phrases.append(temp)
                
                return [p.strip() for p in phrases if len(p.strip()) >= min_length]

            # Process each chunk
            for chunk_id in highlight_chunks:
                str_chunk_id = str(chunk_id)
                if str_chunk_id not in chunk_mapping:
                    continue
                    
                chunk_text = clean_text(chunk_mapping[str_chunk_id])
                phrases = extract_phrases(chunk_text)
                
                # Find matches page by page
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_rect = page.rect
                    page_dict = page.get_text("dict")
                    
                    for phrase in phrases:
                        # Try exact and fuzzy matching
                        text_instances = page.search_for(phrase, quads=True)
                        
                        if not text_instances:
                            # Try with partial matching
                            words = phrase.split()
                            if len(words) > 3:
                                partial_phrase = ' '.join(words[:3])
                                text_instances = page.search_for(partial_phrase, quads=True)
                        
                        logger.info(phrases, text_instances)
                        for quads in text_instances:
                            # Convert quad coordinates to rect
                            rect = quads.rect
                            
                            # Add some padding for visibility
                            padding = 2
                            highlight = {
                                'page': page_num,
                                'coords': {
                                    'x1': max(0, rect.x0 - padding),
                                    'y1': max(0, page_rect.height - rect.y1 - padding),
                                    'x2': min(page_rect.width, rect.x1 + padding),
                                    'y2': min(page_rect.height, page_rect.height - rect.y0 + padding)
                                },
                                'text': phrase[:50],
                                'chunk_id': str_chunk_id
                            }
                            highlights.append(highlight)
            
            doc.close()
            logger.error(f"document_processor.py - DocumentProcessor - get_pdf_highlights2 ~1: Generated {len(highlights)} highlights")
            return highlights
            
        except Exception as e:
            logger.error(f"document_processor.py - DocumentProcessor - get_pdf_highlights2 ~2: Error in get_pdf_highlights: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def highlight_pdf2(self, pdf_content: str, highlights: List[Dict]) -> Optional[str]:
        """Apply highlights to PDF and return the modified PDF as base64 string"""
        if not highlights:
            return pdf_content

        try:
            # Parse PDF content
            content_type, content_string = pdf_content.split(',', 1)
            pdf_bytes = base64.b64decode(content_string)
            
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            
            # Debug print
            logger.info(f"Applying {len(highlights)} highlights")
            
            for highlight in highlights:
                page_num = highlight['page']
                coords = highlight['coords']
                
                if page_num < len(doc):
                    page = doc[page_num]
                    rect = fitz.Rect(
                        coords['x1'],
                        coords['y1'],
                        coords['x2'],
                        coords['y2']
                    )
                    
                    # Add yellow highlight with good visibility
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0.8, 0))  # Bright yellow
                    annot.set_opacity(0.3)  # More visible
                    annot.update()
                    
                    # Debug print for first highlight
                    if highlights.index(highlight) == 0:
                        logger.info(f"Applied first highlight at page {page_num}, coords: {coords}")

            output_buffer = io.BytesIO()
            doc.save(output_buffer)
            doc.close()
            output_buffer.seek(0)
            
            highlighted_content = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            return f"data:application/pdf;base64,{highlighted_content}"

        except Exception as e:
            logger.info(f"Error in highlight_pdf: {str(e)}")
            import traceback
            traceback.print_exc()
            return pdf_content
    
    def _find_text_position(self, doc: fitz.Document, text: str) -> List[Dict]:
        """Find all occurrences of text in the document with their positions."""
        positions = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_instances = page.search_for(text)
            
            for rect in text_instances:
                coords = {
                    'x1': rect.x0,
                    'y1': rect.y0,
                    'x2': rect.x1,
                    'y2': rect.y1
                }
                
                if self._is_valid_coords(coords):
                    positions.append({
                        'page': page_num,
                        'coords': coords,
                        'text': text
                    })
        
        return positions

    def _is_valid_coords(self, coords: Dict) -> bool:
        """Validate coordinate values."""
        try:
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Basic sanity checks
            if x1 >= x2 or y1 >= y2:
                return False
                
            # Check minimum size (adjust thresholds as needed)
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                return False
                
            return True
            
        except (KeyError, TypeError):
            return False

    def _is_overlapping(self, text: str, position: Dict) -> bool:
        """Check if text overlaps with an existing position."""
        pos_text = position.get('text', '').lower()
        return text.lower() in pos_text or pos_text in text.lower()

    def _combine_nearby_rects(self, rects: List[fitz.Rect], threshold: float = 5.0) -> List[fitz.Rect]:
        """Combine rectangles that are close to each other."""
        if not rects:
            return []

        # Sort rectangles by vertical position, then horizontal
        sorted_rects = sorted(rects, key=lambda r: (r.y0, r.x0))
        combined = [sorted_rects[0]]

        for rect in sorted_rects[1:]:
            last = combined[-1]
            # Check if rectangles are close enough to combine
            if (abs(rect.y0 - last.y0) <= threshold and 
                (abs(rect.x0 - last.x1) <= threshold or 
                 abs(rect.x1 - last.x0) <= threshold)):
                # Merge rectangles
                combined[-1] = fitz.Rect(
                    min(last.x0, rect.x0),
                    min(last.y0, rect.y0),
                    max(last.x1, rect.x1),
                    max(last.y1, rect.y1)
                )
            else:
                combined.append(rect)

        return combined
    
    def _merge_overlapping_highlights2(self, highlights: List[Dict]) -> List[Dict]:
        """Merge overlapping or nearby highlights on the same page."""
        if not highlights:
            return []

        # Group highlights by page
        page_highlights = {}
        for h in highlights:
            page = h['page']
            if page not in page_highlights:
                page_highlights[page] = []
            page_highlights[page].append(h)

        merged_highlights = []
        for page, page_highlights_list in page_highlights.items():
            # Convert to Rect objects for easier manipulation
            rects = [fitz.Rect(
                h['coords']['x1'], 
                h['coords']['y1'],
                h['coords']['x2'], 
                h['coords']['y2']
            ) for h in page_highlights_list]

            # Merge overlapping rectangles
            merged_rects = self._combine_nearby_rects(rects)

            # Convert back to highlight format
            for rect in merged_rects:
                merged_highlights.append({
                    'page': page,
                    'coords': {
                        'x1': rect.x0,
                        'y1': rect.y0,
                        'x2': rect.x1,
                        'y2': rect.y1
                    },
                    'chunk_id': page_highlights_list[0]['chunk_id']  # Use first chunk_id
                })

        return merged_highlights
    
    def _extract_text_with_layout(self, doc) -> List[List[dict]]:
        """Extract text from PDF while preserving layout"""
        pages_content = []
        
        for page in doc:
            blocks = page.get_text("dict", sort=True)["blocks"]
            page_text = []
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            spans_text = []
                            for span in line["spans"]:
                                if span["text"].strip():
                                    spans_text.append({
                                        "text": span["text"],
                                        "font_size": span["size"],
                                        "is_bold": "bold" in span["font"].lower(),
                                        "is_italic": "italic" in span["font"].lower(),
                                        "bbox": span["bbox"]
                                    })
                            
                            if spans_text:
                                page_text.append(spans_text)
            
            pages_content.append(page_text)
        
        return pages_content

    def _extract_images(self, doc) -> List[dict]:
        """Extract images from PDF"""
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_format = base_image["ext"]
                    
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    images.append({
                        'data': f"data:image/{image_format};base64,{image_b64}",
                        'page': page_num,
                        'format': image_format,
                        'size': len(image_bytes)
                    })
                    
                except Exception as e:
                    logger.error(f"document_processor.py - DocumentProcessor - _extract_images ~2 : Error extracting image {img_index + 1} from page {page_num + 1}: {str(e)}")
                    continue
        
        return images

    def _extract_tables(self, doc) -> List[pd.DataFrame]:
        """Extract tables from PDF"""
        tables = []
        
        for page in doc:
            text = page.get_text("text")
            if '\t' in text or '  ' in text:
                rows = [line.split('\t') for line in text.split('\n') if line.strip()]
                if rows:
                    df = pd.DataFrame(rows)
                    if not df.empty and len(df.columns) > 1:
                        tables.append(df)
        
        return tables
    
    def _extract_tables_camelot(self, pdf_bytes: bytes) -> List[pd.DataFrame]:
        """Extract tables using Camelot library"""
        pdf_buffer = io.BytesIO(pdf_bytes)
        tables = camelot.read_pdf(pdf_buffer, pages='all', flavor='stream')
        extracted_tables = []
        
        for table in tables:
            df = table.df
            # Clean up the dataframe
            df = df.replace(r'\n', ' ', regex=True)
            df = df.replace(r'\s+', ' ', regex=True)
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, (df != '').any()]
            df = df.dropna(how='all')
            df.columns = [f"Column {i+1}" for i in range(len(df.columns))]
            
            if not df.empty:
                extracted_tables.append(df)
        
        return extracted_tables
    
    def _extract_tables_basic(self, pdf_bytes: bytes) -> List[pd.DataFrame]:
        """Basic table extraction when Camelot is not available"""
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        
        tables = []
        for page in reader.pages:
            text = page.extract_text()
            rows = [line.split() for line in text.split('\n') if line.strip()]
            if rows:
                df = pd.DataFrame(rows)
                if not df.empty:
                    tables.append(df)
        
        return tables