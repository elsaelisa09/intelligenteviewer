#services/vector_store.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"
__license__  = "MIT"

from datetime import datetime
import json
import shutil
from pathlib import Path
import tempfile
import uuid
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import io
import base64
import logging
from typing import Tuple, List, Optional, Dict, Any
import re
from services.structure_aware_processor import StructureAwareProcessor, StructuredChunk
from app.storage_config import BASE_STORAGE_DIR
import traceback
import os 
import gc   
from services.language_service import LanguageService
import time


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize embeddings model
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    cache_folder="./model_cache"
)

# Text splitting configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

class VectorStoreService:
    def __init__(self):
        self.embeddings = embeddings
        self.text_splitter = text_splitter
        self.TEMP_DIR = Path(tempfile.gettempdir()) / "faiss_indices"
        self.TEMP_DIR.mkdir(exist_ok=True)
        self.structure_processor = StructureAwareProcessor()
        self.logger = logging.getLogger(__name__)
        self._session_cache = {}
        self.vectorstores = {}
        self._metadata_cache = {}
        self.language_service = LanguageService()
        self._query_translation_cache = {}
        
        # New caching mechanisms for performance
        self._chunk_mapping_cache = {}  # Cache for chunk mappings
        self._query_embedding_cache = {}  # Cache for query embeddings
        self._document_language_cache = {}  # Cache for document languages
        
        # Performance optimization flags
        self._skip_language_detection_for_small_docs = True
        self._enable_fast_path = True  # Enable performance optimizations
        
        # Initialize FAISS with nprobe=1 for faster searches at slight accuracy cost
        self._faiss_nprobe = 1

    def clear_all_caches(self):
        """Clear all in-memory caches to force fresh data loading"""
        self._session_cache = {}
        self._metadata_cache = {}
        self._query_translation_cache = {}
        self._chunk_mapping_cache = {}
        self._query_embedding_cache = {}
        self._document_language_cache = {}
        if hasattr(self, '_document_chunks_cache'):
            self._document_chunks_cache = {}
        if hasattr(self, '_user_group_cache'):
            self._user_group_cache = {}
        self.logger.info("All in-memory caches have been cleared")
        
    def _get_user_paths(self, filename: str, user_id: str, db_session : str) -> Tuple[Path, Path, Path]:
        """Get user-specific paths for all storage types"""
        from app.storage_config import VECTOR_STORE_DIR, ORIGINAL_FILES_DIR, CHUNK_MAPS_DIR, get_group_for_user
        group_name = get_group_for_user(user_id, db_session)
        # Create user-specific directories
        user_vector_dir = VECTOR_STORE_DIR / str(group_name)
        user_original_dir = ORIGINAL_FILES_DIR / str(group_name)
        user_chunks_dir = CHUNK_MAPS_DIR / str(group_name)
        
        # Create session directory under vector store
        filename_vector_dir = user_vector_dir / filename
        
        # Ensure all directories exist
        for path in [user_vector_dir, user_original_dir, user_chunks_dir, filename_vector_dir]:
            path.mkdir(parents=True, exist_ok=True)
            
        return filename_vector_dir, user_original_dir, user_chunks_dir

    def validate_position(self, position):
        """Validate position coordinates"""
        if not position or 'coords' not in position:
            return False
            
        coords = position['coords']
        required_keys = ['x1', 'y1', 'x2', 'y2']
        
        # Check all required coordinates exist and are numeric
        if not all(key in coords for key in required_keys):
            return False
            
        try:
            # Validate coordinate values
            x1, y1, x2, y2 = [float(coords[key]) for key in required_keys]
            
            # Check coordinates make sense
            if x1 >= x2 or y1 >= y2:
                return False
                
            # Check reasonable bounds (adjust as needed)
            if any(coord < 0 or coord > 1000 for coord in [x1, y1, x2, y2]):
                return False
                
            return True
        except (ValueError, TypeError):
            return False
    
    def map_chunk_positions(self, chunk_text, text_positions, filename):
        """Map text positions to chunk with improved accuracy and filename"""
        chunk_positions = []
        remaining_text = chunk_text
        text_pos_copy = text_positions.copy()
        
        logger.debug(f"Vector_store.py - VectorStoreService - map_chunk_position : Mapping positions for chunk text: {chunk_text[:100]}...")
        
        while remaining_text and text_pos_copy:
            matched = False
            for pos_idx, pos_item in enumerate(text_pos_copy):
                item_text = pos_item['text'].strip()
                if not item_text:
                    continue
                    
                # Try exact match first
                start_idx = remaining_text.find(item_text)
                if start_idx >= 0:
                    if self.validate_position(pos_item):
                        chunk_positions.append({
                            'page': pos_item['page'],
                            'coords': pos_item['coords'],
                            'matched_text': item_text,
                            'filename': filename
                        })
                        logger.debug(f"Vector_store.py - VectorStoreService - map_chunk_position : Matched text: {item_text[:30]}... at position {pos_item['coords']}")
                    remaining_text = remaining_text[start_idx + len(item_text):].lstrip()
                    text_pos_copy = text_pos_copy[pos_idx + 1:]
                    matched = True
                    break
            
            if not matched:
                # If no exact match found, try with the next character sequence
                remaining_text = remaining_text[1:]
                
        logger.debug(f"Vector_store.py - VectorStoreService - map_chunk_position : Found {len(chunk_positions)} valid positions for chunk")
        return chunk_positions

    def _get_user_group(self, user_id: str, db_session) -> str:
        """Efficiently get user group with caching"""
        if not hasattr(self, '_user_group_cache'):
            self._user_group_cache = {}
            
        # Return cached value if available
        if user_id in self._user_group_cache:
            return self._user_group_cache[user_id]
            
        # Get group from database
        try:
            from app.storage_config import get_group_for_user
            group_name = get_group_for_user(user_id, db_session)
            
            # Cache the result
            self._user_group_cache[user_id] = group_name
            return group_name
        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - _get_user_group : Error getting user group: {str(e)}")
            return "public"  # Default fallback
        
    def create_vectorstore_and_mapping(self, text: str, filename: str, user_id: str = None, db_session = None, default_language = None) -> Tuple[str, Dict]:
        """Create vector store with optimized performance and caching"""
        try:
            self.logger.info(f"Vector_store.py - VectorStoreService - create_vectorstore_and_mapping: Creating vector store for {filename}")
            
            # Skip cleanup for better performance, handle separately in a maintenance task
            # self.cleanup_old_indices()
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            group_name = self._get_user_group(user_id, db_session)
            
            # Process document based on type using more efficient approach
            start_time = time.time()
            
            # Create a primary cache key for this document
            doc_cache_key = f"{filename}_{hash(text[:1000]) if isinstance(text, str) else id(text)}"
            
            # Check if we've already processed this document
            if hasattr(self, '_document_chunks_cache') and doc_cache_key in self._document_chunks_cache:
                chunks = self._document_chunks_cache[doc_cache_key]
                self.logger.info(f"Vector_store.py - VectorStoreService - create_vectorstore_and_mapping : Using cached chunks for {filename}")
            else:
                # Initialize cache if needed
                if not hasattr(self, '_document_chunks_cache'):
                    self._document_chunks_cache = {}
                    
                # Process document
                if isinstance(text, str) and text.startswith('data:application/pdf;base64,'):
                    # PDF processing
                    content_type, content_string = text.split(',', 1)
                    pdf_bytes = base64.b64decode(content_string)
                    
                    # Use context manager for auto-close
                    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
                        chunks = self.structure_processor.process_document(doc, filename)
                else:
                    # Text processing
                    if not isinstance(text, str):
                        try:
                            text = text.decode('utf-8')
                        except (UnicodeDecodeError, AttributeError):
                            text = str(text)
                    
                    text_chunk = {
                        'title': 'Document',
                        'content': text,
                        'page': 0,
                        'bbox': None,
                        'level': 1,
                        'parent': None
                    }
                    chunks = self.structure_processor.process_document(text_chunk, filename)

                # Cache the chunks
                self._document_chunks_cache[doc_cache_key] = chunks
                
                # Limit cache size
                if len(self._document_chunks_cache) > 10:  # Only keep recent docs
                    keys = list(self._document_chunks_cache.keys())
                    for key in keys[:-5]:  # Remove all but the 5 most recent
                        self._document_chunks_cache.pop(key, None)
            
            # Fast path for empty chunks
            if not chunks:
                # Simplified fallback
                texts = self.text_splitter.split_text(text if isinstance(text, str) else str(text))
                chunks = []
                
                # Batch embedding for better performance
                if texts:
                    try:
                        # Try batch embedding if available
                        if hasattr(self.embeddings, 'embed_documents'):
                            embeddings_list = self.embeddings.embed_documents(texts)
                            for i, (chunk_text, embedding) in enumerate(zip(texts, embeddings_list)):
                                chunks.append(StructuredChunk(
                                    id=f"chunk_{i}",
                                    text=chunk_text,
                                    metadata={
                                        'page': 0,
                                        'position': i
                                    },
                                    section_title="Document",
                                    embedding=embedding
                                ))
                        else:
                            # Fallback to individual embedding
                            for i, chunk_text in enumerate(texts):
                                chunks.append(StructuredChunk(
                                    id=f"chunk_{i}",
                                    text=chunk_text,
                                    metadata={
                                        'page': 0,
                                        'position': i
                                    },
                                    section_title="Document",
                                    embedding=self.embeddings.embed_query(chunk_text)
                                ))
                    except Exception as embed_err:
                        self.logger.error(f"Error during embedding: {str(embed_err)}")
                        # Last resort fallback
                        for i, chunk_text in enumerate(texts):
                            chunks.append(StructuredChunk(
                                id=f"chunk_{i}",
                                text=chunk_text,
                                metadata={
                                    'page': 0,
                                    'position': i
                                },
                                section_title="Document"
                            ))

            if not chunks:
                raise ValueError("Vector_store.py - VectorStoreService - create_vectorstore_and_mapping : No chunks created from document")
            
            # Language detection - improved approach for Indonesian
            # Sample from multiple chunks to get better language detection
            sample_chunks = []
            # Always include first chunk
            if len(chunks) > 0:
                sample_chunks.append(chunks[0])
            
            # Add middle chunk if available
            if len(chunks) > 2:
                sample_chunks.append(chunks[len(chunks) // 2])
                
            # Add last chunk if available
            if len(chunks) > 1:
                sample_chunks.append(chunks[-1])

            # Add some more samples if we have many chunks
            if len(chunks) > 10:
                sample_idx = len(chunks) // 3
                sample_chunks.append(chunks[sample_idx])
                
                sample_idx = 2 * len(chunks) // 3
                sample_chunks.append(chunks[sample_idx])

            # Get a good amount of text from samples
            sample_text = " ".join(chunk.text for chunk in sample_chunks)
            doc_lang = default_language
            if default_language != "en":
                # First try improved Indonesian detection with patterns
                indonesian_patterns = [
                    r'\b(yang|dengan|untuk|dari|kepada|adalah|tersebut|ini|itu)\b',
                    r'\b(apa|siapa|mengapa|bagaimana|kapan|dimana|kemana)\b',
                    r'\b(dan|atau|tetapi|namun|karena|sebab|jika|maka)\b',
                    r'\b(di|ke|pada|dalam|tentang|mengenai|oleh)\b'
                ]
                
                # Quick check for Indonesian patterns in filename and content
                filename_lower = filename.lower()
                # Check content for Indonesian patterns
                indonesian_matches = 0
                for pattern in indonesian_patterns:
                    if re.search(pattern, sample_text.lower()):
                        indonesian_matches += 1
                        
                if indonesian_matches >= 2:
                    # Strong indicator of Indonesian content
                    doc_lang = "id"
                    self.logger.info(f"Vector_store.py - VectorStoreService - create_vectorstore_and_mapping : "
                                    f"Detected Indonesian document based on patterns: {indonesian_matches} matches, ")
                else:
                    # Use additional standard language (Indonesian) detection as backup
                    try:
                        doc_lang, confidence = self.language_service.detect_language(sample_text)
                    except:
                        # Default to default language
                        doc_lang = default_language
                    self.logger.info(f"Vector_store.py - VectorStoreService - create_vectorstore_and_mapping : "
                                    f"Language detection for {filename}: {doc_lang} (confidence: {confidence})")
                    
            
            self.logger.info(f"Vector_store.py - VectorStoreService - create_vectorstore_and_mapping : "
                            f"Final document language: {doc_lang}")

            # Create chunk mapping and FAISS index more efficiently
            chunk_mapping = {}
            docstore = {}
            index_to_docstore_id = {}
            
            # Pre-allocate numpy array for better memory efficiency
            num_chunks = len(chunks)
            d = None  # We'll determine dimension from first embedding
            
            # First pass: prepare chunk mapping and determine embedding dimension
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{i}"
                
                # Simple language inheritance for chunks - avoid detection per chunk
                chunk_lang = doc_lang
                
                # Create chunk mapping entry
                chunk_mapping[chunk_id] = {
                    'text': chunk.text,
                    'filename': filename,
                    'source_id': session_id,
                    'section_title': chunk.section_title if hasattr(chunk, 'section_title') else None,
                    'parent_section': chunk.parent_section if hasattr(chunk, 'parent_section') else None,
                    'language': chunk_lang,  # Set language for each chunk
                }
                
                # Create document entry
                doc = Document(
                    page_content=chunk.text,
                    metadata={
                        'chunk_id': chunk_id,
                        'filename': filename,
                        'source_id': session_id,
                        'user_id': user_id,
                        'language': chunk_lang  # Set language in metadata too
                    }
                )
                
                docstore[chunk_id] = doc
                index_to_docstore_id[i] = chunk_id
                
                # Determine embedding dimension from first chunk
                if hasattr(chunk, 'embedding') and chunk.embedding is not None and d is None:
                    d = len(chunk.embedding)

            # Second pass: create embeddings array with pre-allocated memory
            if d is None and chunks:
                # No pre-computed embeddings, get dimension from a sample embedding
                sample_embedding = self.embeddings.embed_query(chunks[0].text)
                d = len(sample_embedding)
            
            # Create embeddings array with proper size
            embeddings_array = np.zeros((num_chunks, d), dtype=np.float32)
            
            for i, chunk in enumerate(chunks):
                if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                    embeddings_array[i] = chunk.embedding
                else:
                    embeddings_array[i] = self.embeddings.embed_query(chunk.text)

            # Create optimized FAISS index
            index = faiss.IndexFlatL2(d)
            
            # Set FAISS search parameters for better performance
            if hasattr(index, 'nprobe'):
                index.nprobe = self._faiss_nprobe
                
            # Add vectors to index
            index.add(embeddings_array)

            # Create vector store
            vectorstore = FAISS(
                self.embeddings.embed_query,
                index,
                docstore,
                index_to_docstore_id
            )

            # Add document language to metadata
            metadata = {
                "filename": filename,
                "last_used": datetime.now().isoformat(),
                "doc_count": len(chunks),
                "user_id": user_id,
                "language": doc_lang,  # Store the detected language 
                "create_time": time.time(),
                "embedding_dim": d
            }

            # Save vector store with user-specific paths and enhanced metadata
            self._save_vectorstore(session_id, vectorstore, filename, user_id, group_name, metadata)
            
            # Cache chunk mapping for faster retrieval later
            if not hasattr(self, '_chunk_mapping_cache'):
                self._chunk_mapping_cache = {}
            self._chunk_mapping_cache[session_id] = chunk_mapping
            
            end_time = time.time()
            self.logger.info(f"Vector_store.py - VectorStoreService - create_vectorstore_and_mapping : "
                            f"Vector store creation completed in {end_time - start_time:.2f} seconds")

            return session_id, chunk_mapping

        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - create_vectorstore_and_mapping : Error in vectorstore creation: {str(e)}")
            traceback.print_exc()
            # Cleanup on error
            try:
                if 'session_id' in locals():
                    self.cleanup_vectorstore(session_id, user_id)
            except:
                pass
            raise

        
    def _remove_existing_sessions(self, filename: str):
        """Remove any existing sessions with the same filename"""
        try:
            for session_dir in self.TEMP_DIR.glob("*"):
                try:
                    metadata_path = session_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            if metadata.get("filename") == filename:
                                session_id = session_dir.name
                                self.cleanup_vectorstore(session_id)
                                self.logger.info(f"Vector_store.py - VectorStoreService - _remove_existing_sessions : Removed existing session for {filename}")
                except Exception as e:
                    self.logger.error(f"Error checking session {session_dir}: {str(e)}")
                    # Try to remove problematic directory
                    try:
                        shutil.rmtree(session_dir)
                    except:
                        pass

        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - _remove_existing_sessions : Error removing existing sessions: {str(e)}")
            traceback.print_exc()

    def _extract_positions(self, text: str, filename: str) -> List[Dict]:
        """Extract position information from text"""
        positions = []
        try:
            # For now, return basic position
            positions.append({
                'page': 0,
                'coords': {
                    'x1': 0,
                    'y1': 0,
                    'x2': 100,
                    'y2': 20
                }
            })
            return positions
        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - _extract_positions : Error extracting positions: {str(e)}")
            return positions

    def _validate_position(self, position):
        """Validate position coordinates"""
        if not position or 'coords' not in position:
            return False
            
        coords = position['coords']
        required_keys = ['x1', 'y1', 'x2', 'y2']
        
        # Check all required coordinates exist and are numeric
        if not all(key in coords for key in required_keys):
            return False
            
        try:
            # Validate coordinate values
            x1, y1, x2, y2 = [float(coords[key]) for key in required_keys]
            
            # Check coordinates make sense
            if x1 >= x2 or y1 >= y2:
                return False
                
            # Check reasonable bounds (adjust as needed)
            if any(coord < 0 or coord > 1000 for coord in [x1, y1, x2, y2]):
                return False
                
            return True
        except (ValueError, TypeError):
            return False


    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF with improved position tracking"""
        try:
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_parts.append(page.get_text())
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - _extract_text_from_pdf : Error extracting PDF text: {str(e)}")
            raise

    def preselect_documents_for_query(self, query, doc_state, max_docs=10):
        """
        Efficiently preselect documents to search based on query.
        This dramatically speeds up search when you have many documents.
        
        Args:
            query: User query
            doc_state: Document state dictionary
            max_docs: Maximum number of documents to return
            
        Returns:
            List of selected document IDs to search
        """
        import re
        from collections import Counter
        import time
        
        start_time = time.time()
        
        # Fast path for small document collections
        if len(doc_state) <= max_docs:
            return list(doc_state.keys())
        
        # Normalize query for better matching
        query_lower = query.lower().strip()
        
        # Store document scores
        doc_scores = Counter()
        
        # 1. Direct filename mentions (highest priority)
        for session_id, info in doc_state.items():
            if not isinstance(info, dict):
                continue
                
            filename = info.get('filename', '').lower()
            if not filename:
                continue
                
            # Direct filename match
            if filename in query_lower:
                # This is a strong signal - give it high weight
                doc_scores[session_id] += 100
                
            # Check for filename keyword matches
            keywords = set(re.findall(r'\b[a-z0-9]{3,}\b', filename))
            query_words = set(re.findall(r'\b[a-z0-9]{3,}\b', query_lower))
            
            # Count overlapping keywords
            overlap = keywords.intersection(query_words)
            if overlap:
                doc_scores[session_id] += len(overlap) * 20
        
        # 2. Use metadata if available (medium priority)
        for session_id, info in doc_state.items():
            if not isinstance(info, dict):
                continue
                
            # Recent documents get a boost
            if 'timestamp' in info:
                try:
                    from datetime import datetime
                    timestamp = datetime.fromisoformat(info['timestamp'])
                    age_days = (datetime.now() - timestamp).days
                    # More recent documents get higher priority (max 10 points)
                    recency_score = max(0, 10 - min(10, age_days))
                    doc_scores[session_id] += recency_score
                except:
                    pass
            
            # Documents with similar language to query get a boost
            if 'language' in info:
                doc_lang = info.get('language')
                print("####################line code 649")
                print(doc_lang)
                # If we have a cached query language, compare with it
                if hasattr(self, '_last_query_language') and self._last_query_language:
                    if doc_lang == self._last_query_language:
                        doc_scores[session_id] += 15
            
            # Boost based on file size - medium-sized docs often most relevant
            if 'file_size' in info:
                size = info['file_size']
                # Give highest scores to mid-sized documents (not too small, not too large)
                if 10000 <= size <= 500000:  # ~10KB to ~500KB
                    doc_scores[session_id] += 5
            
            # Previous usage boost - documents that have been selected before
            if 'highlight_count' in info:
                doc_scores[session_id] += min(10, info['highlight_count'])
        
        # 3. Fallback: add any documents not yet scored
        for session_id in doc_state:
            if session_id not in doc_scores:
                doc_scores[session_id] = 1  # Default minimal score
        
        # Get the top-scoring documents
        selected_docs = [doc_id for doc_id, _ in doc_scores.most_common(max_docs)]
        
        # If we didn't select enough documents, add more
        if len(selected_docs) < min(max_docs, len(doc_state)):
            remaining = [doc_id for doc_id in doc_state if doc_id not in selected_docs]
            selected_docs.extend(remaining[:max_docs - len(selected_docs)])
        
        end_time = time.time()
        logger.info(f"Vector_store.py - VectorStoreService - preselect_documents_for_query : Document preselection completed in {end_time - start_time:.4f}s, selected {len(selected_docs)} out of {len(doc_state)} documents")
        
        return selected_docs

    def _save_vectorstore(self, session_id, vectorstore: FAISS, filename: str, user_id: str = None, 
                         group_name="public", metadata=None):
        """Save vector store with user-specific paths and enhanced metadata"""
        try:
            if user_id:
                # Use user-specific paths
                session_dir, _, _ = self._get_user_paths(filename, user_id, group_name)
            else:
                # Fallback to existing temp directory
                session_dir = self.TEMP_DIR / session_id
                session_dir.mkdir(exist_ok=True)
            
            # Save FAISS index
            index_path = session_dir / "index.bin"
            faiss.write_index(vectorstore.index, str(index_path))
            
            # Save document store data
            data_path = session_dir / "data.json"
            docstore_data = []
            
            for doc_id, doc in vectorstore.docstore.items():
                docstore_data.append({
                    "doc_id": doc_id,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            data = {
                "docstore": docstore_data,
                "index_to_docstore_id": vectorstore.index_to_docstore_id
            }
            
            with open(data_path, "w") as f:
                json.dump(data, f, indent=4, default=str)
            
            # Save metadata
            metadata_path = session_dir / "metadata.json"
            if not metadata:
                metadata = {
                    "filename": filename,
                    "last_used": datetime.now().isoformat(),
                    "doc_count": len(docstore_data),
                    "user_id": user_id
                }
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - _save_vectorstore : Error saving vectorstore: {str(e)}")
            raise

    def load_vectorstore(self, session_id: str) -> Tuple[Optional[FAISS], Optional[Dict]]:
        """Load vector store with lazy loading and caching"""
        try:
            # Check cache first
            if session_id in self._session_cache:
                cached = self._session_cache[session_id]
                # Update last used time
                cached["metadata"]["last_used"] = datetime.now().isoformat()
                return cached["vectorstore"], cached["metadata"]

            from app.storage_config import VECTOR_STORE_DIR
            
            # Check metadata cache first to avoid file operations
            if session_id in self._metadata_cache:
                metadata_info = self._metadata_cache[session_id]
                session_dir = metadata_info['path']
                
                # Only load if we're actually going to use this vector store
                logger.info(f"Vector_store.py - VectorStoreService - load_vectorstore : Using cached path info for {session_id}: {session_dir}")
            else:
                # Try to determine if this is a filename (persistent document)
                if '.' in session_id:  # Simple check if session_id looks like a filename
                    # Search all group directories only for metadata files first
                    # instead of loading the full vector store
                    for group_dir in VECTOR_STORE_DIR.glob('*'):
                        if not group_dir.is_dir():
                            continue
                            
                        # Check for filename match
                        filename_dir = group_dir / session_id
                        if filename_dir.exists() and filename_dir.is_dir():
                            metadata_path = filename_dir / "metadata.json"
                            if metadata_path.exists():
                                # Store path in cache but don't load the actual data yet
                                self._metadata_cache[session_id] = {
                                    'path': filename_dir,
                                    'last_check': datetime.now().isoformat()
                                }
                                session_dir = filename_dir
                                break
                
                # If still not found, try other lookups
                if session_id not in self._metadata_cache:
                    session_dir = self.TEMP_DIR / session_id
                    
                    # Only if not in temp dir, do full search
                    if not session_dir.exists():
                        for group_dir in VECTOR_STORE_DIR.glob('*'):
                            if not group_dir.is_dir():
                                continue
                                
                            # Search all subdirectories only checking metadata files
                            for potential_dir in group_dir.glob('*'):
                                if not potential_dir.is_dir():
                                    continue
                                    
                                metadata_path = potential_dir / "metadata.json"
                                if metadata_path.exists():
                                    try:
                                        with open(metadata_path, 'r') as f:
                                            metadata = json.load(f)
                                            if metadata.get('filename') == session_id or metadata.get('session_id') == session_id:
                                                session_dir = potential_dir
                                                self._metadata_cache[session_id] = {
                                                    'path': potential_dir,
                                                    'last_check': datetime.now().isoformat()
                                                }
                                                break
                                    except Exception as e:
                                        logger.info(f"Vector_store.py - VectorStoreService - load_vectorstore : Error reading metadata: {str(e)}")
                            
                            if session_id in self._metadata_cache:
                                break
            
            # Now that we have the session_dir, load the actual vector store
            if not 'session_dir' in locals() or not session_dir.exists():
                logger.info(f"Vector_store.py - VectorStoreService - load_vectorstore : No vector store found for {session_id}")
                return None, None
                
            #logger.info(f"Loading vector store from: {session_dir}")
            index_path = session_dir / "index.bin"
            data_path = session_dir / "data.json"
            metadata_path = session_dir / "metadata.json"

            if not all(p.exists() for p in [index_path, data_path, metadata_path]):
                logger.info(f"Vector_store.py - VectorStoreService - load_vectorstore : Missing vector store files in {session_dir}")
                return None, None

            # Load index and documents - this is the heavy operation
            index = faiss.read_index(str(index_path))

            # Load document store data - only load minimal needed data
            with open(data_path, "r") as f:
                data = json.load(f)

            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            metadata["last_used"] = datetime.now().isoformat()
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            # Reconstruct document store
            docstore = {}
            for doc_data in data["docstore"]:
                docstore[doc_data["doc_id"]] = Document(
                    page_content=doc_data["page_content"],
                    metadata=doc_data.get("metadata", {})
                )

            # Create vector store
            vectorstore = FAISS(
                self.embeddings.embed_query,
                index,
                docstore,
                data.get("index_to_docstore_id", {})
            )

            # Cache the loaded data
            self._session_cache[session_id] = {
                "vectorstore": vectorstore,
                "metadata": metadata
            }

            return vectorstore, metadata

        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - load_vectorstore : Error loading vectorstore for {session_id}: {str(e)}")
            traceback.print_exc()
            return None, None
        
    def _save_chunks_to_store(self, chunks, session_id):
        """Save structured chunks to vector store."""
        try:
            # Create session directory
            session_dir = self.TEMP_DIR / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Prepare embeddings and document store
            embeddings_list = []
            docstore = {}
            index_to_docstore_id = {}
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                if not chunk.text.strip():
                    continue
                    
                # Store chunk data
                chunk_data = {
                    'id': chunk.id,
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'section_title': chunk.section_title,
                    'parent_section': chunk.parent_section,
                    'entities': chunk.entities,
                    'references': chunk.references
                }
                
                # Add to document store
                docstore[chunk.id] = Document(
                    page_content=chunk.text,
                    metadata=chunk_data
                )
                
                # Store embedding
                embeddings_list.append(chunk.embedding)
                index_to_docstore_id[i] = chunk.id
            
            if not embeddings_list:
                raise ValueError("Vector_store.py - VectorStoreService - _save_chunks_to_store : No valid embeddings created")
            
            # Create FAISS index
            embeddings_array = np.array(embeddings_list).astype("float32")
            d = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_array)
            
            # Create vector store
            vectorstore = FAISS(
                self.embeddings.embed_query,
                index,
                docstore,
                index_to_docstore_id
            )
            
            # Save index and metadata
            index_path = session_dir / "index.bin"
            data_path = session_dir / "data.json"
            metadata_path = session_dir / "metadata.json"
            
            # Save FAISS index
            faiss.write_index(vectorstore.index, str(index_path))
            
            # Save document store data
            docstore_data = []
            for doc_id, doc in vectorstore.docstore.items():
                docstore_data.append({
                    "doc_id": doc_id,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Save data and metadata
            data = {
                "docstore": docstore_data,
                "index_to_docstore_id": vectorstore.index_to_docstore_id
            }
            
            metadata = {
                "last_used": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "embedding_dim": d
            }
            
            with open(data_path, "w") as f:
                json.dump(data, f, indent=4, default=str)
                
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Vector_store.py - VectorStoreService - _save_chunks_to_store : Successfully saved {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chunks to store: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def _create_doc_from_text(self, text: str, filename: str) -> fitz.Document:
        """Create a document object from text content."""
        try:
            # Create a temporary PDF from text
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), text)
            return doc
        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - _create_doc_from_text : Error creating document from text: {str(e)}")
            raise


    def _extract_pdf_positions(self, doc: fitz.Document, filename: str) -> List[Dict]:
        """Extract text positions from PDF with structure awareness"""
        try:
            positions = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    if span["text"].strip():
                                        # Get span coordinates
                                        bbox = span["bbox"]
                                        
                                        # Get structural information
                                        is_header = span["size"] > 12
                                        is_bold = "bold" in span["font"].lower()
                                        
                                        positions.append({
                                            'text': span["text"].strip(),
                                            'page': page_num,
                                            'coords': {
                                                'x1': bbox[0],
                                                'y1': bbox[1],
                                                'x2': bbox[2],
                                                'y2': bbox[3]
                                            },
                                            'font_size': span["size"],
                                            'is_header': is_header,
                                            'is_bold': is_bold,
                                            'block_type': block.get("type", 0)
                                        })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - _extract_pdf_positions : Error extracting PDF positions: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _map_text_positions(self, chunk_text: str, text_positions: List[Dict]) -> List[Dict]:
        """Map text positions with improved precision"""
        try:
            chunk_positions = []
            # Clean and normalize texts
            chunk_text = ' '.join(chunk_text.split())
            
            # Only look for substantial pieces of text (at least 3 words)
            min_text_length = 3  # minimum words
            
            # Split chunk into sentences or phrases
            sentences = [s.strip() for s in re.split(r'[.!?;]', chunk_text) if s.strip()]
            
            for sentence in sentences:
                if len(sentence.split()) >= min_text_length:
                    matched_position = None
                    longest_match = 0
                    
                    # Look for the longest matching text position
                    for pos in text_positions:
                        pos_text = ' '.join(pos['text'].split())
                        if not pos_text:
                            continue
                        
                        # Try exact match first
                        if pos_text in sentence:
                            if len(pos_text) > longest_match:
                                matched_position = pos
                                longest_match = len(pos_text)
                    
                    # Only add position if we found a substantial match
                    if matched_position and 'coords' in matched_position:
                        chunk_positions.append({
                            'page': matched_position['page'],
                            'coords': matched_position['coords'],
                            'text': matched_position['text']
                        })
            
            # Remove duplicate positions
            unique_positions = []
            seen = set()
            for pos in chunk_positions:
                pos_key = (pos['page'], 
                        pos['coords']['x1'], 
                        pos['coords']['y1'],
                        pos['coords']['x2'], 
                        pos['coords']['y2'])
                if pos_key not in seen:
                    seen.add(pos_key)
                    unique_positions.append(pos)
            
            return unique_positions
            
        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - _map_pdf_positions : Error mapping text positions: {str(e)}")
            return []
    
    def save_vectorstore(self, session_id, vectorstore):
        """Save vectorstore to disk"""
        session_dir = self.TEMP_DIR / session_id
        index_path = session_dir / "index.bin"
        data_path = session_dir / "data.json"

        faiss.write_index(vectorstore.index, str(index_path))

        docstore_data = []
        for doc_id, doc in vectorstore.docstore.items():
            docstore_data.append({
                "doc_id": doc_id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })

        data = {
            "docstore": docstore_data,
            "index_to_docstore_id": vectorstore.index_to_docstore_id
        }

        with open(data_path, "w") as f:
            json.dump(data, f, indent=4, default=str)

    def process_document_content(content, filename):
        """Process document content to ensure it's in readable text form"""
        try:
            # Check if content is base64 encoded
            if isinstance(content, str) and content.startswith('data:'):
                content_type, content_string = content.split(',', 1)
                decoded = base64.b64decode(content_string)
                
                # Process based on file extension
                file_extension = filename.split('.')[-1].lower()
                
                if file_extension == 'pdf':
                    # For PDFs, use PyMuPDF to extract text
                    try:
                        doc = fitz.open(stream=io.BytesIO(decoded), filetype="pdf")
                        text_parts = []
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            text_parts.append(page.get_text())
                        doc.close()
                        return "\n\n".join(text_parts)
                    except Exception as e:
                        logger.info(f"Error extracting PDF text: {str(e)}")
                
                elif file_extension in ['docx', 'doc']:
                    # For DOCX, use docx2txt to extract text
                    try:
                        import docx2txt
                        return docx2txt.process(io.BytesIO(decoded))
                    except Exception as e:
                        logger(f"Vector_store.py - VectorStoreService - process_document_content : Error extracting DOCX text: {str(e)}")
                        
            # If content is already text or processing failed, return as is
            return content
            
        except Exception as e:
            logger.info(f"Vector_store.py - VectorStoreService - process_document_content : Error processing document content: {str(e)}")
            return content

    def _sanitize_text(self, text):
        """
        Sanitize chunk text efficiently to ensure it's human-readable
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove base64 encoded content with minimal regex
        if 'base64,' in text or 'CmVuZG9iag' in text:
            return ""
        
        # Simplified sanitization - just remove non-printable characters
        return ''.join(char for char in text if ord(char) > 31 and ord(char) < 127 or char in '\n\t ')

    def _fast_score_chunk(self, chunk: Dict) -> float:
        """Optimized scoring function focused on performance"""
        try:
            # Use the vector similarity as the base score (already calculated)
            base_score = chunk.get('score', 0.5)
            
            # Simple length-based adjustment - efficient and effective
            content = chunk.get('content', chunk.get('text', ''))
            content_len = len(content) if content else 0
            
            # Fast length scoring - prefer chunks between 100-800 chars
            # Too short = may lack context, too long = may be less focused
            if content_len < 100:
                length_factor = content_len / 100  # Penalize very short chunks
            elif content_len <= 800:
                length_factor = 1.0  # Ideal length range
            else:
                length_factor = max(0.7, 800 / content_len)  # Gradually penalize longer chunks
            
            # Very simple information density approximation (avoid splitting content)
            if content and len(content) > 20:
                # Count capital letters as a proxy for information-rich content
                capital_ratio = sum(1 for c in content if c.isupper()) / len(content)
                # More capitals often mean more proper nouns, acronyms, etc.
                density_factor = min(1.2, 1.0 + capital_ratio * 3.0)
            else:
                density_factor = 0.8
            
            # Very fast content type detection without regex
            content_lower = content.lower() if content else ""
            if "definition" in content_lower or "refers to" in content_lower:
                content_factor = 1.15  # Definitions are valuable
            elif "example" in content_lower or "instance" in content_lower:
                content_factor = 1.1   # Examples are helpful
            elif ":" in content or ";" in content:
                content_factor = 1.05  # Lists/structured content often valuable
            else:
                content_factor = 1.0
            
            # Simple weighting - prioritize similarity while considering content quality
            final_score = base_score * 0.7 + (length_factor * density_factor * content_factor) * 0.3
            
            return final_score
            
        except Exception as e:
            # Fallback to base similarity score on any error
            return chunk.get('score', 0.5)
            

    def process_query_for_chunks(self, query: str, chunks: List[Dict]) -> Tuple[str, str, Dict]:
        """
        Process query for a set of chunks, ensuring language compatibility with improved logic.
        
        Args:
            query (str): User query
            chunks (List[Dict]): List of chunks to search
            
        Returns:
            Tuple[str, str, Dict]: (processed_query, query_lang, translations)
                - processed_query: The query to use (original or translated)
                - query_lang: Detected query language
                - translations: Dictionary of translations for caching
        """
        # Detect query language with increased confidence threshold
        query_lang, query_confidence = self.detect_language(query)
        if not query_lang :
            query_lang = "id"
        
        # Determine main language of chunks
        chunks_lang = self.get_main_language(chunks)
        
        logger.info(f"Query language: {query_lang} (confidence: {query_confidence}), Chunks language: {chunks_lang}")
        
        # Initialize translation cache for this query
        translations = {
            "query": {
                "original": query,
                "lang": query_lang
            }
        }
        
        # Enhanced translation decision logic
        should_translate = False
        
        # Conditions for translation:
        # 1. Languages are different
        # 2. Query language confidence is low (< 0.7)
        # 3. Chunks language is significantly different from query language
        if query_lang != chunks_lang:
            # Low confidence detection
            if query_confidence < 0.7:
                should_translate = True
                logger.info(f"Vector_store.py - VectorStoreService - process_query_for_chunks : Translation triggered: Low language detection confidence ({query_confidence})")
            
            # Check if it's a clear mismatch between languages
            elif len(chunks) > 0:
                # Add a language difference check
                chunk_langs = [self.detect_language(chunk.get('text', ''))[0] for chunk in chunks]
                unique_langs = set(chunk_langs)
                
                if len(unique_langs) == 1 and list(unique_langs)[0] != query_lang:
                    should_translate = True
                    logger.info(f"Vector_store.py - VectorStoreService - process_query_for_chunks : Translation triggered: Consistent chunk language different from query")
        
        # Perform translation if needed
        if should_translate:
            translated_query = self.translate_text(query, query_lang, chunks_lang)
            
            # Update translations cache
            translations["query"]["translated"] = translated_query
            translations["query"]["translated_lang"] = chunks_lang
            
            logger.info(f"Translated query from {query_lang} to {chunks_lang}: {query} -> {translated_query}")
            
            return translated_query, query_lang, translations
        
        # No translation needed
        logger.info("Vector_store.py - VectorStoreService - process_query_for_chunks : No translation needed. Using original query.")
        return query, query_lang, translations

    def _load_chunk_mapping_fast(self, session_id, user_id, db_session):
        """Efficiently load chunk mapping with caching and minimal IO"""
        import json
        
        # Check cache for this session
        if hasattr(self, '_chunk_mapping_cache') and session_id in self._chunk_mapping_cache:
            return self._chunk_mapping_cache[session_id]
        
        # Initialize cache if needed
        if not hasattr(self, '_chunk_mapping_cache'):
            self._chunk_mapping_cache = {}
        
        chunk_mapping = {}
        
        # Only try to load from disk if we have necessary parameters
        if user_id and session_id and db_session:
            try:
                from app.storage_config import CHUNK_MAPS_DIR, get_group_for_user
                group_name = get_group_for_user(user_id, db_session)
                
                # Try different potential chunk map file paths
                # Use a list to prioritize paths
                chunk_file_paths = [
                    CHUNK_MAPS_DIR / str(group_name) / f"{session_id}_chunks.json",
                    CHUNK_MAPS_DIR / str(group_name) / session_id,
                    CHUNK_MAPS_DIR / str(group_name) / f"{session_id}.json"
                ]
                
                for chunk_file_path in chunk_file_paths:
                    if chunk_file_path.exists():
                        # Use a more efficient file reading approach
                        try:
                            with open(chunk_file_path, 'r') as f:
                                chunk_mapping = json.load(f)
                                # Cache the result
                                self._chunk_mapping_cache[session_id] = chunk_mapping
                                logger.info(f"Loaded and cached chunk mapping from {chunk_file_path}")
                                break
                        except json.JSONDecodeError:
                            logger.info(f"Vector_store.py - VectorStoreService - _load_chunk_mapping_fast : Error decoding JSON from {chunk_file_path}")
                            continue
            except Exception as e:
                logger.info(f"Vector_store.py - VectorStoreService - _load_chunk_mapping_fast : Error loading chunk mapping: {str(e)}")
        
        # Limit cache size
        if len(self._chunk_mapping_cache) > 50:
            # Simple LRU-like behavior - remove oldest entries
            keys_to_remove = list(self._chunk_mapping_cache.keys())[:-25]
            for key in keys_to_remove:
                self._chunk_mapping_cache.pop(key, None)
        
        return chunk_mapping

    def _document_language_detection(self, chunks, default_languange):
        """
        Accurately detect document language from content chunks with improved Indonesian detection.
        
        Args:
            chunks (list): List of document chunks
            
        Returns:
            str: Detected language code
        """
        if not chunks or len(chunks) == 0:
            return "en"
        
        # Sample more chunks for better detection
        max_samples = min(5, len(chunks))
        # Sample from beginning, middle, and end
        sample_indices = [0]
        if len(chunks) > 2:
            sample_indices.append(len(chunks) // 2)
        if len(chunks) > 1:
            sample_indices.append(len(chunks) - 1)
        
        # Add some random samples if we have enough chunks
        if len(chunks) > 5:
            import random
            for _ in range(min(2, max_samples - len(sample_indices))):
                idx = random.randint(1, len(chunks) - 2)
                if idx not in sample_indices:
                    sample_indices.append(idx)
        
        sample_chunks = [chunks[i] for i in sample_indices]
        sample_text = " ".join(chunk.text for chunk in sample_chunks)
        
        # Use enhanced language detection with logging
        try:
            doc_lang, confidence = self.language_service.detect_language(sample_text, log_detection=True)
            if not query_lang :
                query_lang = default_languange
        except:
            # Default to Indonesian if detection fails
            query_lang = default_languange
        
        self.logger.info(f"Vector_store.py - VectorStoreService - _document_language_detection : "
                        f"Detected document language: {doc_lang} with confidence {confidence}")
        
        # Cache this document's language
        self._document_language_cache = getattr(self, '_document_language_cache', {})
        cache_key = hash(sample_text[:100])
        self._document_language_cache[cache_key] = doc_lang
    

    def get_relevant_chunks(self, vectorstore, query, chunk_mapping=None, user_id=None, session_id=None, db_session=None, k=1, threshold=0.3, default_language = "id"):
        """Optimized: Get relevant chunks with enhanced language handling and improved performance"""
        import time
        import numpy as np
        import json
        from pathlib import Path

        if not vectorstore:
            logger.info("No vector store provided")   
            return [], "", [], []
                        
        try:
            start_time = time.time()
            
            # Cache query embeddings if they've been computed before
            query_cache_key = query.strip().lower()
            if query_cache_key in self._query_embedding_cache:
                query_embedding = self._query_embedding_cache[query_cache_key]
                logger.info(f"Vector_store.py - VectorStoreService - get_relevant_chunks : Using cached query embedding for: {query[:30]}...")
            else:
                # If chunk_mapping is not provided, try to load it more efficiently
                if chunk_mapping is None:
                    chunk_mapping = self._load_chunk_mapping_fast(session_id, user_id, db_session)
                
                # Get the document's language - with more thorough detection
                doc_language = default_language  # Default
                try:
                    if hasattr(vectorstore, 'metadata') and isinstance(vectorstore.metadata, dict):
                        doc_language = vectorstore.metadata.get('language')
                    else:
                        # Sample some document chunks to detect language
                        sample_chunks = []
                        for doc_id, doc in vectorstore.docstore.items():
                            if len(sample_chunks) >= 5:
                                break
                            sample_chunks.append({
                                'text': doc.page_content,
                                'metadata': doc.metadata
                            })
                        
                        if sample_chunks:
                            doc_language = self.language_service.get_main_language(sample_chunks, log_detection=True)
                            logger.info(f"Vector_store.py - VectorStoreService - get_relevant_chunks : "
                                        f"Detected document language from samples: {doc_language}")
                except Exception as e:
                    logger.error(f"Vector_store.py - VectorStoreService - get_relevant_chunks : "
                                f"Error detecting document language: {str(e)}")
                
                # Always detect query language for proper translation
                try:
                    query_lang, query_confidence = self.language_service.detect_language(query)
                    if not query_lang or query_lang == doc_language:
                        # For Indonesian filenames, override with 'id'
                        filename_lower = filename.lower()
                        indonesian_indicators = ['pengelolaan', 'keselamatan', 'penggunaan', 'pekerjaan', 'untuk', 'dengan', 'dalam']
                        if any(indicator in filename_lower for indicator in indonesian_indicators):
                            query_lang = "id"
                except:
                    # Default to Indonesian if detection fails
                    query_lang = "id"

                logger.info(f"Vector_store.py - VectorStoreService - get_relevant_chunks : "
                            f"Query language: {query_lang} (confidence: {query_confidence}), "
                            f"Document language: {doc_language}")
                
                # Decide on translation - be more aggressive with translation
                # to ensure Indonesian queries get translated properly when needed
                if query_lang != doc_language:
                    # Translation needed - always translate if languages don't match
                    translated_query = self.language_service.translate_text(query, query_lang, doc_language)
                    self._query_translation_cache[query_cache_key] = {
                        'original': query,
                        'translated': translated_query,
                        'source_lang': query_lang,
                        'target_lang': doc_language
                    }
                    query_to_embed = translated_query
                    logger.info(f"Vector_store.py - VectorStoreService - get_relevant_chunks : "
                                f"Translated query for embedding: {query[:30]}... -> {translated_query[:30]}...")
                else:
                    # No translation needed
                    query_to_embed = query
                    logger.info(f"Vector_store.py - VectorStoreService - get_relevant_chunks : "
                                f"No translation needed, using original query: {query[:30]}...")
                
                # Generate embedding for the appropriate query (original or translated)
                query_embedding = self.embeddings.embed_query(query_to_embed)
                
                # Cache the embedding
                self._query_embedding_cache[query_cache_key] = query_embedding
                
                # Limit cache size to avoid memory issues
                if len(self._query_embedding_cache) > 100:
                    # Remove oldest entries (simplistic approach)
                    keys_to_remove = list(self._query_embedding_cache.keys())[:-50]
                    for key in keys_to_remove:
                        self._query_embedding_cache.pop(key, None)
            
            # Perform vector search - making sure index is valid
            if not hasattr(vectorstore, 'index') or vectorstore.index is None:
                logger.error("Vector_store.py - VectorStoreService - get_relevant_chunks : Vector store index is None or missing")
                return [], "", [], []
                
            query_vector = np.ascontiguousarray([query_embedding], dtype="float32")
            
            # Get more initial results and filter later for better quality
            search_k = min(k * 3, 15)
            
            search_start = time.time()
            try:
                D, I = vectorstore.index.search(query_vector, search_k)
            except Exception as search_error:
                logger.error(f"Vector_store.py - VectorStoreService - get_relevant_chunks : FAISS search error: {str(search_error)}")
                return [], "", [], []
                
            search_end = time.time()
            logger.info(f"Vector_store.py - VectorStoreService - get_relevant_chunks : FAISS search time: {search_end - search_start:.4f}s")
            
            # Fast path: if no results, return early
            if len(I[0]) == 0 or all(idx == -1 for idx in I[0]):
                logger.info("Vector_store.py - VectorStoreService - get_relevant_chunks : No search results found")
                return [], "", [], []
            
            # Process the indices efficiently (avoid multiple loops)
            valid_indices = [i for i in I[0] if i != -1]
            if not valid_indices:
                logger.info("Vector_store.py - VectorStoreService - get_relevant_chunks : No valid indices found")
                return [], "", [], []
            
            # Create a mapping from index to document ID 
            index_to_doc_id = {}
            for i in valid_indices:
                try:
                    idx_str = str(i)
                    if not hasattr(vectorstore, 'index_to_docstore_id'):
                        logger.error("Vector_store.py - VectorStoreService - get_relevant_chunks : index_to_docstore_id missing")
                        return [], "", [], []
                        
                    if idx_str in vectorstore.index_to_docstore_id:
                        index_to_doc_id[i] = vectorstore.index_to_docstore_id[idx_str]
                except (KeyError, ValueError) as e:
                    logger.error(f"Vector_store.py - VectorStoreService - get_relevant_chunks : Error with index {i}: {str(e)}")
                    continue
            
            # Fast path: if no valid document IDs found, return early
            if not index_to_doc_id:
                logger.info("Vector_store.py - VectorStoreService - get_relevant_chunks : No valid document IDs found")
                return [], "", [], []
            
            # Efficient batch retrieval of documents
            chunk_candidates = []
            all_chunk_ids = []
            all_scores = []
            all_filenames = []
            
            # Track which docs we've seen to avoid duplicates
            doc_id_to_index = {}
            
            # Fast path for single document
            unique_docs = set()
            
            # Validate docstore exists
            if not hasattr(vectorstore, 'docstore') or vectorstore.docstore is None:
                logger.error("Vector_store.py - VectorStoreService - get_relevant_chunks : Vector store docstore is None or missing")
                return [], "", [], []
            
            # Process in order of similarity
            for idx, distance in zip(I[0], D[0]):
                if idx == -1:
                    continue
                    
                doc_id = index_to_doc_id.get(idx)
                if not doc_id:
                    continue
                    
                doc = vectorstore.docstore.get(doc_id)
                if not doc:
                    continue
                    
                # Calculate similarity score (higher is better)
                similarity = float(np.exp(-distance))

                # Skip low similarity scores
                if similarity < threshold:
                    continue
                    
                # Get metadata efficiently
                metadata = doc.metadata
                filename = metadata.get('filename', 'Unknown')
                unique_docs.add(filename)
                
                # Store chunk info
                all_chunk_ids.append(doc_id)
                all_scores.append(similarity)
                all_filenames.append(filename)
                print("line 1594")
                print(all_chunk_ids)
                # Get chunk text 
                chunk_text = doc.page_content
                
                # For chunking mapping, only look up if we need it
                if chunk_mapping and doc_id in chunk_mapping:
                    chunk_lang = chunk_mapping[doc_id].get('language', default_language)
                    source_id = chunk_mapping[doc_id].get('source_id')
                else:
                    chunk_lang = metadata.get('language', default_language)
                    source_id = metadata.get('source_id')
                
                # Add to candidates list
                chunk_candidates.append({
                    'chunk_id': doc_id,
                    'content': chunk_text,
                    'text': chunk_text,
                    'score': similarity,
                    'filename': filename,
                    'source_id': source_id,
                    'language': chunk_lang,
                    'final_score': similarity  # Pre-populate with base score
                })
                # We only need k results
                if len(chunk_candidates) >= k:
                    break
            
            # If we have scoring functions and still more than k candidates,
            # apply fast scoring to re-rank the top candidates
            if len(chunk_candidates) > k:
                # Only apply scoring if we have the method
                if hasattr(self, '_fast_score_chunk') and callable(self._fast_score_chunk):
                    for chunk in chunk_candidates:
                        chunk['final_score'] = self._fast_score_chunk(chunk)
                
                # Sort by final score and take top k
                chunk_candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
                chunk_candidates = chunk_candidates[:k]
                # Re-align the main return values
                all_chunk_ids = [c['chunk_id'] for c in chunk_candidates]
                all_scores = [c['final_score'] for c in chunk_candidates]
                all_filenames = [c['filename'] for c in chunk_candidates]
            
            # Fast context building for multiple documents
            source_id_map = {}
            for chunk in chunk_candidates:
                source_id = chunk.get('source_id')
                if source_id not in source_id_map:
                    source_id_map[source_id] = []
                source_id_map[source_id].append(chunk)
            
            # Build context by grouping chunks from same source
            context_parts = []
            
            for source_id, chunks in source_id_map.items():
                if not source_id or not chunks:
                    continue
                    
                # Sort chunks to maintain original document order
                chunks.sort(key=lambda x: x.get('chunk_id', ''))
                
                # Group source chunks
                source_chunks = []
                filename = chunks[0].get('filename', 'Unknown')
                
                for chunk in chunks:
                    chunk_id = chunk.get('chunk_id', '')
                    chunk_text = chunk.get('text', chunk.get('content', ''))
                    chunk_lang = chunk.get('language', 'id')
                    # Format chunk with ID 
                    if not chunk_id.startswith('chunk_'):
                        chunk_id = f"chunk_{chunk_id}"
                    
                    source_chunks.append(
                        f'[{chunk_id} from "{filename}" (lang:"{chunk_lang}")] {chunk_text}'
                    )
                
                # Add to context
                context_parts.append(
                    f'\n=== From Document: "{filename}" ===\n' + 
                    "\n".join(source_chunks)
                )
            
            context = "\n\n".join(context_parts)
            
            # If we have no context parts but have chunks, create a minimal context
            if not context_parts and chunk_candidates:
                # Create a minimal context with just the top chunk
                top_chunk = chunk_candidates[0]
                chunk_id = top_chunk.get('chunk_id', '')
                filename = top_chunk.get('filename', 'Unknown')
                chunk_text = top_chunk.get('text', top_chunk.get('content', ''))
                chunk_lang = top_chunk.get('language', self._current_document_language)
                
                # Format for context
                if not chunk_id.startswith('chunk_'):
                    chunk_id = f"chunk_{chunk_id}"
                    
                context = f'=== From Document: "{filename}" ===\n[{chunk_id} from "{filename}" (lang:{chunk_lang})] {chunk_text}'
            
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Vector_store.py - VectorStoreService - get_relevant_chunks : Total get_relevant_chunks time: {total_time:.4f}s")
            
            return all_chunk_ids, context, all_scores, all_filenames
        
        except Exception as e:
            logger.error(f"Vector_store.py - VectorStoreService - get_relevant_chunks : Error in get_relevant_chunks: {str(e)}")
            import traceback
            traceback.print_exc()
            # Always return the expected tuple format, even on error
            return [], "", [], []
        
    def _score_chunk(self, chunk: Dict, query: str) -> float:
        """Score chunk based on multiple relevance criteria."""
        try:
            base_score = chunk['score']  # Initial similarity score
            content = chunk['content']
            section_title = chunk.get('section_title', '')
            
            # Boost factors
            section_relevance = 1.0
            content_relevance = 1.0
            position_relevance = 1.0
            
            # Check if chunk contains declarative statements
            if self._contains_declarative_statement(content):
                content_relevance *= 1.2
            
            # Check if chunk is from a relevant section
            if section_title and self._is_relevant_section(section_title, query):
                section_relevance *= 1.15
            
            # Check position context (e.g., beginning of document/section)
            if self._is_introductory_content(chunk):
                position_relevance *= 1.1
            
            # Calculate information density
            info_density = self._calculate_info_density(content)
            
            # Combine scores with weights
            final_score = (
                base_score * 0.4 +
                content_relevance * 0.3 +
                section_relevance * 0.15 +
                position_relevance * 0.15
            ) * info_density
            
            return final_score
            
        except Exception as e:
            logger.info(f"Vector_store.py - VectorStoreService -  _score_chunk : Error scoring chunk: {str(e)}")
            return chunk['score']  # Return base score if scoring fails
            

    def _contains_declarative_statement(self, text: str) -> bool:
        """Check if text contains declarative statements."""
        # Look for sentence patterns that tend to be declarative
        patterns = [
            r'^(?:This|The|It|We|I)\s+(?:is|am|are|was|were)\s+',  # Statements starting with subject + be verb
            r'^(?:This|The)\s+(?:document|letter|report|analysis)\s+',  # Document purpose statements
            r'(?:purpose|goal|objective|aim)\s+(?:is|was|are|were)\s+to\s+',  # Purpose declarations
            r'(?:presents|describes|explains|outlines|summarizes)\s+'  # Descriptive verbs
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE | re.MULTILINE) 
                for pattern in patterns)

    def _is_relevant_section(self, section_title: str, query: str) -> bool:
        """Check if section title is relevant to query."""
        # Convert to sets of words for comparison
        title_words = set(section_title.lower().split())
        query_words = set(query.lower().split())
        
        # Calculate word overlap
        overlap = len(title_words & query_words)
        
        return overlap > 0

    def _is_introductory_content(self, chunk: Dict) -> bool:
        """Check if chunk contains introductory content."""
        metadata = chunk.get('metadata', {})
        position = metadata.get('position', -1)
        
        # Consider first few chunks of document/section as introductory
        if position >= 0 and position <= 2:
            return True
            
        return False

    def _calculate_info_density(self, text: str) -> float:
        """Calculate information density of text."""
        try:
            # Count substantive words (excluding common stop words)
            words = text.split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            substantive_words = [w for w in words if w.lower() not in stop_words]
            
            # Calculate density ratio
            if words:
                density = len(substantive_words) / len(words)
                return min(1.0, density + 0.5)  # Normalize and boost
                
            return 1.0
            
        except Exception:
            return 1.0

    def _select_diverse_chunks(self, chunks: List[Dict], k: int) -> List[Dict]:
        """Select diverse set of chunks to ensure good coverage."""
        if len(chunks) <= k:
            return chunks
            
        selected = [chunks[0]]  # Always include top chunk
        
        while len(selected) < k:
            # Find chunk with highest score that adds new information
            max_score = -1
            best_chunk = None
            
            for chunk in chunks:
                if chunk in selected:
                    continue
                    
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(chunk, selected)
                combined_score = chunk['final_score'] * diversity_score
                
                if combined_score > max_score:
                    max_score = combined_score
                    best_chunk = chunk
            
            if best_chunk:
                selected.append(best_chunk)
            else:
                break
        
        return selected

    def _calculate_diversity_score(self, candidate: Dict, selected: List[Dict]) -> float:
        """Calculate how much new information a chunk adds."""
        try:
            candidate_words = set(candidate['content'].lower().split())
            
            # Calculate overlap with existing selected chunks
            max_overlap = 0
            for chunk in selected:
                selected_words = set(chunk['content'].lower().split())
                if selected_words:
                    overlap = len(candidate_words & selected_words) / len(selected_words)
                    max_overlap = max(max_overlap, overlap)
            
            # Return diversity score (lower overlap = higher diversity)
            return 1.0 - (max_overlap * 0.8)  # Scale factor to avoid too much penalty
            
        except Exception:
            return 0.5  # Default middle value
        
    def _is_highlight_duplicate(self, new_highlight: Dict, existing_highlights: List[Dict]) -> bool:
        """Check if a highlight overlaps significantly with existing ones"""
        for existing in existing_highlights:
            if existing['page'] == new_highlight['page']:
                # Check for overlap
                e_coords = existing['coords']
                n_coords = new_highlight['coords']
                
                # Calculate overlap
                x_overlap = max(0, min(e_coords['x2'], n_coords['x2']) - max(e_coords['x1'], n_coords['x1']))
                y_overlap = max(0, min(e_coords['y2'], n_coords['y2']) - max(e_coords['y1'], n_coords['y1']))
                
                if x_overlap > 0 and y_overlap > 0:
                    # Calculate overlap area
                    overlap_area = x_overlap * y_overlap
                    new_area = (n_coords['x2'] - n_coords['x1']) * (n_coords['y2'] - n_coords['y1'])
                    
                    # If overlap is more than 50% of the new highlight, consider it a duplicate
                    if overlap_area / new_area > 0.5:
                        return True
        return False

    def cleanup_old_indices(self):
        """Clean up indices older than 1 hour and remove orphaned sessions"""
        try:
            now = datetime.now()
            for session_dir in self.TEMP_DIR.glob("*"):
                try:
                    metadata_file = session_dir / "metadata.json"
                    should_delete = False
                    
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        last_used = datetime.fromisoformat(metadata.get("last_used", "2000-01-01T00:00:00"))
                        if (now - last_used).total_seconds() > 3600:  # 1 hour
                            should_delete = True
                    else:
                        # No metadata file means it's an orphaned session
                        should_delete = True
                    
                    if should_delete:
                        shutil.rmtree(session_dir)
                        self.logger.info(f"Vector_store.py - VectorStoreService - cleanup_old_indices ~ error 1: Cleaned up old session: {session_dir}")
                        
                except Exception as e:
                    self.logger.error(f"Vector_store.py - VectorStoreService - cleanup_old_indices : Error checking session {session_dir}: {str(e)}")
                    # Try to remove problematic directory
                    try:
                        shutil.rmtree(session_dir)
                    except:
                        pass
                        
        except Exception as e:
            self.logger.error(f"Vector_store.py - VectorStoreService - cleanup_old_indices ~ error 2: Error in cleanup_old_indices: {str(e)}")
            traceback.print_exc()

    
    def cleanup_vectorstore(self, session_id: str, user_id: str = None, db_session = None):
        """Clean up vector store with user-specific paths"""
        try:
            self.logger.info(f"Cleaning up vector store for session: {session_id}")
            from app.storage_config import get_group_for_user
            group_name = get_group_for_user(user_id, db_session)
            # Remove from session cache
            self._session_cache.pop(session_id, None)
            
            # Determine directory path
            if user_id:
                session_dir, _, _ = self._get_user_paths(session_id, user_id, group_name)
            else:
                raise ValueError("Vector_store.py - VectorStoreService - cleanup_vectorstore : User ID and database session are required")
            
            # Remove the directory and contents
            if session_dir.exists():
                try:
                    # Force garbage collection
                    gc.collect()
                    
                    # Remove directory
                    shutil.rmtree(session_dir)
                    self.logger.info(f"Vector_store.py - VectorStoreService - cleanup_vectorstore : Removed session directory: {session_dir}")
                    
                except Exception as e:
                    self.logger.error(f"Vector_store.py - VectorStoreService - cleanup_vectorstore : Error removing session directory: {str(e)}")
                    # Try to remove files individually
                    for file in session_dir.glob("*"):
                        try:
                            file.unlink()
                        except:
                            pass
                    try:
                        session_dir.rmdir()
                    except:
                        pass
            
            # Force final garbage collection
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up vector store: {str(e)}")
            traceback.print_exc()
    
def sync_user_files(user_id: str, db_session) -> Dict[str, Dict]:
    """Sync and return all files for a user using filename as key for persistent documents
    with optimized loading - only metadata is loaded initially"""
    try:
        from app.storage_config import VECTOR_STORE_DIR, ORIGINAL_FILES_DIR, CHUNK_MAPS_DIR, get_group_for_user
        group_name = get_group_for_user(user_id, db_session)
        
        logger.info(f"Vector_store.py - VectorStoreService - sync_user_files : Syncing files for user {user_id} in group {group_name}")
        
        # Get user directories
        user_vector_dir = VECTOR_STORE_DIR / str(group_name)
        user_original_dir = ORIGINAL_FILES_DIR / str(group_name)
        user_chunks_dir = CHUNK_MAPS_DIR / str(group_name)
        
        # Initialize result dictionary
        synced_files = {}
        
        # Ensure directories exist
        user_vector_dir.mkdir(parents=True, exist_ok=True)
        user_original_dir.mkdir(parents=True, exist_ok=True)
        user_chunks_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Vector_store.py - VectorStoreService - sync_user_files : Looking for files in {user_original_dir}")
        
        # Read original files - but only load metadata initially
        if user_original_dir.exists():
            for file_path in user_original_dir.glob('*'):
                if file_path.is_file():
                    try:
                        filename = file_path.name
                        logger.info(f"Vector_store.py - VectorStoreService - sync_user_files : Found file: {filename}")
                        
                        # For persistent documents, use filename directly as session_id
                        session_id = filename
                        
                        # Check if vector store exists - but don't load it
                        vector_store_exists = False
                        vector_store_path = user_vector_dir / filename
                        
                        if vector_store_path.exists() and (vector_store_path / "index.bin").exists():
                            vector_store_exists = True
                            logger.info(f"Vector_store.py - VectorStoreService - sync_user_files : Found vector store at: {vector_store_path}")
                        
                        # Check for chunk mapping - but don't load content
                        chunk_file = user_chunks_dir / f"{filename}_chunks.json"
                        chunk_mapping_exists = chunk_file.exists()
                        
                        # Get basic file info without loading content
                        file_size = file_path.stat().st_size
                        file_extension = file_path.suffix.lower()
                        
                        # Store file information - without actual content
                        synced_files[session_id] = {
                            "filename": filename,
                            "content": None,  # Don't load content now
                            "file_size": file_size,
                            "timestamp": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                            "user_id": user_id,
                            "file_path": str(file_path),
                            "file_extension": file_extension,
                            "vector_store_path": str(vector_store_path) if vector_store_exists else None,
                            "vector_store_exists": vector_store_exists,
                            "chunk_mapping_exists": chunk_mapping_exists,
                            "chunk_mapping_path": str(chunk_file) if chunk_mapping_exists else None,
                            "source": "group",  # Mark as a group (persistent) source
                            "group_name": group_name,  # Store the group name
                            "content_loaded": False  # Flag to indicate content is not loaded
                        }
                        
                        logger.info(f"Vector_store.py - VectorStoreService - sync_user_files : Added file metadata for {filename} to sync result")
                        
                    except Exception as e:
                        logger.error(f"Vector_store.py - VectorStoreService - sync_user_files : Error syncing file {file_path}: {str(e)}")
                        traceback.print_exc()
                        continue
        
        logger.info(f"Vector_store.py - VectorStoreService - sync_user_files : Synced metadata for {len(synced_files)} files for user {user_id}")
        return synced_files
        
    except Exception as e:
        logger.error(f"Error syncing user files: {str(e)}")
        traceback.print_exc()
        return {}
    
def check_files_exist(user_id: str, session_id: str, db_session: str) -> Dict[str, bool]:
    """Check if all necessary files exist"""
    try:
        vector_path, original_path, chunks_path = _get_storage_paths(session_id, user_id, db_session)
        logger.info(vector_path, original_path, chunks_path )
        return {
            "vector_store": os.path.exists(os.path.join(vector_path, "index.bin")),
            "metadata": os.path.exists(os.path.join(vector_path, "metadata.json")),
            "chunks": os.path.exists(os.path.join(chunks_path, f"{session_id}_chunks.json"))
        }
    except Exception as e:
        logger.info(f"Vector_store.py - VectorStoreService - sync_user_files : Error checking files: {str(e)}")
        return {"vector_store": False, "metadata": False, "chunks": False}
    
def _get_storage_paths(session_id: str, user_id: str, db_session: str) -> Tuple[Path, Path, Path]:
    """Get paths for vector store, original file, and chunk mapping"""
    from app.storage_config import VECTOR_STORE_DIR, ORIGINAL_FILES_DIR, CHUNK_MAPS_DIR, get_group_for_user
    group_name = get_group_for_user(user_id, db_session)
    # Create user-specific directories
    user_vector_dir = VECTOR_STORE_DIR / str(group_name)
    user_original_dir = ORIGINAL_FILES_DIR / str(group_name)
    user_chunks_dir = CHUNK_MAPS_DIR / str(group_name)
    # Create session directory under vector store
    session_vector_dir = user_vector_dir / session_id
    
    # Get file paths
    vector_path = session_vector_dir
    original_path = user_original_dir / f"{session_id}"  # Assuming PDF for now
    chunks_path = user_chunks_dir
    
    return vector_path, original_path, chunks_path

