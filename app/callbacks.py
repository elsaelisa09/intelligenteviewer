__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"
__license__  = "MIT"

#callbacks.py

from dash import Input, Output, State, ctx, ALL, MATCH
from dash import html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash
import json
import base64
import docx2txt
import io
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from services.llm_service import LLMServiceFactory, create_llm_service
from utils.visualization import DocumentVisualizer,create_system_notification
from utils.text_helpers import TextProcessor
from app import config, layout
from app.document_view import create_document_list_layout
from urllib.parse import urlparse
import requests
import os
from pathlib import Path
import mimetypes
from datetime import datetime
import traceback
import pandas as pd
import re
from services.vector_store import sync_user_files, VectorStoreService
from app.document_callbacks import sync_documents_on_load
import warnings
from auth.route_settings import get_welcome_messages
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

MAX_DOC_SIZE = 50 * 1024 * 1024

# Initialize global variables
DocProc = DocumentProcessor()
vect_serv = VectorStoreService()
logger.info("Callbacks.py - default:  Loading services...")
llm_serv = LLMServiceFactory.create_llm_service()

def parse_contents(contents, filename):
    try:
        logger.info(f"Parsing Contents for {filename}")
        
        if not contents:
            logger.info("WARNING: No contents received!")
            return None, [], [], ''

        try:
            content_type, content_string = contents.split(',', 1)
            decoded = base64.b64decode(content_string)
        except Exception as e:
            logger.info(f"Error processing content: {str(e)}")
            return None, [], [], ''

        if len(decoded) > MAX_DOC_SIZE:
            raise Exception("File too large (max 50MB)")

        if filename.lower().endswith('.pdf'):
            content, images, tables, plain_text = DocProc.process_pdf(decoded)
            return content, images, tables, plain_text

        elif filename.lower().endswith(('.txt', '.md')):
            content = decoded.decode("utf-8")
            return content, [], [], content

        elif filename.lower().endswith('.docx'):
            content = docx2txt.process(io.BytesIO(decoded))
            return content, [], [], content

        else:
            raise Exception("Unsupported file type")

    except Exception as e:
        logger.info(f"Error in parse_contents: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def get_group_welcome_message(user_id, db_session=None):
    """
    Get the welcome message for a user's group
    
    Args:
        user_id (str): User ID to get message for
        db_session: Database session (optional)
        
    Returns:
        str: Welcome message for the user's group or default message
    """
    try:
        # Import needed modules
        from pathlib import Path
        import json
        import os
        
        # Create database session if not provided
        if not db_session:
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import create_engine
            from auth.config import AUTH_CONFIG
            engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
            DBSession = sessionmaker(bind=engine)
            db_session = DBSession()
        
        # Get the user's group using the existing function
        from app.storage_config import get_group_for_user
        group_name = get_group_for_user(user_id, db_session)
        
        logger.info(f"Callbacks.py - get_group_welcome_message : User {user_id} belongs to group: {group_name}")
        
        # Load welcome messages from settings
        storage_dir = Path("storage")
        settings_dir = storage_dir / "settings"
        welcome_messages_file = settings_dir / "group_welcome_messages.json"
        
        # Make sure the settings directory exists
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Callbacks.py - get_group_welcome_message : Looking for welcome messages at: {welcome_messages_file.absolute()}")
        
        # Check if welcome messages file exists
        if welcome_messages_file.exists():
            try:
                with open(welcome_messages_file, "r") as f:
                    welcome_messages = json.load(f)
                    
                # Look for this group's message
                if group_name in welcome_messages:
                    logger.info(f"Found welcome message for group {group_name}")
                    return welcome_messages[group_name]
            except Exception as e:
                logger.info(f"Error loading welcome messages: {str(e)}")
        
        # Try loading from auth.route_settings as a fallback
        try:
            from auth.route_settings import get_welcome_messages
            welcome_messages = get_welcome_messages()
            if group_name in welcome_messages:
                logger.info(f"Found welcome message for group {group_name} in route_settings")
                return welcome_messages[group_name]
        except Exception as e:
            logger.error(f"Error loading welcome messages from route_settings ~1: {str(e)}")
        
        # Fallback to default message if file doesn't exist or group not found
        default_message = f"Welcome to {group_name}! You can ask questions about documents in this group."
        logger.info(f"Callbacks.py - get_group_welocme_message : Using default welcome message for group {group_name}")
        return default_message
        
    except Exception as e:
        logger.error(f"Callbacks.py - get_group_welocme_message : Error getting welcome message ~2 : {str(e)}")
        import traceback
        traceback.print_exc()
        return "Welcome! You can ask questions about your documents."
    
def get_retrieval_settings(group_name):
    """
    Get document retrieval settings from the LLM settings.
    Returns default values if settings are not found.
    
    Returns:
        dict: Dictionary with retrieval settings including:
              - chunks_per_doc: Number of chunks to retrieve per document
              - max_total_chunks: Maximum total chunks to include in context
              - similarity_threshold: Minimum similarity threshold for chunks
    """
    try:
        # Import settings function
        from auth.route_settings import get_group_llm_settings
        
        # Get LLM settings
        retrieval_settings = get_group_llm_settings(group_name)
        
        # Get values with defaults
        chunks_per_doc = retrieval_settings["chunks_per_doc"]
        max_total_chunks = retrieval_settings["max_total_chunks"]
        similarity_threshold = retrieval_settings["similarity_threshold"]
        lang = retrieval_settings["default_language"]
        
        logger.info(f"Callbacks.py - get_retrieval_settings : Retrieved document retrieval settings:")
        logger.info(f"Callbacks.py - get_retrieval_settings : - chunks_per_doc: {chunks_per_doc}")
        logger.info(f"Callbacks.py - get_retrieval_settings : - max_total_chunks: {max_total_chunks}")
        logger.info(f"Callbacks.py - get_retrieval_settings : - similarity_threshold: {similarity_threshold}")
        logger.info(f"Callbacks.py - get_retrieval_settings : - default group language: {lang}")

        return {
            "chunks_per_doc": chunks_per_doc,
            "max_total_chunks": max_total_chunks,
            "similarity_threshold": similarity_threshold,
            "lang" : lang
        }
    except Exception as e:
        logger.error(f"Callbacks.py - get_retrieval_settings : Error getting retrieval settings: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return default values
        return {
            "chunks_per_doc": 3,
            "max_total_chunks": 10,
            "similarity_threshold": 0.75,
            "lang" : "en"
        }
    
def create_feedback_buttons(index):
    """Create like/dislike buttons with a consistent style"""
    return html.Div([
        html.Button([
            html.I(className="fas fa-thumbs-up me-1"),
            "Like"
        ],
        id={
            'type': 'like-button',
            'index': index
        },
        className="btn btn-outline-success btn-sm me-2"
        ),
        html.Button([
            html.I(className="fas fa-thumbs-down me-1"),
            "Dislike"
        ],
        id={
            'type': 'dislike-button',
            'index': index
        },
        className="btn btn-outline-danger btn-sm"
        ),
        html.Div(
            "Thank you for your feedback!",
            id={'type': 'feedback-message', 'index': index},
            style={'display': 'none'},
            className="text-muted mt-2 small"
        )
    ], className="mt-2")

#save like details to csv
def save_feedback_to_csv(query, response, feedback_type, document_name):
    """Save feedback to CSV file"""
    feedback_file = "feedback_data.csv"
    timestamp = datetime.now().isoformat()
    
    # Prepare the new feedback data
    new_data = {
        'timestamp': [timestamp],
        'query': [query],
        'response': [response],
        'feedback': [feedback_type],
        'document': [document_name]
    }
    
    try:
        # If file exists, append to it
        if os.path.exists(feedback_file):
            df = pd.read_csv(feedback_file)
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            # Create new file with headers
            df = pd.DataFrame(new_data)
        
        # Save to CSV
        df.to_csv(feedback_file, index=False)
        return True
    except Exception as e:
        logger.error(f"Callbacks.py - save_feedback_to_csv : Error saving feedback: {str(e)}")
        return False

def load_document_content(session_id: str, doc_info: Dict) -> Dict:
    """Lazily load document content only when needed"""
    
    # If content is already loaded, return as is
    if doc_info.get('content_loaded', False) and doc_info.get('content') is not None:
        return doc_info
    
    try:
        # Get the file path
        file_path = doc_info.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return doc_info
        
        # Load content based on file extension
        file_extension = doc_info.get('file_extension', '').lower()
        if not file_extension and 'filename' in doc_info:
            file_extension = os.path.splitext(doc_info['filename'])[1].lower()
        
        with open(file_path, 'rb') as f:
            content = f.read()
            
        if file_extension == '.pdf':
            # For PDFs, convert to base64 string
            content_type = "application/pdf"
            content_b64 = f"data:{content_type};base64,{base64.b64encode(content).decode()}"
            doc_info['content'] = content_b64
        elif file_extension in ['.txt', '.md']:
            # For text files, decode to string
            try:
                doc_info['content'] = content.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback to base64 if decoding fails
                content_type = "text/plain"
                content_b64 = f"data:{content_type};base64,{base64.b64encode(content).decode()}"
                doc_info['content'] = content_b64
        elif file_extension == '.docx':
            # For DOCX, use docx2txt
            try:
                import docx2txt
                import io
                doc_info['content'] = docx2txt.process(io.BytesIO(content))
            except Exception as e:
                logger.info(f"Error processing DOCX: {str(e)}")
                # Fallback to base64
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                content_b64 = f"data:{content_type};base64,{base64.b64encode(content).decode()}"
                doc_info['content'] = content_b64
        else:
            # For unknown types, store as base64
            content_type = "application/octet-stream"
            content_b64 = f"data:{content_type};base64,{base64.b64encode(content).decode()}"
            doc_info['content'] = content_b64
            
        # Update the content loaded flag
        doc_info['content_loaded'] = True
        return doc_info
        
    except Exception as e:
        logger.error(f"Callbacks.py - load_document_content : Error loading content for {session_id}: {str(e)}")
        traceback.print_exc()
        return doc_info
    
def load_chunk_mapping(session_id, user_id, db_session):
    """
    Load chunk mapping from persistent storage for a document
    
    Args:
        session_id (str): Session ID or filename
        user_id (str): User ID
        db_session: Database session
        
    Returns:
        dict: Loaded chunk mapping or None if not found
    """
    try:
        from app.storage_config import CHUNK_MAPS_DIR, get_group_for_user
        group_name = get_group_for_user(user_id, db_session)
        
        # Try with filename_chunks.json first
        chunk_file_path = CHUNK_MAPS_DIR / str(group_name) / f"{session_id}_chunks.json"
        
        if not chunk_file_path.exists():
            # Try with session_id as is
            chunk_file_path = CHUNK_MAPS_DIR / str(group_name) / f"{session_id}"
            if not chunk_file_path.exists():
                logger.info(f"Callbacks.py - load_chunk_mapping : No chunk mapping found for {session_id} in group {group_name}")
                return None
        
        logger.info(f"Callbacks.py - load_chunk_mapping : Loading chunk mapping from: {chunk_file_path}")
        with open(chunk_file_path, 'r') as f:
            chunk_mapping = json.load(f)
        
        return chunk_mapping
        
    except Exception as e:
        logger.error(f"Callbacks.py - load_chunk_mapping : Error loading chunk mapping: {str(e)}")
        traceback.print_exc()
        return None

def register_welcome_message_callbacks(app):
    """Register callbacks related to welcome messages"""
    from dash import Input, Output, State, html
    from utils.visualization import create_system_notification
    
    @app.callback(
        Output("chat-history", "children", allow_duplicate=True),
        [Input("page-loaded-trigger", "data")],
        [State("chat-history", "children"),
         State("auth-state", "data"),
         State("document-state", "data"),
         State("url", "pathname")],
        prevent_initial_call=True
    )
    def show_welcome_message(trigger, chat_history, auth_state, doc_state, pathname):
        """Show a welcome message when the page loads"""
        if not trigger or not auth_state or not auth_state.get('authenticated'):
            return chat_history or []
        
        # Only show on main page
        if pathname != '/':
            return chat_history or []
            
        # Don't show if chat history already has messages
        if chat_history and len(chat_history) > 0:
            # Check if any of the messages are not system notifications
            has_user_messages = False
            for msg in chat_history:
                if isinstance(msg, dict) and 'props' in msg:
                    if 'className' in msg['props'] and 'system-notification' not in msg['props']['className']:
                        has_user_messages = True
                        break
            
            if has_user_messages:
                return chat_history
        
        try:
            # Get current user's ID
            current_user_id = auth_state.get('user_id')
            if not current_user_id:
                return chat_history or []
            
            # Get welcome message for this user's group
            welcome_message = get_group_welcome_message(current_user_id)
            
            logger.info(f"Callbacks.py - register_welcome_message_callback - show_welcome_message : Retrieved welcome message: {welcome_message}")
            
            # Create welcome message notification
            welcome_notification = create_system_notification(
                welcome_message,
                type="info",
                action="welcome"
            )
            
            # Return new chat history with welcome message
            # First clear any system notifications
            filtered_chat_history = []
            for msg in chat_history or []:
                if isinstance(msg, dict) and 'props' in msg:
                    if 'className' in msg['props'] and 'system-notification' in msg['props']['className']:
                        continue
                filtered_chat_history.append(msg)
            
            # Add welcome message at the beginning
            return [welcome_notification] + filtered_chat_history
            
        except Exception as e:
            logger.error(f"Callbacks.py - register_welcome_message_callback - show_welcome_message : Error showing welcome message: {str(e)}")
            import traceback
            traceback.print_exc()
            return chat_history or []
        
def register_feedback_callbacks(app):
    @app.callback(
        [Output({'type': 'feedback-message', 'index': MATCH}, 'style'),
         Output({'type': 'like-button', 'index': MATCH}, 'disabled'),
         Output({'type': 'dislike-button', 'index': MATCH}, 'disabled')],
        [Input({'type': 'like-button', 'index': MATCH}, 'n_clicks'),
         Input({'type': 'dislike-button', 'index': MATCH}, 'n_clicks')],
        [State('chat-history', 'children'),
         State('document-selector', 'value'),
         State('document-state', 'data')]
    )
    def handle_feedback(like_clicks, dislike_clicks, chat_history, selected_doc, doc_state):
        if not like_clicks and not dislike_clicks:
            raise PreventUpdate
            
        triggered = ctx.triggered_id
        feedback_index = triggered['index']
        feedback_type = 'like' if triggered['type'] == 'like-button' else 'dislike'
        
        try:
            # Find the corresponding query-response pair
            for i, msg in enumerate(chat_history):
                if isinstance(msg, dict) and msg.get('props', {}).get('id', {}).get('index') == feedback_index:
                    query = chat_history[i-1]['props']['children']
                    response = msg['props']['children'][0]['props']['children']
                    
                    # Get document name
                    doc_name = doc_state[selected_doc]['filename'] if selected_doc and selected_doc in doc_state else 'Unknown'
                    
                    # Save feedback
                    if save_feedback_to_csv(query, response, feedback_type, doc_name):
                        return {'display': 'block'}, True, True
                    
            return {'display': 'none'}, False, False
            
        except Exception as e:
            logger.error(f"Callbacks.py - register_feedback_callbacks - handle_feedback : Error handling feedback: {str(e)}")
            return {'display': 'none'}, False, False

def process_pdf_highlights(doc_proc, content, used_chunk_ids, chunk_mapping_for_highlights, assistant_reply):
    """Process highlights for PDF documents with improved error handling"""
    try:
        logger.info(f"Callbacks.py - process_pdf_highlights : Processing PDF highlights for {len(used_chunk_ids)} chunks")
        highlights = doc_proc.get_pdf_highlights(
            content, 
            used_chunk_ids, 
            chunk_mapping_for_highlights, 
            assistant_reply
        )
        return highlights
    except Exception as e:
        logger.error(f"Callbacks.py - process_pdf_highlights ~1 : Error in PDF highlighting: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback: Create basic highlights for visualization
        highlights = []
        try:
            # For PDF content, create minimal highlight data
            for chunk_id in used_chunk_ids:
                chunk_data = chunk_mapping_for_highlights.get(chunk_id, {})
                page_num = chunk_data.get('metadata', {}).get('page', 0)
                
                # Create a simple highlight on the appropriate page
                highlights.append({
                    'page': page_num,
                    'coords': {
                        'x1': 50,  # Arbitrary position
                        'y1': 100,
                        'x2': 500,
                        'y2': 150
                    },
                    'chunk_id': chunk_id
                })
                logger.info(f"Callbacks.py - process_pdf_highlights : Created fallback highlight for chunk {chunk_id} on page {page_num}")
        except:
            logger.error("Callbacks.py - process_pdf_highlights ~2 : Failed to create fallback highlights")
        
        return highlights
    
def extract_chunk_text(chunk_id, chunk_mapping, assistant_reply, filename=None):
    """
    Extract chunk text from multiple sources with priority to persistent storage
    
    Args:
        chunk_id (str): Chunk ID to extract
        chunk_mapping (dict): Chunk mapping dictionary from memory
        assistant_reply (str): Full LLM response
        filename (str, optional): Filename to load persistent chunk mapping
    
    Returns:
        str: Extracted chunk text or empty string
    """
    import os
    import json
    from app.storage_config import CHUNK_MAPS_DIR, get_group_for_user
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine
    from auth.config import AUTH_CONFIG
    from auth.models import Base
    
    # Ensure chunk_id has 'chunk_' prefix
    chunk_key = chunk_id if chunk_id.startswith('chunk_') else f'chunk_{chunk_id}'
    
    # 1. Check persistent storage first if filename is provided
    if filename:
        try:
            # Get current user's group
            engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
            DBSession = sessionmaker(bind=engine)
            db_session = DBSession()
            
            # Get group name 
            group_name = get_group_for_user(db_session.query(Base).first().id, db_session)
            
            # Construct potential chunk map file paths
            chunk_map_paths = [
                CHUNK_MAPS_DIR / str(group_name) / f"{filename}_chunks.json",
                CHUNK_MAPS_DIR / str(group_name) / filename
            ]
            
            # Try loading from persistent storage
            for chunk_map_path in chunk_map_paths:
                if chunk_map_path.exists():
                    with open(chunk_map_path, 'r') as f:
                        persistent_chunk_map = json.load(f)
                        
                    if chunk_key in persistent_chunk_map:
                        chunk_text = persistent_chunk_map[chunk_key].get('text', '')
                        if chunk_text:
                            logger.info(f"Callbacks.py - extract_chunk_text : Found chunk text in persistent storage: {chunk_text[:100]}...")
                            return chunk_text
        except Exception as e:
            logger.error(f"Callbacks.py - extract_chunk_text : Error loading persistent chunk mapping ~1: {e}")
    
    # 2. Check in-memory chunk mapping
    if chunk_key in chunk_mapping:
        chunk_text = chunk_mapping[chunk_key].get('text', '')
        if chunk_text:
            logger.info(f"Callbacks.py - extract_chunk_text : Found chunk text in memory mapping: {chunk_text[:100]}...")
            return chunk_text
    
    # 3. Try extracting from assistant reply
    try:
        # Regex to find chunk text between chunk identifiers
        chunk_pattern = rf'\[chunk_{chunk_id} from [^\]]*\](.*?)(?=\[chunk_|\Z)'
        match = re.search(chunk_pattern, assistant_reply, re.DOTALL)
        if match:
            chunk_text = match.group(1).strip()
            logger.info(f"Callbacks.py - extract_chunk_text : Found chunk text in assistant reply: {chunk_text[:100]}...")
            return chunk_text
    except Exception as e:
        logger.error(f"Callbacks.py - extract_chunk_text : Error extracting text from assistant reply ~2: {e}")
    
    # 4. Fallback to chunk mapping iteration
    for key, value in chunk_mapping.items():
        if chunk_id in key or key.endswith(chunk_id):
            chunk_text = value.get('text', '')
            if chunk_text:
                logger.info(f"Callbacks.py - extract_chunk_text : Found chunk text through iteration: {chunk_text[:100]}...")
                return chunk_text
    
    logger.info(f"Callbacks.py - extract_chunk_text : No text found for chunk {chunk_id}")
    return ""

    
def process_highlights(doc_state, chunk_mapping, used_chunks, assistant_reply, document):
    """
    Create comprehensive highlights covering at least 50% of the chunk text
    with improved handling for on-the-fly and persistent PDF documents
    """
    import re
    
    try:
        logger.info("Callbacks.py - process_highlights : \n### HIGHLIGHT PROCESSING START ###")
        logger.info(f"Callbacks.py - process_highlights : Target document: {document}")
        logger.info(f"Callbacks.py - process_highlights : Used chunks: {used_chunks}")
        
        # Find the matching document in doc_state
        matching_session_id = None
        
        for session_id, info in doc_state.items():
            filename = info.get('filename', '')
            if filename == document:
                matching_session_id = session_id
                break
        
        if not matching_session_id:
            logger.info(f"Callbacks.py - process_highlights : No matching document found for: {document}")
            return doc_state
        
        # Get document info
        doc_info = doc_state[matching_session_id]
        
        # Ensure document content is loaded
        if not doc_info.get('content_loaded', False) or not doc_info.get('content'):
            try:
                from app.callbacks import load_document_content
                doc_info = load_document_content(matching_session_id, doc_info)
                doc_state[matching_session_id] = doc_info
                logger.info(f"Callbacks.py - process_highlights : Content loaded for {document}")
            except Exception as e:
                logger.error(f"Callbacks.py - process_highlights ~1 : Error loading content: {str(e)}")
                return doc_state
        
        # Process all used chunks that apply to this document
        highlights = []
        content = doc_info.get('content', '')
        
        if not content:
            logger.info(f"Callbacks.py - process_highlights : No content available for {document}")
            return doc_state
        
        # Determine if this is a PDF document
        is_pdf = document.lower().endswith('.pdf')
        
        # For PDF documents, create a document-specific chunk mapping with source_id
        if is_pdf:
            # Create a modified chunk mapping with proper source_id
            doc_chunk_mapping = {}
            
            for chunk_id in used_chunks:
                if chunk_id in chunk_mapping:
                    # Make a copy of the chunk data to avoid modifying the original
                    chunk_data = dict(chunk_mapping[chunk_id])
                    
                    # Set source_id explicitly to match the document's session_id
                    chunk_data['source_id'] = matching_session_id
                    
                    # Add to the document-specific mapping
                    doc_chunk_mapping[chunk_id] = chunk_data
                    logger.info(f"Callbacks.py - process_highlights : Added chunk {chunk_id} to doc mapping with source_id={matching_session_id}")
            
            if doc_chunk_mapping:
                # Process all chunks at once for efficiency
                try:
                    # Get all highlights for this document
                    logger.info(f"Callbacks.py - process_highlights : Getting PDF highlights for {len(doc_chunk_mapping)} chunks")
                    highlights = DocProc.get_pdf_highlights(
                        content,
                        list(doc_chunk_mapping.keys()),
                        doc_chunk_mapping,
                        assistant_reply
                    )
                    
                    if highlights:
                        logger.info(f"Callbacks.py - process_highlights : Found {len(highlights)} highlights for PDF document")
                        doc_state[matching_session_id]["highlights"] = highlights
                    else:
                        logger.info("Callbacks.py - process_highlights : No highlights found for PDF document")
                        # Try fallback method for PDFs
                        try:
                            fallback_highlights = []
                            for chunk_id in used_chunks:
                                if chunk_id in chunk_mapping:
                                    chunk_text = chunk_mapping[chunk_id].get('text', '')
                                    logger.info(f"Callbacks.py - process_highlights : Trying fallback with chunk text: {chunk_text[:50]}...")
                                    
                                    # Create a basic highlight with metadata we have
                                    highlight = {
                                        'page': 0,  # Default to first page if not known
                                        'coords': {
                                            'x1': 50,
                                            'y1': 100,
                                            'x2': 500,
                                            'y2': 150
                                        },
                                        'text': chunk_text[:100] if chunk_text else 'Highlighted text',
                                        'chunk_id': chunk_id,
                                        'source_id': matching_session_id
                                    }
                                    
                                    # Try to get page number from metadata if available
                                    if 'metadata' in chunk_mapping[chunk_id]:
                                        metadata = chunk_mapping[chunk_id]['metadata']
                                        if 'page' in metadata:
                                            highlight['page'] = metadata['page']
                                    
                                    fallback_highlights.append(highlight)
                            
                            if fallback_highlights:
                                logger.info(f"Callbacks.py - process_highlights : Created {len(fallback_highlights)} fallback highlights")
                                doc_state[matching_session_id]["highlights"] = fallback_highlights
                        except Exception as fallback_error:
                            logger.error(f"Callbacks.py - process_highlights ~1 : Fallback highlighting failed: {str(fallback_error)}")
                except Exception as pdf_error:
                    logger.error(f"Callbacks.py - process_highlights ~2 : Error getting PDF highlights: {str(pdf_error)}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.info("Callbacks.py - process_highlights : No chunks found in mapping for this document")
        else:
            # Handle text document highlighting
            for chunk_id in used_chunks:
                try:
                    # Get chunk text
                    chunk_text = None
                    
                    # Try to get chunk text from chunk_mapping
                    if chunk_id in chunk_mapping:
                        chunk_text = chunk_mapping[chunk_id].get('text', '')
                        logger.info(f"Callbacks.py - process_highlights : Found chunk text in mapping: {chunk_text[:50]}...")
                    
                    if chunk_text and isinstance(content, str) and isinstance(chunk_text, str):
                        # First, try to find the entire chunk text in the content
                        # Normalize both to handle whitespace differences
                        normalized_chunk = ' '.join(chunk_text.split())
                        normalized_content = ' '.join(content.split())
                        
                        chunk_pos = normalized_content.find(normalized_chunk)
                        if chunk_pos >= 0:
                            # We found the entire chunk text!
                            highlight = {
                                'start': chunk_pos,
                                'end': chunk_pos + len(normalized_chunk),
                                'text': normalized_chunk,
                                'chunk_id': chunk_id
                            }
                            highlights.append(highlight)
                            logger.info(f"Callbacks.py - process_highlights : Found and highlighted entire chunk text at position {chunk_pos}")
                        else:
                            # Try to break the chunk into paragraphs and highlight those
                            paragraphs = re.split(r'\r?\n\r?\n', chunk_text)
                            highlighted_paragraphs = []
                            
                            start = []
                            end = []
                            for para in paragraphs:
                                if not para.strip():
                                    continue
                                    
                                # Normalize paragraph text
                                normalized_para = ' '.join(para.split())
                                para_pos = normalized_content.find(normalized_para)
                                
                                if para_pos >= 0:
                                    logger.info(para_pos)
                                    start.append(para_pos)
                                    end.append(para_pos + len(normalized_para))
                            if start:
                                highlight = {
                                    'start': min(start),
                                    'end': max(end),
                                    'text': paragraphs,
                                    'chunk_id': chunk_id
                                }
                                highlights.append(highlight)
                                highlighted_paragraphs.append(paragraphs)
                                logger.info(f"Callbacks.py - process_highlights : Highlighted paragraph at position {para_pos}")
                                
                            # If we didn't find any paragraphs, try sentences
                            if not highlighted_paragraphs:
                                sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
                                for sentence in sentences:
                                    if len(sentence.strip()) < 15:
                                        continue
                                        
                                    # Normalize sentence text
                                    normalized_sentence = ' '.join(sentence.split())
                                    sentence_pos = normalized_content.find(normalized_sentence)
                                    
                                    if sentence_pos >= 0:
                                        start.append(sentence_pos)
                                        end.append(sentence_pos + len(normalized_sentence))
                                        
                                if start:
                                    highlight = {
                                        'start': min(start),
                                        'end': max(end),
                                        'text': sentences,
                                        'chunk_id': chunk_id
                                    }
                                    highlights.append(highlight)
                                    logger.info(f"Callbacks.py - process_highlights : Highlighted sentence at position {sentence_pos}")
                    else:
                        logger.error(f"Callbacks.py - process_highlights ~3: No chunk text or content is not a string")
                except Exception as e:
                    logger.error(f"Callbacks.py - process_highlights ~4: Error processing chunk {chunk_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Update text document with highlights
            if highlights:
                # Sort highlights by start position for better organization
                highlights.sort(key=lambda h: h['start'])
                
                # Check how much of the chunk text we're highlighting
                if 'chunk_text' in locals() and chunk_text:
                    total_highlighted = sum(len(str(h['text'])) for h in highlights)
                    coverage = total_highlighted / len(chunk_text)
                    logger.info(f"Callbacks.py - process_highlights : Highlighting approximately {coverage:.1%} of the chunk text")
                
                logger.info(f"Added {len(highlights)} highlights to {document}")
                doc_state[matching_session_id]['highlights'] = highlights
        
        logger.info("Callbacks.py - process_highlights : ### HIGHLIGHT PROCESSING COMPLETE ###")
        return doc_state
        
    except Exception as e:
        logger.error(f"Callbacks.py - process_highlights ~5: Error in process_highlights: {str(e)}")
        import traceback
        traceback.print_exc()
        return doc_state
    
    
def handle_url_load(app):
    @app.callback(
        [
            Output("document-list", "children", allow_duplicate=True),
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Output("document-selector", "options", allow_duplicate=True),
            Output("upload-error", "children", allow_duplicate=True),
            Output("upload-error", "className", allow_duplicate=True),
            Output("url", "pathname", allow_duplicate=True),
            Output("expanded-upload-area", "style", allow_duplicate=True),
            Output("upload-area-is-open", "data", allow_duplicate=True)
        ],
        Input("url-load-button", "n_clicks"),
        [
            State("url-input", "value"),
            State("document-list", "children"),
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("document-selector", "options"),
        ],
        prevent_initial_call=True
    )
    def load_from_urls(n_clicks, urls, doc_list, doc_state, vstore_state, chunk_state, 
                      chat_history, current_options):
        if not n_clicks or not urls:
            raise PreventUpdate
        
        if urls not in [ '/', '']:
            raise PreventUpdate
            
        try:
            doc_state = doc_state or {}
            vstore_state = vstore_state or {}
            chunk_state = chunk_state or {}
            chat_history = chat_history or []
            current_options = current_options or []
            
            urls = [url.strip() for url in urls.split('\n') if url.strip()]
            final_doc_list = doc_list.copy() if doc_list else []
            
            def get_filename_from_cd(cd):
                """Extract filename from content-disposition header"""
                if not cd:
                    return None
                fname = re.findall('filename=(.+)', cd)
                if len(fname) == 0:
                    return None
                return fname[0].strip('"')
            
            def extract_redirect_url(html_content):
                """Extract redirect URL from HTML content"""
                try:
                    match = re.search(r'redirectUrl=\'(.*?)\'', html_content)
                    if match:
                        return match.group(1)
                    match = re.search(r'url=(.*?)"', html_content)
                    if match:
                        return match.group(1)
                except:
                    return None
                return None
            
            for url in urls:
                try:
                    # First request with allow_redirects=True
                    session = requests.Session()
                    response = session.get(url, timeout=30, allow_redirects=True)
                    response.raise_for_status()
                    
                    content_type = response.headers.get('content-type', '').lower()
                    
                    # If we got HTML and it contains a redirect
                    if 'text/html' in content_type:
                        redirect_url = extract_redirect_url(response.text)
                        if redirect_url:
                            logger.info(f"Callbacks.py - load_from_urls : Following redirect to: {redirect_url}")
                            response = session.get(redirect_url, timeout=30, allow_redirects=True)
                            response.raise_for_status()
                            content_type = response.headers.get('content-type', '').lower()
                    
                    # Check if we finally got a PDF
                    if 'application/pdf' not in content_type:
                        raise ValueError(f"Callbacks.py - load_from_urls : URL does not point to a PDF file: {url}")
                    
                    content = response.content
                    if len(content) > MAX_DOC_SIZE:
                        raise ValueError(f"Callbacks.py - load_from_urls : File too large (max 50MB)")
                    
                    # Get filename
                    filename = get_filename_from_cd(response.headers.get('content-disposition'))
                    if not filename:
                        filename = os.path.basename(urlparse(response.url).path)
                    if not filename or not filename.lower().endswith('.pdf'):
                        filename = f"document_{len(doc_state)}.pdf"
                    
                    # Convert to base64
                    content_b64 = f"data:application/pdf;base64,{base64.b64encode(content).decode()}"
                    
                    # Process document
                    session_id, chunk_mapping = vect_serv.create_vectorstore_and_mapping(content_b64, filename)
                    vect_serv.clear_all_caches()
                    # Store document
                    doc_state[session_id] = {
                        "filename": filename,
                        "content": content_b64,
                        "source": "url",
                        "url": url,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    vstore_state[session_id] = session_id
                    chunk_state[session_id] = json.dumps(chunk_mapping)
                    
                    final_doc_list.append(layout.create_document_item(filename, session_id))
                    chat_history.append(create_system_notification(
                        f"Added document from URL: {filename}",
                        type="success",
                        action="add"
                    ))
                    
                except Exception as e:
                    error_msg = f"Callbacks.py - handle_url_load - load_from_urls ~1: Error processing URL {url}: {str(e)}"
                    logger.error(error_msg)
                    chat_history.append(create_system_notification(
                        error_msg,
                        type="error"
                    ))
            
            selector_options = [
                {"label": info["filename"], "value": session_id}
                for session_id, info in doc_state.items()
            ]
            
            return (
                final_doc_list,
                doc_state,
                vstore_state,
                chunk_state,
                chat_history,
                selector_options,
                "",
                "upload-error d-none",
                "",
                {"display": "none"},
                False
            )
            
        except Exception as e:
            error_msg = f"Callbacks.py - handle_url_load - load_from_urls ~2: Error loading URLs: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return (
                doc_list,
                doc_state,
                vstore_state,
                chunk_state,
                chat_history,
                current_options,
                error_msg,
                "upload-error",
                urls,
                {"display": "block"},
                True
            )
        

def handle_folder_load(app):
    @app.callback(
        [
            Output("document-list", "children", allow_duplicate=True),
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Output("document-selector", "options", allow_duplicate=True),
            Output("upload-error", "children", allow_duplicate=True),
            Output("upload-error", "className", allow_duplicate=True),
            Output("folder-path-input", "value", allow_duplicate=True),
            Output("expanded-upload-area", "style", allow_duplicate=True),
            Output("upload-area-is-open", "data", allow_duplicate=True)
        ],
        Input("folder-load-button", "n_clicks"),
        [
            State("folder-path-input", "value"),
            State("document-list", "children"),
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("document-selector", "options"),
            State("url", "pathname")
        ],
        prevent_initial_call=True
    )
    def load_from_folder(n_clicks, folder_path, doc_list, doc_state, vstore_state, chunk_state, 
                        chat_history, current_options, urls):
        if not n_clicks or not folder_path:
            raise PreventUpdate
        
        if urls not in [ '/', '']:
            raise PreventUpdate
        try:
            # Initialize states
            doc_state = doc_state or {}
            vstore_state = vstore_state or {}
            chunk_state = chunk_state or {}
            chat_history = chat_history or []
            current_options = current_options or []
            
            folder_path = folder_path.strip().replace('\\', '/')
            folder = Path(folder_path)
            
            if not folder.exists():
                raise ValueError(f"Callbacks.py - handle_folder_load - load_from_folder : Folder not found: {folder_path}")
            if not folder.is_dir():
                raise ValueError(f"Callbacks.py - handle_folder_load - load_from_folder : Not a directory: {folder_path}")
            
            supported_extensions = {'.pdf', '.txt', '.md', '.docx'}
            files_processed = 0
            
            new_doc_list = doc_list.copy() if doc_list else []
            
            for file_path in folder.glob('**/*'):
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            
                        filename = file_path.name
                        
                        # Convert to base64 for processing
                        if file_path.suffix.lower() == '.pdf':
                            mime_type = "application/pdf"
                            content_b64 = f"data:{mime_type};base64,{base64.b64encode(content).decode()}"
                        else:
                            content_b64 = content.decode('utf-8')
                        
                        # Process document
                        session_id, chunk_mapping = vect_serv.create_vectorstore_and_mapping(content_b64, filename)
                        
                        # Store document information
                        doc_state[session_id] = {
                            "filename": filename,
                            "content": content_b64,
                            "source": "folder",
                            "path": str(file_path),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        vstore_state[session_id] = session_id
                        chunk_state[session_id] = json.dumps(chunk_mapping)
                        
                        # Add new document item
                        new_doc_item = layout.create_document_item(filename, session_id)
                        new_doc_list.append(new_doc_item)
                        
                        chat_history.append(create_system_notification(f"Added document: {filename}", type="success", action="add"))
                        files_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Callbacks.py - handle_folder_load - load_from_folder ~1: Error processing file {file_path}: {str(e)}")
                        chat_history.append(create_system_notification(f"Error: {str(e)}", type="error"))
            
            if files_processed == 0:
                raise ValueError(f"Callbacks.py - handle_folder_load - load_from_folder : No supported files found in {folder_path}")
            
            # Update selector options
            selector_options = [
                {"label": info["filename"], "value": session_id}
                for session_id, info in doc_state.items()
            ]
            
            return (
                new_doc_list,
                doc_state,
                vstore_state,
                chunk_state,
                chat_history,
                selector_options,
                "",
                "upload-error d-none",
                "",
                {"display": "none"},
                False
            )
            
        except Exception as e:
            error_msg = f"Callbacks.py - handle_folder_load - load_from_folder ~2: Error loading from folder: {str(e)}"
            logger.error(ferror_msg)
            return (
                doc_list,
                doc_state,
                vstore_state,
                chunk_state,
                chat_history,
                current_options,
                error_msg,
                "upload-error",
                folder_path,
                {"display": "block"},
                True
            )

def handle_document_removal(app):
    @app.callback(
        [
            Output("document-list", "children", allow_duplicate=True),
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Output("document-selector", "options", allow_duplicate=True),
            Output("document-selector", "value", allow_duplicate=True),
            Output("document-viewer-container", "children", allow_duplicate=True),
            Output("document-data", "data", allow_duplicate=True),
            Output("pdf-highlights", "data", allow_duplicate=True),
            Output("upload-trigger", "data", allow_duplicate=True),  # Add trigger for upload reset
        ],
        [
            Input({"type": "remove-document", "index": ALL}, "n_clicks"),
            Input({"type": "remove-group", "index": ALL}, "n_clicks"),  # Add explicit group removal input
        ],
        [
            State("document-list", "children"),
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("document-selector", "options"),
            State("document-selector", "value"),
            State("url", "pathname"),
            State("auth-state", "data"),
            State("delete-group-id", "data")
        ],
        prevent_initial_call=True
    )
    def remove_document(doc_n_clicks, group_n_clicks, doc_list, doc_state, vstore_state, chunk_state, 
                       chat_history, current_options, current_selected, urls, auth_state, delete_group_id):
        if (not any(doc_n_clicks) and not any(group_n_clicks)) or not doc_state:
            raise PreventUpdate
        if urls not in [ '/', '']:
            raise PreventUpdate
            
        triggered = ctx.triggered_id
        logger.info(f"Document removal triggered by: {triggered}")
        
        # Handle specific document removal
        if isinstance(triggered, dict) and triggered.get('type') == 'remove-document':
            session_id = triggered.get('index', None)
            
            if session_id and session_id in doc_state:
                try:
                    filename = doc_state[session_id]["filename"]
                    
                    # Check if this is a persistent document
                    is_persistent = doc_state[session_id].get('source') in ['folder', 'group'] or 'file_path' in doc_state[session_id]
                    
                    if is_persistent:
                        # For persistent documents, only remove from current session but not from storage
                        logger.info(f"Removing persistent document {filename} from current session only")
                        chat_history.append(create_system_notification(
                            f"Removed {filename} from current session. The document remains stored in the group folder.",
                            type="info",
                            action="remove"
                        ))
                        
                        # Remove from current states only
                        doc_state.pop(session_id, None)
                        vstore_state.pop(session_id, None)
                        chunk_state.pop(session_id, None)
                    else:
                        # For non-persistent documents, clean up completely
                        current_user_id = auth_state.get('user_id') if auth_state else None
                        doc_user_id = doc_state[session_id].get('user_id', current_user_id)
                        
                        # Create a database session
                        from sqlalchemy.orm import sessionmaker
                        from auth.config import AUTH_CONFIG
                        from auth.models import Base
                        from sqlalchemy import create_engine
                        engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
                        DBSession = sessionmaker(bind=engine)
                        db_session = DBSession()
                        
                        # Clean up the vector store
                        vect_serv = VectorStoreService()
                        vect_serv.cleanup_vectorstore(session_id, user_id=doc_user_id, db_session=db_session)
                        
                        # Remove from all states
                        doc_state.pop(session_id, None)
                        vstore_state.pop(session_id, None)
                        chunk_state.pop(session_id, None)
                        
                        # Notify user
                        chat_history.append(create_system_notification(
                            f"Removed document: {filename}",
                            type="success",
                            action="remove"
                        ))
                except Exception as e:
                    logger.error(f"Error removing document: {str(e)}")
                    traceback.print_exc()
                    chat_history.append(create_system_notification(
                        f"Error removing document: {str(e)}",
                        type="error"
                    ))
                    return dash.no_update
                    
        # Handle group removal
        elif isinstance(triggered, dict) and triggered.get('type') == 'remove-group':
            group_name = triggered.get('index', None)
            
            if group_name:
                try:
                    # Find all documents in this group
                    session_ids_to_remove = []
                    for session_id, info in doc_state.items():
                        if info.get('group_name') == group_name:
                            session_ids_to_remove.append(session_id)
                    
                    if not session_ids_to_remove:
                        logger.info(f"No documents found for group {group_name}")
                        raise PreventUpdate
                    
                    # Remove all documents in this group
                    for session_id in session_ids_to_remove:
                        filename = doc_state[session_id].get("filename", "Unknown")
                        
                        # Only remove from current session states
                        doc_state.pop(session_id, None)
                        vstore_state.pop(session_id, None)
                        chunk_state.pop(session_id, None)
                    
                    # Add notification
                    chat_history.append(create_system_notification(
                        f"Removed group: {group_name} ({len(session_ids_to_remove)} documents)",
                        type="info",
                        action="remove"
                    ))
                except Exception as e:
                    logger.error(f"Error removing group: {str(e)}")
                    traceback.print_exc()
                    chat_history.append(create_system_notification(
                        f"Error removing group: {str(e)}",
                        type="error"
                    ))
                    return dash.no_update
                    
        # Force garbage collection
        import gc
        gc.collect()
                
        # Create new document list - using layout.create_document_item with proper grouping
        final_doc_list = []
        
        # Group documents by their group name
        grouped_docs = {}
        for session_id, info in doc_state.items():
            if not isinstance(info, dict) or not info.get('filename'):
                continue
                
            is_persistent = info.get('source') in ['folder', 'group'] or 'path' in info
            group_name = info.get('group_name', 'Uploaded')
            
            if group_name not in grouped_docs:
                grouped_docs[group_name] = []
                
            grouped_docs[group_name].append({
                'session_id': session_id,
                'info': info,
                'is_persistent': is_persistent
            })
        
        # Process groups and documents
        for group_name, group_docs in grouped_docs.items():
            # Separate persistent and non-persistent documents
            non_persistent_docs = [doc for doc in group_docs if not doc['is_persistent']]
            persistent_docs = [doc for doc in group_docs if doc['is_persistent']]
            
            # Add group header for persistent groups
            if persistent_docs and not non_persistent_docs:
                group_header = html.Div([
                    html.I(className="fas fa-folder me-2"),
                    html.Span(group_name, className="fw-bold"),
                    html.Button(
                        html.I(className="fas fa-trash"),
                        id={
                            'type': 'remove-group',
                            'index': group_name
                        },
                        className="btn btn-link text-danger p-0 ms-2",
                        title=f"Remove all documents from {group_name}"
                    )
                ], className="mb-2 text-muted")
                final_doc_list.append(group_header)
            
            # Add all documents in this group
            for doc in group_docs:
                doc_item = layout.create_document_item(
                    doc['info']['filename'], 
                    doc['session_id'], 
                    is_persistent=doc['is_persistent'],
                    group_name=group_name if doc['is_persistent'] else None
                )
                final_doc_list.append(doc_item)
        
        # If no documents at all, show a message
        if not final_doc_list and not doc_state:
            final_doc_list = [
                html.Div(
                    "No documents uploaded. Click 'Add Documents' to get started.",
                    className="text-center text-muted py-4"
                )
            ]
        
        # Update selector options - IMPORTANT: This is the key fix
        selector_options = [
            {"label": info["filename"], "value": sid}
            for sid, info in doc_state.items()
        ]
        
        # Check if current selected document was removed
        new_selected = current_selected if current_selected in doc_state else None
        
        # Reset viewer if the selected document was removed
        empty_viewer = layout.create_pdf_viewer()
        
        # Generate new trigger value
        trigger_value = datetime.now().isoformat()
        
        logger.info(f"Updated document selector with {len(selector_options)} options")
        
        return (
            final_doc_list,       # document list
            doc_state,            # document state
            vstore_state,         # vectorstore state
            chunk_state,          # chunk mapping state
            chat_history,         # chat history
            selector_options,     # selector options - NOW PROPERLY UPDATED
            new_selected,         # selector value
            empty_viewer,         # viewer
            None,                 # document data
            None,                 # highlights
            trigger_value,        # upload reset trigger
        )

def register_upload_handlers(app):
    # Reset upload component when triggered
    @app.callback(
        [
            Output("upload-document", "contents", allow_duplicate=True),
            Output("upload-document", "filename", allow_duplicate=True)
        ],
        Input("upload-trigger", "data"),
        prevent_initial_call=True
    )
    def reset_upload(trigger_value):
        """Reset upload component when triggered"""
        if trigger_value:
            return None, None
        raise PreventUpdate
        
def register_toggle_callback(app):
    @app.callback(
        [
            Output("expanded-upload-area", "style", allow_duplicate=True),
            Output("upload-area-is-open", "data", allow_duplicate=True)
        ],
        [
            Input("upload-toggle-button", "n_clicks"),
            Input("upload-document", "filename"),
            Input("url-load-button", "n_clicks"),
            Input("folder-load-button", "n_clicks")
        ],
        [
            State("upload-area-is-open", "data"),
            State("expanded-upload-area", "style")
        ],
        prevent_initial_call=True
    )
    def toggle_upload_area(n_clicks, selected_files, url_clicks, folder_clicks, 
                          is_open, current_style):
        triggered_id = ctx.triggered_id
        
        if (triggered_id == "upload-document" and selected_files) or \
           (triggered_id in ["url-load-button", "folder-load-button"] and \
            (url_clicks or folder_clicks)):
            return {"display": "none"}, False
            
        elif triggered_id == "upload-toggle-button":
            is_open = not is_open
            style = {
                "display": "block" if is_open else "none",
                "backgroundColor": "white",
                "border": "1px solid #dee2e6",
                "borderRadius": "8px",
                "padding": "1rem",
                "marginTop": "1rem"
            }
            return style, is_open
            
        return current_style or {"display": "none"}, is_open

    @app.callback(
        [
            Output("upload-tab-content", "style"),
            Output("url-tab-content", "style"),
            Output("folder-tab-content", "style"),
            Output("server-tab-content", "style"),
            Output("tab-upload-link", "active"),
            Output("tab-url-link", "active"),
            Output("tab-folder-link", "active"),
            Output("tab-server-link", "active"),
        ],
        [
            Input("tab-upload-link", "n_clicks"),
            Input("tab-url-link", "n_clicks"),
            Input("tab-folder-link", "n_clicks"),
            Input("tab-server-link", "n_clicks"),
        ],
        prevent_initial_call=True
    )
    def switch_tab(upload_clicks, url_clicks, folder_clicks, server_clicks):
        triggered_id = ctx.triggered_id
        logger.info(f"Tab switch triggered by: {triggered_id}")

        hide_style = {"display": "none"}
        show_style = {"display": "block"}
        
        if triggered_id == "tab-upload-link":
            return show_style, hide_style, hide_style, hide_style, True, False, False, False
        elif triggered_id == "tab-url-link":
            return hide_style, show_style, hide_style, hide_style, False, True, False, False
        elif triggered_id == "tab-folder-link":
            return hide_style, hide_style, show_style, hide_style, False, False, True, False
        elif triggered_id == "tab-server-link":
            return hide_style, hide_style, hide_style, show_style, False, False, False, True
            
        # Default to upload tab
        return show_style, hide_style, hide_style, True, False, False, False

    # Add a callback to ensure tab content visibility
    @app.callback(
        Output("upload-tabs", "children"),
        [Input("upload-tabs", "active_tab")],
        prevent_initial_call=True
    )
    def update_tab_content(active_tab):
        logger.info(f"Tab changed to: {active_tab}")  # Debug print
        return dash.no_update  # Just refresh the tabs

    @app.callback(
        Output("upload-status", "data"),
        [Input("upload-document", "contents")],
        [State("upload-status", "data")]
    )
    def start_progress_tracking(contents, current_status):
        if not contents:
            return {"progress": 0, "status": ""}
            
        return {"progress": 0, "status": "Starting upload..."}


def register_progress(app):
    @app.callback(
        [
            Output("upload-progress-bar", "value", allow_duplicate=True),
            Output("upload-progress-bar", "style", allow_duplicate=True),
            Output("progress-status", "children",allow_duplicate=True),
            Output("progress-status", "style",allow_duplicate=True),
            Output("progress-interval", "disabled", allow_duplicate=True)
        ],
        [
            Input("upload-document", "contents"),
            Input("progress-interval", "n_intervals"),
            Input("upload-status", "data")
        ],
        [State("upload-document", "filename")]
    )
    def update_progress(contents, n_intervals, status_data, filenames):
        if not contents or not status_data:
            return 0, {"display": "none"}, "", {"display": "none"}, True

        # Get the triggered input
        triggered = ctx.triggered_id

        if triggered == "upload-document":
            # Start of upload
            return (
                0,
                {"display": "block", "height": "4px", "width": "100%"},
                "Starting upload...",
                {"display": "block"},
                False
            )

        progress = status_data.get("progress", 0)
        status_text = status_data.get("status", "")

        if progress >= 100:
            return (
                100,
                {"display": "block", "height": "4px", "width": "100%"},
                "Processing uploaded file/s complete!",
                {"display": "block"},
                True
            )

        # During upload/processing
        phases = [
            (0, 20, "Uploading files..."),
            (20, 40, "Parsing documents..."),
            (40, 70, "Creating chunks..."),
            (70, 90, "Building vector store..."),
            (90, 100, "Finalizing...")
        ]

        # Find current phase
        current_phase = None
        for start, end, text in phases:
            if start <= progress < end:
                current_phase = (start, end, text)
                break

        if current_phase:
            status_text = current_phase[2]

        return (
            progress,
            {"display": "block", "height": "4px", "width": "100%"},
            status_text,
            {"display": "block"},
            False
        )

def register_callbacks(app):
    
    @app.callback(
        [
            Output("document-list", "children", allow_duplicate=True),
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Output("document-selector", "options", allow_duplicate=True),
            Output("upload-error", "children", allow_duplicate=True),
            Output("upload-error", "style", allow_duplicate=True),
            Output("expanded-upload-area", "style", allow_duplicate=True),
            Output("upload-area-is-open", "data", allow_duplicate=True),
            Output("upload-status", "data", allow_duplicate=True) 
        ],
        [Input("upload-document", "contents")],  
        [
            State("upload-document", "filename"),
            State("document-list", "children"),
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("document-selector", "options"),
            State("url", "pathname")
        ],
        prevent_initial_call=True
    )
    def handle_upload(contents, filenames, doc_list, doc_state, vstore_state, chunk_state, 
                    chat_history, current_options, urls):
        
        if not contents or not filenames:
            raise PreventUpdate
        if urls not in [ '/', '']:
            raise PreventUpdate
        logger.info(f"Callbacks.py - register_callbacks - handle_upload : Upload callback triggered for files: {filenames}")  # Debug log
        
        closed_style = { "className": "upload-area-closed" }
        status_data = {"progress": 0, "status": "Starting..."}

        try:
            # Initialize states
            doc_state = doc_state or {}
            vstore_state = vstore_state or {}
            chunk_state = chunk_state or {}
            chat_history = chat_history or []
            current_options = current_options or []
            error_msg = ""
            error_style = {"display": "none"}
            
            # Ensure contents and filenames are lists
            if not isinstance(contents, list):
                contents = [contents]
                filenames = [filenames]
            
            new_doc_list = doc_list.copy() if doc_list else []
            
            # Process each file
            for i, (content, filename) in enumerate(zip(contents, filenames)):
                try:
                    logger.info(f"Callbacks.py - register_callbacks - handle_upload : Processing file: {filename}")  # Debug log
                    
                    # Update progress
                    progress = 20 + (i / len(contents)) * 20
                    status_data = {"progress": progress, "status": f"Processing {filename}..."}
                    
                    # Check and remove existing file sessions
                    existing_session = None
                    for session_id, info in doc_state.items():
                        if info.get("filename") == filename:
                            logger.info(f"Callbacks.py - register_callbacks - handle_upload : Found existing session for {filename}")  # Debug log
                            existing_session = session_id
                            break
                    
                    if existing_session:
                        logger.info(f"Callbacks.py - register_callbacks - handle_upload : Cleaning up existing session: {existing_session}")  # Debug log
                        vect_serv.cleanup_vectorstore(existing_session)
                        doc_state.pop(existing_session, None)
                        vstore_state.pop(existing_session, None)
                        chunk_state.pop(existing_session, None)
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                    
                    logger.info(f"Creating new vectorstore for {filename}")  # Debug log
                    
                    # Process the document
                    if filename.lower().endswith('.pdf'):
                        session_id, chunk_mapping = vect_serv.create_vectorstore_and_mapping(content, filename)
                        doc_content = content
                    else:
                        _, _, _, plain_text = parse_contents(content, filename)
                        session_id, chunk_mapping = vect_serv.create_vectorstore_and_mapping(plain_text, filename)
                        doc_content = plain_text
                    
                    logger.info(f"Callbacks.py - register_callbacks - handle_upload : Created new session: {session_id}")  # Debug log
                    
                    # Update progress
                    progress = 40 + (i / len(contents)) * 30
                    status_data = {"progress": progress, "status": "Creating chunks..."}

                    # Update states
                    doc_state[session_id] = {
                        "filename": filename,
                        "content": doc_content,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    vstore_state[session_id] = session_id
                    chunk_state[session_id] = json.dumps(chunk_mapping)
                    
                    # Update UI
                    new_doc_item = layout.create_document_item(filename, session_id)
                    new_doc_list.append(new_doc_item)
                    chat_history.append(create_system_notification(
                        f"Added document: {filename}", 
                        type="success", 
                        action="add"
                    ))
                    
                    # Update progress
                    progress = 70 + (i / len(contents)) * 20
                    status_data = {"progress": progress, "status": "Building vector store..."}
                    
                except Exception as e:
                    logger.error(f"Callbacks.py - register_callbacks - handle_upload ~1: Error processing {filename}: {str(e)}")  # Debug log
                    traceback.print_exc()
                    error_msg = f"Error processing {filename}: {str(e)}"
                    error_style = {"display": "block", "color": "red"}
                    chat_history.append(create_system_notification(
                        f"Error: {str(e)}", 
                        type="error"
                    ))
            
            # Final progress update
            status_data = {"progress": 100, "status": "Processing complete!"}
            
            # Update selector options
            selector_options = [
                {"label": info["filename"], "value": session_id}
                for session_id, info in doc_state.items()
            ]
            
            logger.info("Callbacks.py - register_callbacks - handle_upload : Upload processing completed successfully")  # Debug log
            
            return (
                new_doc_list,
                doc_state,
                vstore_state,
                chunk_state,
                chat_history,
                selector_options,
                error_msg,
                error_style,
                {"display": "none"},
                False,
                status_data
            )
            
        except Exception as e:
            logger.error(f"Callbacks.py - register_callbacks - handle_upload ~2: Error in upload handling: {str(e)}")  # Debug log
            traceback.print_exc()
            error_msg = f"Error: {str(e)}"
            error_style = {"display": "block", "color": "red"}
            return (
                doc_list,
                doc_state,
                vstore_state,
                chunk_state,
                chat_history,
                current_options,
                error_msg,
                error_style,
                {"display": "block"},
                True,
                status_data
            )    
    @app.callback(
        [
            Output("doc-selector-container", "style", allow_duplicate=True),
            Output("doc-selector-dropdown", "options", allow_duplicate=True),
            Output("doc-selector-dropdown", "value", allow_duplicate=True),
            Output("query-input", "value", allow_duplicate=True)
        ],
        [
            Input("query-input", "value"),
            Input("doc-selector-dropdown", "value"),
        ],
        [
            State("document-state", "data"),
            State("url-input", "value")
        ],
        prevent_initial_call=True
    )

    def handle_doc_selector(query, selected_doc, doc_state, urls):
        if urls in [ '/documents', '/admin']:
            raise PreventUpdate
        triggered = ctx.triggered_id
        base_style = {
            "position": "absolute",
            "width": "300px",
            "backgroundColor": "white",
            "border": "1px solid #dee2e6",
            "borderRadius": "4px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "zIndex": "1000",
            "marginTop": "2px"
        }
        
        # If a document was selected from dropdown
        if triggered == "doc-selector-dropdown" and selected_doc:
            base_style["display"] = "none"
            doc_name = doc_state[selected_doc]["filename"]
            return base_style, [], None, f"@{doc_name} "
        
        # Show dropdown when @ is typed
        if query and query.startswith("@"):
            base_style["display"] = "block"
            options = [
                {"label": info["filename"], "value": session_id}
                for session_id, info in doc_state.items()
            ] if doc_state else []
            return base_style, options, None, dash.no_update
        
        # Hide dropdown for all other cases
        base_style["display"] = "none"
        return base_style, [], None, dash.no_update



    @app.callback(
        [
            Output("chat-history", "children", allow_duplicate=True),
            Output("query-input", "value", allow_duplicate=True),
            Output("document-state", "data", allow_duplicate=True),
            Output("document-selector", "value",allow_duplicate=True),
            Output("query-processing-status", "data", allow_duplicate=True) 
        ],
        [Input("submit-btn", "n_clicks"), Input("query-input", "n_submit")],
        [
            State("query-input", "value"),
            State("chat-history", "children"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("document-state", "data"),
            State("url", "pathname"),
            State("auth-state", "data"),
        ],
        prevent_initial_call=True
    )
    # Modified handle_query function in callbacks.py

    def handle_query(n_clicks, n_submit, query, chat_history, vectorstore_state, chunk_mapping_state, doc_state, urls, auth_state):
        if not query or (not n_clicks and not n_submit):
            raise PreventUpdate
        if urls not in [ '/', '']:
            raise PreventUpdate
        if query and query.endswith('\n'):
            return chat_history, query.rstrip('\n'), doc_state, dash.no_update, dash.no_update             
        try:
            # Helper function to parse chunk information from LLM response
            def parse_llm_chunk_info(chunk_str):
                """
                Parse chunk information from the LLM response format
                Example: 'chunk_0 from RecommendationLetterAngelica.pdf'
                Returns: (chunk_id, document_name)
                """
                try:
                    # Handle possible quoted filenames
                    if ' from ' in chunk_str:
                        parts = chunk_str.split(' from ', 1)
                        chunk_id = parts[0].strip()
                        document_name = parts[1].strip()
                        
                        # Handle quoted filenames: remove quotes if present
                        if document_name.startswith('"') and document_name.endswith('"'):
                            document_name = document_name[1:-1]
                        elif document_name.startswith("'") and document_name.endswith("'"):
                            document_name = document_name[1:-1]
                        
                        # Clean up document name - remove any trailing characters
                        document_name = re.sub(r'[,\]\)\s]+$', '', document_name)
                        
                        # If chunk_id starts with "chunk_" but has no number, try to extract it
                        if chunk_id.startswith('chunk_') and not chunk_id[6:].isdigit():
                            # Try to find a digit elsewhere in the string
                            digit_match = re.search(r'(\d+)', chunk_str)
                            if digit_match:
                                chunk_id = f"chunk_{digit_match.group(1)}"
                        
                        return chunk_id, document_name
                        
                    # Alternative formats that might be used
                    # Check for "in" or "of" as alternatives to "from"
                    for separator in [' in ', ' of ']:
                        if separator in chunk_str:
                            parts = chunk_str.split(separator, 1)
                            chunk_id = parts[0].strip()
                            document_name = parts[1].strip()
                            
                            # Handle quoted filenames
                            if document_name.startswith('"') and document_name.endswith('"'):
                                document_name = document_name[1:-1]
                            elif document_name.startswith("'") and document_name.endswith("'"):
                                document_name = document_name[1:-1]
                                
                            document_name = re.sub(r'[,\]\)\s]+$', '', document_name)
                            return chunk_id, document_name
                    
                    # If we couldn't find a document name, check if there's a digit we can use as chunk_id
                    digit_match = re.search(r'(?<!\d)(\d+)(?!\d)', chunk_str)
                    if digit_match:
                        chunk_id = f"chunk_{digit_match.group(1)}"
                        return chunk_id, None
                        
                    app.clientside_callback(
                        """
                        function() {
                            setTimeout(function() {
                                var chatHistory = document.getElementById('chat-history');
                                if (chatHistory) {
                                    chatHistory.scrollTop = chatHistory.scrollHeight;
                                }
                            }, 100);
                            return window.dash_clientside.no_update;
                        }
                        """,
                        Output('chat-history', 'data-scroll', allow_duplicate=True),
                        Input('scroll-to-bottom', 'key'),
                        prevent_initial_call=True
                    )
                    # If no document name or chunk number could be found, just return the chunk ID
                    chat_history.append(html.Div(id="scroll-to-bottom", key=str(datetime.now().timestamp()), style={"display": "none"}))
                    return chunk_str.strip(), None
                
                except Exception as e:
                    logger.error(f"Callbacks.py - handle_query - handle_upload ~1: Error parsing chunk info '{chunk_str}': {str(e)}")
                    return chunk_str.strip(), None
                    
            # Function to determine document relevance
            def get_document_relevance(doc_name, used_chunk_indices, doc_references, doc_state, most_relevant_doc_id):
                """
                Determine document relevance score based on usage frequency, selection state and match quality
                
                Args:
                    doc_name: Name of the document
                    used_chunk_indices: List of chunk indices used
                    doc_references: List of document references
                    doc_state: Document state dictionary
                    most_relevant_doc_id: ID of the most relevant document
                    
                Returns:
                    float: Relevance score (higher is more relevant)
                """
                # Base relevance is the frequency of document citation
                relevance = doc_references.count(doc_name)
                
                # Give priority to documents with highlights
                for session_id, info in doc_state.items():
                    if info.get("filename") == doc_name and info.get("highlights"):
                        relevance += 5
                        break
                        
                # Give highest priority to selected document
                if most_relevant_doc_id and doc_name == doc_state.get(most_relevant_doc_id, {}).get("filename"):
                    relevance += 10
                    
                return relevance
        
            status_data = {"progress": 0, "status": "Starting query processing..."}
            logger.info("Callbacks.py - register_callbacks - handle_query : Starting query processing...")
            chat_history = chat_history or []
            doc_specific = False
            selected_doc_id = None
            
            # Update progress: Query analysis
            status_data = {"progress": 25, "status": "Analyzing query..."}
            # Check if this is a document-specific query
            if query.startswith("@"):
                parts = query.split(" ", 1)
                if len(parts) > 1:
                    doc_name = parts[0][1:]  # Remove @ symbol
                    query = parts[1]  # Get actual query
                    doc_specific = True
                    logger.info(f"Callbacks.py - register_callbacks - handle_query : Document-specific query for: {doc_name}")
                    
                    # Find document ID by name
                    for session_id, info in doc_state.items():
                        if info["filename"] == doc_name:
                            selected_doc_id = session_id
                            break
            
#            chat_history.append(html.Hr(style={"margin": "20px 0"}))

            # Update progress: Document search
            status_data = {"progress": 50, "status": "Searching through documents..."}
            if not vectorstore_state:
                raise ValueError("Callbacks.py - register_callbacks - handle_query : Please upload documents first.")
            
            # Clear previous highlights
            for session_id in doc_state:
                if "highlights" in doc_state[session_id]:
                    doc_state[session_id]["highlights"] = None

            # Create a database session
            from sqlalchemy.orm import sessionmaker
            from auth.models import Base
            from sqlalchemy import create_engine
            from auth.config import AUTH_CONFIG 
            engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
            DBSession = sessionmaker(bind=engine)
            db_session = DBSession()


            # Get user info for loading persistent data
            current_user_id = auth_state.get('user_id')
            if not current_user_id:
                raise PreventUpdate
            
            from app.storage_config import get_group_for_user
            user_group = get_group_for_user(current_user_id, db_session)
            logger.info(f"User {current_user_id} belongs to group: {user_group}")
            llm_serv = LLMServiceFactory.create_llm_service_for_group(user_group)

            retrieval_settings = get_retrieval_settings(group_name = user_group)
            chunks_per_doc = retrieval_settings["chunks_per_doc"]
            max_total_chunks = retrieval_settings["max_total_chunks"]
            similarity_threshold = retrieval_settings["similarity_threshold"]
            doc_lang = retrieval_settings["lang"]

            logger.info(f"Callbacks.py - register_callbacks - handle_query : Using retrieval settings: chunks_per_doc={chunks_per_doc}, " +
                f"max_total_chunks={max_total_chunks}, " +
                f"similarity_threshold={similarity_threshold}")

            # Initialize collections to store chunks from all documents
            chunk_index_map = {}
            all_chunk_ids = []
            all_context_parts = []
            all_scores = []
            all_filenames = []
            combined_chunk_mapping = {}
            current_chunk_index = 1
            
            # Process only selected document or all documents
            if doc_specific and selected_doc_id:
                doc_ids = [selected_doc_id]
            else:
                doc_ids = list(vectorstore_state.keys())

            logger.info(f"Callbacks.py - register_callbacks - handle_query : Processing documents: {doc_ids}")
        

            # Keep track of document sources for each chunk

            for session_id in doc_ids:
                if session_id in vectorstore_state:
                    logger.info(f"Callbacks.py - register_callbacks - handle_query : Processing document {session_id}")
                    
                    # Check if this is a persistent document
                    is_persistent = False
                    filename = None
                    if session_id in doc_state:
                        is_persistent = doc_state[session_id].get('source') in ['folder', 'group'] or 'file_path' in doc_state[session_id]
                        filename = doc_state[session_id].get('filename')
                    
                    # For persistent documents, use filename as the lookup key
                    lookup_id = filename if is_persistent and filename else session_id
                    logger.info(f"Callbacks.py - register_callbacks - handle_query : Using lookup ID: {lookup_id} for vector store loading")
                    
                    vectorstore, metadata = vect_serv.load_vectorstore(lookup_id)
                    
                    if vectorstore:
                        try:
                            logger.info("Callbacks.py - register_callbacks - handle_query : Loading chunk mapping...")
                            chunk_mapping = None
                            
                            if is_persistent and filename:
                                # For persistent docs, look for chunk mapping with filename
                                try:
                                    from app.storage_config import CHUNK_MAPS_DIR, get_group_for_user
                                    group_name = get_group_for_user(current_user_id, db_session)
                                    
                                    # Try different naming patterns for chunk map file
                                    chunk_file_paths = [
                                        CHUNK_MAPS_DIR / str(group_name) / f"{filename}_chunks.json",
                                        CHUNK_MAPS_DIR / str(group_name) / f"{session_id}_chunks.json",
                                        CHUNK_MAPS_DIR / str(group_name) / filename,
                                        CHUNK_MAPS_DIR / str(group_name) / session_id
                                    ]
                                    
                                    for chunk_file_path in chunk_file_paths:
                                        if chunk_file_path.exists():
                                            with open(chunk_file_path, 'r') as f:
                                                chunk_mapping = json.load(f)
                                                logger.info(f"Callbacks.py - register_callbacks - handle_query : Loaded persistent chunk mapping from {chunk_file_path}")
                                                break
                                except Exception as e:
                                    logger.error(f"Callbacks.py - register_callbacks - handle_query ~2: Error loading persistent chunk mapping: {str(e)}")
                            
                            # Fallback to memory chunk mapping if needed
                            if not chunk_mapping and session_id in chunk_mapping_state:
                                try:
                                    if isinstance(chunk_mapping_state[session_id], str):
                                        chunk_mapping = json.loads(chunk_mapping_state[session_id])
                                    else:
                                        chunk_mapping = chunk_mapping_state[session_id]
                                except Exception as e:
                                    logger.error(f"Callbacks.py - register_callbacks - handle_query ~3: Error parsing chunk mapping from state: {str(e)}")
                                    traceback.print_exc()
                            
                            if not chunk_mapping:
                                logger.info(f"Callbacks.py - register_callbacks - handle_query : No chunk mapping found for {session_id}, skipping")
                                continue
    

                            if is_persistent and filename:
                                chunk_ids, context, scores, filenames = vect_serv.get_relevant_chunks(
                                    vectorstore, 
                                    query, 
                                    chunk_mapping = chunk_mapping,
                                    user_id=current_user_id, 
                                    session_id=filename,
                                    db_session=db_session,
                                    k=chunks_per_doc,                   
                                    threshold=similarity_threshold,
                                    default_language = doc_lang
                                )
                            else:
                                chunk_ids, context, scores, filenames = vect_serv.get_relevant_chunks(
                                    vectorstore,
                                    query,
                                    k=chunks_per_doc,                   
                                    threshold=similarity_threshold,
                                    default_language = doc_lang)

                            #select top chunks 
                            if chunk_ids and scores:
                                # Create a list of tuples (chunk_id, context_part, score, filename) for sorting
                                combined_chunks = list(zip(chunk_ids, context.split('\n\n=== From Document:'), scores, filenames))
                                
                                # Sort by score in descending order
                                combined_chunks.sort(key=lambda x: x[2], reverse=True)
                                
                                # Limit to max_total_chunks
                                combined_chunks = combined_chunks[:max_total_chunks]
                                
                                # Unpack the sorted and limited results
                                chunk_ids = [c[0] for c in combined_chunks]
                                context = '\n\n=== From Document:'.join([c[1] for c in combined_chunks])
                                scores = [c[2] for c in combined_chunks]
                                filenames = [c[3] for c in combined_chunks]
                                
                                logger.info(f"Callbacks.py - register_callbacks - handle_query : Limited to top {len(chunk_ids)} chunks out of original set based on max_total_chunks={max_total_chunks}")

                            # Store chunks for this document
                            if chunk_ids and context:
                                # Split context into separate chunks
                                if context.startswith("=== From Document:"):
                                    context_parts = context.split("\n\n=== From Document:")
                                else:
                                    context_parts = [""] + context.split("\n\n=== From Document:")
                                
                                # Skip empty first part if needed
                                if context_parts[0] == "":
                                    context_parts = context_parts[1:]
                                
                                # Add formatting back to parts (except first one)
                                for i in range(len(context_parts)):
                                    if i > 0 or not context.startswith("=== From Document:"):
                                        context_parts[i] = "=== From Document:" + context_parts[i]
                                
                                # Store everything for later sorting
                                for i, chunk_id in enumerate(chunk_ids):
                                    # Get corresponding context part, score and filename
                                    ctx_part = context_parts[i] if i < len(context_parts) else context_parts[0]
                                    score = scores[i] if i < len(scores) else 0.0
                                    file = filenames[i] if i < len(filenames) else filenames[0] if filenames else doc_state[session_id]["filename"]
                                    
                                    all_chunk_ids.append(chunk_id)
                                    all_context_parts.append(ctx_part)
                                    all_scores.append(score)
                                    all_filenames.append(file)
                                    
                                    # Map this chunk
                                    chunk_index_map[current_chunk_index] = {
                                        'chunk_id': chunk_id,
                                        'session_id': session_id,
                                        'filename': file
                                    }
                                    
                                    current_chunk_index += 1
                                
                                # Store chunk mapping for this document
                                combined_chunk_mapping[lookup_id] = chunk_mapping

                        except Exception as e:
                            logger.error(f"Callbacks.py - register_callbacks - handle_query ~4: Error processing document {session_id}: {str(e)}")
                            traceback.print_exc()
                            continue
            
            if all_chunk_ids:
                # Create list of (chunk_id, context, score, filename) tuples
                combined_chunks = list(zip(all_chunk_ids, all_context_parts, all_scores, all_filenames))
                
                # Sort by score in descending order
                combined_chunks.sort(key=lambda x: x[2], reverse=True)
                
                # Limit to max_total_chunks
                combined_chunks = combined_chunks[:max_total_chunks]
                
                logger.info(f"Callbacks.py - register_callbacks - handle_query : Limited to {len(combined_chunks)} highest-scoring chunks out of {len(all_chunk_ids)} total chunks")
                
                # Unpack the limited results
                all_relevant_chunks = [c[0] for c in combined_chunks]
                all_context = [c[1] for c in combined_chunks]
                doc_references = [c[3] for c in combined_chunks]
                
                # Make sure chunk indices in the mapping match the final selection
                used_chunk_indices = set(all_relevant_chunks)
                chunk_index_map = {k: v for k, v in chunk_index_map.items() 
                                if v['chunk_id'] in used_chunk_indices}
            else:
                all_relevant_chunks = []
                all_context = []
                doc_references = []

            if not all_context:
                logger.info("Callbacks.py - register_callbacks - handle_query : No relevant context found")  
                chat_history.append(html.Div(f"User: {query}", className="user-message"))
                chat_history.append(html.Div("No relevant information found in the documents.", className="assistant-message"))
                return chat_history, "", doc_state, None, {"progress": 100, "status": "complete"}

            # Update progress: LLM processing
            status_data = {"progress": 75, "status": "Generating response..."}
            logger.info("Callbacks.py - register_callbacks - handle_query : Getting LLM response...")
            combined_context = "\n\n".join(all_context)
            
            assistant_reply, assistant_reply_translated, used_chunk_indices = llm_serv.get_response(combined_context, query)
            status_data = {"progress": 90, "status": "Processing highlights..."}
            logger.info(f"Callbacks.py - register_callbacks - handle_query : Processing highlights for {len(used_chunk_indices)} chunks")

            # Debug the received chunk indices
            logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: LLM returned chunk indices: {used_chunk_indices}")

            # Process used_chunk_indices to get primary document and chunks
            # The LLM now returns chunk indices with their document names
            if used_chunk_indices and len(used_chunk_indices) > 0:
                # Get the primary chunk/document (the first one in the list)
                primary_chunk_index = used_chunk_indices[0]
                primary_chunk_id, primary_document = parse_llm_chunk_info(primary_chunk_index)
                
                logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Primary chunk ID: {primary_chunk_id}, Primary document: {primary_document}")
                
                # If we have a primary document name and chunk ID
                if primary_document and primary_chunk_id:
                    # Find the session_id for this document
                    primary_session_id = None
                    for session_id, info in doc_state.items():
                        if info.get('filename') == primary_document:
                            primary_session_id = session_id
                            logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Found matching session for {primary_document}: {primary_session_id}")
                            break
                            
                    # Find the chunk mapping for this document
                    primary_lookup_id = None
                    if primary_session_id:
                        # First try session_id directly
                        if primary_session_id in combined_chunk_mapping:
                            primary_lookup_id = primary_session_id
                        else:
                            # Try by filename
                            for lookup_id in combined_chunk_mapping.keys():
                                if lookup_id.endswith(primary_document) or lookup_id == primary_document:
                                    primary_lookup_id = lookup_id
                                    break
                        
                    # If we found the document and its mapping, process the highlights
                    if primary_session_id and primary_lookup_id and primary_lookup_id in combined_chunk_mapping:
                        logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Using {primary_document} as primary document for highlighting")
                        selected_chunk_mapping = combined_chunk_mapping[primary_lookup_id]
                        
                        # Extract all chunk IDs from used_chunk_indices
                        used_chunk_ids = []
                        for chunk_index in used_chunk_indices:
                            chunk_id, doc_name = parse_llm_chunk_info(chunk_index)
                            
                            # Only process chunks from the primary document
                            if doc_name and doc_name != primary_document:
                                logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Skipping highlight for chunk from different document: {doc_name}")
                                continue
                            
                            # Clean up chunk_id if needed
                            if chunk_id.startswith('chunk_'):
                                # Check if this exact chunk ID exists in the mapping
                                if chunk_id in selected_chunk_mapping:
                                    used_chunk_ids.append(chunk_id)
                                    logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Found exact match for chunk ID {chunk_id}")
                                elif chunk_id[6:].isdigit():
                                    # Try with numeric part only
                                    chunk_num = chunk_id[6:]
                                    # Look for variations like chunk_XX, chunk_0XX, etc.
                                    for mapping_id in selected_chunk_mapping.keys():
                                        if mapping_id.endswith(f"_{chunk_num}"):
                                            used_chunk_ids.append(mapping_id)
                                            logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Matched chunk ID {mapping_id} using number {chunk_num}")
                                            break
                                    
                                    # If still not found, add the original ID for fallback processing
                                    if not any(id.endswith(f"_{chunk_num}") for id in used_chunk_ids):
                                        used_chunk_ids.append(chunk_id)
                                        logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Using original chunk ID {chunk_id} for fallback")
                            else:
                                # For non-standard format, try direct match
                                if chunk_id in selected_chunk_mapping:
                                    used_chunk_ids.append(chunk_id)
                                else:
                                    logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Non-standard chunk ID not found: {chunk_id}")
                        
                        # If we have valid chunk IDs, process the highlights using our new approach
                        if used_chunk_ids:
                            logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Processing {len(used_chunk_ids)} chunk IDs: {used_chunk_ids}")
                            # Call the new process_highlights function with the primary document
                            doc_state = process_highlights(doc_state, selected_chunk_mapping, used_chunk_ids, assistant_reply, primary_document)
                        else:
                            logger.info("Callbacks.py - handle_query : Debug: No valid chunk IDs found for highlighting")
                            # Use first 3 chunks as fallback
                            fallback_chunks = list(selected_chunk_mapping.keys())[:3]
                            logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Using fallback chunks: {fallback_chunks}")
                            doc_state = process_highlights(doc_state, selected_chunk_mapping, fallback_chunks, assistant_reply, primary_document)
                    else:
                        logger.info(f"Callbacks.py - register_callbacks - handle_query : Debug: Couldn't find primary document {primary_document} in chunk mapping")

            # Find the most relevant document for display
            most_relevant_doc_id = primary_session_id if 'primary_session_id' in locals() else None
            
            # If we couldn't determine a primary document, use the first document with highlights
            if not most_relevant_doc_id:
                for session_id in doc_state:
                    if doc_state[session_id].get("highlights"):
                        most_relevant_doc_id = session_id
                        break
            
            # If still no relevant document found, use the first document referenced
            if not most_relevant_doc_id and doc_references:
                for session_id, info in doc_state.items():
                    if info["filename"] == doc_references[0]:
                        most_relevant_doc_id = session_id
                        break
            
            logger.info("Callbacks.py - register_callbacks - handle_query : Updating chat history...")
            # Update chat history
            prefix = f"[@{doc_state[selected_doc_id]['filename']}] " if doc_specific and selected_doc_id in doc_state else ""
            
            response_id = f"response-{len(chat_history)}"
            
            # First, get unique document references and sort by relevance
            unique_docs = list(set(doc_references))
            unique_docs.sort(key=lambda doc: get_document_relevance(doc, used_chunk_indices, doc_references, doc_state, most_relevant_doc_id), reverse=True)

            # Create the source citation component
            if most_relevant_doc_id and doc_state.get(most_relevant_doc_id):
                relevant_doc_name = doc_state[most_relevant_doc_id]["filename"]
                source_citation = html.Div(
                    [
                        "Source: ",
                        html.Span(
                            relevant_doc_name,
                            style={'fontWeight': 'bold'}
                        ),
                        # Only add the dropdown for additional sources if there are more than one
                        html.Span(
                            [
                                " ",
                                html.Button(
                                    ["+", html.Span(str(len(unique_docs) - 1), className="ms-1")],
                                    id={'type': 'toggle-sources', 'index': response_id},
                                    className="btn btn-sm btn-outline-secondary ms-1",
                                    style={'fontSize': '0.75rem', 'padding': '0px 5px'},
                                    n_clicks=0,
                                ) if len(unique_docs) > 1 else "",
                            ]
                        ),
                        # Hidden div to display all sources when toggled
                        html.Div(
                            [
                                html.Hr(style={'margin': '5px 0'}),
                                "All Sources: ",
                                *[
                                    html.Div(
                                        [
                                            html.Span(
                                                doc_name,
                                                style={
                                                    'fontWeight': 'bold' if doc_name == relevant_doc_name else 'normal',
                                                    'color': '#666' if doc_name != relevant_doc_name else 'inherit'
                                                }
                                            ),
                                            html.Span(
                                                f" (cited {doc_references.count(doc_name)} times)",
                                                style={'fontSize': '0.85rem', 'color': '#666'}
                                            ) if doc_references.count(doc_name) > 1 else ""
                                        ],
                                        className="mt-1"
                                    )
                                    for doc_name in unique_docs
                                ]
                            ],
                            id={'type': 'all-sources', 'index': response_id},
                            style={'display': 'none', 'marginTop': '5px'}
                        )
                    ],
                    className="source-citation"
                )
            else:
                # Fallback if no most relevant document is identified
                source_citation = html.Div(
                    [
                        "Sources: ",
                        html.Span(", ".join(unique_docs[:1])) if unique_docs else html.Span("None")
                    ],
                    className="source-citation"
                )

            # Create the chat message components
            chat_history.extend([
                html.Div([
                    # Response text
                    html.Div(
                        f"{assistant_reply_translated}",
                        style={"marginBottom": "8px"}
                    ),
                    # Updated Sources section with only the most relevant document
                    source_citation,
                    # Add feedback buttons
                    create_feedback_buttons(response_id)
                ], 
                className="chat-message llm-response",
                id={'type': 'response-container', 'index': response_id}),
                html.Hr(style={"margin": "20px 0"})
            ])
            status_data = {"progress": 100, "status": "complete"}
            logger.info("Query processing complete")
            
            return chat_history, "", doc_state, most_relevant_doc_id, status_data
                        
        except Exception as e:
            logger.error(f"Callbacks.py - register_callbacks - handle_query ~5: Error in handle_query: {str(e)}")
            traceback.print_exc()
            error_msg = f"Error: {str(e)}" if str(e) else "An unexpected error occurred"
            chat_history.append(
                html.Div(
                    error_msg,
                    className="chat-message error-message"
                )
            )
            return chat_history, query, doc_state, dash.no_update, {"progress": 100, "status": "error"}
    
    @app.callback(
        [Output({'type': 'all-sources', 'index': MATCH}, 'style')],
        [Input({'type': 'toggle-sources', 'index': MATCH}, 'n_clicks')],
        [State({'type': 'all-sources', 'index': MATCH}, 'style')]
    )
    def toggle_all_sources(n_clicks, current_style):
        """Toggle display of all sources when the + button is clicked"""
        if not n_clicks or n_clicks % 2 == 0:
            return [{'display': 'none', 'marginTop': '5px'}]
        else:
            return [{'display': 'block', 'marginTop': '5px'}]
  

    def get_document_relevance(doc_name, used_chunk_indices, doc_references, doc_state, most_relevant_doc_id):
        """
        Determine document relevance score based on usage frequency, selection state and match quality
        
        Args:
            doc_name: Name of the document
            used_chunk_indices: List of chunk indices used
            doc_references: List of document references
            doc_state: Document state dictionary
            most_relevant_doc_id: ID of the most relevant document
            
        Returns:
            float: Relevance score (higher is more relevant)
        """
        # Base relevance is the frequency of document citation
        relevance = doc_references.count(doc_name)
        
        # Give priority to documents with highlights
        for session_id, info in doc_state.items():
            if info.get("filename") == doc_name and info.get("highlights"):
                relevance += 5
                break
                
        # Give highest priority to selected document
        if most_relevant_doc_id and doc_name == doc_state.get(most_relevant_doc_id, {}).get("filename"):
            relevance += 10
            
        return relevance

    @app.callback(
        [
            Output("document-viewer-container", "children", allow_duplicate=True),
            Output("document-data", "data", allow_duplicate=True),
            Output("pdf-highlights", "data", allow_duplicate=True),
            Output("text-highlights", "data", allow_duplicate=True),
            Output("document-state", "data", allow_duplicate=True),  # Add output to update doc state
        ],
        [Input("document-selector", "value")],
        [State("document-state", "data"),
        State("auth-state", "data"),
        State("url", "pathname")],
        prevent_initial_call=True
    )
    def update_document_viewer(selected_doc, doc_state, auth_state, urls):
        """Update document viewer with lazy loading of document content"""
        
        if not selected_doc or not doc_state or selected_doc not in doc_state:
            raise PreventUpdate
        if urls in ['/documents', '/admin']:
            raise PreventUpdate
        
        doc_info = doc_state[selected_doc]
        filename = doc_info["filename"]
        highlights = doc_info.get("highlights", [])
        
        logger.info(f"Callbacks.py - register_callbacks - update_document_viewer : Processing document: {filename}")
        logger.info(f"Callbacks.py - register_callbacks - update_document_viewer : Found {len(highlights) if highlights else 0} highlights")
        
        # Check if content needs to be loaded
        content_loaded = doc_info.get('content_loaded', False)
        if not content_loaded or doc_info.get('content') is None:
            logger.info(f"Content not loaded for {filename}, loading now")
            # Load content just-in-time
            updated_doc_info = load_document_content(selected_doc, doc_info)
            # Update the document in the state
            doc_state[selected_doc] = updated_doc_info
            content = updated_doc_info.get('content', '')
        else:
            logger.info(f"Callbacks.py - register_callbacks - update_document_viewer : Content already loaded for {filename}")
            content = doc_info.get('content', '')
        
        # Create the appropriate viewer based on file type
        if filename.lower().endswith('.pdf'):
            # Ensure PDF content has proper base64 prefix
            if isinstance(content, bytes):
                content = f"data:application/pdf;base64,{base64.b64encode(content).decode()}"
            elif isinstance(content, str) and not content.startswith('data:'):
                try:
                    # Try to decode if it's a base64 string without prefix
                    decoded = base64.b64decode(content)
                    content = f"data:application/pdf;base64,{content}"
                except:
                    # If not valid base64, it might be already in the right format
                    pass
                    
            # Create fresh PDF viewer
            viewer = html.Div([
                # PDF viewer container
                html.Div(
                    id='pdf-js-viewer',
                    style={
                        'width': '100%', 
                        'height': '700px',
                        'border': '1px solid #ccc',
                        'backgroundColor': '#ffffff',
                        'position': 'relative',
                        'display': 'block'
                    }
                ),
                # Hidden text viewer
                html.Div(
                    id='text-viewer',
                    style={'display': 'none'}
                ),
                # Stores
                dcc.Store(id='document-data', storage_type='memory'),
                dcc.Store(id='pdf-highlights', storage_type='memory'),
                dcc.Store(id='text-highlights', storage_type='memory'),
                dcc.Store(id='pdf-viewer-result', storage_type='memory'),
                dcc.Store(id='text-viewer-scroll', storage_type='memory'),

                # Make sure this dummy output element exists:
                html.Div(id='_dummy-output', style={'display': 'none'}),
                
                # Control elements with unique keys for forced refresh
                html.Div(
                    id='pdf-highlight-trigger', 
                    style={'display': 'none'}, 
                    key=datetime.now().isoformat()
                ),
                html.Div(
                    id='_clear-stores', 
                    style={'display': 'none'}, 
                    key=datetime.now().isoformat()
                )
            ])
            logger.info(f"Callbacks.py - register_callbacks - update_document_viewer : Created PDF viewer with {len(highlights) if highlights else 0} highlights")
            return viewer, content, json.dumps(highlights) if highlights else None, None, doc_state
            
        else:
            if isinstance(content, bytes):
                try:
                    content = content.decode('utf-8')
                except UnicodeDecodeError:
                    content = "Binary content cannot be displayed as text. Please use appropriate viewer."

            # Create text viewer
            viewer = html.Div([
                # Hidden PDF viewer
                html.Div(
                    id='pdf-js-viewer',
                    style={'display': 'none'}
                ),
                # Text viewer container
                html.Div(
                    id='text-viewer',
                    style={
                        'width': '100%',
                        'height': '700px',
                        'border': '1px solid #ccc',
                        'backgroundColor': '#ffffff',
                        'padding': '20px',
                        'overflowY': 'auto',
                        'position': 'relative',
                        'fontSize': '14px',
                        'lineHeight': '1.5',
                        'display': 'block'
                    }
                ),
                # Stores
                dcc.Store(id='document-data', storage_type='memory'),
                dcc.Store(id='pdf-highlights', storage_type='memory'),
                dcc.Store(id='text-highlights', storage_type='memory'),
                
                # Control elements
                html.Div(id='pdf-highlight-trigger', style={'display': 'none'}),
                html.Div(id='_clear-stores', style={'display': 'none'}),
                dcc.Store(id='pdf-viewer-result', storage_type='memory'),
                dcc.Store(id='text-viewer-scroll', storage_type='memory'),

                # Make sure this dummy output element exists:
                html.Div(id='_dummy-output', style={'display': 'none'})
            ])
            logger.info("Callbacks.py - register_callbacks - update_document_viewer : Created text viewer")
            return viewer, content, None, highlights if highlights else None, doc_state
        
    @app.callback(
        [
            Output("text-viewer", "children", allow_duplicate=True),
            Output("text-viewer", "style", allow_duplicate=True)
        ],
        [
            Input("document-data", "data"),
            Input("text-highlights", "data")
        ],
        [
            State("document-selector", "value"),
            State("document-state", "data")
        ],
        prevent_initial_call=True
    )
    def update_text_content(content, highlights, selected_doc, doc_state):
        """Handle text document content and continuous highlighting"""
        if not content or not selected_doc or not doc_state or selected_doc not in doc_state:
            raise PreventUpdate
                    
        doc_info = doc_state[selected_doc]
        filename = doc_info["filename"]
        
        # Only process non-PDF documents
        if filename.lower().endswith('.pdf'):
            raise PreventUpdate
                    
        logger.info(f"Callbacks.py - register_callbacks - update_text_content : Updating text content for {filename}")
        
        try:
            base_style = {
                'width': '100%',
                'height': '700px',
                'border': '1px solid #ccc',
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'overflowY': 'auto',
                'whiteSpace': 'pre-wrap',
                'fontFamily': 'monospace',
                'display': 'block',
                'lineHeight': '1.5'
            }
            
            # Use appropriate text content source
            if filename.lower().endswith('.docx') and 'raw_text' in doc_info:
                text_content = doc_info['raw_text']
            else:
                text_content = content
            
            # If content is base64 encoded, handle differently
            if isinstance(text_content, str) and text_content.startswith('data:'):
                # For base64 encoded content, we can't easily show it directly
                if 'raw_text' in doc_info:
                    text_content = doc_info['raw_text']
                    logger.info("Callbacks.py - register_callbacks - update_text_content : Using raw_text instead of base64 content")
                else:
                    try:
                        import base64
                        content_parts = text_content.split('base64,', 1)
                        if len(content_parts) > 1:
                            base64_content = content_parts[1]
                            decoded = base64.b64decode(base64_content)
                            
                            # Try to decode as text
                            try:
                                text_content = decoded.decode('utf-8')
                            except UnicodeDecodeError:
                                try:
                                    text_content = decoded.decode('latin-1')
                                except:
                                    text_content = "Binary content cannot be displayed as text. Please use appropriate viewer."
                        else:
                            text_content = "Cannot display content (invalid encoding)"
                    except:
                        text_content = "Cannot display content (decoding error)"
            
            lines = text_content.split('\n')
            formatted_content = []
            content_pos = 0
            
            # Create a mapping of positions to highlight information
            highlight_map = {}
            
            if highlights:
                # For each character position, determine if it should be highlighted
                for h in highlights:
                    start = h['start']
                    end = h['end']
                    
                    # Mark each position in the range as needing to be highlighted
                    for pos in range(start, end):
                        highlight_map[pos] = {
                            'chunk_id': h.get('chunk_id'),
                            'is_core': h.get('is_core', False)
                        }
                
                # Generate highlighted content line by line
                for line_num, line in enumerate(lines):
                    line_content = []
                    current_pos = 0
                    line_start = content_pos
                    line_end = line_start + len(line)
                    
                    # Track highlight state
                    in_highlight = False
                    current_highlight = None
                    highlight_start = None
                    
                    # Process each character position in the line
                    for i in range(len(line)):
                        pos = line_start + i
                        is_highlighted = pos in highlight_map
                        
                        # Detect highlight transitions
                        if not in_highlight and is_highlighted:
                            # Starting a new highlight section
                            if current_pos < i:
                                # Add text before highlight
                                line_content.append(html.Span(line[current_pos:i]))
                            
                            in_highlight = True
                            highlight_start = i
                            current_highlight = highlight_map[pos]
                        
                        elif in_highlight and (not is_highlighted or 
                                            (highlight_map.get(pos) != current_highlight)):
                            # Ending a highlight section or changing to a different highlight
                            highlight_text = line[highlight_start:i]
                            
                            # Add the highlighted text span
                            if highlight_text.strip():
                                style = {
                                    'backgroundColor': 'rgba(255, 200, 0, 0.4)' if current_highlight.get('is_core', False) else 'rgba(255, 255, 0, 0.3)',
                                    'padding': '2px 0',
                                    'borderRadius': '2px',
                                    'display': 'inline',
                                }
                                
                                line_content.append(
                                    html.Span(
                                        highlight_text,
                                        style=style,
                                        className='core-highlight' if current_highlight.get('is_core') else 'text-highlight'
                                    )
                                )
                            
                            # Reset for new section
                            current_pos = i
                            
                            if is_highlighted:
                                # Start a new highlight immediately
                                in_highlight = True
                                highlight_start = i
                                current_highlight = highlight_map[pos]
                            else:
                                in_highlight = False
                    
                    # Handle the end of line
                    if in_highlight:
                        # Finish the current highlight
                        highlight_text = line[highlight_start:]
                        
                        if highlight_text.strip():
                            style = {
                                'backgroundColor': 'rgba(255, 200, 0, 0.4)' if current_highlight.get('is_core', False) else 'rgba(255, 255, 0, 0.3)',
                                'padding': '2px 0',
                                'borderRadius': '2px',
                                'display': 'inline',
                            }
                            
                            line_content.append(
                                html.Span(
                                    highlight_text,
                                    style=style,
                                    className='core-highlight' if current_highlight.get('is_core') else 'text-highlight'
                                )
                            )
                    elif current_pos < len(line):
                        # Add remaining unhighlighted text
                        line_content.append(html.Span(line[current_pos:]))
                    
                    # Add the line to formatted content
                    formatted_content.append(
                        html.Div(
                            line_content, 
                            style={
                                'minHeight': '1.5em',
                                'display': 'block',
                                'width': '100%'
                            }
                        )
                    )
                    
                    # Update position for next line
                    content_pos = line_end + 1  # +1 for newline character
            else:
                # No highlights - simple line formatting
                formatted_content = [
                    html.Div(
                        html.Span(line), 
                        style={
                            'minHeight': '1.5em',
                            'display': 'block',
                            'width': '100%'
                        }
                    )
                    for line in lines
                ]
            
            return formatted_content, base_style
                
        except Exception as e:
            logger.error(f"Callbacks.py - register_callbacks - update_text_content : Error updating text content: {str(e)}")
            traceback.print_exc()
            # Return plain text as fallback
            return html.Pre(text_content if isinstance(text_content, str) else str(text_content)), base_style

    @app.callback(
        Output("pdf-js-viewer", "style", allow_duplicate=True),
        [Input("document-selector", "value")],
        [State("document-state", "data")],
        prevent_initial_call=True
    )
    def toggle_pdf_viewer(selected_doc, doc_state):
        """Show/hide PDF viewer based on document type"""
        if not selected_doc or not doc_state or selected_doc not in doc_state:
            raise PreventUpdate
            
        doc_info = doc_state[selected_doc]
        filename = doc_info["filename"]
        
        if filename.lower().endswith('.pdf'):
            return {
                'width': '100%',
                'height': '700px',
                'border': '1px solid #ccc',
                'backgroundColor': '#ffffff',
                'position': 'relative',
                'display': 'block'
            }
        else:
            return {'display': 'none'}
        
    @app.callback(
        Output("chat-history", "children", allow_duplicate=True),  # Make sure this ID exists
        [Input("document-state", "data")],
        [State("chat-history", "children"),
         State("url", "pathname")
         ],
        prevent_initial_call=True
    )
    def show_persistent_docs_message(doc_state, chat_history, pathname):
        """Add a welcome message for persistent documents"""
        if pathname in [ '/documents', '/admin']:
            raise PreventUpdate
    
        if not doc_state or not isinstance(doc_state, dict) or len(doc_state) == 0:
            return chat_history or []
        
        # Count persistent documents
        persistent_docs = [info for info in doc_state.values() 
                        if isinstance(info, dict) and 
                        (info.get('source') in ['folder', 'group'] or 'path' in info)]
        
        if persistent_docs:
            # Only add the message if chat history is empty
            if not chat_history or len(chat_history) == 0:
                from utils.visualization import create_system_notification
                return [create_system_notification(
                    f"{len(persistent_docs)} persistent documents loaded from your group storage",
                    type="success",
                    action="load"
                )]
        
        return chat_history or []
    
    def create_persistent_docs_notification():
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2 text-primary"),
                "Your persistent documents have been loaded automatically.",
            ], className="alert alert-info alert-dismissible fade show", role="alert"),
            html.Button(
                html.Span("×", className="visually-hidden"),
                className="btn-close",
                **{"data-bs-dismiss": "alert"},
                **{"data-bs-dismiss": "alert", "aria-label": "Close"}
                )
        ], id="persistent-docs-notification", className="mb-3")

def register_auto_load_callback(app):
    """Register a simplified callback to load documents on app start"""
    @app.callback(
        [
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True)
        ],
        [Input("page-loaded-trigger", "data")],
        [
            State("auth-state", "data"),
            State("document-state", "data"),
            State("url", "pathname")
        ],
        prevent_initial_call=True
    )
    def auto_load_documents(trigger_data, auth_state, doc_state, pathname):
        """Simplified function to load documents"""
        if not trigger_data or not auth_state or not auth_state.get('authenticated'):
            raise PreventUpdate
        if pathname != '/':
            raise PreventUpdate  
        # Don't reload if documents already loaded
        if doc_state and len(doc_state) > 0:
            raise PreventUpdate
            
        try:
            # Get user info
            current_user_id = auth_state.get('user_id')
            if not current_user_id:
                raise PreventUpdate
                
            # Load documents
            from sqlalchemy.orm import sessionmaker
            from auth.models import Base
            from sqlalchemy import create_engine
            from auth.config import AUTH_CONFIG 
            engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
            DBSession = sessionmaker(bind=engine)
            db_session = DBSession()
            
            # Sync documents
            doc_state, vstore_state, chunk_state = sync_documents_on_load('/', auth_state, {})
            
            return doc_state, vstore_state, chunk_state
            
        except Exception as e:
            logger.error(f"Callbacks.py - register_auto_load_callback - auto_load_document :Error loading documents: {str(e)}")
            traceback.print_exc()
            return {}, {}, {}
            
    @app.callback(
        [
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("document-selector", "options", allow_duplicate=True)
        ],
        [Input("document-state", "data")],
        [
            State("auth-state", "data"),
            State("vectorstore-state", "data"),
            State("document-selector", "options"),
            State("url", "pathname")
        ],
        prevent_initial_call=True
    )
    def validate_vector_stores(doc_state, auth_state, vstore_state, selector_options, pathname):
        """Validate vector stores and ensure they exist"""
        if not doc_state or not auth_state:
            raise PreventUpdate
        if pathname != '/':
            raise PreventUpdate
        current_user_id = auth_state.get('user_id')
        if not current_user_id:
            raise PreventUpdate
            
        # Initialize states
        vstore_state = vstore_state or {}
        
        # Get database session
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        from auth.config import AUTH_CONFIG
        engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
        DBSession = sessionmaker(bind=engine)
        db_session = DBSession()
        
        try:
            # Create vector store service
            
            # Check each document
            for session_id, info in doc_state.items():
                if session_id in vstore_state:
                    # Vector store exists, verify it can be loaded
                    try:
                        # Try to load vector store
                        vs, metadata = vect_serv.load_vectorstore(session_id)
                        if not vs:
                            logger.error(f"Callbacks.py - register_auto_load_callback -  validate_vector_stores : Vector store for session {session_id} failed to load, removing from state")
                            vstore_state.pop(session_id, None)
                    except Exception as e:
                        logger.info(f"Callbacks.py - register_auto_load_callback -  validate_vector_stores ~1 : Error loading vector store for session {session_id}: {str(e)}")
                        vstore_state.pop(session_id, None)
                elif info.get('vector_store_exists'):
                    # Vector store exists but not in current state
                    vstore_state[session_id] = session_id
            
            # Update selector options if needed
            selector_options = [
                {"label": info["filename"], "value": session_id}
                for session_id, info in doc_state.items()
                if session_id in vstore_state  # Only include documents with valid vector stores
            ]
            
            return vstore_state, selector_options
            
        except Exception as e:
            logger.error(f"Callbacks.py - register_auto_load_callback -  validate_vector_stores ~1Error validating vector stores: {str(e)}")
            return vstore_state, selector_options
        
def register_chat_autoscroll(app):
    """Register a dead-simple auto-scroll function that actually works"""
    
    # This is the simplest possible solution
    app.clientside_callback(
        """
        function(children) {
            // Force the browser to wait a moment before scrolling
            setTimeout(function() {
                const chatHistory = document.getElementById('chat-history');
                if (chatHistory) {
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                    console.log("Auto-scrolled chat history");
                }
            }, 200);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-history', 'style', allow_duplicate=True),  
        Input('chat-history', 'children'),
        State('chat-history', 'style'),
        prevent_initial_call=True
    )

def register_query_progress(app):
    @app.callback(
        [
            Output("query-progress-bar", "value", allow_duplicate=True),
            Output("query-progress-bar", "style", allow_duplicate=True),
            Output("progress-status", "children", allow_duplicate=True),
            Output("progress-status", "style", allow_duplicate=True),
            Output("query-progress-interval", "disabled", allow_duplicate=True)
        ],
        [
            Input("submit-btn", "n_clicks"),
            Input("query-input", "n_submit"),
            Input("query-progress-interval", "n_intervals"),
            Input("query-processing-status", "data")
        ],
        [State("query-input", "value")]
    )
    def update_query_progress(n_clicks, n_submit, n_intervals, status_data, query_value):
        if not (n_clicks or n_submit) and not status_data:
            return 0, {"display": "none"}, "", {"display": "none"}, True

        triggered = ctx.triggered_id
        
        # Start of query processing
        if triggered in ["submit-btn", "query-input"]:
            if not query_value:
                return 0, {"display": "none"}, "", {"display": "none"}, True
                
            return (
                0,
                {"display": "block", "height": "4px", "width": "100%"},
                "Processing query...",
                {"display": "block"},
                False
            )

        progress = status_data.get("progress", 0) if status_data else 0
        status_text = status_data.get("status", "") if status_data else ""

        # Query processing phases
        phases = [
            (0, 1, "Waiting for query.."),
            (1, 25, "Analyzing query..."),
            (25, 50, "Searching through documents..."),
            (50, 75, "Processing context..."),
            (75, 95, "Generating response..."),
            (95, 100, "Finalizing..."),
        ]

        # If we hit 100%, stop the interval
        if progress >= 100:
            return (
                100,
                {"display": "block", "height": "4px", "width": "100%"},
                "Response ready!",
                {"display": "block"},
                True
            )

        # Find the current phase
        current_phase = None
        for start, end, text in phases:
            if start <= progress < end:
                current_phase = (start, end, text)
                break

        if current_phase:
            status_text = current_phase[2]
            # Increment progress within the current phase
            progress_step = (current_phase[1] - current_phase[0]) / 10
            progress = min(progress + progress_step, current_phase[1])

        return (
            progress,
            {"display": "block", "height": "4px", "width": "100%"},
            status_text,
            {"display": "block"},
            False
        )
    """Register the clientside callback for textarea expansion"""
    app.clientside_callback(
        """
        function(value) {
            const textarea = document.querySelector('#query-input');
            if (!textarea) return window.dash_clientside.no_update;
            
            // Reset height temporarily to get the correct scrollHeight
            textarea.style.height = '50px';
            
            // Get the scroll height and line height
            const scrollHeight = textarea.scrollHeight;
            const lineHeight = parseInt(getComputedStyle(textarea).lineHeight);
            const lines = Math.ceil(scrollHeight / lineHeight);
            
            // Calculate new height
            let newHeight;
            if (lines <= 3) {
                newHeight = '50px';  // Default height for 1-3 lines
            } else {
                newHeight = Math.min(scrollHeight, 200) + 'px';  // Expand but limit to maxHeight
            }
            
            textarea.style.height = newHeight;
            return window.dash_clientside.no_update;
        }
        """,
        Output("query-input", "style", allow_duplicate=True),
        Input("query-input", "value"),
        prevent_initial_call=True
    )

    # PDF viewer client callback
    # Server-side trigger for PDF viewer update
    @app.callback(
        Output("pdf-highlight-trigger", "children", allow_duplicate=True),
        [Input("document-data", "data"),
        Input("pdf-highlights", "data")],
        prevent_initial_call=True
    )
    def trigger_pdf_viewer_update(pdfData, highlightsJson):
        """Trigger PDF viewer update with highlights and auto-scrolling"""
        # Simply return a timestamp to trigger the client-side callback
        return datetime.now().isoformat()

    # Client-side callback for PDF viewer with auto-scrolling
    app.clientside_callback(
        """
        function(triggerTime, pdfData, highlightsJson) {
            console.log('Client callback triggered');
            
            if (!pdfData) return;
            
            const loadPDF = async () => {
                if (!window.pdfViewerInitialized) {
                    console.log('Waiting for viewer initialization');
                    setTimeout(loadPDF, 100);
                    return;
                }
                
                try {
                    const viewer = window.currentViewer;
                    if (viewer) {
                        console.log('Cleaning up existing viewer');
                        await viewer.cleanup();
                    }
                    
                    console.log('Initializing new viewer');
                    const newViewer = window.initPDFViewer('pdf-js-viewer');
                    window.currentViewer = newViewer;
                    
                    const content_string = pdfData.split(',')[1];
                    const decoded = atob(content_string);
                    const array = new Uint8Array(decoded.length);
                    for (let i = 0; i < decoded.length; i++) {
                        array[i] = decoded.charCodeAt(i);
                    }
                    
                    console.log('Loading document into viewer');
                    await newViewer.loadDocument(array);
                    
                    if (highlightsJson) {
                        try {
                            const highlights = JSON.parse(highlightsJson);
                            console.log('Applying highlights:', highlights);
                            if (Array.isArray(highlights) && highlights.length > 0) {
                                await newViewer.setHighlights(highlights);
                                
                                // Auto-scroll to the first highlight after a short delay
                                setTimeout(async () => {
                                    try {
                                        // Get first highlight's page
                                        const firstHighlight = highlights[0];
                                        console.log("First highlight:", firstHighlight);
                                        
                                        if (firstHighlight && firstHighlight.page !== undefined) {
                                            console.log('Auto-scrolling to highlight on page:', firstHighlight.page);
                                            console.log('Highlight position:', firstHighlight.coords);
                                            
                                            // Check if functions exist
                                            console.log('jumpToPage exists:', typeof newViewer.jumpToPage === 'function');
                                            console.log('scrollToPosition exists:', typeof newViewer.scrollToPosition === 'function');
                                            
                                            // Jump to the page of the first highlight
                                            await newViewer.jumpToPage(firstHighlight.page);
                                            
                                            // Scroll to highlight position within the page
                                            if (firstHighlight.coords) {
                                                const { x1, y1 } = firstHighlight.coords;
                                                await newViewer.scrollToPosition(firstHighlight.page, y1);
                                            }
                                        } else {
                                            console.log("No valid highlight found for scrolling");
                                        }
                                    } catch (scrollError) {
                                        console.error('Error auto-scrolling to highlight:', scrollError);
                                        console.error(scrollError.stack); // Log the stack trace
                                    }
                                }, 1000); // Small delay to ensure highlights are rendered
                            }
                        } catch (error) {k  
                            console.error('Error applying highlights:', error);
                        }
                    }
                } catch (error) {
                    console.error('Error in PDF loading process:', error);
                }
            };
            
            // Start the loading process
            loadPDF();
            
            return Date.now().toString();
        }
        """,
        Output("pdf-viewer-result", "data"),  # Use a new output ID to avoid conflicts
        [Input("pdf-highlight-trigger", "children")],
        [State("document-data", "data"),
        State("pdf-highlights", "data")],
        prevent_initial_call=True
    )
    app.clientside_callback(
    """
    function(selectedDoc, docState) {
        if (!selectedDoc || !docState) return;
        
        const docInfo = docState[selectedDoc];
        if (!docInfo) return;
        
        const isPDF = docInfo.filename.toLowerCase().endsWith('.pdf');
        
        // Always clean up existing viewer
        if (window.currentViewer) {
            try {
                window.currentViewer.cleanup();
                window.currentViewer = null;
            } catch (error) {
                console.error('Error cleaning up viewer:', error);
            }
        }
        
        // Update viewer visibility
        document.getElementById('pdf-js-viewer').style.display = isPDF ? 'block' : 'none';
        document.getElementById('text-viewer').style.display = isPDF ? 'none' : 'block';
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("_clear-stores", "children", allow_duplicate=True),
    [Input("document-selector", "value")],
    [State("document-state", "data")]
)

    app.clientside_callback(
    """
    function(scrollData) {
        if (!scrollData || !scrollData.scroll_to) return;
        
        // Short delay to ensure rendering is complete
        setTimeout(() => {
            try {
                const textViewer = document.getElementById('text-viewer');
                if (!textViewer) return;
                
                // Find all highlight elements
                const highlights = textViewer.querySelectorAll('.text-highlight');
                if (highlights && highlights.length > 0) {
                    // Scroll to the first highlight
                    highlights[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
                    console.log('Scrolled to first text highlight');
                } else {
                    // If no highlight elements found, try scrolling to position
                    const position = scrollData.scroll_to;
                    
                    // Find the element containing the text at this position
                    // This is more complex since we need to find which div contains this position
                    const allDivs = textViewer.querySelectorAll('div');
                    let totalLength = 0;
                    let targetDiv = null;
                    
                    for (const div of allDivs) {
                        const divText = div.textContent || '';
                        if (totalLength <= position && totalLength + divText.length >= position) {
                            targetDiv = div;
                            break;
                        }
                        totalLength += divText.length + 1; // +1 for newline
                    }
                    
                    if (targetDiv) {
                        targetDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        console.log('Scrolled to position:', position);
                    } else {
                        // Fallback: just scroll to top
                        textViewer.scrollTop = 0;
                    }
                }
            } catch (error) {
                console.error('Error in text auto-scroll:', error);
            }
        }, 300);
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("_dummy-output", "children",allow_duplicate=True),
    Input("text-viewer-scroll", "data"),
    prevent_initial_call=True
    )


    # Add this to your register_callbacks function in callbacks.py
    app.clientside_callback(
        """
        function(highlights) {
            if (!highlights || highlights.length === 0) return;
            
            console.log("Direct text auto-scroll triggered for", highlights.length, "highlights");
            
            // Wait for the DOM to be updated with highlights
            setTimeout(() => {
                try {
                    // Try to find highlight elements in the text viewer
                    const textViewer = document.getElementById('text-viewer');
                    if (!textViewer) {
                        console.log("Text viewer not found");
                        return;
                    }
                    
                    const highlightElements = textViewer.querySelectorAll('.text-highlight');
                    console.log("Found", highlightElements.length, "highlight elements");
                    
                    if (highlightElements && highlightElements.length > 0) {
                        // Scroll to the first highlight element
                        highlightElements[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
                        console.log("Scrolled to first highlight element");
                    } else {
                        // Fallback to position-based scrolling
                        const firstHighlightPos = highlights[0].start;
                        console.log("Trying position-based scroll to", firstHighlightPos);
                        // ... position-based scrolling logic ...
                    }
                } catch (error) {
                    console.error("Error in auto-scroll:", error);
                }
            }, 800); // Longer delay to ensure DOM is updated
            
            return window.dash_clientside.no_update;
        }
        """,
        Output("_dummy-output", "children", allow_duplicate=True),
        Input("text-highlights", "data"),
        prevent_initial_call=True
    )

def register_immediate_question_display(app):
    """Add client-side JavaScript to show questions immediately"""
    
    app.clientside_callback(
        """
        function(n_clicks, n_submit, query) {
            if ((!n_clicks && !n_submit) || !query) return window.dash_clientside.no_update;
            
            // Get the chat history div
            const chatHistory = document.getElementById('chat-history');
            if (!chatHistory) return window.dash_clientside.no_update;
            
            // Create a temporary question element
            const questionDiv = document.createElement('div');
            questionDiv.className = 'chat-message user-query';
            questionDiv.innerHTML = 'Question: ' + query;
            
            // Add it to the chat history
            chatHistory.appendChild(questionDiv);
            
            // Create a loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'chat-message loading-message';
            loadingDiv.innerHTML = '<div style="display: flex; align-items: center;"><span style="margin-right: 8px;">Generating response</span><div class="spinner-dots"><span>.</span><span>.</span><span>.</span></div></div>';
            loadingDiv.id = 'temp-loading-indicator';
            
            // Add loading indicator
            chatHistory.appendChild(loadingDiv);
            
            // Scroll to bottom
            chatHistory.scrollTop = chatHistory.scrollHeight;
            
            // Create style for loading animation if it doesn't exist
            if (!document.getElementById('loading-animation-style')) {
                const style = document.createElement('style');
                style.id = 'loading-animation-style';
                style.textContent = `
                    .spinner-dots span {
                        animation: pulse 1.4s infinite;
                        animation-fill-mode: both;
                    }
                    .spinner-dots span:nth-child(2) {
                        animation-delay: 0.2s;
                    }
                    .spinner-dots span:nth-child(3) {
                        animation-delay: 0.4s;
                    }
                    @keyframes pulse {
                        0%, 80%, 100% { opacity: 0; }
                        40% { opacity: 1; }
                    }
                    .loading-message {
                        padding: 10px 15px;
                        background-color: #f8f9fa;
                        border-left: 3px solid #6c757d;
                        border-radius: 4px;
                    }
                `;
                document.head.appendChild(style);
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('_dummy-output', 'children', allow_duplicate=True),
        [Input('submit-btn', 'n_clicks'), Input('query-input', 'n_submit')],
        [State('query-input', 'value')],
        prevent_initial_call=True
    )

    app.clientside_callback(
            """
            function(children) {
                // Remove the temporary elements
                const loadingIndicator = document.getElementById('temp-loading-indicator');
                if (loadingIndicator) {
                    loadingIndicator.remove();
                }
                
                // Cleanup any duplicate questions (if they exist)
                setTimeout(() => {
                    const userQueries = document.querySelectorAll('.user-query');
                    if (userQueries.length > 0) {
                        // Keep only the unique questions
                        const textContents = new Set();
                        for (let i = 0; i < userQueries.length; i++) {
                            const text = userQueries[i].textContent;
                            if (textContents.has(text)) {
                                userQueries[i].remove();
                            } else {
                                textContents.add(text);
                            }
                        }
                    }
                }, 100);
                
                return window.dash_clientside.no_update;
            }
            """,
            Output('_dummy-output', 'children', allow_duplicate=True),
            Input('chat-history', 'children'),
            prevent_initial_call=True
        )