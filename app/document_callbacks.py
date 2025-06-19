# document_callbacks.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"
__license__  = "MIT"

import dash
from dash import Input, Output, State, ctx, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html
import json
import base64
from datetime import datetime
from services.vector_store import sync_user_files, check_files_exist, VectorStoreService
from app.storage_config import init_storage, save_file, delete_file, get_user_directory
from app.document_view import create_status_badge, load_document_tags, save_document_tags, create_tag_badge
import traceback
import os
from app.storage_config import BASE_STORAGE_DIR, get_group_for_user
from pathlib import Path
from dash.dependencies import ClientsideFunction
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.storage_config import get_group_for_user
from app.document_view import load_document_tags, create_document_list_layout
from auth.config import AUTH_CONFIG  # Make sure this is imported for the database URI
from app.utils import parse_contents
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize storage directories
init_storage()

engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
DBSession = sessionmaker(bind=engine)
db_session = DBSession()

def format_size(size_in_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} TB"

def update_table_with_tags(doc_state, tags_data, user_id):
    """Update the document table with tags"""
    if not doc_state or not isinstance(doc_state, dict):
        return [html.Tr([
            html.Td(
                "No documents found. Click 'Upload New Document' to add some!",
                colSpan=7,
                className="text-center text-muted py-4"
            )
        ])]
        
    new_table_body = []
    for session_id, info in doc_state.items():
        doc_info = doc_state[session_id]
        session_id = doc_info['filename']
        if info.get('user_id') == user_id:
            try:
                # Get file status
                file_status = check_files_exist(user_id, session_id, db_session )
                
                # Create status badges
                status_badges = [
                    create_status_badge(
                        file_status['chunks'],
                        "Chunks"
                    ),
                    create_status_badge(
                        file_status['vector_store'],
                        "Vector DB"
                    ),
                    create_status_badge(
                        file_status['metadata'],
                        "Metadata"
                    )
                ]
                
                # Get tags for this document
                doc_tags = tags_data.get(session_id, ["Untagged"])
                tag_badges = [create_tag_badge(tag) for tag in doc_tags]
                
                # Calculate size safely
                content = info.get('content', '')
                if isinstance(content, str):
                    size = len(content)
                elif isinstance(content, bytes):
                    size = len(content)
                else:
                    size = 0
                
                new_table_body.append(
                    html.Tr([
                        # File name column
                        html.Td([
                            html.I(className="fas fa-file me-2"),
                            info['filename']
                        ]),
                        # Type column
                        html.Td(info['filename'].split('.')[-1].upper()),
                        # Size column
                        html.Td(format_size(size)),
                        # Date column
                        html.Td(
                            datetime.fromisoformat(info['timestamp']).strftime("%Y-%m-%d %H:%M")
                        ),
                        # Status column
                        html.Td(status_badges),
                        # Tags column
                        html.Td([
                            html.Div(tag_badges, className="d-flex flex-wrap gap-1"),
                            html.Div([
                                dbc.Button(
                                    html.I(className="fas fa-tags"),
                                    color="link",
                                    size="sm",
                                    id={
                                        'type': 'edit-tags-btn',
                                        'index': session_id
                                    },
                                    title="Edit Tags",
                                    className="p-0 ms-2",
                                    n_clicks=0  # Explicitly set to 0 to prevent auto-triggering
                                )
                            ])
                        ]),
                        # Actions column
                        html.Td([
                            dbc.ButtonGroup([
                                dbc.Button(
                                    html.I(className="fas fa-sync-alt"),
                                    color="primary",
                                    outline=True,
                                    size="sm",
                                    id={
                                        'type': 'sync-doc-btn',
                                        'index': session_id
                                    },
                                    className="me-2",
                                    title="Synchronize Files",
                                    n_clicks=0  # Explicitly set to 0
                                ),
                                dbc.Button(
                                    html.I(className="fas fa-trash"),
                                    color="danger",
                                    outline=True,
                                    size="sm",
                                    id={
                                        'type': 'delete-doc-btn',
                                        'index': session_id
                                    },
                                    title="Delete Document",
                                    n_clicks=0  # Explicitly set to 0
                                )
                            ])
                        ])
                    ])
                )
            except Exception as e:
                logger.error(f"settings_service.py - update_table_with_tags : Error building table row for document {session_id}: {str(e)}")
                traceback.print_exc()

    if not new_table_body:
        new_table_body = [html.Tr([
            html.Td(
                "No documents found. Click 'Upload New Document' to add some!",
                colSpan=7,
                className="text-center text-muted py-4"
            )
        ])]
        
    return new_table_body

def register_document_view_callbacks(app):
    @app.callback(
        [
        Output("doc-table-body", "children", allow_duplicate=True),
        Output("document-state-persistent", "data", allow_duplicate=True),
        Output("vectorstore-state-persistent", "data", allow_duplicate=True),
        Output("chunk-mapping-state-persistent", "data", allow_duplicate=True),
        Output("upload-modal-persistent", "is_open", allow_duplicate=True),
        Output("upload-progress-bar-persistent", "value", allow_duplicate=True),
        Output("upload-progress-bar-persistent", "style", allow_duplicate=True),
        Output("progress-status-persistent", "children", allow_duplicate=True),
        Output("progress-status-persistent", "style", allow_duplicate=True),
        Output("upload-error-persistent", "children", allow_duplicate=True),
        Output("redirect", "pathname", allow_duplicate=True),
        Output("document-tags-state", "data", allow_duplicate=True),
        Output("upload-spinner", "style", allow_duplicate=True),
        Output("upload-document-persistent", "style", allow_duplicate=True),
        Output("progress-interval-persistent", "disabled", allow_duplicate=True)
        ],
        [   
        Input("doc-list-upload-btn-persistent", "n_clicks"),
        Input("upload-document-persistent", "contents")
        ],
        [
        State("upload-document-persistent", "filename"),
        State("doc-table-body", "children"),
        State("document-state-persistent", "data"),
        State("vectorstore-state-persistent", "data"),
        State("chunk-mapping-state-persistent", "data"),
        State("auth-state", "data"),
        State("upload-modal-persistent", "is_open"),
        State("document-tags-state", "data")
        ],
        prevent_initial_call=True
    )
    def handle_document_upload(upload_btn_clicks, contents, filenames, table_body, doc_state, vstore_state, chunk_state, auth_state, modal_is_open, tags_data):
        triggered = ctx.triggered_id
        
        # Initialize tags_data if not present
        if not tags_data:
            tags_data = {}
                
        # Handle upload button click
        if triggered == "doc-list-upload-btn-persistent":
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                not modal_is_open,
                0,
                {"display": "none"},
                "",
                {"display": "none"},
                "",
                dash.no_update,
                dash.no_update,
                {"display": "none"},
                {},
                True
            )
        # Handle file upload
        if not contents or not filenames:
            raise PreventUpdate

        try:
            # Initialize states
            doc_state = doc_state or {}
            vstore_state = vstore_state or {}
            chunk_state = chunk_state or {}
            current_user_id = auth_state.get('user_id') if auth_state else None
            
            if not current_user_id:
                raise ValueError("User not authenticated")
            
            engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
            DBSession = sessionmaker(bind=engine)
            db_session = DBSession()
            group_name = get_group_for_user(current_user_id, db_session)

            # Ensure contents and filenames are lists
            if not isinstance(contents, list):
                contents = [contents]
                filenames = [filenames]

            vect_serv = VectorStoreService()
            
            # Process each file
            for i, (content, filename) in enumerate(zip(contents, filenames)):
                try:
                    progress = (i + 1) / len(contents) * 100
                    # Decode base64 content
                    content_type, content_string = content.split(',')
                    decoded = base64.b64decode(content_string)
                    
                    # Save original file
                    file_path = save_file(decoded, filename, current_user_id, 'original', group_name)
                    
                    # Process document
                    if filename.lower().endswith('.pdf'):
                        status_text = f"Processing PDF: {filename}"
                        session_id, chunk_mapping = vect_serv.create_vectorstore_and_mapping(
                            content, 
                            filename,
                            user_id=current_user_id
                        )
                        doc_content = content
                    else:
                        status_text = f"Processing document: {filename}"
                        _, _, _, plain_text = parse_contents(content, filename)
                        session_id, chunk_mapping = vect_serv.create_vectorstore_and_mapping(
                            plain_text, 
                            filename,
                            user_id=current_user_id
                        )
                        doc_content = plain_text
                    
                    # Save chunk mapping
                    chunk_file = f"{filename}_chunks.json"
                    save_file(
                        json.dumps(chunk_mapping).encode('utf-8'),
                        chunk_file,
                        current_user_id,
                        'chunks',
                        group_name
                    )
                    
                    # Update states with user ID and file paths
                    doc_state[session_id] = {
                        "filename": filename,
                        "content": doc_content,
                        "timestamp": datetime.now().isoformat(),
                        "user_id": current_user_id,
                        "file_path": str(file_path),
                        "chunk_path": str(chunk_file),
                        "vector_store_id": session_id
                    }
                    
                    # Add default tag for the new document
                    tags_data[session_id] = ["Untagged"]
                    
                    vstore_state[session_id] = session_id
                    chunk_state[session_id] = json.dumps(chunk_mapping)
                except Exception as e:
                    logger.info(f"Error processing {filename}: {str(e)}")
                    traceback.print_exc()
                    return (
                        table_body,
                        doc_state,
                        vstore_state,
                        chunk_state,
                        False,
                        0,
                        {"display": "none"},
                        f"Error: {str(e)}",
                        {"display": "block"},
                        f"Error processing {filename}: {str(e)}",
                        dash.no_update,
                        tags_data,
                        {"display": "none"},
                        {},
                        True
                    )

            # Save updated tags data
            save_document_tags(current_user_id, tags_data)

            # Update table body with tags
            new_table_body = update_table_with_tags(doc_state, tags_data, current_user_id)

            return (
                new_table_body,
                doc_state,
                vstore_state,
                chunk_state,
                False,              # Close modal
                100,                # Set progress to 100%
                {"display": "none"},
                "Upload complete!",
                {"display": "block", "color": "green", "fontWeight": "bold"},
                "",
                "/documents",       # Redirect to documents page to refresh
                tags_data,          # Return updated tags data
                {"display": "none"},# Hide spinner
                {},                 # Reset upload area display
                True                # Disable interval
            )
                
        except Exception as e:
            logger.error(f"settings_service.py - register_document_view_callbacks - handle_document_view_callbacks : Error in document view upload: {str(e)}")
            traceback.print_exc()
            return (
                table_body,
                doc_state,
                vstore_state,
                chunk_state,
                False,
                0,
                {"display": "none"},
                f"Error: {str(e)}",
                {"display": "block"},
                f"Error: {str(e)}",
                dash.no_update,
                tags_data,
                {"display": "none"},
                {},
                True
            )

    # Add admin check to delete button callback
    @app.callback(
        [
            Output("delete-confirm-modal", "is_open"),
            Output("delete-doc-id-persistent", "data"),
            Output("upload-error-persistent", "children")  # Add error output
        ],
        [Input({"type": "delete-doc-btn", "index": ALL}, "n_clicks")],
        [
            State("delete-confirm-modal", "is_open"),
            State("is-admin-state", "data")  # This now contains our expanded admin check
        ],
        prevent_initial_call=True
    )
    def toggle_delete_modal(delete_clicks, is_open, is_admin):
        if not any(click for click in delete_clicks if click):
            raise PreventUpdate
                
        # Check if user is admin or group admin
        if not is_admin:
            return False, None, "Error: Only administrators can delete documents."
                
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "delete-doc-btn":
            return True, triggered["index"], ""
                
        return False, None, ""
    
    # Add admin check to upload button callback
    @app.callback(
        [
            Output("upload-modal-persistent", "is_open", allow_duplicate=True),
            Output("upload-error-persistent", "children", allow_duplicate=True)
        ],
        [Input("doc-list-upload-btn-persistent", "n_clicks")],
        [
            State("upload-modal-persistent", "is_open"),
            State("is-admin-state", "data")  # This now contains our expanded admin check
        ],
        prevent_initial_call=True
    )
    def toggle_upload_modal(n_clicks, is_open, is_admin):
        if n_clicks is None:
            return is_open, ""
        
        if n_clicks:
            # Only open modal if user is admin or group admin (now correctly checked)
            if is_admin:
                return not is_open, ""
            else:
                return False, "Error: Only administrators can upload documents."
        return is_open, ""
    
    
    # Add a callback that checks admin status on direct document page load
    @app.callback(
        Output("is-admin-state", "data", allow_duplicate=True),
        [Input("url", "pathname")],
        [State("auth-state", "data")],
        prevent_initial_call=True
    )
    def check_admin_on_page_load(pathname, auth_state):
        """Check admin status when navigating to document page"""
        if pathname != "/documents":
            raise PreventUpdate
            
        current_user_id = auth_state.get('user_id') if auth_state else None
        
        if not current_user_id:
            logger.info("No user_id found for admin check")
            return False
            
        logger.error(f"settings_service.py - register_document_view_callbacks - check_admin_on_page_load : Checking admin status on page load for user: {current_user_id}")
        
        from app.document_view import is_group_admin, get_current_db_session
        db_session = get_current_db_session()
        
        # Use the is_group_admin function that checks both global admin and group admin status
        admin_status = is_group_admin(current_user_id, db_session)
        
        logger.info(f"Admin check result: {admin_status}")
        
        return admin_status

    @app.callback(
        [
            Output("doc-table-body", "children", allow_duplicate=True),
            Output("document-state-persistent", "data", allow_duplicate=True),
            Output("vectorstore-state-persistent", "data", allow_duplicate=True),
            Output("chunk-mapping-state-persistent", "data", allow_duplicate=True),
            Output("delete-confirm-modal", "is_open", allow_duplicate=True),
            Output("delete-doc-id-persistent", "data", allow_duplicate=True),
            Output("document-tags-state", "data", allow_duplicate=True),
            Output("deletion-complete-flag", "data"),
            Output("delete-progress-interval", "disabled", allow_duplicate=True)
        ],
        [
            Input("delete-status-store", "data"),
            Input("cancel-delete-btn", "n_clicks")
        ],
        [
            State("delete-doc-id-persistent", "data"),
            State("document-state-view", "data"),
            State("vectorstore-state-persistent", "data"),
            State("chunk-mapping-state-persistent", "data"),
            State("auth-state", "data"),
            State("document-tags-state", "data"),
            State("deletion-complete-flag", "data"),  # Add this state to check if we've already deleted
            State("is-admin-state", "data")
        ],
        prevent_initial_call=True
    )
    def handle_document_deletion(status_data, cancel_clicks, doc_id, doc_state, vstore_state, 
                            chunk_state, auth_state, tags_data, deletion_flag, is_admin):
        # Get trigger information
        triggered = ctx.triggered_id
        logger.info(f"settings_service.py - register_document_view_callbacks - handle_document_deletion : Deletion callback triggered by: {triggered}")
        if not is_admin:
            # Return an error message for non-admins trying to upload
            return (
                table_body,
                doc_state,
                vstore_state,
                chunk_state,
                False,              # Close modal
                0,                  # Set progress to 0
                {"display": "none"},
                "",
                {"display": "none"},
                "Error: Only administrators can upload documents.",
                dash.no_update,     # Don't redirect
                tags_data,          # Keep existing tags
                {"display": "none"},# Hide spinner
                {},                 # Reset upload area display
                True                # Disable interval
            )

        # Check if we've already completed this deletion
        if deletion_flag and deletion_flag.get("action") in ["success", "not_found", "error"]:
            logger.info(f"settings_service.py - register_document_view_callbacks - handle_document_deletion : Deletion already completed with status: {deletion_flag.get('action')}")
            raise PreventUpdate
        
        # Handle cancel button
        if triggered == "cancel-delete-btn" and cancel_clicks:
            return (
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                False, None, dash.no_update, {"action": "cancel", "doc_id": doc_id},
                True  # Disable interval
            )
        
        # Handle deletion progress
        if triggered == "delete-status-store":
            progress = status_data.get("progress", 0)
            logger.info(f"Deletion progress: {progress}%")
            
            # Only process deletion when progress reaches 100%
            if progress < 100:
                raise PreventUpdate
            
            logger.info("settings_service.py - register_document_view_callbacks - handle_document_deletion : Deletion progress complete, executing actual deletion...")
            
            # Now we continue with actual deletion
            try:
                # First, check if we're dealing with None values
                if not doc_state:
                    doc_state = {}
                if not vstore_state:
                    vstore_state = {}
                if not chunk_state:
                    chunk_state = {}
                if not tags_data:
                    tags_data = {}
                
                logger.info(f"Starting deletion process for document ID: {doc_id}")
                
                current_user_id = auth_state.get('user_id') if auth_state else None
                
                if not current_user_id:
                    raise ValueError("User not authenticated")
                
                if doc_id in doc_state:
                    doc_info = doc_state[doc_id]
                    
                    # Verify user ownership
                    if doc_info.get('user_id') != current_user_id:
                        logger.info(f"User {current_user_id} does not own document {doc_id}")
                        raise PreventUpdate
                    
                    # Get file details
                    filename = doc_info['filename']
                    logger.info(f"Deleting document: {filename}")
                    
                    # Get the vector store service
                    from services.vector_store import VectorStoreService
                    vect_serv = VectorStoreService()
                    
                    # Clean up vector store files
                    try:
                        logger.info(f"settings_service.py - register_document_view_callbacks - handle_document_deletion : Cleaning up vector store for session ID: {doc_id}")
                        vect_serv.cleanup_vectorstore(doc_id, user_id=current_user_id)
                    except Exception as e:
                        logger.error(f"settings_service.py - register_document_view_callbacks - handle_document_deletion : Error cleaning up vector store: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    # Cleanup chunkmap and original files
                    from app.storage_config import VECTOR_STORE_DIR, ORIGINAL_FILES_DIR, CHUNK_MAPS_DIR, get_group_for_user
                    group_name = get_group_for_user(current_user_id, db_session)
                    user_original_dir = ORIGINAL_FILES_DIR / str(group_name)
                    user_original_chunk = CHUNK_MAPS_DIR / str(group_name)
                    original_file = Path(f"{user_original_dir}/{filename}")
                    original_chunk = Path(f"{user_original_chunk}/{filename}_chunks.json")
                    delete_file_or_folder(original_file)
                    delete_file_or_folder(original_chunk)
                    
                    # Remove from state
                    doc_state.pop(doc_id, None)
                    vstore_state.pop(doc_id, None)
                    chunk_state.pop(doc_id, None)
                    
                    # Also remove tags for this document
                    if doc_id in tags_data:
                        tags_data.pop(doc_id, None)
                        # Save updated tags data
                        save_document_tags(current_user_id, tags_data)
                    
                    # Also check for filename in tags
                    if filename in tags_data:
                        tags_data.pop(filename, None)
                        # Save updated tags data
                        save_document_tags(current_user_id, tags_data)

                    # Update table body with current doc state and tags
                    new_table_body = update_table_with_tags(doc_state, tags_data, current_user_id)
                    
                    logger.info(f"settings_service.py - register_document_view_callbacks - handle_document_deletion : Deletion of document {doc_id} completed successfully")
                    return (
                        new_table_body, doc_state, vstore_state, chunk_state, 
                        False, None, tags_data, {"action": "success", "doc_id": doc_id},
                        True  # Disable interval
                    )
                else:
                    logger.info(f"settings_service.py - register_document_view_callbacks - handle_document_deletion : Document ID {doc_id} not found in document state")
                    return (
                        dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                        False, None, dash.no_update, {"action": "not_found", "doc_id": doc_id},
                        True  # Disable interval
                    )
                    
            except Exception as e:
                logger.info(f"Error deleting document: {str(e)}")
                import traceback
                traceback.print_exc()
                return (
                    dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                    False, None, dash.no_update, {"action": "error", "doc_id": doc_id, "error": str(e)},
                    True  # Disable interval
                )
        
        # Default case
        return (
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
            dash.no_update
        )
    
    @app.callback(
        [
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Output("removed-groups-state", "data", allow_duplicate=True)
        ],
        [Input({"type": "remove-group", "index": ALL}, "n_clicks")],
        [
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("auth-state", "data"),
            State("removed-groups-state", "data")
        ],
        prevent_initial_call=True
    )
    def handle_group_removal(group_clicks, doc_state, vstore_state, chunk_state, chat_history, auth_state, removed_groups):
        """Remove all documents from a specific group from the current session only"""
        if not any(group_clicks) or not doc_state:
            raise PreventUpdate
        
        triggered_id = ctx.triggered_id
        if not isinstance(triggered_id, dict) or triggered_id.get("type") != "remove-group":
            raise PreventUpdate
        
        group_name = triggered_id.get("index")
        
        # Initialize removed_groups if None
        removed_groups = removed_groups or []
        
        try:
            # Initialize result states
            updated_doc_state = doc_state.copy()
            updated_vstore_state = vstore_state.copy()
            updated_chunk_state = chunk_state.copy()
            
            # Prepare a dictionary to store removed group information
            removed_group_info = {
                "name": group_name,
                "documents": {},
                "vectorstore": {}
            }
            
            # Identify documents to remove from the current session
            sessions_to_remove = []
            for session_id, info in list(doc_state.items()):
                if info.get('group_name') == group_name:
                    # Mark for removal from current session
                    sessions_to_remove.append(session_id)
                    
                    # Store document and vectorstore state
                    removed_group_info["documents"][session_id] = info.copy()
                    
                    # Store vectorstore state if exists
                    if session_id in vstore_state:
                        removed_group_info["vectorstore"][session_id] = vstore_state[session_id]
            
            # Remove identified sessions from chunk mapping
            for session_id in sessions_to_remove:
                updated_doc_state.pop(session_id, None)
                updated_vstore_state.pop(session_id, None)
                updated_chunk_state.pop(session_id, None)
            
            # Add the group to removed groups if not already present
            existing_removed_groups = [group.get('name') for group in removed_groups]
            if group_name not in existing_removed_groups:
                removed_groups.append(removed_group_info)
            
            # Add notification to chat history
            system_notification = html.Div(
                f"Removed {len(sessions_to_remove)} documents from group: {group_name}",
                className="alert alert-info"
            )
            chat_history.append(system_notification)
            
            return (
                updated_doc_state,
                updated_vstore_state,
                updated_chunk_state,
                chat_history,
                removed_groups
            )
        
        except Exception as e:
            logger.error(f"settings_service.py - register_document_view_callbacks - handle_document_deletion : Error removing group: {str(e)}")
            import traceback
            traceback.print_exc()
            return dash.no_update
        
    @app.callback(
        [
            Output("tag-edit-modal", "is_open"),
            Output("edit-tags-doc-id", "data"),
            Output("tag-input", "value"),
            Output("current-tags-display", "children")
        ],
        [
            Input({"type": "edit-tags-btn", "index": ALL}, "n_clicks"),
            Input("cancel-tag-edit-btn", "n_clicks"),
            Input("save-tags-btn", "n_clicks"),
            Input("deletion-complete-flag", "data")  # Add this input
        ],
        [
            State("tag-edit-modal", "is_open"),
            State("document-tags-state", "data"),
            State("edit-tags-doc-id", "data")
        ],
        prevent_initial_call=True
    )
    def toggle_tag_edit_modal(edit_tag_clicks, cancel_clicks, save_clicks, deletion_flag, 
                            is_open, tags_data, current_doc_id):
        # Add more robust trigger detection
        triggered = ctx.triggered_id
        
        # Print full debugging information
        logger.info(f"Tag modal callback triggered by: {triggered}")
        logger.info(f"Current modal state: {is_open}")
        
        # Debug trigger prop IDs
        if ctx.triggered_prop_ids:
            logger.info(f"Trigger prop IDs: {ctx.triggered_prop_ids}")
            logger.info(f"Trigger prop ID keys: {list(ctx.triggered_prop_ids.keys())}")
        
        # If deletion just happened, prevent opening tag modal
        if triggered == "deletion-complete-flag" and deletion_flag:
            logger.info(f"settings_service.py - register_document_view_callbacks - toggle_tag_edit_modal : Deletion event detected: {deletion_flag}")
            if deletion_flag.get("action") in ["success", "not_found", "error"]:
                # This was triggered by a deletion, don't open the modal
                return False, None, "", ""
            # Otherwise it's just initialization, ignore
            raise PreventUpdate
        
        # Simplified approach: skip the complex pattern matching
        # Instead, directly check if it's an edit button and if there are actual clicks
        if isinstance(triggered, dict) and triggered.get("type") == "edit-tags-btn":
            logger.info(f"settings_service.py - register_document_view_callbacks - toggle_tag_edit_modal : Edit button clicked: {triggered}")
            # Get the document ID from the triggered component
            doc_id = triggered["index"]
            
            # Simple check: if we were triggered by this component, check if it has clicks
            # Find its index in the list of all edit buttons
            button_index = None
            for i, clicks in enumerate(edit_tag_clicks):
                if clicks and clicks > 0:
                    button_index = i
                    break
            
            # If no buttons were actually clicked, prevent update
            if button_index is None:
                logger.info("settings_service.py - register_document_view_callbacks - toggle_tag_edit_modal : No buttons actually clicked, preventing update")
                raise PreventUpdate
                
            # Only open the modal if it's not already open
            modal_open = True
            
            # Get current tags for this document
            if tags_data and doc_id in tags_data:
                current_tags = tags_data[doc_id]
                tag_input = ", ".join(current_tags)
                
                # Create visual display of current tags
                tag_badges = [create_tag_badge(tag) for tag in current_tags]
                current_tags_display = html.Div([
                    html.Div("Current tags:", className="mb-2 fw-bold"),
                    html.Div(tag_badges, className="d-flex flex-wrap gap-1")
                ])
            else:
                # No tags yet
                tag_input = ""
                current_tags_display = html.Div("No tags assigned yet")
            
            return modal_open, doc_id, tag_input, current_tags_display
        
        # Handle cancel button click
        elif triggered == "cancel-tag-edit-btn" and cancel_clicks:
            return False, dash.no_update, dash.no_update, dash.no_update
        
        # Handle save button click
        elif triggered == "save-tags-btn" and save_clicks:
            return False, dash.no_update, dash.no_update, dash.no_update
        
        # Default case - no change
        raise PreventUpdate
    
    @app.callback(
        [
            Output("document-tags-state", "data"),
            Output("doc-table-body", "children")
        ],
        [Input("save-tags-btn", "n_clicks")],
        [
            State("tag-input", "value"),
            State("edit-tags-doc-id", "data"),
            State("document-tags-state", "data"),
            State("document-state-view", "data"),
            State("auth-state", "data")
        ],
        prevent_initial_call=True
    )
    def save_tags(save_clicks, tag_input, doc_id, tags_data, doc_state, auth_state):
        if not save_clicks or not doc_id:
            raise PreventUpdate
        
        logger.info(f"settings_service.py - register_document_view_callbacks - save_tags : Saving tags for document {doc_id}")
        logger.info(f"settings_service.py - register_document_view_callbacks - save_tags : Current tags_data: {tags_data}")
        logger.info(f"settings_service.py - register_document_view_callbacks - save_tags : Document state keys: {list(doc_state.keys()) if doc_state else 'None'}")
        
        # Initialize tags data if not present
        if not tags_data:
            tags_data = {}
        
        # Parse tags from input
        if tag_input:
            tags = [tag.strip() for tag in tag_input.split(",") if tag.strip()]
        else:
            tags = ["Untagged"]  # Default tag if none provided
        
        # Update tags data
        logger.info(f"Setting tags for doc_id {doc_id} to {tags}")
        tags_data[doc_id] = tags
        
        # Save to storage
        user_id = auth_state.get('user_id') if auth_state else None
        if user_id:
            # Load existing tags first to avoid overwriting other documents
            existing_tags = load_document_tags(user_id)
            
            # Update only the current document's tags
            existing_tags[doc_id] = tags
            
            # Save back to storage
            save_document_tags(user_id, existing_tags)
            
            # Use the existing tags data for consistency
            tags_data = existing_tags
        
        # Update table body with new tags
        new_table_body = update_table_with_tags(doc_state, tags_data, user_id)
        
        logger.info(f"settings_service.py - register_document_view_callbacks - save_tags : Tags saved successfully. Updated tags_data: {tags_data}")
        return tags_data, new_table_body
    
    @app.callback(
        [
            Output("upload-spinner", "style"),
            Output("upload-document-persistent", "style"),
            Output("progress-interval-persistent", "disabled")
        ],
        [Input("upload-document-persistent", "contents")],
        prevent_initial_call=True
    )
    def show_upload_animation(contents):
        """Show loading animation when file is uploaded"""
        if contents:
            return (
                {"display": "block"},  # Show spinner
                {"display": "none"},   # Hide upload area
                False                 # Enable progress interval
            )
        return (
            {"display": "none"},  # Hide spinner
            {},                   # Default upload area display
            True                  # Disable progress interval
        )

    @app.callback(
        [
            Output("upload-progress-bar-persistent", "value"),
            Output("upload-progress-bar-persistent", "style"),
            Output("progress-status-persistent", "children"),
            Output("progress-status-persistent", "style")
        ],
        [Input("progress-interval-persistent", "n_intervals")],
        [State("upload-progress-bar-persistent", "value")],
        prevent_initial_call=True
    )
    def update_progress_bar(n_intervals, current_value):
        """Update progress bar during file processing"""
        if n_intervals is None:
            raise PreventUpdate
        
        # Simulated progress - increments more slowly as it gets higher
        # to simulate the slowing down of processing larger files
        if current_value < 20:
            new_value = min(current_value + 10, 100)
        elif current_value < 50:
            new_value = min(current_value + 5, 100)
        elif current_value < 80:
            new_value = min(current_value + 2, 100)
        else:
            new_value = min(current_value + 1, 95)  # Never quite reaches 100 until done
        
        # Show appropriate status message
        if new_value < 30:
            status = "Uploading document..."
        elif new_value < 60:
            status = "Processing content..."
        elif new_value < 90:
            status = "Creating vector store..."
        else:
            status = "Finalizing..."
        
        return (
            new_value,
            {"display": "block"},
            status,
            {"display": "block"}
        )
    
    @app.callback(
        [
            Output("upload-success-animation", "style"),
            Output("upload-indicators", "style")
        ],
        [Input("upload-progress-bar-persistent", "value")],
        prevent_initial_call=True
    )
    def show_success_animation(progress_value):
        """Show success animation when upload completes"""
        if progress_value == 100:
            # Small delay to allow seeing the 100% before switching
            # (You would implement this with a more complex pattern in production)
            return (
                {"display": "block"},  # Show success animation
                {"display": "none"}    # Hide indicators
            )
        return (
            {"display": "none"},   # Hide success animation
            {"display": "block"}   # Show indicators
    )

    @app.callback(
        [
            Output("delete-confirmation-content", "style"),
            Output("delete-progress-content", "style"),
            Output("delete-progress-interval", "disabled"),
            Output("delete-modal-progress", "value"),
            Output("delete-modal-status", "children"),
            Output("delete-status-store", "data")
        ],
        [Input("confirm-delete-btn", "n_clicks")],
        [State("delete-status-store", "data")],
        prevent_initial_call=True
    )
    def show_delete_progress(confirm_clicks, status_data):
        """Show deletion progress when confirm button is clicked"""
        if not confirm_clicks:
            raise PreventUpdate
        
        logger.info(f"Delete confirmed with {confirm_clicks} clicks")
        logger.info(f"Starting deletion progress animation")
        
        # Start the deletion process visualization
        return (
            {"display": "none"},  # Hide confirmation content
            {"display": "block"},  # Show progress content
            False,  # Enable progress interval
            0,  # Initial progress value
            "Starting deletion process...",  # Initial status
            {"progress": 0, "status": "Starting deletion process..."}  # Initial status data
        )

    @app.callback(
        Output("delete-success-animation", "style", allow_duplicate=True),
        [Input("deletion-complete-flag", "data")],
        prevent_initial_call=True
    )
    def show_hide_delete_success(deletion_flag):
        """Show success animation when deletion is complete"""
        if not deletion_flag:
            raise PreventUpdate
        
        # Only show for successful deletions
        if deletion_flag.get("action") == "success":
            # Return visible for now, we'll hide it with a separate interval
            return {"display": "block"}
        
        # Default case
        return {"display": "none"}

    @app.callback(
        [
            Output("delete-modal-progress", "value", allow_duplicate=True),
            Output("delete-modal-status", "children", allow_duplicate=True),
            Output("delete-status-store", "data", allow_duplicate=True)
        ],
        [Input("delete-progress-interval", "n_intervals")],
        [
            State("delete-status-store", "data"),
            State("deletion-complete-flag", "data")  # Add this state
        ],
        prevent_initial_call=True
    )
    def update_delete_progress(n_intervals, status_data, deletion_flag):
        """Update deletion progress based on interval ticks"""
        if n_intervals is None:
            raise PreventUpdate
        
        # Check if deletion is already complete
        if deletion_flag and deletion_flag.get("action") in ["success", "not_found", "error", "cancel"]:
            logger.info("Deletion complete, stopping progress updates")
            raise PreventUpdate
        
        # Get current progress
        current_progress = status_data.get("progress", 0)
        
        # If we're already at 100%, don't update further
        if current_progress >= 100:
            logger.info("settings_service.py - register_document_view_callbacks - update_delete_progress : Progress already at 100%, no further updates needed")
            raise PreventUpdate
        
        # Calculate new progress value (simulated)
        if current_progress < 30:
            new_progress = min(current_progress + 15, 100)  # Faster at the beginning
            status = "Removing document data..."
        elif current_progress < 60:
            new_progress = min(current_progress + 10, 100)  # Medium speed
            status = "Cleaning up vector store..."
        elif current_progress < 90:
            new_progress = min(current_progress + 5, 100)  # Slower near the end
            status = "Removing file references..."
        else:
            new_progress = 100
            status = "Finalizing..."
        
        # Update status data
        updated_status_data = {"progress": new_progress, "status": status}
        
        logger.info(f"settings_service.py - register_document_view_callbacks - update_delete_progress : Updated progress to {new_progress}%")
        return new_progress, status, updated_status_data
    
    @app.callback(
        [
            Output("delete-success-animation", "style"),
            Output("delete-confirm-modal", "is_open", allow_duplicate=True)
        ],
        [Input("deletion-complete-flag", "data")],
        prevent_initial_call=True
    )
    def show_delete_success(deletion_flag):
        """Show success animation when deletion is complete"""
        if not deletion_flag:
            raise PreventUpdate
        
        # Only show for successful deletions
        if deletion_flag.get("action") == "success":
            # Close the modal and show success animation
            return {"display": "block"}, False
        
        # Default case
        return {"display": "none"}, dash.no_update
    
    @app.callback(
        [
            Output("delete-success-animation", "style", allow_duplicate=True),
            Output("success-hide-interval", "disabled")
        ],
        [Input("success-hide-interval", "n_intervals")],
        [State("delete-success-animation", "style")],
        prevent_initial_call=True
    )
    def hide_success_animation(n_intervals, current_style):
        """Hide the success animation after a few seconds"""
        if n_intervals is None or not current_style or current_style.get("display") != "block":
            raise PreventUpdate
        
        # After one interval, hide the animation and disable further intervals
        return {"display": "none"}, True
    
    @app.callback(
        [
            Output("deletion-complete-flag", "data", allow_duplicate=True),
            Output("delete-status-store", "data", allow_duplicate=True)
        ],
        [Input("delete-confirm-modal", "is_open")],
        [State("delete-doc-id-persistent", "data")],
        prevent_initial_call=True
    )
    def reset_deletion_state(is_open, doc_id):
        """Reset deletion state when opening modal for a new deletion"""
        if is_open:
            logger.info(f"settings_service.py - register_document_view_callbacks - reset_deletion_state : Resetting deletion state for document: {doc_id}")
            return None, {"progress": 0, "status": ""}
        
        # If closing the modal, don't update
        raise PreventUpdate

def register_document_clientside_callbacks(app):
    """Register clientside callbacks for document view"""
    app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='refresh_document_page'
        ),
        Output('dummy-output', 'children'),
        [Input('upload-document-persistent', 'contents')]
    )

def sync_documents_on_load(pathname, auth_state, document_state):
    """Sync documents based on user's group for any route"""
    # Handle routes other than /documents specifically
    if pathname != '/documents' and pathname != '/':
        raise PreventUpdate
        
    current_user_id = auth_state.get('user_id') if auth_state else None

    if not current_user_id:
        raise PreventUpdate
        
    # Get database session
    from sqlalchemy.orm import sessionmaker
    from auth.models import Base
    engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
    DBSession = sessionmaker(bind=engine)
    db_session = DBSession()

    # Sync user files regardless of current pathname
    synced_files = sync_user_files(current_user_id, db_session)
    
    # Initialize document states
    document_state = {}
    vectorstore_state = {}
    chunk_state = {}
    
    for session_id, info in synced_files.items():
        # Create a serializable version of the document info
        serializable_info = {}
        for key, value in info.items():
            # Convert bytes to base64 string if needed
            if isinstance(value, bytes):
                serializable_info[key] = base64.b64encode(value).decode('utf-8')
            else:
                serializable_info[key] = value
                
        document_state[session_id] = serializable_info
        
        if info.get('vector_store_exists'):
            vectorstore_state[session_id] = session_id
            
        if info.get('chunk_mapping'):
            # Ensure chunk_mapping is JSON serializable
            if isinstance(info['chunk_mapping'], bytes):
                chunk_state[session_id] = base64.b64encode(info['chunk_mapping']).decode('utf-8')
            elif isinstance(info['chunk_mapping'], dict):
                chunk_state[session_id] = json.dumps(info['chunk_mapping'])
            else:
                # Assume it's already a JSON string
                chunk_state[session_id] = info['chunk_mapping']

    return document_state, vectorstore_state, chunk_state

def delete_file_or_folder(path):
    """
    Delete a file or folder at the given path.
    
    Args:
        path (str): Path to the file or folder to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        if os.path.isfile(path):
            # Delete a file
            os.remove(path)
            logger.info(f"File '{path}' has been successfully deleted.")
            return True
        elif os.path.isdir(path):
            # Delete a directory and all its contents
            shutil.rmtree(path)
            logger.info(f"Directory '{path}' and all its contents have been successfully deleted.")
            return True
        else:
            logger.info(f"The path '{path}' does not exist.")
            return False
    except Exception as e:
        logger.error(f"settings_service.py - default- delete_file_or_folder : Error occurred while deleting '{path}': {e}")
        return False
    
def register_auto_sync_callback(app):
    """
    Register a callback to automatically sync documents when the Documents page loads
    """
    from dash import Input, Output, State
    from dash.exceptions import PreventUpdate
    
    @app.callback(
        Output("doc-table-body", "children", allow_duplicate=True),
        [Input("page-loaded-trigger", "data")],
        [
            State("url", "pathname"),
            State("document-state-view", "data"),
            State("auth-state", "data")
        ],
        prevent_initial_call=True
    )
    def auto_sync_documents(trigger_data, pathname, doc_state, auth_state):
        """Automatically sync documents when the Documents page loads"""
        if not trigger_data or pathname != '/documents':
            raise PreventUpdate
            
        try:
            current_user_id = auth_state.get('user_id') if auth_state else None
            
            if not current_user_id or not doc_state:
                raise PreventUpdate
                
            # Get database session
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from auth.config import AUTH_CONFIG
            
            engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
            DBSession = sessionmaker(bind=engine)
            db_session = DBSession()
            
            # Get vector store service
            from services.vector_store import VectorStoreService
            # For each document, check if vector store exists and rebuild if missing
            from app.document_view import load_document_tags
            tags_data = load_document_tags(current_user_id)
            
            logger.info(f"settings_service.py - register_auto_sync_callbacks - auto_sync_documents : Auto-syncing documents for user {current_user_id}")
            updated_docs = 0
            
            for session_id, info in doc_state.items():
                doc_info = doc_state[session_id]
                filename = doc_info['filename']
                if info.get('user_id') == current_user_id and info.get('content'):
                    # Check if files exist
                    from services.vector_store import check_files_exist
                    file_status = check_files_exist(current_user_id, filename, db_session)
                    
                    # If vector store is missing, rebuild it
                    if not file_status['vector_store'] and info.get('content'):
                        logger.info(f"Rebuilding missing vector store for {info['filename']}")
                        vect_serv = VectorStoreService()
                        content = info['content']
                        filename = info['filename']
                        
                        # Recreate vector store
                        vect_serv.create_vectorstore_and_mapping(content, filename, user_id=current_user_id)
                        updated_docs += 1
            
            if updated_docs > 0:
                logger.info(f"settings_service.py - register_auto_sync_callbacks - auto_sync_documents : Auto-synced {updated_docs} documents")
            
            # Update table with current status
            new_table_body = update_table_with_tags(doc_state, tags_data, current_user_id)
            return new_table_body
            
        except Exception as e:
            logger.error(f"settings_service.py - register_document_view_callbacks - auto_sync_documents :  Error in auto_sync_documents: {str(e)}")
            import traceback
            traceback.print_exc()
            raise PreventUpdate

def register_server_group_callbacks(app):
    @app.callback(
        [
            Output("server-group-selector", "options"),
            Output("server-group-selector", "disabled")
        ],
        [Input("tab-server-link", "n_clicks")],
        [
            State("auth-state", "data"),
            State("removed-groups-state", "data")
        ]
    )
    def populate_server_groups(active_tab, auth_state, removed_groups):
        """Populate available server groups when Server tab is selected"""
        logger.info("=" * 50)
        logger.info("Populating Server Groups")
        logger.info(f"Active Tab: {active_tab}")
        logger.info(f"Auth State: {auth_state}")
        logger.info(f"Removed Groups: {removed_groups}")

        if not active_tab or not auth_state:
            logger.info("Preventing update - wrong tab or no auth state")
            raise PreventUpdate

        try:
            from services.group_service import GroupService
            group_service = GroupService()
            
            current_user_id = auth_state.get('user_id')
            
            # Filter for removed groups
            removed_groups = removed_groups or []
            filtered_groups = [
                {"name": group.get('name')} 
                for group in removed_groups
            ]
            
            logger.info(f"Filtered Groups: {filtered_groups}")
            
            # Convert groups to dropdown options
            group_options = [
                {"label": group['name'], "value": group['name']} 
                for group in filtered_groups
            ]
            
            # Disable selector if no groups are available
            disabled = len(group_options) == 0
            
            return group_options, disabled
        
        except Exception as e:
            logger.error(f"document_callbacks.py - register_server_groups_callbacks - populate_server_groups : ERROR in populate_server_groups: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], True
        
    @app.callback(
        [
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Output("server-group-status", "children", allow_duplicate=True),
            Output("expanded-upload-area", "style", allow_duplicate=True),
            Output("removed-groups-state", "data", allow_duplicate=True)
        ],
        [Input("add-server-group-btn", "n_clicks")],
        [
            State("server-group-selector", "value"),
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("auth-state", "data"),
            State("removed-groups-state", "data")
        ],
        prevent_initial_call=True
    )
    def add_server_group(n_clicks, group_name, doc_state, vstore_state, chunk_state, chat_history, auth_state, removed_groups):
        """Add a group from the server, restoring previous vector store"""
        logger.info("=" * 50)
        logger.info("Adding Server Group")
        logger.info(f"Clicks: {n_clicks}")
        logger.info(f"Group Name: {group_name}")

        if not n_clicks or not group_name:
            raise PreventUpdate
        
        try:
            # Initialize states if not existing
            doc_state = doc_state or {}
            vstore_state = vstore_state or {}
            chunk_state = chunk_state or {}
            
            # Find the removed group information
            removed_group_info = None
            updated_removed_groups = []
            for group in removed_groups:
                if group.get('name') == group_name:
                    removed_group_info = group
                else:
                    updated_removed_groups.append(group)
            
            if not removed_group_info:
                raise ValueError(f"Group {group_name} not found in removed groups")
            
            # Restore documents and vectorstore
            for session_id, doc_info in removed_group_info.get('documents', {}).items():
                # Add document back to document state
                doc_state[session_id] = doc_info
                
                # Restore vectorstore state if exists
                if session_id in removed_group_info.get('vectorstore', {}):
                    vstore_state[session_id] = removed_group_info['vectorstore'][session_id]
            
            # Add notification to chat history
            chat_history.append(
                html.Div(
                    f"Restored {len(removed_group_info.get('documents', {}))} documents from group {group_name}",
                    className="alert alert-success"
                )
            )
            
            # Hide the expanded upload area
            upload_area_style = {"display": "none"}
            
            return (
                doc_state, 
                vstore_state, 
                chunk_state,  # Keep chunk mapping empty 
                chat_history, 
                html.Div(f"Group {group_name} restored successfully!", className="text-success"),
                upload_area_style,
                updated_removed_groups
            )
        
        except Exception as e:
            logger.error(f"settings_service.py - register_server_group_callbacks - add_server_group : ERROR adding server group: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return (
                doc_state, 
                vstore_state, 
                chunk_state, 
                chat_history, 
                html.Div(f"Error: {str(e)}", className="text-danger"),
                {"display": "none"},
                removed_groups
            )