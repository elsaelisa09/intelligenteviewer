# grouped_documents.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from dash import html, dcc, dash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import traceback
import json
from services.vector_store import VectorStoreService
import dash
from dash import Input, Output, State, ctx, ALL, MATCH, callback_context
from dash.exceptions import PreventUpdate
import traceback
import json
from services.vector_store import VectorStoreService

def create_group_document_item(group_name, doc_count, is_active=True):
    """Create a grouped document item for the document list with improved button functionality"""
    import dash_bootstrap_components as dbc
    from dash import html
    
    # Generate a unique ID for this group
    group_id = f"group-{group_name}"
    
    return html.Div(
        [
            html.Div(
                [
                    # Group icon with folder appearance
                    html.I(className="fas fa-folder me-2", title="Document Group"),
                    
                    # Group name and document count
                    html.Span([
                        html.Span(group_name, className="fw-bold"),
                        html.Span(f" ({doc_count} documents)", className="text-muted small")
                    ], className="text-truncate", style={"maxWidth": "200px"}),
                    
                    # Toggle switch for including/excluding the group
                    dbc.Switch(
                        id={'type': 'group-toggle', 'index': group_id},
                        value=is_active,
                        className="ms-2",
                        style={"transform": "scale(0.8)"}
                    ),
                    
                    # Delete button for the entire group - using dbc.Button instead of html.Button
                    dbc.Button(
                        html.I(className="fas fa-trash"),
                        id={'type': 'remove-group', 'index': group_id},
                        color="link",
                        className="text-danger p-0 ms-2",
                        title="Remove entire group",
                        n_clicks=0  # Initialize with zero clicks
                    )
                ],
                className="d-flex align-items-center bg-light rounded px-2 py-1"
            )
        ],
        id={'type': 'group-item', 'index': group_id},
        className="d-inline-block me-2 mb-2",
        style={"whiteSpace": "nowrap"}
    )

def get_documents_by_group(doc_state):
    """Organize documents by group"""
    documents_by_group = {}
    individual_docs = []
    
    for session_id, info in doc_state.items():
        if isinstance(info, dict) and info.get('filename'):
            # Check if this document belongs to a group
            group_name = info.get('group_name')
            is_persistent = info.get('source') in ['folder', 'group'] or 'file_path' in info or 'path' in info
            
            if is_persistent and group_name:
                # Add to group collection
                if group_name not in documents_by_group:
                    documents_by_group[group_name] = []
                documents_by_group[group_name].append((session_id, info))
            else:
                # This is an individually uploaded document
                individual_docs.append((session_id, info))
    
    return documents_by_group, individual_docs

def register_group_operations_callbacks(app):
    """Register callbacks for group-based operations with page guards"""
    
    # Add this helper function for page availability check
    def is_page_available(current_page_id, page_patterns):
        """Check if current page ID matches any of the allowed patterns"""
        if not current_page_id:
            return False
            
        if isinstance(page_patterns, str):
            page_patterns = [page_patterns]
            
        for pattern in page_patterns:
            if pattern in current_page_id:
                return True
                
        return False
    

    @app.callback(
        [
            Output("delete-group-confirm-modal", "is_open"),
            Output("delete-group-id", "data")
        ],
        [
            Input({"type": "remove-group", "index": ALL}, "n_clicks")
        ],
        [
            State("delete-group-confirm-modal", "is_open"),
            State("current-page-id", "children")
        ],
        prevent_initial_call=True
    )
    def toggle_group_delete_modal(remove_clicks, is_open, current_page_id):
        """Open delete confirmation modal for group deletion - simplified version"""
        # Import callback_context directly
        from dash import callback_context as ctx
        
        # Only proceed if we're on the main page
        if current_page_id != "main-page":
            print(f"Not on main page (current: {current_page_id}), skipping modal toggle")
            raise PreventUpdate
        
        # Check if any buttons were clicked
        if not ctx.triggered or not any(click for click in remove_clicks if click):
            print("No remove-group buttons clicked")
            raise PreventUpdate
        
        # Find which button was clicked
        triggered_prop_id = ctx.triggered[0]["prop_id"]
        print(f"Button clicked: {triggered_prop_id}")
        
        # Extract the component ID from the prop_id
        try:
            import json
            import re
            
            # Parse the component ID from the prop_id (prop_id format: '{"type":"remove-group","index":"group-name"}.n_clicks')
            match = re.search(r'(\{.*\})\.', triggered_prop_id)
            if match:
                component_id_str = match.group(1)
                component_id = json.loads(component_id_str)
                
                if component_id.get("type") == "remove-group":
                    group_id = component_id.get("index")
                    if group_id and group_id.startswith("group-"):
                        group_name = group_id[6:]  # Remove 'group-' prefix
                        print(f"Will open delete modal for group: {group_name}")
                        return True, group_name
        except Exception as e:
            print(f"Error parsing component ID: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Default: no change
        return is_open, None
    
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
            Output("upload-trigger", "data", allow_duplicate=True)
        ],
        [Input("delete-group-id", "data")],
        [
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("document-selector", "options"),
            State("document-selector", "value"),
            State("auth-state", "data"),
            State("current-page-id", "children")  # Check current page
        ],
        prevent_initial_call=True
    )
    def delete_group_documents(group_name, doc_state, vstore_state, chunk_state, 
                            chat_history, current_options, current_selected, auth_state, current_page_id):
        """Delete all documents in a group"""
        from dash import callback_context as ctx
        
        # Only proceed if we're on the main page
        if current_page_id != "main-page":
            print(f"Not on main page (current: {current_page_id}), skipping group deletion")
            raise PreventUpdate
            
        # Check if the callback was actually triggered with valid data
        if not ctx.triggered or not group_name or not doc_state:
            print("Group deletion callback triggered but no valid group name or document state")
            raise PreventUpdate
            
        print(f"Deleting all documents in group: {group_name}")
        
        try:
            # Get the user ID
            current_user_id = auth_state.get('user_id') if auth_state else None
            if not current_user_id:
                print("No user ID found, cannot delete group documents")
                raise PreventUpdate
                
            # Create database session
            from sqlalchemy.orm import sessionmaker
            from auth.config import AUTH_CONFIG
            from sqlalchemy import create_engine
            engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
            DBSession = sessionmaker(bind=engine)
            db_session = DBSession()
            
            # Create vector store service
            vect_serv = VectorStoreService()
            
            # Find all documents in this group
            docs_to_delete = []
            for session_id, info in doc_state.items():
                if isinstance(info, dict) and info.get('group_name') == group_name:
                    docs_to_delete.append(session_id)
            
            print(f"Found {len(docs_to_delete)} documents to delete in group {group_name}")
            
            # Delete each document
            for session_id in docs_to_delete:
                try:
                    # Clean up vector store
                    vect_serv.cleanup_vectorstore(session_id, user_id=current_user_id, db_session=db_session)
                    
                    # Remove from state
                    doc_state.pop(session_id, None)
                    vstore_state.pop(session_id, None)
                    chunk_state.pop(session_id, None)
                    
                    print(f"Deleted document {session_id} from group {group_name}")
                except Exception as e:
                    print(f"Error deleting document {session_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Update document list
            from app.layout import create_document_item
            
            # Organize remaining documents by group
            documents_by_group, individual_docs = get_documents_by_group(doc_state)
            
            # Create new document list
            document_list = []
            
            # Add groups first
            for g_name, documents in documents_by_group.items():
                # Create group item
                group_item = create_group_document_item(g_name, len(documents))
                document_list.append(group_item)
                
                # Add individual documents with group indicator
                for s_id, info in documents:
                    doc_item = create_document_item(
                        info['filename'], 
                        s_id, 
                        is_persistent=True,
                        group_name=g_name
                    )
                    document_list.append(doc_item)
            
            # Add individually uploaded documents
            for s_id, info in individual_docs:
                doc_item = create_document_item(
                    info['filename'], 
                    s_id, 
                    is_persistent=(info.get('source') in ['folder', 'group'] or 'file_path' in info)
                )
                document_list.append(doc_item)
            
            # Update selector options
            selector_options = [
                {"label": info["filename"], "value": s_id}
                for s_id, info in doc_state.items()
            ]
            
            # Create system notification for chat history
            from utils.visualization import create_system_notification
            notification = create_system_notification(
                f"Removed all documents from group: {group_name}",
                type="success",
                action="remove"
            )
            
            if chat_history:
                chat_history.append(notification)
            else:
                chat_history = [notification]
            
            # Reset viewer
            from app.layout import create_pdf_viewer
            empty_viewer = create_pdf_viewer()
            
            # Generate new trigger value for upload reset
            from datetime import datetime
            trigger_value = datetime.now().isoformat()
            
            return (
                document_list,       # document list
                doc_state,           # document state
                vstore_state,        # vectorstore state
                chunk_state,         # chunk mapping state
                chat_history,        # chat history
                selector_options,    # selector options
                None,                # selector value (reset)
                empty_viewer,        # reset viewer
                None,                # document data
                None,                # highlights
                trigger_value        # upload reset trigger
            )
            
        except Exception as e:
            print(f"Error deleting group documents: {str(e)}")
            import traceback
            traceback.print_exc()
            raise PreventUpdate

    @app.callback(
        Output("document-list", "children", allow_duplicate=True),
        [Input({"type": "group-toggle", "index": ALL}, "value")],
        [
            State("document-state", "data"),
            State("current-page-id", "children")
        ],
        prevent_initial_call=True
    )
    def toggle_group_visibility(toggle_values, doc_state, current_page_id):
        """Toggle visibility of documents in a group"""
        # Import callback_context directly
        from dash import callback_context as ctx
        
        if current_page_id != "main-page":
            raise PreventUpdate
                
        if not doc_state:
            raise PreventUpdate
        
        # Get the trigger information
        triggered = ctx.triggered_id
        
        if not isinstance(triggered, dict) or triggered.get('type') != 'group-toggle':
            raise PreventUpdate
        
        # Get the group name from the component ID
        group_id = triggered.get('index')
        if not group_id or not group_id.startswith('group-'):
            raise PreventUpdate
                
        group_name = group_id[6:]  # Remove the 'group-' prefix
        
        # Get the toggle value from the first triggered item
        is_active = False
        if ctx.triggered and len(ctx.triggered) > 0:
            is_active = ctx.triggered[0]['value']
        
        print(f"Toggle group {group_name} visibility: {is_active}")
        
        # Store all toggle states to preserve other groups' state
        toggle_states = {}
        toggle_states[group_name] = is_active  # Set the toggled group state
        
        # Organize documents by group
        from app.grouped_documents import get_documents_by_group, create_group_document_item
        documents_by_group, individual_docs = get_documents_by_group(doc_state)
        
        # Create the updated document list
        document_list = []
        
        # Add group items
        for g_name, documents in documents_by_group.items():
            # Get the visibility state for this group (default to True if not toggled)
            g_is_active = toggle_states.get(g_name, True)
            if g_name == group_name:
                g_is_active = is_active
            
            # Create group item
            group_item = create_group_document_item(
                g_name, 
                len(documents), 
                is_active=g_is_active
            )
            document_list.append(group_item)
            
            # Only add documents if group is active
            if g_is_active:
                from app.layout import create_document_item
                for session_id, info in documents:
                    doc_item = create_document_item(
                        info['filename'], 
                        session_id, 
                        is_persistent=True,
                        group_name=g_name
                    )
                    document_list.append(doc_item)
        
        # Add individually uploaded documents
        from app.layout import create_document_item
        for session_id, info in individual_docs:
            doc_item = create_document_item(
                info['filename'], 
                session_id, 
                is_persistent=(info.get('source') in ['folder', 'group'] or 'file_path' in info)
            )
            document_list.append(doc_item)
        
        return document_list

    @app.callback(
        [
            Output("delete-group-confirm-modal", "is_open", allow_duplicate=True),
            Output("delete-group-id", "data", allow_duplicate=True)
        ],
        [
            Input("cancel-group-delete-btn", "n_clicks"),
            Input("confirm-group-delete-btn", "n_clicks")
        ],
        [
            State("delete-group-confirm-modal", "is_open"),
            State("delete-group-id", "data"),
            State("current-page-id", "children")
        ],
        prevent_initial_call=True
    )
    def handle_delete_modal_buttons(cancel_clicks, confirm_clicks, is_open, current_group_id, current_page_id):
        """Handle the modal buttons separately for better control"""
        # Import callback_context directly
        from dash import callback_context as ctx
        
        # Only proceed if we're on the main page and the modal is open
        if current_page_id != "main-page" or not is_open:
            raise PreventUpdate
        
        # Determine which button was clicked
        if not ctx.triggered:
            raise PreventUpdate
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "cancel-group-delete-btn" and cancel_clicks:
            print("Cancel button clicked")
            return False, None
        
        if button_id == "confirm-group-delete-btn" and confirm_clicks:
            print(f"Confirm button clicked for group: {current_group_id}")
            # Keep the group ID but close the modal
            # The delete_group_documents callback will pick up the group ID
            return False, current_group_id
        
        # Default: no change
        return is_open, current_group_id