__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"
__license__  = "MIT"

# document_view.py

import dash_bootstrap_components as dbc
from dash import html, dcc
from datetime import datetime
from services.vector_store import check_files_exist
from app.storage_config import DOC_STATUS_DIR, BASE_STORAGE_DIR
import json
from pathlib import Path
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from auth.config import AUTH_CONFIG
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
DBSession = sessionmaker(bind=engine)
db_session = DBSession()

def create_status_badge(status: bool, label: str) -> html.Span:
    """Create a status badge for file status"""
    return html.Span(
        label,
        className=f"badge {'bg-success' if status else 'bg-danger'} me-1",
        style={
            "fontSize": "0.75rem",
            "padding": "0.25rem 0.5rem"
        }
    )

def create_tag_badge(tag: str) -> html.Span:
    """Create a badge for document tags"""
    return html.Span(
        tag,
        className=f"badge bg-info me-1",
        style={
            "fontSize": "0.75rem",
            "padding": "0.25rem 0.5rem",
            "backgroundColor": "#17a2b8"
        }
    )

def create_user_flag(label: str) -> html.Span:
    return html.Span(
        label,
        className=f"badge {'bg-success' if label else 'bg-danger'} me-1",
        style={
            "fontSize": "0.75rem",
            "padding": "0.25rem 0.5rem"
        }
    )
def format_size(size_in_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} TB"

def create_tag_filter(tags_data):
    """Create a dropdown filter for document tags"""
    # Collect all unique tags across all documents
    all_tags = set()
    for doc_id, tags in tags_data.items():
        all_tags.update(tags)
    
    # Sort tags alphabetically
    sorted_tags = sorted(list(all_tags))
    
    # Create dropdown options
    options = [{"label": "All Tags", "value": "all"}] + [
        {"label": tag, "value": tag} for tag in sorted_tags
    ]
    
    return html.Div([
        html.Label("Filter by Tag:", className="me-2"),
        dcc.Dropdown(
            id="tag-filter-dropdown",
            options=options,
            value="all",
            clearable=False,
            style={"width": "200px", "display": "inline-block"}
        )
    ], className="mb-3")

def create_upload_modal_content():
    """Create the upload modal content with improved loading animation"""
    return html.Div([
        # Upload area
        dcc.Upload(
            id="upload-document-persistent",
            multiple=True,
            children=html.Div([
                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                html.Div("Drag and Drop Documents Here", className="fw-bold"),
                html.Div("or", className="text-muted my-2"),
                dbc.Button(
                    "Browse Files",
                    color="primary",
                    size="sm",
                    className="mt-1"
                ),
                html.Div(
                    "Supported formats: PDF, TXT, MD, DOCX",
                    className="text-muted small mt-2"
                )
            ], className="upload-content text-center py-4")
        ),
        
        # Loading spinner and progress indicators
        html.Div([
            # Spinner
            html.Div([
                dbc.Spinner(
                    spinner_style={"width": "3rem", "height": "3rem"},
                    color="primary",
                ),
                html.Div("Processing your document...", className="mt-2 text-center")
            ], id="upload-spinner", style={"display": "none"}, className="text-center py-3"),
            
            # Progress bar
            dbc.Progress(
                id="upload-progress-bar-persistent",
                value=0,
                style={"display": "none"},
                className="mt-3",
                animated=True,
                striped=True
            ),
            
            # Status text
            html.Div(
                id="progress-status-persistent",
                children="",
                style={"display": "none"},
                className="mt-2 text-muted small"
            ),
            
            # Error message
            html.Div(
                id="upload-error-persistent",
                className="text-danger mt-2"
            ),
        ], id="upload-indicators"),
        create_upload_success_animation(),
        # Interval for progress updates
        dcc.Interval(
            id="progress-interval-persistent",
            interval=500,  # 500ms interval
            n_intervals=0,
            disabled=True
        )
    ])

def create_document_list_layout(doc_state, user_id):
    """Create a layout for the documents list page"""
    
    COLORS = {
        'primary': '#6b5b95',
        'secondary': '#b8b8d1',
        'background': '#f8f7fc',
        'border': '#e6e4ed',
        'disabled': '#cccccc'  # New color for disabled buttons
    }
    
    # Load document tags from storage
    tags_data = load_document_tags(user_id)
    
    # Check if user is a group admin
    db_session = get_current_db_session()
    is_admin = is_group_admin(user_id, db_session)
    
    # Filter documents for the current user
    user_documents = []
    if doc_state and isinstance(doc_state, dict):
       
        for session_id, info in doc_state.items():
            if isinstance(info, dict) and info.get('filename'):
                if user_id is None or info.get('user_id') == user_id:
                    # Check file statuses
                    file_status = check_files_exist(user_id, session_id, db_session)
                    
                    # Get document tags or use default
                    doc_tags = tags_data.get(session_id, ["Untagged"])
                    
                    doc_info = {
                        'session_id': session_id,
                        'filename': info['filename'],
                        'upload_date': info.get('timestamp', datetime.now().isoformat()),
                        'type': info['filename'].split('.')[-1].upper(),
                        'size': len(info.get('content', '')) if isinstance(info.get('content'), str) else 0,
                        'file_status': file_status,
                        'tags': doc_tags
                    }
                    user_documents.append(doc_info)
    
    # Sort documents by upload date (newest first)
    user_documents.sort(key=lambda x: x['upload_date'], reverse=True)
    
    # Create tag filter component
    tag_filter = create_tag_filter(tags_data)

    # Create upload modal
    upload_modal_content = create_upload_modal_content()
    
    # Create tag edit modal
    tag_edit_modal = dbc.Modal([
        dbc.ModalHeader("Edit Document Tags"),
        dbc.ModalBody([
            html.P("Enter tags separated by commas:"),
            dbc.Input(
                id="tag-input",
                type="text",
                placeholder="e.g. important, report, draft",
                className="mb-3"
            ),
            html.Div(id="current-tags-display", className="mb-3"),
            html.Small(
                "Tags help you organize your documents. You can use them to filter and find documents more easily.",
                className="text-muted"
            )
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "Cancel",
                id="cancel-tag-edit-btn",
                className="me-2"
            ),
            dbc.Button(
                "Save Tags",
                id="save-tags-btn",
                color="primary"
            )
        ]),
    ], id="tag-edit-modal", is_open=False)
    
    # Create table structure
    table_header = html.Thead([
        html.Tr([
            html.Th("File Name"),
            html.Th("Type"),
            html.Th("Size"),
            html.Th("Upload Date"),
            html.Th("Status"),
            html.Th("Tags"),
            html.Th("Actions", style={"width": "150px"})
        ])
    ])
    
    # Create table body
    table_body = []
    for doc in user_documents:
        # Create status badges
        status_badges = [
            create_status_badge(
                doc['file_status']['chunks'],
                "Chunks"
            ),
            create_status_badge(
                doc['file_status']['vector_store'],
                "Vector DB"
            ),
            create_status_badge(
                doc['file_status']['metadata'],
                "Metadata"
            )
        ]
        
        # Create tag badges
        tag_badges = [create_tag_badge(tag) for tag in doc['tags']]
        
        # Set action buttons based on admin status
        if is_admin:
            action_buttons = dbc.ButtonGroup([
                dbc.Button(
                    html.I(className="fas fa-sync-alt"),
                    color="primary",
                    outline=True,
                    size="sm",
                    id={
                        'type': 'sync-doc-btn',
                        'index': doc['session_id']
                    },
                    className="me-2",
                    title="Synchronize Files"
                ),
                dbc.Button(
                    html.I(className="fas fa-trash"),
                    color="danger",
                    outline=True,
                    size="sm",
                    id={
                        'type': 'delete-doc-btn',
                        'index': doc['session_id']
                    },
                    title="Delete Document"
                )
            ])
        else:
            # Disabled buttons for non-admin users
            action_buttons = dbc.ButtonGroup([
                dbc.Button(
                    html.I(className="fas fa-sync-alt"),
                    color="secondary",
                    outline=True,
                    size="sm",
                    disabled=True,
                    className="me-2",
                    title="Synchronize Files (Admin Only)"
                ),
                dbc.Button(
                    html.I(className="fas fa-trash"),
                    color="secondary",
                    outline=True,
                    size="sm",
                    disabled=True,
                    title="Delete Document (Admin Only)"
                )
            ])
        
        table_body.append(html.Tr([
            # File name column
            html.Td([
                html.I(className="fas fa-file me-2"),
                doc['filename']
            ]),
            # Type column
            html.Td(doc['type']),
            # Size column
            html.Td(format_size(doc['size'])),
            # Date column
            html.Td(
                datetime.fromisoformat(doc['upload_date']).strftime("%Y-%m-%d %H:%M")
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
                            'index': doc['session_id']
                        },
                        title="Edit Tags",
                        className="p-0 ms-2"
                    )
                ])
            ]),
            # Actions column
            html.Td(action_buttons)
        ]))
    
    if not table_body:
        table_body = [html.Tr([
            html.Td(
                "No documents found. Click 'Upload New Document' to add some!",
                colSpan=7,
                className="text-center text-muted py-4"
            )
        ])]
    
    # Determine upload button properties based on admin status
    upload_button_color = "primary" if is_admin else "secondary"
    upload_button_style = {
        "backgroundColor": COLORS['primary'] if is_admin else COLORS['disabled'],
        "borderColor": COLORS['primary'] if is_admin else COLORS['disabled'],
    }
    upload_button_disabled = not is_admin
    
    # Add admin status indicator if needed
    admin_status_display = html.Div([
        html.Span(
            "Admin Access" if is_admin else "Read-Only Access", 
            className=f"badge {'bg-success' if is_admin else 'bg-warning'} me-2"
        )
    ], className="mb-3") if not is_admin else None
    
    return html.Div([
        # Header section
        html.Div([
            html.H2("My Documents", className="mb-4"),
            admin_status_display,  # Show admin status for non-admin users
            tag_filter,
            html.Div([
            html.Div(
                id="document-list-persistent",
                className="document-list-inline-persistent",
                children=[]
            )
        ], className="d-inline-block ms-3 flex-grow-1"),
            html.Div([
                # Document stats
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Total Documents", className="card-title"),
                                html.H2(str(len(user_documents)), className="mb-0")
                            ])
                        ], className="text-center")
                    , width=4),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Recent Uploads", className="card-title"),
                                html.H2(
                                    str(sum(1 for doc in user_documents 
                                        if datetime.fromisoformat(doc['upload_date']).date() == datetime.now().date())),
                                    className="mb-0"
                                )
                            ])
                        ], className="text-center")
                    , width=4),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("File Types", className="card-title"),
                                html.H2(
                                    str(len(set(doc['type'] for doc in user_documents))),
                                    className="mb-0"
                                )
                            ])
                        ], className="text-center")
                    , width=4),
                ], className="mb-4"),
                
                # Upload button with conditional styling and disabling
                dbc.Button([
                    html.I(className="fas fa-upload me-2"),
                    "Upload New Document"
                ],
                id="doc-list-upload-btn-persistent",
                color=upload_button_color,
                style=upload_button_style,
                disabled=upload_button_disabled,  # Disable if not admin
                className="mb-4"
                ),
                
                # Add tooltip for non-admin users explaining why button is disabled
                dbc.Tooltip(
                    "Only group administrators can upload documents",
                    target="doc-list-upload-btn-persistent",
                    placement="top",
                    is_open=not is_admin
                ) if not is_admin else None,
            ]),
            
            # Document list table
            html.Div(
                dbc.Table(
                    [table_header, html.Tbody(table_body, id="doc-table-body")],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    className="mb-4"
                ),
                id="doc-table-container",
                className="table-responsive"
            ),
        ], className="p-4", style={"backgroundColor": "white", "borderRadius": "8px"}),
        
        html.Div([
            # Spinner
            html.Div([
                dbc.Spinner(
                    spinner_style={"width": "3rem", "height": "3rem"},
                    color="danger",  # Red color to indicate deletion
                ),
                html.Div("Deleting document...", className="mt-2 text-center")
            ], id="delete-spinner", style={"display": "none"}, className="text-center py-3"),
            
            # Progress bar
            dbc.Progress(
                id="delete-progress-bar",
                value=0,
                style={"display": "none"},
                className="mt-3",
                animated=True,
                striped=True,
                color="danger"  # Red color for deletion
            ),
            
            # Status text
            html.Div(
                id="delete-status",
                children="",
                style={"display": "none"},
                className="mt-2 text-muted small"
            ),
            
            # Success animation (checkmark)
            html.Div([
                html.I(className="fas fa-check-circle fa-3x text-success"),
                html.Div("Document Successfully Deleted!", className="mt-2 fw-bold"),
                html.Div("The document has been removed from your library.", className="mt-1 text-muted small")
            ], id="delete-success-animation", style={"display": "none"}, className="text-center py-4"),
            
            # Interval for progress updates
            dcc.Interval(
                id="delete-progress-interval",
                interval=300,  # 300ms interval (faster than upload for better UX)
                n_intervals=0,
                disabled=True
            ),
            
            # Store for delete status
            dcc.Store(id="delete-status-store", data={"progress": 0, "status": ""})
        ], id="delete-indicators", style={"display": "none"}),

        # Modals
        dbc.Modal([
            dbc.ModalHeader("Upload New Document"),
            dbc.ModalBody(upload_modal_content),
        ], id="upload-modal-persistent", is_open=False),
        
        dbc.Modal([
            dbc.ModalHeader("Confirm Deletion"),
            
            # Normal confirmation content
            html.Div([
                dbc.ModalBody([
                    html.P("Are you sure you want to delete this document? This action cannot be undone."),
                    html.P("All associated data including vector stores and metadata will be removed.", 
                        className="text-danger")
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Cancel",
                        id="cancel-delete-btn",
                        className="me-2"
                    ),
                    dbc.Button(
                        "Delete",
                        id="confirm-delete-btn",
                        color="danger"
                    ),
                ]),
            ], id="delete-confirmation-content"),
            
            # This div will show during deletion process
            html.Div([
                dbc.ModalBody([
                    # Center all content
                    html.Div([
                        # Spinner
                        dbc.Spinner(
                            spinner_style={"width": "3rem", "height": "3rem"},
                            color="danger",
                        ),
                        html.Div("Deleting document...", className="mt-3 fw-bold"),
                        
                        # Progress bar
                        dbc.Progress(
                            id="delete-modal-progress",
                            value=0,
                            className="mt-3",
                            animated=True,
                            striped=True,
                            color="danger"
                        ),
                        
                        # Status text
                        html.Div(
                            id="delete-modal-status",
                            children="",
                            className="mt-2 text-muted"
                        ),
                    ], className="text-center py-3"),
                ]),
            ], id="delete-progress-content", style={"display": "none"}),
        ], id="delete-confirm-modal", is_open=False),
        
        # Tag edit modal
        tag_edit_modal,
        
        # Stores
        dcc.Store(id="document-state-persistent", storage_type="memory"),
        dcc.Store(id="vectorstore-state-persistent", storage_type="memory"),
        dcc.Store(id="chunk-mapping-state-persistent", storage_type="memory"),
        dcc.Store(id="delete-doc-id-persistent", data=None),
        dcc.Store(id="edit-tags-doc-id", data=None),
        dcc.Store(id="document-tags-state", data=tags_data),
        dcc.Store(id="upload-trigger-persistent", data=None),
        dcc.Store(id="upload-status-persistent", data={"progress": 0, "status": ""}),
        dcc.Location(id='redirect', refresh=True),
        html.Div(id='dummy-output', style={'display': 'none'}),
        dcc.Store(id="deletion-complete-flag", data=None),
        # Add a store to track admin status
        dcc.Store(id="is-admin-state", data=is_admin)
    ])

def is_group_admin(user_id, db_session):
    """
    Check if a user is an admin of any group
    
    Args:
        user_id (str or int): The user ID to check
        db_session: The database session to use
        
    Returns:
        bool: True if the user is a group admin, False otherwise
    """
    try:
        # Import here to avoid circular imports
        from auth.group_management import GroupService
        
        # First check if user is a global admin
        from auth.models import User
        user = db_session.query(User).filter(User.id == user_id).first()
        if user and user.role == 'admin':
            logger.info(f"User {user_id} is a global admin")
            return True
        
        # Get group service and check if user is an admin of any group
        group_service = GroupService()
        
        # Convert user_id to string (it might be an integer in some contexts)
        user_id_str = str(user_id)
        
        # Get all groups and check if user is an admin of any
        for group_id, group_info in group_service.groups.items():
            group_admins = group_info.get('group_admins', [])
            if user_id_str in group_admins:
                logger.info(f"document_view.py - is_group_admin : User {user_id} is an admin of group {group_id}")
                return True
        
        logger.error(f"document_view.py - is_group_admin : User {user_id} is not an admin of any group")
        return False
        
    except Exception as e:
        logger.error(f"Error checking group admin status: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def load_document_tags(user_id):
    """Load document tags from storage"""
    TAGS_DIR = Path(BASE_STORAGE_DIR) / "tags"
    
    # Ensure tags directory exists
    os.makedirs(TAGS_DIR, exist_ok=True)
    
    tags_file = TAGS_DIR / f"user_{user_id}_tags.json"
    
    if tags_file.exists():
        try:
            with open(tags_file, 'r') as f:
                tags_data = json.load(f)
                logger.info(f"document_view.py - load_document_tags : Loaded tags from {tags_file}: {tags_data}")
                return tags_data
        except Exception as e:
            logger.error(f"document_view.py - load_document_tags ~1: Error loading tags from {tags_file}: {str(e)}")
            return {}
    
    logger.error(f"document_view.py - load_document_tags ~2 : No tags file found at {tags_file}")
    return {}

def save_document_tags(user_id, tags_data):
    """Save document tags to storage"""
    TAGS_DIR = Path(BASE_STORAGE_DIR) / "tags"
    
    # Ensure tags directory exists
    os.makedirs(TAGS_DIR, exist_ok=True)
    
    tags_file = TAGS_DIR / f"user_{user_id}_tags.json"
    
    # Debug what we're saving
    logger.info(f"Saving tags to {tags_file}: {tags_data}")
    
    try:
        with open(tags_file, 'w') as f:
            json.dump(tags_data, f)
        logger.info(f"document_view.py - save_document_tags : Tags saved successfully to {tags_file}")
        return True
    except Exception as e:
        logger.error(f"document_view.py - save_document_tags : Error saving tags to {tags_file}: {str(e)}")
        return False

def create_upload_success_animation():
    """Create a success animation component for completed uploads"""
    return html.Div([
        html.Div([
            html.I(className="fas fa-check-circle fa-3x text-success"),
            html.Div("Upload Successful!", className="mt-2 fw-bold"),
            html.Div("Your document has been processed and is ready to use.", className="mt-1 text-muted small")
        ], className="text-center py-4")
    ], id="upload-success-animation", style={"display": "none"})

def get_current_db_session():
    """Get a new database session"""
    from sqlalchemy.orm import sessionmaker
    from auth.models import Base
    engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
    DBSession = sessionmaker(bind=engine)
    return DBSession()