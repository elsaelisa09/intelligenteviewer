__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import List, Dict, Tuple, Optional, Set

from dash import html
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_query_progress():
    """Create a progress indicator for query processing"""
    return html.Div([
        dbc.Progress(
            id="query-progress-bar",
            value=0,
            style={
                "height": "4px",
                "width": "100%",
                "marginBottom": "8px",
                "display": "none"
            },
            className="query-progress"
        ),
        html.Div(
            id="query-status",
            style={
                "fontSize": "0.875rem",
                "color": "#666",
                "marginBottom": "8px",
                "display": "none"
            },
            className="d-flex align-items-center mb-3"
        ),
        dcc.Store(id="query-processing-status", data={"progress": 0, "status": ""}),
        dcc.Interval(
            id="query-progress-interval",
            interval=300,  # 300ms refresh rate
            n_intervals=0,
            disabled=True
        )
    ])

def create_progress_bar():
    """Create a progress bar component for upload and processing status"""
    return html.Div([
        dbc.Progress(
            id="upload-progress-bar",
            value=0,
            style={
                "height": "4px",
                "width": "100%",
                "marginBottom": "8px",
                "display": "none"
            },
            className="upload-progress"
        ),
        html.Div(
            id="progress-status",
            style={
                "fontSize": "0.875rem",
                "color": "#666",
                "marginBottom": "8px",
                "display": "none"
            }
        ),
        dcc.Store(id="upload-status", data={"progress": 0, "status": ""}),
        dcc.Interval(
            id="progress-interval",
            interval=500,  # 500ms refresh rate
            n_intervals=0,
            disabled=True
        )
    ])

def create_text_viewer():
    """Create a viewer component for non-PDF text documents"""
    return html.Div([
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
                'fontFamily': 'monospace',
                'whiteSpace': 'pre-wrap',
                'display': 'none'  # Initially hidden
            }
        ),
        html.Div(
            id='text-highlights',
            style={'display': 'none'}
        )
    ])

def apply_text_highlights(content: str, highlights: List[Dict]) -> str:
    """Apply highlights to text content with improved formatting"""
    if not highlights:
        return content

    try:
        # Convert content to list for manipulation
        content_list = list(content)
        
        # Sort highlights by start position (reverse order to maintain positions)
        sorted_highlights = sorted(highlights, key=lambda x: x['start'], reverse=True)
        
        # Insert highlight markers
        for highlight in sorted_highlights:
            start = highlight['start']
            end = highlight['end']
            
            if 0 <= start < len(content) and start < end <= len(content):
                # Create highlight with unique identifier
                highlight_id = f"highlight-{start}-{end}"
                
                # Determine highlight class based on type
                highlight_class = 'core-highlight' if highlight.get('is_core') else 'text-highlight'
                
                # Insert markers
                content_list.insert(end, '</span>')
                content_list.insert(start, f'<span id="{highlight_id}" class="{highlight_class}">')
        
        # Join content back together
        highlighted_content = ''.join(content_list)
        
        return highlighted_content
        
    except Exception as e:
        logger.debug(f"layout.py - apply_text_highlights : Error applying text highlights: {str(e)}")
        return content

def find_text_positions(content: str, search_text: str) -> List[Dict]:
    """Find all occurrences of text with improved matching"""
    positions = []
    try:
        if not search_text or not content:
            return positions

        search_text = search_text.strip()
        content = content.strip()
        
        # Try exact match first
        start = 0
        while True:
            pos = content.find(search_text, start)
            if pos == -1:
                break
            
            positions.append({
                'start': pos,
                'end': pos + len(search_text),
                'text': search_text
            })
            start = pos + 1
        
        # If no exact matches found, try fuzzy matching
        if not positions:
            words = search_text.split()
            if len(words) >= 3:
                # Try matching with first few words
                partial_text = ' '.join(words[:3])
                start = 0
                while True:
                    pos = content.find(partial_text, start)
                    if pos == -1:
                        break
                    
                    # Find a reasonable end position
                    end_pos = content.find('.', pos)
                    if end_pos == -1:
                        end_pos = min(pos + len(search_text) * 2, len(content))
                    
                    positions.append({
                        'start': pos,
                        'end': end_pos,
                        'text': content[pos:end_pos].strip()
                    })
                    start = pos + 1
        
        return positions
        
    except Exception as e:
        logger.debug(f"layout.py - find_text_positions : Error finding text positions: {str(e)}")
        return positions

def create_pdf_viewer():
    """Create PDF viewer with all necessary stores"""
    return html.Div([
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
                'fontFamily': 'monospace',
                'whiteSpace': 'pre-wrap',
                'display': 'none'
            }
        ),
        
        # All necessary stores
        dcc.Store(id='document-data', storage_type='memory'),
        dcc.Store(id='pdf-highlights', storage_type='memory'),
        dcc.Store(id='text-highlights', storage_type='memory'),  # Add this store
        dcc.Store(id='document-viewer-data', storage_type='memory'),
        
        # Control elements
#        html.Div(id='pdf-highlight-trigger', style={'display': 'none'}),
        html.Div(id='_clear-stores', style={'display': 'none'})
    ])

def create_document_viewer():
    """Create document viewer with complete store components"""
    # Style definitions
    highlight_styles = '''
        .text-highlight {
            background-color: rgba(255, 255, 0, 0.3);
            padding: 2px 0;
            border-radius: 2px;
            transition: background-color 0.2s;
        }
        .text-highlight:hover {
            background-color: rgba(255, 255, 0, 0.5);
        }
        .core-highlight {
            background-color: rgba(255, 200, 0, 0.4);
            padding: 2px 0;
            border-radius: 2px;
            transition: background-color 0.2s;
        }
        .core-highlight:hover {
            background-color: rgba(255, 200, 0, 0.6);
        }
    '''
    
    return html.Div([
        # Add styles
        html.Style(highlight_styles),
        
        # PDF viewer container
        html.Div(
            id="pdf-js-viewer",
            style={
                "width": "100%",
                "height": "700px",
                "display": "none"
            }
        ),
        
        # Text viewer container
        html.Div(
            id="text-viewer",
            style={
                "width": "100%",
                "height": "700px",
                "display": "none"
            }
        ),
        
        # Add ALL necessary stores
        dcc.Store(id='document-viewer-data', storage_type='memory'),
        dcc.Store(id='document-data', storage_type='memory'),  # Add this store
        dcc.Store(id='pdf-highlights', storage_type='memory'),
        dcc.Store(id='text-highlights', storage_type='memory'),
        
        # Hidden div for cleanup
        html.Div(id='_clear-stores', style={'display': 'none'}),
        
        # Add highlight trigger
#        html.Div(id='pdf-highlight-trigger', style={'display': 'none'})
    ], id="document-viewer-container")

def create_group_delete_modal():
    """Create a confirmation modal for group deletion"""
    import dash_bootstrap_components as dbc
    from dash import html, dcc
    
    modal = dbc.Modal([
        dbc.ModalHeader("Confirm Group Deletion"),
        dbc.ModalBody([
            html.P("Are you sure you want to delete all documents in this group?"),
            html.P("This will remove the documents from the current session only.", className="text-muted"),
            html.P("The original files in storage will not be affected.", className="text-muted")
        ]),
        dbc.ModalFooter([
            dbc.Button(
                "Cancel",
                id="cancel-group-delete-btn",
                className="me-2"
            ),
            dbc.Button(
                "Delete",
                id="confirm-group-delete-btn",
                color="danger"
            ),
        ]),
    ], id="delete-group-confirm-modal", is_open=False)
    
    # Add the store for the group ID
    group_id_store = dcc.Store(id="delete-group-id", data=None)
    
    return html.Div([modal, group_id_store])

def create_document_item(filename, session_id, is_persistent=False, group_name=None):
    """Create a document list item with group indicators and proper styling"""
    # Add an icon based on whether the document is persistent or uploaded
    icon_class = "fas fa-database me-1" if is_persistent else "fas fa-file me-2"
    icon_title = "Persistent Document" if is_persistent else "Uploaded Document"
    
    # Add group indicator if document belongs to a group
    group_indicator = ""
    if group_name:
        group_indicator = html.Span(
            [" ", html.I(className="fas fa-folder-open fa-xs me-1"), group_name],
            className="text-muted ms-1 small",
            style={"fontSize": "0.75rem"}
        )
    
    return html.Div(
        [
            html.Div(
                [
                    html.I(className=icon_class, title=icon_title),
                    html.Span([
                        html.Span(filename, className="text-truncate", style={"maxWidth": "150px"}),
                        group_indicator if group_name else ""
                    ]),
                    html.Button(
                        html.I(className="fas fa-times"),
                        id={
                            'type': 'remove-document',
                            'index': session_id
                        },
                        className="btn btn-link text-danger p-0 ms-2",
                    )
                ],
                className="d-flex align-items-center bg-light rounded px-2 py-1"
            )
        ],
        id={'type': 'document-item', 'index': session_id},
        className="d-inline-block me-2 mb-1",
        style={
            "whiteSpace": "nowrap", 
            "marginLeft": "20px" if group_name else "0px",  # Indent if part of a group
            "borderLeft": f"3px solid #e6e6e6" if group_name else "none"  # Visual indicator for group membership
        }
    )

def create_upload_area():
    """Create an enhanced upload area with multiple file source options"""
    
    COLORS = {
        'violet': '#6b5b95',
        'violet_hover': '#5d4d85',
        'border': '#e6e4ed',
        'background': '#f8f7fc',
    }

    style = {
        'upload-container': {
            'position': 'relative',
            'marginBottom': '1rem'
        },
        'expanded-upload': {
            'position': 'absolute',
            'top': '100%',
            'left': '0',
            'right': '0',
            'backgroundColor': 'white',
            'border': f'1px solid {COLORS["border"]}',
            'borderRadius': '0.25rem',
            'padding': '1rem',
            'marginTop': '0.5rem',
            'zIndex': '1000',
            'boxShadow': '0 2px 4px rgba(107, 91, 149, 0.1)'
        }
    }

    upload_tab_content = dcc.Upload(
        id="upload-document",
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
    )

    url_tab_content = html.Div([
        dbc.Textarea(
            id="url-input",
            placeholder="Enter URL(s) - one per line",
            className="mb-2",
            style={"height": "100px", "resize": "vertical"},
        ),
        dbc.Button(
            [html.I(className="fas fa-download me-2"), "Load from URLs"],
            id="url-load-button",
            color="primary",
            size="sm",
            className="mt-2"
        )
    ], className="p-4")

    folder_tab_content = html.Div([
        dbc.Input(
            id="folder-path-input",
            type="text",
            placeholder="Enter folder path",
            className="mb-2"
        ),
        dbc.Button(
            [html.I(className="fas fa-folder-open me-2"), "Load from Folder"],
            id="folder-load-button",
            color="primary",
            size="sm",
            className="mt-2"
        )
    ], className="p-4")

    server_group_tab_content = html.Div([
        dbc.Select(
            id="server-group-selector",
            placeholder="Select a group to add",
            className="mb-3"
        ),
        dbc.Button(
            [html.I(className="fas fa-plus me-2"), "Add Group"],
            id="add-server-group-btn",
            color="primary",
            size="sm",
            className="mt-2"
        ),
        html.Div(
            id="server-group-status",
            className="mt-2"
        )
    ], className="p-4")

    return html.Div([
        # Top container for button and document list
        html.Div([
            # Left side - Upload Button
            html.Div([
                dbc.Button([
                    html.I(className="fas fa-upload me-2"),
                    "Add Documents"
                ], 
                id="upload-toggle-button",
                color="primary",
                style={
                    "minWidth": "200px",
                    "whiteSpace": "nowrap",
                    "padding": "8px 16px",
                    "display": "inline-block",
                    "backgroundColor": COLORS['violet'],
                    "borderColor": COLORS['violet'],
                    "color": "white",
                    "transition": "all 0.2s ease",
                    "boxShadow": "0 2px 4px rgba(107, 91, 149, 0.15)",
                })
            ], className="d-inline-block"),
            
            # Right side - Document List
            html.Div([
                html.Div(
                    id="document-list",
                    className="document-list-inline",
                    children=[]
                )
            ], className="d-inline-block ms-3 flex-grow-1")
        ], className="d-flex align-items-center mb-3"),

        # Progress Bar Component
        create_progress_bar(),

        # Expanded Upload Area
        html.Div([
            # Tab Selection
            dbc.Nav(
                [
                    dbc.NavItem(
                        dbc.NavLink(
                            "Upload Files",
                            id="tab-upload-link",
                            active=True,
                            href="#",
                        )
                    ),
                    dbc.NavItem(
                        dbc.NavLink(
                            "From URLs",
                            id="tab-url-link",
                            href="#",
                        )
                    ),
                    dbc.NavItem(
                        dbc.NavLink(
                            "From Folder",
                            id="tab-folder-link",
                            href="#",
                        )
                    ),
                    dbc.NavItem(
                        dbc.NavLink(
                            "From Server",
                            id="tab-server-link",
                            href="#",
                        )
                    ),
                ],
                pills=True,
                fill=True,
                id="upload-tabs",
                className="mb-3",
            ),

            # Tab Content Container
            html.Div([
                html.Div(upload_tab_content, id="upload-tab-content", style={"display": "block"}),
                html.Div(url_tab_content, id="url-tab-content", style={"display": "none"}),
                html.Div(folder_tab_content, id="folder-tab-content", style={"display": "none"}),
                html.Div(server_group_tab_content, id="server-tab-content", style={"display": "none"}),
            ], id="tab-content"),
        ],
        id="expanded-upload-area", 
        style={"display": "none", "backgroundColor": "white", "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
        
        # Error message
        html.Div(id="upload-error", className="upload-error d-none"),
        
        # Store component to track toggle state
        dcc.Store(id="upload-area-is-open", data=False)
    ], className="upload-container")
    
def create_layout():
    """Create the application layout with enhanced styling"""
    
    # Custom styles
    SHADOW_STYLE = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    CARD_STYLE = {
        "borderRadius": "8px",
        "boxShadow": SHADOW_STYLE,
        "backgroundColor": "#ffffff",
        "height": "100%",
        "padding": "20px",
    }
    COLORS = {
        'primary': '#6b5b95',      # Main violet - balanced between professional and engaging
        'secondary': '#9b8bb4',    # Lighter violet for secondary elements
        'background': '#f8f7fc',   # Very light violet tint for background
        'surface': '#ffffff',      # Pure white for contrast
        'text': '#2d283e',        # Deep violet-black for primary text
        'text_secondary': '#564f6f', # Muted violet for secondary text
        'border': '#e6e4ed',      # Light violet-gray for borders
        'shadow': '0 4px 12px rgba(107, 91, 149, 0.08)', # Violet-tinted shadow
        'highlight': '#8677aa',    # Mid-tone violet for highlights
        'hover': '#5d4d85'        # Darker violet for hover states
    }
    
    CARD_STYLE = {
        "borderRadius": "12px",
        "boxShadow": COLORS['shadow'],
        "backgroundColor": COLORS['surface'],
        "border": f"1px solid {COLORS['border']}",
        "height": "100%",
        "padding": "24px",
        "transition": "box-shadow 0.3s ease"
    }
    
    HEADER_STYLE = {
        "padding": "0.75rem 0",
        "background": f"linear-gradient(135deg, {COLORS['background']} 0%, #f1effa 100%)",
        "borderRadius": "8px",
        "marginBottom": "1rem",
        "boxShadow": COLORS['shadow'],
    }
    return dbc.Container(
        [
            # Header section
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H1(
                                "DokuAI 2025",
                                style={
                                    "fontWeight": "bold",
                                    "color": COLORS['primary'],
                                    "fontSize": "2rem",
                                    "textAlign": "center",
                                    "marginBottom": "0.25rem"
                                }
                            ),
                            html.P(
                                "Upload multiple documents, ask questions across all of them",
                                className="lead mb-0",
                                style={
                                "fontSize": "1rem",  # Smaller subtitle
                                "color": COLORS['text'],
                                "opacity": "0.75",
                                "textAlign": "center",
                                }
                            ),
                        ],
                        style=HEADER_STYLE
                    ),
                    width=12,
                )
            ),
            
            # Main content row
            dbc.Row(
                [
                    # Left column - Chat Section
                    dbc.Col(
                        dbc.Card(
                            [
                                # Enhanced upload area
                                create_upload_area(),
                                
                                # Chat history with improved scrolling
                                html.Div(
                                    id="chat-history",
                                    style={
                                        "height": "500px",
                                        "overflowY": "auto",
                                        "border": f"1px solid {COLORS['border']}",
                                        "padding": "24px",
                                        "marginBottom": "24px",
                                        "borderRadius": "12px",
                                        "backgroundColor": COLORS['background'],
                                        "scrollBehavior": "smooth"
                                    },
                                ),
                                
                                # Input group with keyboard event handling
                                dbc.InputGroup(
                                    [
                                        dbc.Textarea(
                                            id="query-input",
                                            placeholder="Ask a question about your documents...",
                                            style={
                                                "borderRadius": "10px 0 0 10px",
                                                "border": f"1px solid {COLORS['border']}",
                                                "padding": "12px 16px",
                                                "fontSize": "1rem",
                                                "backgroundColor": COLORS['background'],
                                                "color": COLORS['text'],
                                                "minHeight": "50px",  # Initial height
                                                "maxHeight": "200px", # Maximum height before scrolling
                                                "resize": "auto",     # Disable manual resizing
                                                "overflowY": "auto",  # Add scrollbar when content exceeds maxHeight
                                                "lineHeight": "1.5"   # Consistent line height
                                            },
                                            n_submit=0,
                                            debounce=False,
                                            rows=1,                   # Start with one row
                                            className="auto-expand"   # Class for auto-expansion
                                        ),
                                        dbc.Button(
                                            [
                                                html.I(className="fas fa-paper-plane me-2"),
                                                "Send"
                                            ],
                                            id="submit-btn",
                                            color="primary",
                                            style={
                                                "borderRadius": "0 10px 10px 0",
                                                "backgroundColor": COLORS['primary'],
                                                "border": "none",
                                                "padding": "12px 24px",
                                                "transition": "all 0.2s ease"
                                            }
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                create_query_progress(),
                                html.Div(
                                    dbc.Select(
                                        id="doc-selector-dropdown",
                                        options=[],
                                        value=None,
                                        style={
                                            "width": "100%",
                                            "borderRadius": "8px",
                                            "border": f"1px solid {COLORS['border']}",
                                            "backgroundColor": COLORS['background']
                                        }
                                    ),
                                    id="doc-selector-container",
                                    style={
                                        "position": "absolute",
                                        "width": "300px",
                                        "backgroundColor": COLORS['surface'],
                                        "border": f"1px solid {COLORS['border']}",
                                        "borderRadius": "8px",
                                        "boxShadow": COLORS['shadow'],
                                        "zIndex": "1000",
                                        "display": "none",
                                        "marginTop": "4px"
                                    }
                                ),
                                
                                # State management stores
                                dcc.Store(id='vectorstore-state', storage_type='memory'),
                                dcc.Store(id='document-state', storage_type='memory'),
                                dcc.Store(id='chunk-mapping-state', storage_type='memory'),
                                dcc.Store(id='upload-trigger', data=None),
                                dcc.Store(id="delete-group-id", data=None), 
                                dcc.Store(id='removed-groups-state', storage_type='session', data=[]),
                                html.Div(id="scroll-trigger"),
                                
                                # Add group deletion modal
                                create_group_delete_modal(),
                            ],
                            body=True,
                            style=CARD_STYLE
                        ),
                        md=6,
                        className="mb-4"
                    ),
                    
                    # Right column - Document Viewer Section
                    dbc.Col(
                        dbc.Card(
                            [
                                # Document selector
                                dbc.Select(
                                    id="document-selector",
                                    placeholder="Select a document to view...",
                                    style={
                                        "marginBottom": "1rem",
                                        "borderRadius": "8px",
                                        "border": f"1px solid {COLORS['border']}",
                                        "padding": "10px",
                                        "backgroundColor": COLORS['background']
                                    }
                                ),
                                
                                # Document viewer container
                                html.Div(
                                    id="document-viewer-container",
                                    children=create_pdf_viewer(),
                                    style={
                                        "height": "700px",
                                        "border": f"1px solid {COLORS['border']}",
                                        "borderRadius": "12px",
                                        "backgroundColor": COLORS['background'],
                                        "position": "relative",
                                        "overflow": "hidden"
                                    },
                                ),
                            ],
                            body=True,
                            style=CARD_STYLE
                        ),
                        md=6,
                        className="mb-4"
                    ),
                ],
                className="g-4",
            ),
        ],
        fluid=True,
        className="py-4",
        style={"backgroundColor": COLORS['background']}
    )