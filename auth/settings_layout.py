# auth/settings_layout.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import json
import os
from pathlib import Path
from .admin import COLORS
import datetime
import glob 
import dash
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base storage directory for settings
BASE_STORAGE_DIR = Path('storage')
SETTINGS_DIR = BASE_STORAGE_DIR / 'settings'
GROUP_MESSAGES_FILE = SETTINGS_DIR / 'group_welcome_messages.json'
LLM_SETTINGS_FILE = SETTINGS_DIR / 'llm_settings.json'
RETRIEVAL_SETTINGS_FILE = SETTINGS_DIR / 'retrieval_settings.json'

def ensure_groups_exist():
    """Make sure at least one group exists in the system"""
    try:
        # First try to load groups using the existing method
        groups = load_groups_directly()
        
        # If groups were found, return them
        if groups and len(groups) > 0:
            logger.info(f"Found {len(groups)} existing groups")
            return groups
        
        # If no groups were found, create a default group
        logger.info("No groups found, creating default groups")
        default_groups = {
            "default_group": {
                "name": "Default Group",
                "description": "Default group for document organization",
                "group_admins": [],
                "created_at": str(datetime.datetime.now())
            },
            "general_group": {
                "name": "General",
                "description": "General purpose group",
                "group_admins": [],
                "created_at": str(datetime.datetime.now())
            }
        }
        
        # Save these default groups
        ensure_settings_dirs()
        groups_dir = BASE_STORAGE_DIR / 'groups'
        groups_dir.mkdir(parents=True, exist_ok=True)
        
        groups_file = groups_dir / 'user_groups.json'
        with open(groups_file, 'w') as f:
            json.dump(default_groups, f, indent=4)
        
        logger.info(f"Settings.layout.py - default - ensure_groups_exist : Created and saved default groups to {groups_file}")
        return default_groups
        
    except Exception as e:
        logger.error(f"Settings.layout.py - default - ensure_groups_exist : Error in ensure_groups_exist: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return a failsafe group
        return {
            "failsafe_group": {
                "name": "Failsafe Group",
                "description": f"Created due to error: {str(e)}",
                "group_admins": [],
                "created_at": str(datetime.datetime.now())
            }
        }
    
def load_groups_directly():
    """Load groups directly from the file system, bypassing GroupService"""
    try:
        import json
        from pathlib import Path
        import os
        
        # Try multiple possible paths
        possible_paths = [
            Path('storage/groups/user_groups.json'),
            Path('./storage/groups/user_groups.json'),
            Path('../storage/groups/user_groups.json')
        ]
        
        # Print current working directory for debugging
        logger.info(f"Settings.layout.py - default - load_groups_directly : Current working directory: {os.getcwd()}")
        
        # Check each path
        for path in possible_paths:
            logger.info(f"Checking path: {path.absolute()}")
            if path.exists():
                logger.info(f"Found groups file at: {path.absolute()}")
                with open(path, 'r') as f:
                    groups = json.load(f)
                    logger.info(f"Loaded {len(groups)} groups from file")
                    return groups
        
        # If no file found, look in directories
        logger.info("Settings.layout.py - default - load_groups_directly : No groups file found in standard locations, searching directories...")
        storage_dir = None
        for path in [Path('storage'), Path('./storage'), Path('../storage')]:
            if path.exists() and path.is_dir():
                storage_dir = path
                break
        
        if storage_dir:
            logger.info(f"Found storage directory at: {storage_dir.absolute()}")
            groups_dir = storage_dir / 'groups'
            if groups_dir.exists() and groups_dir.is_dir():
                logger.info(f"Found groups directory at: {groups_dir.absolute()}")
                # List all files in groups directory
                for file in groups_dir.iterdir():
                    logger.info(f"Found file: {file.name}")
                    if file.name == 'user_groups.json':
                        logger.info(f"Found user_groups.json at: {file.absolute()}")
                        with open(file, 'r') as f:
                            groups = json.load(f)
                            logger.info(f"Loaded {len(groups)} groups from file")
                            return groups
        
        # If still no file found, create a sample group
        logger.info("Settings.layout.py - default - load_groups_directly : No groups file found, creating sample group")
        return {
            "default_group": {
                "name": "Default Group",
                "description": "This is a sample group created because no groups file was found",
                "group_admins": [],
                "created_at": str(datetime.datetime.now())
            }
        }
    except Exception as e:
        logger.info(f"Error loading groups directly: {str(e)}")
        import traceback
        logger.info(traceback.format_exc())
        
        # Create a sample group on error
        logger.error("Settings.layout.py - default - load_groups_directly : Error occurred, creating sample group")
        return {
            "error_group": {
                "name": "Error Group",
                "description": f"This is a sample group created due to error: {str(e)}",
                "group_admins": [],
                "created_at": str(datetime.datetime.now())
            }
        }

def ensure_settings_dirs():
    """Ensure settings directories exist"""
    BASE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

def load_group_messages():
    """Load group welcome messages from file"""
    ensure_settings_dirs()
    try:
        if GROUP_MESSAGES_FILE.exists():
            with open(GROUP_MESSAGES_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Settings.layout.py - default - load_group_messages : Error loading group messages: {e}")
        return {}

def load_llm_settings():
    """Load LLM API settings from file"""
    ensure_settings_dirs()
    try:
        if LLM_SETTINGS_FILE.exists():
            with open(LLM_SETTINGS_FILE, 'r') as f:
                return json.load(f)
        # Default settings
        return {
            "provider": "azure-gpt",
            "azure-gpt": {
                "api_key": "",
                "endpoint": "",
                "deployment_name": "",
                "api_version": "2023-05-15"
            },
            "claude": {
                "api_key": "",
                "model": "claude-3-opus-20240229"
            },
            "gemini": {
                "api_key": "",
                "model": "gemini-pro"
            },
            "llama": {
                "endpoint": "http://localhost:8000/v1",
                "model": "llama3-70b"
            }
        }
    except Exception as e:
        logger.error(f"Settings.layout.py - default - load_llm_settings : Error loading LLM settings: {e}")
        return {}

def load_retrieval_settings():
    """Load retrieval settings from file"""
    ensure_settings_dirs()
    try:
        if RETRIEVAL_SETTINGS_FILE.exists():
            with open(RETRIEVAL_SETTINGS_FILE, 'r') as f:
                return json.load(f)
        # Default settings
        return {
            "relevant_chunks_per_doc": 3,
            "similarity_threshold": 0.75,
            "max_total_chunks": 10
        }
    except Exception as e:
        logger.error(f"Settings.layout.py - default - load_retrieval_settings : Error loading retrieval settings: {e}")
        return {}

def save_group_messages(messages):
    """Save group welcome messages to file"""
    ensure_settings_dirs()
    try:
        with open(GROUP_MESSAGES_FILE, 'w') as f:
            json.dump(messages, f, indent=4)
        return True
    except Exception as e:
        logger.info(f"Settings.layout.py - default - save_group_messages : Error saving group messages: {e}")
        return False

def save_llm_settings(settings):
    """Save LLM API settings to file"""
    ensure_settings_dirs()
    try:
        with open(LLM_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        logger.info(f"Settings.layout.py - default - save_llm_settings : Error saving LLM settings: {e}")
        return False

def save_retrieval_settings(settings):
    """Save retrieval settings to file"""
    ensure_settings_dirs()
    try:
        with open(RETRIEVAL_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        logger.info(f"Settings.layout.py - default - save_retrieval_settings : Error saving retrieval settings: {e}")
        return False

def create_settings_layout():
    """Create a settings page layout with tabs for different settings categories"""
    
    return html.Div(
        style={
            'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["white"]} 100%)',
            'minHeight': '100vh',
            'padding': '1.5rem',
        },
        children=[
            dbc.Container([
                # Header with title and save button
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(
                                className="fas fa-cogs fa-2x",
                                style={'color': COLORS['primary'], 'marginRight': '15px'}
                            ),
                            html.H2("System Settings", style={
                                'color': COLORS['primary'],
                                'fontWeight': '600',
                                'display': 'inline-block',
                                'verticalAlign': 'middle',
                                'margin': '0'
                            })
                        ], style={'display': 'flex', 'alignItems': 'center'})
                    ], width=8),
                    dbc.Col([
                        dbc.Button(
                            ["Save All Settings ", html.I(className="fas fa-save")],
                            id="save-all-settings-button",
                            color="primary",
                            className="float-end",
                            n_clicks=0,
                            style={
                                'backgroundColor': COLORS['primary'],
                                'borderColor': COLORS['primary'],
                                'boxShadow': COLORS['shadow'],
                            }
                        )
                    ], width=4, className="d-flex align-items-center justify-content-end")
                ], className="mb-4 align-items-center"),
                
                # Settings Tabs
                dbc.Card([
                    dbc.CardBody([
                        dbc.Tabs([
                            # Group Welcome Messages Tab
                            dbc.Tab(
                                label="Group Welcome Messages",
                                tab_id="group-messages-tab",
                                label_style={"font-weight": "bold"},
                                children=[
                                    html.Div([
                                        html.P("Configure custom welcome messages for each group.", className="text-muted mb-4"),
                                        
                                        # Welcome Messages Table
                                        dash_table.DataTable(
                                            id='welcome-messages-table',
                                            columns=[
                                                {"name": "Group ID", "id": "group_id"},
                                                {"name": "Group Name", "id": "group_name"},
                                                {"name": "Welcome Message", "id": "welcome_message", "editable": True},
                                            ],
                                            data=[],
                                            style_table={'overflowX': 'auto'},
                                            style_cell={
                                                'textAlign': 'left',
                                                'padding': '12px',
                                                'whiteSpace': 'normal',
                                                'fontFamily': '"Segoe UI", Arial, sans-serif',
                                                'fontSize': '14px',
                                                'height': 'auto',  # Allow multi-line content
                                            },
                                            style_cell_conditional=[
                                                {
                                                    'if': {'column_id': 'welcome_message'},
                                                    'minWidth': '400px', 
                                                    'width': '60%',
                                                    'textAlign': 'left',
                                                    'whiteSpace': 'pre-line'
                                                }
                                            ],
                                            style_header={
                                                'backgroundColor': COLORS['light_accent'],
                                                'fontWeight': 'bold',
                                                'color': COLORS['text'],
                                                'borderBottom': f'1px solid {COLORS["secondary"]}',
                                            },
                                            style_data_conditional=[
                                                {
                                                    'if': {'row_index': 'odd'},
                                                    'backgroundColor': 'rgba(248, 247, 252, 0.5)'
                                                }
                                            ],
                                            editable=True,
                                            row_selectable=False,
                                            page_size=10,
                                        ),
                                        html.Div([
                                            html.I(className="fas fa-info-circle me-2", style={'color': COLORS['primary']}),
                                            html.Span("You can directly edit welcome messages by clicking on the cell. Changes are saved when you click 'Save All Settings'.")
                                        ], className="mt-3", style={"fontSize": "0.9rem", "color": COLORS['text']}),
                                        
                                        # Edit Selected Message Button
                                        dbc.Button(
                                            ["Edit in Full Editor ", html.I(className="fas fa-edit")],
                                            id="edit-message-button",
                                            color="warning",
                                            className="mt-3",
                                            style={
                                                'fontWeight': '500',
                                                'boxShadow': COLORS['shadow'],
                                            }
                                        ),
                                    ], className="p-3")
                                ]
                            ),
                            
                            # LLM API Settings Tab
                            dbc.Tab(
                                label="LLM Providers",
                                tab_id="llm-settings-tab",
                                label_style={"font-weight": "bold"},
                                children=[
                                    html.Div([
                                        html.P("Configure LLM API provider settings.", className="text-muted mb-4"),
                                        
                                        # LLM Provider Selection
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Active LLM Provider", style={'fontWeight': '600', 'color': COLORS['text']}),
                                                dbc.Select(
                                                    id="llm-provider-select",
                                                    options=[
                                                        {"label": "Azure OpenAI", "value": "azure-gpt"},
                                                        {"label": "Anthropic Claude", "value": "claude"},
                                                        {"label": "Google Gemini", "value": "gemini"},
                                                        {"label": "On-premise LLAMA", "value": "llama"}
                                                    ],
                                                    value="azure-gpt",
                                                    style={
                                                        'borderColor': COLORS['light_accent'],
                                                        'fontSize': '1rem',
                                                        'padding': '0.75rem',
                                                        'marginBottom': '1.5rem',
                                                        'width': '100%'
                                                    }
                                                )
                                            ], md=6)
                                        ]),
                                        
                                        # Provider-specific settings (shown/hidden based on selection)
                                        html.Div([
                                            # Azure OpenAI Settings
                                            html.Div(id="azure-gpt-settings", children=[
                                                dbc.Card([
                                                    dbc.CardHeader(html.H5("Azure OpenAI Settings")),
                                                    dbc.CardBody([
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Label("API Key", style={'fontWeight': '500'}),
                                                                dbc.Input(
                                                                    id="azure-api-key",
                                                                    type="password",
                                                                    placeholder="Enter Azure OpenAI API Key",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                            dbc.Col([
                                                                dbc.Label("Endpoint", style={'fontWeight': '500'}),
                                                                dbc.Input(
                                                                    id="azure-endpoint",
                                                                    type="text",
                                                                    placeholder="https://your-resource.openai.azure.com/",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                        ]),
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Label("Deployment Name", style={'fontWeight': '500'}),
                                                                dbc.Input(
                                                                    id="azure-deployment-name",
                                                                    type="text",
                                                                    placeholder="Enter deployment name",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                            dbc.Col([
                                                                dbc.Label("API Version", style={'fontWeight': '500'}),
                                                                dbc.Input(
                                                                    id="azure-api-version",
                                                                    type="text",
                                                                    placeholder="2023-05-15",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                        ]),
                                                    ]),
                                                ], style={'marginBottom': '1rem', 'border': f'1px solid {COLORS["light_accent"]}'}),
                                            ]),
                                            
                                            # Claude Settings
                                            html.Div(id="claude-settings", children=[
                                                dbc.Card([
                                                    dbc.CardHeader(html.H5("Anthropic Claude Settings")),
                                                    dbc.CardBody([
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Label("API Key", style={'fontWeight': '500'}),
                                                                dbc.Input(
                                                                    id="claude-api-key",
                                                                    type="password",
                                                                    placeholder="Enter Anthropic API Key",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                            dbc.Col([
                                                                dbc.Label("Model", style={'fontWeight': '500'}),
                                                                dbc.Select(
                                                                    id="claude-model",
                                                                    options=[
                                                                        {"label": "Claude 3 Opus", "value": "claude-3-opus-20240229"},
                                                                        {"label": "Claude 3 Sonnet", "value": "claude-3-sonnet-20240229"},
                                                                        {"label": "Claude 3 Haiku", "value": "claude-3-haiku-20240307"}
                                                                    ],
                                                                    value="claude-3-opus-20240229",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                        ]),
                                                    ]),
                                                ], style={'marginBottom': '1rem', 'border': f'1px solid {COLORS["light_accent"]}'}),
                                            ], style={'display': 'none'}),
                                            
                                            # Gemini Settings
                                            html.Div(id="gemini-settings", children=[
                                                dbc.Card([
                                                    dbc.CardHeader(html.H5("Google Gemini Settings")),
                                                    dbc.CardBody([
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Label("API Key", style={'fontWeight': '500'}),
                                                                dbc.Input(
                                                                    id="gemini-api-key",
                                                                    type="password",
                                                                    placeholder="Enter Google API Key",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                            dbc.Col([
                                                                dbc.Label("Model", style={'fontWeight': '500'}),
                                                                dbc.Select(
                                                                    id="gemini-model",
                                                                    options=[
                                                                        {"label": "Gemini Pro", "value": "gemini-pro"},
                                                                        {"label": "Gemini Pro Vision", "value": "gemini-pro-vision"}
                                                                    ],
                                                                    value="gemini-pro",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                        ]),
                                                    ]),
                                                ], style={'marginBottom': '1rem', 'border': f'1px solid {COLORS["light_accent"]}'}),
                                            ], style={'display': 'none'}),
                                            
                                            # LLAMA Settings
                                            html.Div(id="llama-settings", children=[
                                                dbc.Card([
                                                    dbc.CardHeader(html.H5("On-premise LLAMA Settings")),
                                                    dbc.CardBody([
                                                        dbc.Row([
                                                            dbc.Col([
                                                                dbc.Label("Endpoint URL", style={'fontWeight': '500'}),
                                                                dbc.Input(
                                                                    id="llama-endpoint",
                                                                    type="text",
                                                                    placeholder="http://localhost:8000/v1",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                            dbc.Col([
                                                                dbc.Label("Model", style={'fontWeight': '500'}),
                                                                dbc.Input(
                                                                    id="llama-model",
                                                                    type="text",
                                                                    placeholder="llama3-70b",
                                                                    style={'marginBottom': '1rem'},
                                                                ),
                                                            ], md=6),
                                                        ]),
                                                    ]),
                                                ], style={'marginBottom': '1rem', 'border': f'1px solid {COLORS["light_accent"]}'}),
                                            ], style={'display': 'none'}),
                                        ]),
                                        
                                        # Test Connection Button
                                        dbc.Button(
                                            ["Test Connection ", html.I(className="fas fa-plug")],
                                            id="test-llm-connection-button",
                                            color="info",
                                            className="mt-3",
                                            style={
                                                'fontWeight': '500',
                                                'boxShadow': COLORS['shadow'],
                                            }
                                        ),
                                    ], className="p-3")
                                ]
                            ),
                            
                            # Retrieval Settings Tab
                            dbc.Tab(
                                label="Retrieval Parameters",
                                tab_id="retrieval-settings-tab",
                                label_style={"font-weight": "bold"},
                                children=[
                                    html.Div([
                                        html.P("Configure document retrieval and chunking parameters.", className="text-muted mb-4"),
                                        
                                        dbc.Card([
                                            dbc.CardHeader(html.H5("Document Retrieval Settings")),
                                            dbc.CardBody([
                                                dbc.Row([
                                                    dbc.Col([
                                                        dbc.Label("Relevant Chunks per Document (k)", style={'fontWeight': '500'}),
                                                        dbc.InputGroup([
                                                            dbc.Input(
                                                                id="chunks-per-doc-input",
                                                                type="number",
                                                                min=1,
                                                                max=10,
                                                                step=1,
                                                                value=3,
                                                                style={'marginBottom': '1rem'},
                                                            ),
                                                            dbc.InputGroupText("chunks")
                                                        ]),
                                                        html.Small(
                                                            "Number of chunks to retrieve from each relevant document",
                                                            className="text-muted"
                                                        )
                                                    ], md=6),
                                                    dbc.Col([
                                                        dbc.Label("Maximum Total Chunks", style={'fontWeight': '500'}),
                                                        dbc.InputGroup([
                                                            dbc.Input(
                                                                id="max-total-chunks-input",
                                                                type="number",
                                                                min=1,
                                                                max=50,
                                                                step=1,
                                                                value=10,
                                                                style={'marginBottom': '1rem'},
                                                            ),
                                                            dbc.InputGroupText("chunks")
                                                        ]),
                                                        html.Small(
                                                            "Maximum total chunks to include in context",
                                                            className="text-muted"
                                                        )
                                                    ], md=6),
                                                ]),
                                                dbc.Row([
                                                    dbc.Col([
                                                        dbc.Label("Similarity Threshold", style={'fontWeight': '500'}),
                                                        dbc.InputGroup([
                                                            dbc.Input(
                                                                id="similarity-threshold-input",
                                                                type="number",
                                                                min=0,
                                                                max=1,
                                                                step=0.01,
                                                                value=0.75,
                                                                style={'marginBottom': '1rem'},
                                                            ),
                                                            dbc.InputGroupText("(0-1)")
                                                        ]),
                                                        html.Small(
                                                            "Minimum similarity score for including chunks",
                                                            className="text-muted"
                                                        )
                                                    ], md=6),
                                                ]),
                                            ]),
                                        ], style={'marginBottom': '1rem', 'border': f'1px solid {COLORS["light_accent"]}'}),
                                        
                                        # Advanced Settings Card
                                        dbc.Card([
                                            dbc.CardHeader(html.H5("Advanced Settings")),
                                            dbc.CardBody([
                                                dbc.Row([
                                                    dbc.Col([
                                                        dbc.Label("Temperature", style={'fontWeight': '500'}),
                                                        dbc.InputGroup([
                                                            dbc.Input(
                                                                id="temperature-input",
                                                                type="number",
                                                                min=0,
                                                                max=1,
                                                                step=0.1,
                                                                value=0.7,
                                                                style={'marginBottom': '1rem'},
                                                            ),
                                                            dbc.InputGroupText("(0-1)")
                                                        ]),
                                                        html.Small(
                                                            "Controls randomness in LLM responses",
                                                            className="text-muted"
                                                        )
                                                    ], md=6),
                                                    dbc.Col([
                                                        dbc.Label("System Message", style={'fontWeight': '500'}),
                                                        dbc.Textarea(
                                                            id="system-message-input",
                                                            value="You are a helpful assistant that answers questions based on the provided documents.",
                                                            style={'height': '120px', 'marginBottom': '1rem'},
                                                        ),
                                                        html.Small(
                                                            "Default system message for LLM interactions",
                                                            className="text-muted"
                                                        )
                                                    ], md=6),
                                                ]),
                                            ]),
                                        ], style={'marginBottom': '1rem', 'border': f'1px solid {COLORS["light_accent"]}'}),
                                    ], className="p-3")
                                ]
                            ),
                        ],
                        id="settings-tabs",
                        active_tab="group-messages-tab"
                    ),
                    ]),
                ], style={
                    'borderRadius': '12px',
                    'border': f'1px solid {COLORS["light_accent"]}',
                    'boxShadow': COLORS['shadow'],
                    'marginBottom': '2rem'
                }),
                
                # Message Modal for editing welcome messages
                dbc.Modal([
                    dbc.ModalHeader(
                        dbc.ModalTitle(html.Div(id="message-modal-title")),
                        close_button=True
                    ),
                    dbc.ModalBody([
                        dbc.Label("Welcome Message", style={'fontWeight': '500', 'color': COLORS['text']}),
                        dbc.Textarea(
                            id="message-modal-textarea",
                            style={'height': '200px'},
                            className="mb-3"
                        ),
                        html.Div([
                            html.I(className="fas fa-info-circle me-2", style={'color': COLORS['primary']}),
                            html.Span("This message will be shown to users when they first access the group.")
                        ], style={"fontSize": "0.9rem", "color": COLORS['text']}),
                    ]),
                    dbc.ModalFooter([
                        dbc.Button(
                            "Close",
                            id="close-message-modal-button",
                            className="me-2"
                        ),
                        dbc.Button(
                            "Save Changes",
                            id="save-message-modal-button",
                            color="primary"
                        ),
                    ]),
                ], id="welcome-message-modal", is_open=False, size="lg"),
                
                # Feedback Toast
                dbc.Toast(
                    id="settings-feedback-toast",
                    header="Settings Notification",
                    is_open=False,
                    dismissable=True,
                    duration=4000,
                    style={"position": "fixed", "top": 66, "right": 10, "width": 350, "boxShadow": COLORS['shadow']},
                ),
                html.Div([
                    dbc.Button(
                        ["Show Debugging Info ", html.I(className="fas fa-bug")],
                        id="toggle-debug-button",
                        color="secondary",
                        size="sm",
                        className="mt-4"
                    ),
                    html.Div(id="debug-groups-container", style={"display": "none", "marginTop": "1rem"})
                ], className="mt-4"),
                                
                # Store components for state management
                dcc.Store(id="group-messages-store", storage_type="memory"),
                dcc.Store(id="llm-settings-store", storage_type="memory"),
                dcc.Store(id="retrieval-settings-store", storage_type="memory"),
                dcc.Store(id="edit-message-group-id", storage_type="memory"),
                
                # Hidden divs for callbacks
                html.Div(id="settings-load-trigger", style={"display": "none"}),
                html.Div(id="settings-save-trigger", style={"display": "none"}),
            ], fluid=True)
        ]
    )