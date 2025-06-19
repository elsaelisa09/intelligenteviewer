# components/group_llm_settings_ui.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import json
from pathlib import Path
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_directories():
    """Get all directories in the storage folder"""
    storage_dir = Path("storage")
    if not storage_dir.exists():
        storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for any folders that might be groups
    groups = {}
    
    # First, check if a groups directory exists
    groups_dir = storage_dir / "groups"
    if groups_dir.exists() and groups_dir.is_dir():
        groups_file = groups_dir / "user_groups.json"
        if groups_file.exists():
            try:
                with open(groups_file, "r") as f:
                    groups = json.load(f)
                logger.info(f"Found {len(groups)} groups in groups file")
            except Exception as e:
                logger.error(f"Error loading groups file: {e}")
    
    # If still no groups, check vector_stores directory
    if not groups:
        vector_stores_dir = storage_dir / "vector_stores"
        if vector_stores_dir.exists() and vector_stores_dir.is_dir():
            # Get all subdirectories
            for item in vector_stores_dir.iterdir():
                if item.is_dir():
                    group_id = item.name
                    groups[group_id] = {
                        "name": group_id.capitalize(),
                        "description": f"Group for {group_id} documents",
                        "created_at": ""
                    }
            logger.info(f"Found {len(groups)} groups in vector_stores directory")
    
    # Always add default group if not exists
    if "default" not in groups:
        groups["default"] = {
            "name": "Default",
            "description": "Default group for documents",
            "created_at": ""
        }
    
    return groups

def load_group_llm_settings():
    """Load group LLM settings from file"""
    try:
        # Define path to settings file
        storage_dir = Path("storage")
        settings_dir = storage_dir / "settings"
        settings_file = settings_dir / "group_llm_settings.json"
        
        # Create directories if they don't exist
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        if settings_file.exists():
            # Load existing settings
            with open(settings_file, "r") as f:
                settings = json.load(f)
            logger.info(f"Loaded group LLM settings from file")
            return settings
        else:
            # Return default settings if file doesn't exist
            default_settings = {
                "default": "azure-gpt",
                "groups": {},
                "retrieval_params": {
                    "default": {
                        "chunks_per_doc": 3,
                        "max_total_chunks": 10,
                        "similarity_threshold": 0.75,
                        "default_language": "en"
                    }
                }
            }
            logger.info("Group LLM settings file not found, using defaults")
            return default_settings
    except Exception as e:
        logger.error(f"Error loading group LLM settings: {e}")
        logger.error(traceback.format_exc())
        return {
            "default": "azure-gpt",
            "groups": {},
            "retrieval_params": {
                "default": {
                    "chunks_per_doc": 3,
                    "max_total_chunks": 10,
                    "similarity_threshold": 0.75,
                    "default_language": "en"
                }
            }
        }

def save_group_llm_settings(settings):
    """Save group LLM settings to file"""
    try:
        # Define path to settings file
        storage_dir = Path("storage")
        settings_dir = storage_dir / "settings"
        settings_file = settings_dir / "group_llm_settings.json"
        
        # Create directories if they don't exist
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Save settings to file
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=4)
        
        logger.info(f"Saved group LLM settings to file: {settings}")
        return True
    except Exception as e:
        logger.error(f"Error saving group LLM settings: {e}")
        logger.error(traceback.format_exc())
        return False

def create_group_llm_table_data():
    """Create table data for group LLM settings UI"""
    try:
        # Get all groups
        groups = get_all_directories()
        
        # Get LLM settings
        settings = load_group_llm_settings()
        default_provider = settings.get("default", "azure-gpt")
        group_settings = settings.get("groups", {})
        
        # Get retrieval parameters
        retrieval_params = settings.get("retrieval_params", {})
        default_retrieval = retrieval_params.get("default", {
            "chunks_per_doc": 3,
            "max_total_chunks": 10,
            "similarity_threshold": 0.75,
            "default_language": "en"
        })
        
        # Create table data
        table_data = []
        
        # Add default row
        default_retrieval_params = retrieval_params.get("default", default_retrieval)
        table_data.append({
            "group_id": "default",
            "group_name": "Default (fallback)",
            "provider": default_provider,
            "chunks_per_doc": default_retrieval_params.get("chunks_per_doc", 3),
            "max_total_chunks": default_retrieval_params.get("max_total_chunks", 10),
            "similarity_threshold": default_retrieval_params.get("similarity_threshold", 0.75),
            "default_language": default_retrieval_params.get("default_language", "en")
        })
        
        # Add rows for each group
        for group_id, group_info in groups.items():
            if group_id != "default":  # Skip default since it's already added
                provider = group_settings.get(group_id, default_provider)
                
                # Get group-specific retrieval parameters if they exist, otherwise use defaults
                group_retrieval = retrieval_params.get(group_id, default_retrieval_params)
                
                table_data.append({
                    "group_id": group_id,
                    "group_name": group_info.get("name", group_id.capitalize()),
                    "provider": provider,
                    "chunks_per_doc": group_retrieval.get("chunks_per_doc", default_retrieval_params.get("chunks_per_doc", 3)),
                    "max_total_chunks": group_retrieval.get("max_total_chunks", default_retrieval_params.get("max_total_chunks", 10)),
                    "similarity_threshold": group_retrieval.get("similarity_threshold", default_retrieval_params.get("similarity_threshold", 0.75)),
                    "default_language": group_retrieval.get("default_language", default_retrieval_params.get("default_language", "en"))
                })
        
        logger.info(f"Created group LLM table data with {len(table_data)} rows")
        return table_data
    except Exception as e:
        logger.error(f"Error creating group LLM table data: {e}")
        logger.error(traceback.format_exc())
        return [
            {
                "group_id": "default",
                "group_name": "Default (fallback)",
                "provider": "azure-gpt",
                "chunks_per_doc": 3,
                "max_total_chunks": 10,
                "similarity_threshold": 0.75,
                "default_language": "en"
            }
        ]

def create_group_llm_settings_ui():
    """Create a UI component for group LLM settings"""
    try:
        # Get table data
        table_data = create_group_llm_table_data()
        
        return html.Div([
            html.P("Configure which AI model provider to use for each group and set retrieval parameters. This allows different groups to use different settings.", className="mb-4"),
            
            # Table showing group-to-provider mapping with retrieval parameters
            dbc.Card([
                dbc.CardHeader("Group AI Model Configuration"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="group-llm-table",
                        columns=[
                            {"name": "Group ID", "id": "group_id", "editable": False},
                            {"name": "Group Name", "id": "group_name", "editable": False},
                            {
                                "name": "Provider", 
                                "id": "provider", 
                                "editable": True,
                                "presentation": "dropdown"
                            },
                            {
                                "name": "Chunks Per Doc", 
                                "id": "chunks_per_doc", 
                                "editable": True,
                                "type": "numeric"
                            },
                            {
                                "name": "Max Total Chunks", 
                                "id": "max_total_chunks", 
                                "editable": True,
                                "type": "numeric"
                            },
                            {
                                "name": "Similarity Threshold", 
                                "id": "similarity_threshold", 
                                "editable": True,
                                "type": "numeric",
                                "format": {
                                    "specifier": ".2f"
                                }
                            },
                            {
                                "name": "Default Language", 
                                "id": "default_language", 
                                "editable": True,
                                "presentation": "dropdown"
                            },
                        ],
                        data=table_data,
                        dropdown={
                            "provider": {
                                "options": [
                                    {"label": "Azure OpenAI", "value": "azure-gpt"},
                                    {"label": "Anthropic Claude", "value": "claude"},
                                    {"label": "Google Gemini", "value": "gemini"},
                                    {"label": "On-premise LLAMA", "value": "llama"}
                                ],
                                "clearable": False
                            },
                            "default_language": {
                                "options": [
                                    {"label": "English", "value": "en"},
                                    {"label": "Spanish", "value": "es"},
                                    {"label": "French", "value": "fr"},
                                    {"label": "German", "value": "de"},
                                    {"label": "Italian", "value": "it"},
                                    {"label": "Portuguese", "value": "pt"},
                                    {"label": "Dutch", "value": "nl"},
                                    {"label": "Russian", "value": "ru"},
                                    {"label": "Chinese", "value": "zh"},
                                    {"label": "Japanese", "value": "ja"},
                                    {"label": "Korean", "value": "ko"},
                                    {"label": "Arabic", "value": "ar"},
                                    {"label": "Hindi", "value": "hi"},
                                    {"label": "Indonesian", "value": "id"},
                                    {"label": "Turkish", "value": "tr"},
                                    {"label": "Vietnamese", "value": "vi"},
                                    {"label": "Thai", "value": "th"},
                                    {"label": "Malay", "value": "ms"}
                                ],
                                "clearable": False
                            }
                        },
                        style_cell={
                            "textAlign": "left",
                            "padding": "12px",
                            "fontFamily": '"Segoe UI", Arial, sans-serif',
                        },
                        style_header={
                            "backgroundColor": "#e6e4ed",
                            "fontWeight": "bold",
                            "textAlign": "left"
                        },
                        style_data_conditional=[
                            {
                                "if": {"row_index": "odd"},
                                "backgroundColor": "rgba(248, 247, 252, 0.5)"
                            }
                        ],
                        page_size=10,
                        css=[
                            # Enhanced dropdown styling for better user experience
                            {"selector": ".Select-menu-outer", "rule": "display: block !important; visibility: visible !important; z-index: 1000 !important;"},
                            {"selector": ".Select-control", "rule": "border: 1px solid #ddd !important; min-height: 36px !important; cursor: pointer !important;"},
                            {"selector": ".Select.is-focused:not(.is-open) > .Select-control", "rule": "border-color: #6b5b95 !important; box-shadow: 0 0 0 1px #6b5b95 !important;"},
                            {"selector": ".Select-option", "rule": "padding: 10px 12px !important; cursor: pointer !important;"},
                            {"selector": ".Select-option:hover", "rule": "background-color: #f0ebff !important;"},
                            {"selector": ".Select-option.is-selected", "rule": "background-color: #e6e4ed !important; font-weight: bold !important;"},
                            {"selector": ".Select.is-open .Select-menu-outer", "rule": "transition-delay: 0.3s !important; transition: visibility 0s linear 0.3s !important;"},
                            {"selector": ".Select-menu", "rule": "max-height: 250px !important;"},
                            {"selector": ".Select-arrow", "rule": "border-width: 6px 4px 0 !important;"},
                            # Fix dropdown area to prevent accidental closing
                            {"selector": ".Select.is-open", "rule": "pointer-events: auto !important;"},
                            {"selector": ".Select.is-open:after", "rule": "content: ''; position: absolute; top: -15px; left: 0; right: 0; height: 15px; z-index: 999;"}
                        ],
                        tooltip_data=[
                            {
                                "chunks_per_doc": {"value": "Number of most relevant chunks to retrieve from each document (1-10)", "type": "text"},
                                "max_total_chunks": {"value": "Maximum total chunks to include in context (1-50)", "type": "text"},
                                "similarity_threshold": {"value": "Minimum similarity score (0.0-1.0) for including chunks", "type": "text"},
                                "default_language": {"value": "Default language for this group", "type": "text"}
                            } for _ in range(len(table_data))
                        ],
                        tooltip_duration=None
                    ),
                    
                    # Help text
                    html.Div([
                        html.I(className="fas fa-info-circle me-2 text-primary"),
                        html.Span("Groups are automatically detected from your system. Customize the settings for each group and click 'Save' to update.")
                    ], className="mt-3 mb-3 text-muted small"),
                    
                    # Parameter explanation
                    dbc.Alert([
                        html.H6("Parameter Explanation:", className="mb-2 alert-heading"),
                        html.Ul([
                            html.Li([html.Strong("Chunks Per Doc: "), "Number of relevant chunks to retrieve from each document"]),
                            html.Li([html.Strong("Max Total Chunks: "), "Maximum total chunks to include in context"]),
                            html.Li([html.Strong("Similarity Threshold: "), "Minimum similarity score (0-1) for including chunks"]),
                            html.Li([html.Strong("Default Language: "), "Default language for documents in this group"])
                        ], className="mb-0")
                    ], color="info", className="mt-3"),
                    
                    # Save button & feedback
                    html.Div([
                        dbc.Button(
                            "Save Group Settings", 
                            id="save-group-llm-button", 
                            color="primary", 
                            className="mt-2"
                        ),
                        html.Div(id="group-llm-feedback", className="mt-3")
                    ]),
                ])
            ]),
        ])
    except Exception as e:
        logger.error(f"Error creating group LLM settings UI: {e}")
        logger.error(traceback.format_exc())
        return html.Div([
            html.Div("Error loading group LLM settings UI", className="alert alert-danger"),
            html.Pre(str(e) + "\n" + traceback.format_exc())
        ])
    
class GroupLLMSettings:
    """Class for managing group LLM settings"""
    
    @staticmethod
    def get_settings_file_path():
        """Get the path to the group LLM settings file"""
        storage_dir = Path("storage")
        settings_dir = storage_dir / "settings"
        settings_file = settings_dir / "group_llm_settings.json"
        
        # Create directories if they don't exist
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        return settings_file
    
    @staticmethod
    def load_settings():
        """Load group LLM settings from file"""
        settings_file = GroupLLMSettings.get_settings_file_path()
        
        try:
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                logger.info(f"Group LLM settings loaded from {settings_file}")
                return settings
            else:
                # Return default settings if file doesn't exist
                default_settings = {
                    "default": "azure-gpt",  # Default provider for groups not specified
                    "groups": {},             # Map of group_id to provider
                    "retrieval_params": {     # Retrieval parameters per group
                        "default": {
                            "chunks_per_doc": 3,
                            "max_total_chunks": 10,
                            "similarity_threshold": 0.75,
                            "default_language": "en"
                        }
                    }
                }
                
                # Write default settings to file
                with open(settings_file, 'w') as f:
                    json.dump(default_settings, f, indent=4)
                
                logger.info(f"Created default group LLM settings at {settings_file}")
                return default_settings
                
        except Exception as e:
            logger.error(f"Error loading group LLM settings: {str(e)}")
            return {
                "default": "azure-gpt",
                "groups": {},
                "retrieval_params": {
                    "default": {
                        "chunks_per_doc": 3,
                        "max_total_chunks": 10,
                        "similarity_threshold": 0.75,
                        "default_language": "en"
                    }
                }
            }
    
    @staticmethod
    def save_settings(settings):
        """Save group LLM settings to file"""
        settings_file = GroupLLMSettings.get_settings_file_path()
        
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            logger.info(f"Group LLM settings saved to {settings_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving group LLM settings: {str(e)}")
            return False
    
    @staticmethod
    def get_provider_for_group(group_id):
        """Get the active LLM provider for a specific group"""
        settings = GroupLLMSettings.load_settings()
        
        # Check if the group has a specific provider
        if group_id in settings.get("groups", {}):
            provider = settings["groups"][group_id]
            logger.info(f"Using provider '{provider}' for group '{group_id}'")
            return provider
        
        # Use default provider if group not found
        default_provider = settings.get("default", "azure-gpt")
        logger.info(f"Group '{group_id}' not found, using default provider '{default_provider}'")
        return default_provider
    
    @staticmethod
    def get_retrieval_params_for_group(group_id):
        """Get the retrieval parameters for a specific group"""
        settings = GroupLLMSettings.load_settings()
        retrieval_params = settings.get("retrieval_params", {})
        
        # Check if the group has specific retrieval parameters
        if group_id in retrieval_params:
            params = retrieval_params[group_id]
            logger.info(f"Using custom retrieval parameters for group '{group_id}'")
            return params
        
        # Use default retrieval parameters if group not found
        default_params = retrieval_params.get("default", {
            "chunks_per_doc": 3,
            "max_total_chunks": 10,
            "similarity_threshold": 0.75,
            "default_language": "en"
        })
        logger.info(f"Group '{group_id}' not found, using default retrieval parameters")
        return default_params
    
    @staticmethod
    def get_default_language_for_group(group_id):
        """Get the default language for a specific group"""
        params = GroupLLMSettings.get_retrieval_params_for_group(group_id)
        return params.get("default_language", "en")
    
    @staticmethod
    def set_provider_for_group(group_id, provider):
        """Set the active LLM provider for a specific group"""
        settings = GroupLLMSettings.load_settings()
        
        # Initialize groups dict if not present
        if "groups" not in settings:
            settings["groups"] = {}
        
        # Update the provider for the group
        settings["groups"][group_id] = provider
        
        # Save the settings
        success = GroupLLMSettings.save_settings(settings)
        
        if success:
            logger.info(f"Set provider '{provider}' for group '{group_id}'")
        
        return success
    
    @staticmethod
    def set_retrieval_params_for_group(group_id, params):
        """Set the retrieval parameters for a specific group"""
        settings = GroupLLMSettings.load_settings()
        
        # Initialize retrieval_params dict if not present
        if "retrieval_params" not in settings:
            settings["retrieval_params"] = {
                "default": {
                    "chunks_per_doc": 3,
                    "max_total_chunks": 10,
                    "similarity_threshold": 0.75,
                    "default_language": "en"
                }
            }
        
        # Update the retrieval parameters for the group
        settings["retrieval_params"][group_id] = params
        
        # Save the settings
        success = GroupLLMSettings.save_settings(settings)
        
        if success:
            logger.info(f"Set retrieval parameters for group '{group_id}': {params}")
        
        return success
    
    @staticmethod
    def set_default_provider(provider):
        """Set the default LLM provider for groups without specific settings"""
        settings = GroupLLMSettings.load_settings()
        
        # Update the default provider
        settings["default"] = provider
        
        # Save the settings
        success = GroupLLMSettings.save_settings(settings)
        
        if success:
            logger.info(f"Set default provider to '{provider}'")
        
        return success
    
    @staticmethod
    def get_all_group_providers():
        """Get a dictionary of all group-provider mappings"""
        settings = GroupLLMSettings.load_settings()
        
        result = {
            "default": settings.get("default", "azure-gpt"),
            "groups": settings.get("groups", {})
        }
        
        return result