# auth/standalone_settings.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import os
import json
import datetime
import glob
from pathlib import Path
import dash
from dash import html, dcc, dash_table, callback_context, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import traceback
import logging
from dash.exceptions import PreventUpdate
from services.group_llm_settings import GroupLLMSettings
from components.group_llm_settings_ui import create_group_llm_settings_ui, GroupLLMSettings
from components.prompt_settings_ui import create_prompt_settings_ui, register_prompt_settings_callbacks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define colors (same as your admin page)
COLORS = {
    'primary': '#6b5b95',      # Main violet
    'secondary': '#b8b8d1',    # Lighter violet
    'background': '#f8f7fc',   # Very light violet background
    'text': '#2d283e',         # Dark violet text
    'accent': '#8677aa',       # Mid-tone violet
    'light_accent': '#e6e4ed', # Very light violet for borders
    'white': '#ffffff',        # Pure white
    'shadow': '0 4px 12px rgba(107, 91, 149, 0.12)' # Violet-tinted shadow
}

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
                logger.info(f"Route_settings.py - default - get_all_directories : Found {len(groups)} groups in groups file")
            except Exception as e:
                logger.error(f"Route_settings.py - default - get_all_directories : Error loading groups file: {e}")
    
    # If no groups found in the groups file, look for directories in original_files
    if not groups:
        original_files_dir = storage_dir / "original_files"
        if original_files_dir.exists() and original_files_dir.is_dir():
            # Get all subdirectories
            for item in original_files_dir.iterdir():
                if item.is_dir():
                    group_id = item.name
                    groups[group_id] = {
                        "name": group_id.capitalize(),
                        "description": f"Group for {group_id} documents",
                        "created_at": str(datetime.datetime.now())
                    }
            logger.info(f"Route_settings.py - default - get_all_directories : Found {len(groups)} groups in original_files directory")
    
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
                        "created_at": str(datetime.datetime.now())
                    }
            logger.info(f"Route_settings.py - default - get_all_directories : Found {len(groups)} groups in vector_stores directory")
    
    # Create a default group if no groups found
    if not groups:
        logger.info("No groups found, creating default group")
        groups["default"] = {
            "name": "Default",
            "description": "Default group for documents",
            "created_at": str(datetime.datetime.now())
        }
    
    return groups


def create_welcome_table_data():
    """Create table data for welcome messages with defaults for missing messages"""
    # Get groups and welcome messages
    groups = get_all_directories()
    welcome_messages = get_welcome_messages()
    
    logger.info(f"Creating table data from {len(groups)} groups and {len(welcome_messages)} welcome messages")
    
    # Create table data
    table_data = []
    for group_id, group_info in groups.items():
        group_name = group_info.get("name", group_id.capitalize())
        
        # Use existing welcome message if available, otherwise use default
        if group_id in welcome_messages:
            welcome_message = welcome_messages[group_id]
            logger.info(f"Using existing message for {group_id}: {welcome_message[:30]}...")
        else:
            welcome_message = f"Welcome to {group_name}! You can ask questions about documents in this group."
            logger.info(f"Route_settings.py - default - create_welcome_table_data : Using default message for {group_id}")
        
        table_data.append({
            "group_id": group_id,
            "group_name": group_name,
            "welcome_message": welcome_message
        })
    
    logger.info(f"Route_settings.py - default - create_welcome_table_data : Created table data with {len(table_data)} rows")
    return table_data

def get_welcome_messages():
    """Get welcome messages using absolute paths without adding defaults"""
    try:
        # Use absolute paths for reliability
        base_dir = Path(os.getcwd())
        storage_dir = base_dir / "storage"
        settings_dir = storage_dir / "settings"
        welcome_messages_file = settings_dir / "group_welcome_messages.json"
        abs_path = welcome_messages_file.absolute()
        
        logger.info(f"Route_settings.py - default - get_welcome_messages : \n--- LOADING WELCOME MESSAGES ---")
        logger.info(f"Route_settings.py - default - get_welcome_messages : Looking for welcome messages at: {abs_path}")
        
        # Empty dictionary if file doesn't exist
        welcome_messages = {}
        
        # Only load from file if it exists
        if abs_path.exists():
            try:
                with open(abs_path, "r") as f:
                    welcome_messages = json.load(f)
                logger.info(f"Route_settings.py - default - get_welcome_messages : Successfully loaded {len(welcome_messages)} welcome messages")
                logger.info(f"Route_settings.py - default - get_welcome_messages : Messages: {welcome_messages}")
            except Exception as e:
                logger.error(f"Route_settings.py - default - get_welcome_messages : Error loading welcome messages: {e}")
        else:
            logger.info(f"Route_settings.py - default - get_welcome_messages : Welcome messages file does not exist at: {abs_path}")
        
        return welcome_messages
        
    except Exception as e:
        logger.error(f"Route_settings.py - default - get_welcome_messages : ERROR in get_welcome_messages: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

# Replace the save_welcome_messages function in standalone_settings.py with this improved version

def save_welcome_messages(messages):
    """Direct implementation of welcome message saving with absolute paths"""
    try:
        # Use absolute paths for reliability
        base_dir = Path(os.getcwd())
        storage_dir = base_dir / "storage"
        settings_dir = storage_dir / "settings"
        
        # Create directories if they don't exist
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the absolute path to the welcome messages file
        welcome_messages_file = settings_dir / "group_welcome_messages.json"
        abs_path = welcome_messages_file.absolute()
        
        logger.info(f"Route_settings.py - default - save_welcome_messages : \n--- SAVING WELCOME MESSAGES ---")
        logger.info(f"Route_settings.py - default - save_welcome_messages : Current directory: {base_dir}")
        logger.info(f"Route_settings.py - default - save_welcome_messages : Storage directory: {storage_dir}")
        logger.info(f"Route_settings.py - default - save_welcome_messages : Settings directory: {settings_dir}")
        logger.info(f"Route_settings.py - default - save_welcome_messages : Welcome messages file: {abs_path}")
        logger.info(f"Route_settings.py - default - save_welcome_messages : Saving {len(messages)} messages: {messages}")
        
        # Write to file with error handling
        try:
            with open(abs_path, "w") as f:
                json.dump(messages, f, indent=4)
            logger.info(f"Route_settings.py - default - save_welcome_messages : File written successfully to {abs_path}")
        except Exception as write_error:
            logger.error(f"Route_settings.py - default - save_welcome_messages ~1: ERROR writing file: {str(write_error)}")
            return False
        
        # Verify file was written correctly
        try:
            # Check file exists
            if not abs_path.exists():
                logger.info(f"Route_settings.py - default - save_welcome_messages ~2: ERROR: File doesn't exist after writing: {abs_path}")
                return False
            
            # Check file size
            file_size = abs_path.stat().st_size
            logger.info(f"File size: {file_size} bytes")
            
            # Read back content
            with open(abs_path, "r") as f:
                saved_content = json.load(f)
            
            logger.info(f"Route_settings.py - default - save_welcome_messages : Successfully read back {len(saved_content)} messages")
            logger.info(f"Route_settings.py - default - save_welcome_messages : Content verification: {'PASS' if saved_content == messages else 'FAIL'}")
            
            return True
        except Exception as verify_error:
            logger.error(f"Route_settings.py - default - save_welcome_messages ~3: ERROR verifying file: {str(verify_error)}")
            return False
            
    except Exception as e:
        logger.error(f"Route_settings.py - default - save_welcome_messages ~4: ERROR in save_welcome_messages: {str(e)}")
        import traceback
        logger.info(traceback.format_exc())
        return False

def get_group_llm_settings(group_name):
    """Get LLM provider settings with improved debugging"""
    try:
        # Define paths
        from pathlib import Path
        import json
        import os
        
        storage_dir = Path("storage")
        settings_dir = storage_dir / "settings"
        llm_settings_file = settings_dir / "group_llm_settings.json"
        
        logger.info(f"Route_settings.py - default - get_llm_settings : \n--- GETTING GROUP LLM SETTINGS ---")
        logger.info(f"Route_settings.py - default - get_llm_settings : Current working directory: {os.getcwd()}")
        logger.info(f"Route_settings.py - default - get_llm_settings : Looking for settings at: {llm_settings_file.absolute()}")

        settings_dir.mkdir(parents=True, exist_ok=True)
        if llm_settings_file.exists():
            try:
                with open(llm_settings_file, "r") as f:
                    settings = json.load(f)
                
                file_size = llm_settings_file.stat().st_size
                logger.info(f"Successfully loaded LLM settings from {llm_settings_file} ({file_size} bytes)")
                if 'retrieval_params' not in settings or not settings['retrieval_params']:
                    logger.info("Warning: Group retrieval params not found in settings, using default")
                    return {
                        "chunks_per_doc": 3,
                        "max_total_chunks": 10,
                        "similarity_threshold": 0.1,
                        "default_language": "en"
                    }
                else:
                    group_llm_param = settings['retrieval_params'][group_name]
                    return group_llm_param
            except Exception as e:
                logger.error(f"Route_settings.py - default - group_get_llm_settings: Error loading LLM settings file: {str(e)}")
    except Exception as e:
        logger.error(f"Route_settings.py - default - group_get_llm_settings: Error loading LLM settings file: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "chunks_per_doc": 3,
            "max_total_chunks": 10,
            "similarity_threshold": 0.1,
            "default_language": "en"
        }

def get_llm_settings():
    """Get LLM provider settings with improved debugging"""
    try:
        # Define paths
        from pathlib import Path
        import json
        import os
        
        storage_dir = Path("storage")
        settings_dir = storage_dir / "settings"
        llm_settings_file = settings_dir / "llm_settings.json"
        
        logger.info(f"Route_settings.py - default - get_llm_settings : \n--- GETTING LLM SETTINGS ---")
        logger.info(f"Route_settings.py - default - get_llm_settings : Current working directory: {os.getcwd()}")
        logger.info(f"Route_settings.py - default - get_llm_settings : Looking for settings at: {llm_settings_file.absolute()}")
        
        # Ensure settings directory exists
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if settings file exists
        if llm_settings_file.exists():
            try:
                with open(llm_settings_file, "r") as f:
                    settings = json.load(f)
                
                file_size = llm_settings_file.stat().st_size
                logger.info(f"Successfully loaded LLM settings from {llm_settings_file} ({file_size} bytes)")
                logger.info(f"Provider from file: {settings.get('provider', 'NOT FOUND')}")
                
                # Validate the settings
                if 'provider' not in settings or not settings['provider']:
                    logger.info("Warning: Provider not found in settings, using default")
                    settings['provider'] = "azure-gpt"
                
                # Debug: print provider sections
                for provider in ["azure", "claude", "gemini", "llama"]:
                    if provider in settings:
                        logger.info(f"Route_settings.py - default - get_llm_settings : Found {provider} settings: keys = {list(settings[provider].keys())}")
                    else:
                        logger.info(f"Route_settings.py - default - get_llm_settings : Warning: {provider} section missing from settings")
                
                return settings
            except Exception as e:
                logger.error(f"Route_settings.py - default - get_llm_settings ~1: Error loading LLM settings file: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info(f"Route_settings.py - default - get_llm_settings : Settings file does not exist, will create default")
        
        # Create default settings
        default_settings = {
            "provider": "azure-gpt",
            "azure": {
                "api_key": "",
                "endpoint": "",
                "deployment_name": "",
                "api_version": "2023-05-15",
            },
            "claude": {
                "api_key": "",
                "model": "claude-3-opus-20240229",
                "api_version": "2024-02-15",
            },
            "gemini": {
                "api_key": "",
                "project_id": "",
                "model": "gemini-pro",
                "api_version": "v1",
            },
            "llama": {
                "api_key": "",
                "endpoint": "",
                "model": "",
                "api_version": "",
            },
            "retrieval": {
                "chunks_per_doc": 2,
                "max_total_chunks":10 ,
                "similarity_threshold": 0.1
            }
        }
        
        # Save default settings
        try:
            with open(llm_settings_file, "w") as f:
                json.dump(default_settings, f, indent=4)
            
            logger.info(f"Route_settings.py - default - get_llm_settings : Created default LLM settings at {llm_settings_file}")
        except Exception as e:
            logger.error(f"Route_settings.py - default - get_llm_settings ~2 : Error creating default settings file: {str(e)}")
        
        return default_settings
        
    except Exception as e:
        logger.error(f"Route_settings.py - default - get_llm_settings ~3: Error in get_llm_settings: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "provider": "azure-gpt",
            "azure": {
                "api_key": "",
                "endpoint": "",
                "deployment_name": "",
                "api_version": "2023-05-15",
            }
        }

def save_llm_settings(settings):
    """Save LLM settings to storage/settings directory with enhanced debugging"""
    try:
        from pathlib import Path
        import json
        import os
        
        # Debug settings being saved
        logger.info(f"\n--- SAVING LLM SETTINGS ---")
        logger.info(f"Provider being saved: {settings.get('provider', 'NOT FOUND')}")
        for provider in ["azure", "claude", "gemini", "llama"]:
            if provider in settings:
                provider_settings = settings[provider]
                # Redact API keys for security in logs
                keys_to_show = {k: (v[:4] + '****' if k == 'api_key' and v else v) 
                              for k, v in provider_settings.items()}
                logger.info(f"{provider} settings: {keys_to_show}")
        
        if "retrieval" in settings:
            logger.info(f"Retrieval settings: {settings['retrieval']}")
        
        # Setup paths
        storage_dir = Path("storage")
        settings_dir = storage_dir / "settings"
        llm_settings_file = settings_dir / "llm_settings.json"
        
        logger.info(f"Route_settings.py - default - save_llm_settings : Current working directory: {os.getcwd()}")
        logger.info(f"Route_settings.py - default - save_llm_settings : Saving settings to: {llm_settings_file.absolute()}")
        
        # Ensure settings directory exists
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Save settings
        with open(llm_settings_file, "w") as f:
            json.dump(settings, f, indent=4)
        
        # Verify the file was written correctly
        if llm_settings_file.exists():
            file_size = llm_settings_file.stat().st_size
            logger.info(f"Route_settings.py - default - save_llm_settings : File written successfully. Size: {file_size} bytes")
            
            # Verify contents
            with open(llm_settings_file, "r") as f:
                saved_settings = json.load(f)
            
            logger.info(f"Route_settings.py - default - save_llm_settings : Verification: Provider in saved file: {saved_settings.get('provider', 'NOT FOUND')}")
            logger.info(f"Route_settings.py - default - save_llm_settings : Saved LLM settings to {llm_settings_file}")
            return True
        else:
            logger.error(f"Route_settings.py - default - save_llm_settings ~1 : Error: File not created at {llm_settings_file}")
            return False
            
    except Exception as e:
        logger.info(f"Route_settings.py - default - save_llm_settings ~2 : Error saving LLM settings: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Fixed version of save_settings callback function
def save_settings(n_clicks, table_data, 
                 provider, azure_api_key, azure_endpoint, azure_deployment, azure_api_version,
                 claude_api_key, claude_model, claude_api_version,
                 gemini_api_key, gemini_project_id, gemini_model, gemini_api_version,
                 llama_api_key, llama_endpoint, llama_model, llama_api_version,
                 chunks_per_doc, max_total_chunks, similarity_threshold):
    """Save both welcome messages and LLM settings together with enhanced debugging"""
    if not n_clicks:
        return no_update, no_update, no_update
        
    try:
        logger.info(f"\n--- SAVE SETTINGS TRIGGERED ---")
        
        # Part 1: Save welcome messages
        logger.info(f"Route_settings.py - default - save_settings :Save button clicked, processing welcome messages...")
        
        # Convert table data to welcome messages format
        welcome_messages = {}
        for row in table_data:
            group_id = row.get("group_id")
            welcome_message = row.get("welcome_message")
            if group_id and welcome_message:
                welcome_messages[group_id] = welcome_message
                logger.info(f"Adding message for group {group_id}: {welcome_message[:50]}...")
        
        logger.info(f"Route_settings.py - default - save_settings : Processed {len(welcome_messages)} welcome messages")
        
        # Save welcome messages to file
        save_welcome_success = save_welcome_messages(welcome_messages)
        logger.info(f"Welcome messages save result: {'SUCCESS' if save_welcome_success else 'FAILED'}")
        
        # Part 2: Save LLM settings
        logger.info(f"Route_settings.py - default - save_settings :Processing LLM settings...")
        logger.info(f"Route_settings.py - default - save_settings :Selected provider: {provider}")
        logger.info(f"Route_settings.py - default - save_settings :Azure endpoint: {azure_endpoint}")
        logger.info(f"Route_settings.py - default - save_settings :Azure deployment: {azure_deployment}")
        logger.info(f"Route_settings.py - default - save_settings :Claude model: {claude_model}")
        logger.info(f"Route_settings.py - default - save_settings :Retrieval settings: chunks_per_doc={chunks_per_doc}, max_total_chunks={max_total_chunks}")
        
        # Create LLM settings object
        llm_settings = {
            "provider": provider,
            "azure": {
                "api_key": azure_api_key or "",
                "endpoint": azure_endpoint or "",
                "deployment_name": azure_deployment or "",
                "api_version": azure_api_version or "2023-05-15",
            },
            "claude": {
                "api_key": claude_api_key or "",
                "model": claude_model or "claude-3-opus-20240229",
                "api_version": claude_api_version or "2024-02-15",
            },
            "gemini": {
                "api_key": gemini_api_key or "",
                "project_id": gemini_project_id or "",
                "model": gemini_model or "gemini-pro",
                "api_version": gemini_api_version or "v1",
            },
            "llama": {
                "api_key": llama_api_key or "",
                "endpoint": llama_endpoint or "",
                "model": llama_model or "",
                "api_version": llama_api_version or "",
            },
            "retrieval": {
                "chunks_per_doc": chunks_per_doc if chunks_per_doc is not None else 3,
                "max_total_chunks": max_total_chunks if max_total_chunks is not None else 10,
                "similarity_threshold": similarity_threshold if similarity_threshold is not None else 0.75
            }
        }
        
        # Save LLM settings
        save_llm_success = save_llm_settings(llm_settings)
        logger.info(f"Route_settings.py - default - save_settings :LLM settings save result: {'SUCCESS' if save_llm_success else 'FAILED'}")
        
        # Verify the settings were saved correctly
        logger.info("Route_settings.py - default - save_settings :Verifying saved settings...")
        debug_exists, debug_content = debug_settings_file()
        
        # Determine overall success
        if save_welcome_success and save_llm_success:
            logger.info("Route_settings.py - default - save_settings :Save operation successful")
            return True, "Success", "Welcome messages and LLM settings saved successfully"
        elif save_welcome_success:
            logger.info("Route_settings.py - default - save_settings :Only welcome messages saved successfully")
            return True, "Partial Success", "Welcome messages saved, but LLM settings failed"
        elif save_llm_success:
            logger.info("Route_settings.py - default - save_settings :Only LLM settings saved successfully")
            return True, "Partial Success", "LLM settings saved, but welcome messages failed"
        else:
            logger.info("Route_settings.py - default - save_settings :Save operation failed completely")
            return True, "Error", "Failed to save both welcome messages and LLM settings"
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Route_settings.py - default - save_settings :Error saving settings: {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        return True, "Error", f"Error saving settings: {error_msg}"

def debug_settings_file():
    """Print the contents of the settings file for debugging"""
    from pathlib import Path
    import json
    
    try:
        storage_dir = Path("storage")
        settings_dir = storage_dir / "settings"
        llm_settings_file = settings_dir / "llm_settings.json"
        
        logger.debug(f"Route_settings.py - default - debug_settings_file : \n--- DEBUG SETTINGS FILE ---")
        logger.debug(f"Route_settings.py - default - debug_settings_file : Settings file path: {llm_settings_file.absolute()}")
        
        if llm_settings_file.exists():
            with open(llm_settings_file, "r") as f:
                settings = json.load(f)
            
            logger.debug(f"Route_settings.py - default - debug_settings_file : File exists with size: {llm_settings_file.stat().st_size} bytes")
            logger.debug(f"Route_settings.py - default - debug_settings_file : Settings content (first 200 chars): {json.dumps(settings)[:200]}...")
            logger.debug(f"Route_settings.py - default - debug_settings_file : Provider in file: {settings.get('provider', 'NOT FOUND')}")
            
            # Check for key sections
            logger.info(f"Route_settings.py - default - debug_settings_file : Has azure section: {'azure' in settings}")
            logger.info(f"Route_settings.py - default - debug_settings_file : Has claude section: {'claude' in settings}")
            logger.info(f"Route_settings.py - default - debug_settings_file : Has retrieval section: {'retrieval' in settings}")
            
            return True, settings
        else:
            logger.info("Route_settings.py - default - debug_settings_file : Settings file does not exist")
            return False, None
            
    except Exception as e:
        logger.error(f"Route_settings.py - default - debug_settings_file : Error reading settings file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)
    
def create_standalone_settings():
    """Create a standalone settings page"""
    # Get table data using our consistent function
    table_data = create_welcome_table_data()
    
    # Get groups for debug info
    groups = get_all_directories()
    
    return html.Div(
        style={
            'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, {COLORS["white"]} 100%)',
            'minHeight': '100vh',
            'padding': '1.5rem',
        },
        children=[
            dbc.Container([
                # Header
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
                            ["Save Settings ", html.I(className="fas fa-save")],
                            id="save-standalone-settings",
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
                
                # Tabs for different settings sections
                dbc.Tabs([
                    # Welcome Messages Tab
                    dbc.Tab(
                        label="Welcome Messages",
                        tab_id="welcome-messages-tab",
                        label_style={"font-weight": "bold", "color": COLORS["text"]},
                        active_label_style={"font-weight": "bold", "color": COLORS["primary"]},
                        children=[
                            # Welcome Messages Card
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H4("Group Welcome Messages"),
                                    html.P("Configure welcome messages that will be shown to users for each group", className="text-muted mb-0")
                                ]),
                                dbc.CardBody([
                                    # Table of groups and welcome messages
                                    dash_table.DataTable(
                                        id="standalone-welcome-table",
                                        columns=[
                                            {"name": "Group ID", "id": "group_id"},
                                            {"name": "Group Name", "id": "group_name"},
                                            {"name": "Welcome Message", "id": "welcome_message", "editable": True},
                                        ],
                                        data=table_data,
                                        style_cell={
                                            'textAlign': 'left',
                                            'padding': '12px',
                                            'whiteSpace': 'normal',
                                            'height': 'auto',
                                            'fontFamily': '"Segoe UI", Arial, sans-serif',
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
                                        css=[{
                                                'selector': 'input.dash-cell-value', 
                                                'rule': 'text-align: left !important; padding-left: 12px !important;'
                                            }
                                        ],
                                        style_data_conditional=[
                                            {
                                                'if': {'row_index': 'odd'},
                                                'backgroundColor': 'rgba(248, 247, 252, 0.5)'
                                            },
                                            # this for edit mode
                                            {
                                                'if': {'state': 'active'},  # When cell is being edited
                                                'backgroundColor': 'rgba(220, 220, 255, 0.30)',
                                                'border': '1px solid #6b5b95'
                                            }
                                        ],
                                        style_header={
                                            'backgroundColor': COLORS['light_accent'],
                                            'fontWeight': 'bold',
                                            'color': COLORS['text'],
                                            'borderBottom': f'1px solid {COLORS["secondary"]}',
                                        },
                                        editable=True,
                                        row_selectable=False,
                                        page_size=10,
                                    ),
                                    
                                    html.Div([
                                        html.I(className="fas fa-info-circle me-2", style={'color': COLORS['primary']}),
                                        html.Span("Edit welcome messages by clicking on the cells and clicking Save Settings")
                                    ], className="mt-3", style={"fontSize": "0.9rem", "color": COLORS['text']}),
                                ]),
                            ], className="mb-4", style={
                                'borderRadius': '12px',
                                'border': f'1px solid {COLORS["light_accent"]}',
                                'boxShadow': COLORS['shadow'],
                            }),
                        ]
                    ),
                    
                    # LLM Settings Tab
                    dbc.Tab(
                        label="LLM Providers",
                        tab_id="llm-providers-tab",
                        label_style={"font-weight": "bold", "color": COLORS["text"]},
                        active_label_style={"font-weight": "bold", "color": COLORS["primary"]},
                        children=[
                            # LLM Settings Card
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H4("LLM Provider Settings"),
                                    html.P("Configure which AI model to use for answering questions", className="text-muted mb-0")
                                ]),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("AI Provider", style={'fontWeight': '500'}),
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
                                                    'marginBottom': '1rem',
                                                }
                                            ),
                                        ], md=6),
                                    ]),
                                    
                                    # Azure OpenAI Settings Card
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
                                        ])
                                    ], id="azure-settings-card", style={'marginBottom': '1rem', 'border': f'1px solid {COLORS["light_accent"]}'}),
                                    
                                    # Claude Settings Card
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Claude Anthropic Settings")),
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
                                                    dbc.Label("API Version", style={'fontWeight': '500'}),
                                                    dbc.Input(
                                                        id="claude-api-version",
                                                        type="text",
                                                        placeholder="2024-02-15",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                            ]),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Model", style={'fontWeight': '500'}),
                                                    dbc.Select(
                                                        id="claude-model",
                                                        options=[
                                                            {"label": "Claude 3.5 Sonnet", "value": "claude-3-5-sonnet-20240620"},
                                                            {"label": "Claude 3 Opus", "value": "claude-3-opus-20240229"},
                                                            {"label": "Claude 3 Haiku", "value": "claude-3-haiku-20240307"}
                                                        ],
                                                        value="claude-3-opus-20240229",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                            ]),
                                        ])
                                    ], id="claude-settings-card", style={'display': 'none'}),
                                    
                                    # Gemini Settings Card
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("Google Gemini Settings")),
                                        dbc.CardBody([
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("API Key", style={'fontWeight': '500'}),
                                                    dbc.Input(
                                                        id="gemini-api-key",
                                                        type="password",
                                                        placeholder="Enter Google Gemini API Key",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                                dbc.Col([
                                                    dbc.Label("Project ID", style={'fontWeight': '500'}),
                                                    dbc.Input(
                                                        id="gemini-project-id",
                                                        type="text",
                                                        placeholder="Enter Google Cloud Project ID",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                            ]),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Model", style={'fontWeight': '500'}),
                                                    dbc.Select(
                                                        id="gemini-model",
                                                        options=[
                                                            {"label": "Gemini Pro", "value": "gemini-pro"},
                                                            {"label": "Gemini Ultra", "value": "gemini-ultra"},
                                                            {"label": "gemini-2.0-flash", "value": "gemini-2.0-flash"}
                                                        ],
                                                        value="gemini-pro",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                                dbc.Col([
                                                    dbc.Label("API Version", style={'fontWeight': '500'}),
                                                    dbc.Input(
                                                        id="gemini-api-version",
                                                        type="text",
                                                        placeholder="v1",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                            ]),
                                        ])
                                    ], id="gemini-settings-card", style={'display': 'none'}),
                                    
                                    # Llama Settings Card
                                    dbc.Card([
                                        dbc.CardHeader(html.H5("LLAMA API Settings")),
                                        dbc.CardBody([
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("API Key", style={'fontWeight': '500'}),
                                                    dbc.Input(
                                                        id="llama-api-key",
                                                        type="password",
                                                        placeholder="Enter LLAMA API Key",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                                dbc.Col([
                                                    dbc.Label("Endpoint", style={'fontWeight': '500'}),
                                                    dbc.Input(
                                                        id="llama-endpoint",
                                                        type="text",
                                                        placeholder="https://api.llama.ai/",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                            ]),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Model", style={'fontWeight': '500'}),
                                                    dbc.Input(
                                                        id="llama-model",
                                                        type="text",
                                                        placeholder="Enter LLAMA model name",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                                dbc.Col([
                                                    dbc.Label("API Version", style={'fontWeight': '500'}),
                                                    dbc.Input(
                                                        id="llama-api-version",
                                                        type="text",
                                                        placeholder="2024-01-15",
                                                        style={'marginBottom': '1rem'},
                                                    ),
                                                ], md=6),
                                            ]),
                                        ])
                                    ], id="llama-settings-card", style={'display': 'none'}),
                                ]),
                            ], className="mb-4", style={
                                'borderRadius': '12px',
                                'border': f'1px solid {COLORS["light_accent"]}',
                                'boxShadow': COLORS['shadow'],
                            }),
                        ]
                    ),
                    
                    # NEW: Prompts Settings Tab
                    dbc.Tab(
                        label="LLM Prompts",
                        tab_id="llm-prompts-tab",
                        label_style={"font-weight": "bold", "color": COLORS["text"]},
                        active_label_style={"font-weight": "bold", "color": COLORS["primary"]},
                        children=[
                            dbc.Card([
                                dbc.CardBody([
                                    # Include the prompt settings UI component
                                    create_prompt_settings_ui()
                                ]),
                            ], className="mb-4", style={
                                'borderRadius': '12px',
                                'border': f'1px solid {COLORS["light_accent"]}',
                                'boxShadow': COLORS['shadow'],
                            }),
                        ]
                    ),
                    
                    # Group AI Models Tab
                    dbc.Tab(
                        label="Group Models",
                        tab_id="group-models-tab",
                        label_style={"font-weight": "bold", "color": COLORS["text"]},
                        active_label_style={"font-weight": "bold", "color": COLORS["primary"]},
                        children=[
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H4("Group AI Models Settings"),
                                    html.P("Configure which AI model each group uses.", className="text-muted mb-4"),
                                ]),
                                dbc.CardBody([
                                    create_group_llm_settings_ui()
                                ]),
                            ], className="mb-4", style={
                                'borderRadius': '12px',
                                'border': f'1px solid {COLORS["light_accent"]}',
                                'boxShadow': COLORS['shadow'],
                            }),
                        ]
                    ),
                    
                    # Debug Info Tab
                    dbc.Tab(
                        label="Debug Info",
                        tab_id="debug-tab",
                        label_style={"font-weight": "bold", "color": COLORS["text"]},
                        active_label_style={"font-weight": "bold", "color": COLORS["primary"]},
                        children=[
                            # Debug Info Card
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H4("Debug Information"),
                                    html.P("Details about detected groups and settings", className="text-muted mb-0")
                                ]),
                                dbc.CardBody([
                                    html.H5("Detected Groups:"),
                                    html.Ul([
                                        html.Li([
                                            html.Strong(f"{group_id}: "),
                                            html.Span(f"{group_info.get('name', 'Unknown')} - {group_info.get('description', 'No description')}")
                                        ])
                                        for group_id, group_info in groups.items()
                                    ]),
                                    
                                    html.H5("Storage Information:"),
                                    html.P(f"Current Working Directory: {os.getcwd()}"),
                                    html.P(f"Storage Directory: {Path('storage').absolute()}"),
                                    
                                    # Show existing directories
                                    html.H5("Existing Directories:"),
                                    html.Ul([
                                        html.Li(str(path.absolute()))
                                        for path in Path("storage").glob("*")
                                        if path.is_dir()
                                    ]),
                                ]),
                            ], className="mb-4", style={
                                'borderRadius': '12px',
                                'border': f'1px solid {COLORS["light_accent"]}',
                                'boxShadow': COLORS['shadow'],
                            }),
                        ]
                    ),
                ], 
                id="settings-tabs", 
                active_tab="welcome-messages-tab",
                className="custom-tabs",
                style={"border-bottom": f"1px solid {COLORS['light_accent']}"}
                ),
                
                # Success Toast
                dbc.Toast(
                    id="standalone-feedback-toast",
                    header="Settings Saved",
                    is_open=False,
                    dismissable=True,
                    duration=4000,
                    style={"position": "fixed", "top": 66, "right": 10, "width": 350, "boxShadow": COLORS['shadow']},
                ),
            ], fluid=True)
        ]
    )

def register_group_llm_callbacks(app):
    """Register callbacks for the group LLM settings UI"""
    from dash.dependencies import Input, Output, State
    from dash.exceptions import PreventUpdate
    from services.group_llm_settings import GroupLLMSettings
    import logging
    
    logger = logging.getLogger(__name__)
    
    @app.callback(
        [
            Output("group-llm-feedback", "children"),
            Output("group-llm-feedback", "style"),
            Output("group-llm-feedback", "className")
        ],
        Input("save-group-llm-button", "n_clicks"),
        State("group-llm-table", "data"),
        prevent_initial_call=True
    )
    def save_group_llm_settings(n_clicks, table_data):
        """Save the group LLM settings when the save button is clicked"""
        if not n_clicks or not table_data:
            raise PreventUpdate
        
        try:
            # Initialize settings structure
            settings = {
                "default": "azure-gpt",
                "groups": {},
                "retrieval_params": {}
            }
            
            # Process each row in the table
            for row in table_data:
                group_id = row.get("group_id")
                provider = row.get("provider")
                
                # Retrieve retrieval parameters
                chunks_per_doc = int(row.get("chunks_per_doc", 3))
                max_total_chunks = int(row.get("max_total_chunks", 10))
                
                # Make sure similarity_threshold is a float between 0 and 1
                try:
                    similarity_threshold = float(row.get("similarity_threshold", 0.75))
                    similarity_threshold = max(0, min(1, similarity_threshold))  # Clamp between 0 and 1
                except:
                    similarity_threshold = 0.75
                
                default_language = row.get("default_language", "en")
                
                # Validate values are in acceptable ranges
                chunks_per_doc = max(1, min(20, chunks_per_doc))  # Clamp between 1 and 20
                max_total_chunks = max(1, min(50, max_total_chunks))  # Clamp between 1 and 50
                
                if not group_id or not provider:
                    continue
                
                # Store retrieval parameters for this group
                settings["retrieval_params"][group_id] = {
                    "chunks_per_doc": chunks_per_doc,
                    "max_total_chunks": max_total_chunks,
                    "similarity_threshold": similarity_threshold,
                    "default_language": default_language
                }
                
                if group_id == "default":
                    # Handle the default provider
                    settings["default"] = provider
                else:
                    # Add to groups dictionary
                    settings["groups"][group_id] = provider
            
            # Ensure default retrieval parameters exist
            if "default" not in settings["retrieval_params"]:
                settings["retrieval_params"]["default"] = {
                    "chunks_per_doc": 3,
                    "max_total_chunks": 10,
                    "similarity_threshold": 0.75,
                    "default_language": "en"
                }
            
            # Save the settings
            success = GroupLLMSettings.save_settings(settings)
            
            if success:
                logger.info("Group settings saved successfully")
                return (
                    "Group settings saved successfully!",
                    {"display": "block", "padding": "10px", "borderRadius": "5px"},
                    "alert alert-success"
                )
            else:
                logger.error("Failed to save group settings")
                return (
                    "Error: Failed to save group settings.",
                    {"display": "block", "padding": "10px", "borderRadius": "5px"},
                    "alert alert-danger"
                )
                
        except Exception as e:
            logger.error(f"Error saving group settings: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return (
                f"Error: {str(e)}",
                {"display": "block", "padding": "10px", "borderRadius": "5px"},
                "alert alert-danger"
            )
    
    # Add this callback to load the group LLM table data on page load or tab activation
    @app.callback(
        Output("group-llm-table", "data"),
        Input("settings-tabs", "active_tab"),
        prevent_initial_call=True
    )
    def refresh_group_llm_table(active_tab):
        """Refresh the group LLM table when the tab is activated"""
        from components.group_llm_settings_ui import create_group_llm_table_data
        
        if active_tab == "group-models-tab":
            try:
                # Create fresh table data
                table_data = create_group_llm_table_data()
                logger.info(f"Refreshed group LLM table with {len(table_data)} rows")
                return table_data
            except Exception as e:
                logger.error(f"Error refreshing group LLM table: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return []
        raise PreventUpdate
    
def register_standalone_callbacks(app):
    """Register callbacks for the standalone settings page"""
    # Register our combined save callback (for both welcome messages and LLM settings)
    @app.callback(
        [Output("standalone-feedback-toast", "is_open"),
        Output("standalone-feedback-toast", "header"),
        Output("standalone-feedback-toast", "children")],
        [Input("save-standalone-settings", "n_clicks")],
        [State("standalone-welcome-table", "data"),
        State("llm-provider-select", "value"),
        State("azure-api-key", "value"),
        State("azure-endpoint", "value"),
        State("azure-deployment-name", "value"),
        State("azure-api-version", "value"),
        State("claude-api-key", "value"),
        State("claude-model", "value"),
        State("claude-api-version", "value"),
        State("gemini-api-key", "value"),
        State("gemini-project-id", "value"),
        State("gemini-model", "value"),
        State("gemini-api-version", "value"),
        State("llama-api-key", "value"),
        State("llama-endpoint", "value"),
        State("llama-model", "value"),
        State("llama-api-version", "value")],
        prevent_initial_call=True
    )
    def save_settings(n_clicks, table_data, 
                    provider, azure_api_key, azure_endpoint, azure_deployment, azure_api_version,
                    claude_api_key, claude_model, claude_api_version,
                    gemini_api_key, gemini_project_id, gemini_model, gemini_api_version,
                    llama_api_key, llama_endpoint, llama_model, llama_api_version):
        """Save both welcome messages and LLM settings together"""
        if not n_clicks:
            return no_update, no_update, no_update
            
        try:
            logger.info(f"\n--- SAVE SETTINGS TRIGGERED ---")
            
            # Part 1: Save welcome messages
            logger.info(f"Save button clicked, processing welcome messages...")
            
            # Convert table data to welcome messages format
            welcome_messages = {}
            for row in table_data:
                group_id = row.get("group_id")
                welcome_message = row.get("welcome_message")
                if group_id and welcome_message:
                    welcome_messages[group_id] = welcome_message
                    logger.info(f"Adding message for group {group_id}: {welcome_message[:50]}...")
            
            logger.info(f"Processed {len(welcome_messages)} welcome messages")
            
            # Save welcome messages to file
            save_welcome_success = save_welcome_messages(welcome_messages)
            logger.info(f"Welcome messages save result: {'SUCCESS' if save_welcome_success else 'FAILED'}")
            
            # Part 2: Save LLM settings
            logger.info(f"Processing LLM settings...")
            logger.info(f"Selected provider: {provider}")
            logger.info(f"Azure endpoint: {azure_endpoint}")
            logger.info(f"Azure deployment: {azure_deployment}")
            logger.info(f"Claude model: {claude_model}")
            
            # Create LLM settings object
            llm_settings = {
                "provider": provider,
                "azure": {
                    "api_key": azure_api_key or "",
                    "endpoint": azure_endpoint or "",
                    "deployment_name": azure_deployment or "",
                    "api_version": azure_api_version or "2023-05-15",
                },
                "claude": {
                    "api_key": claude_api_key or "",
                    "model": claude_model or "claude-3-opus-20240229",
                    "api_version": claude_api_version or "2024-02-15",
                },
                "gemini": {
                    "api_key": gemini_api_key or "",
                    "project_id": gemini_project_id or "",
                    "model": gemini_model or "gemini-pro",
                    "api_version": gemini_api_version or "v1",
                },
                "llama": {
                    "api_key": llama_api_key or "",
                    "endpoint": llama_endpoint or "",
                    "model": llama_model or "",
                    "api_version": llama_api_version or "",
                }
            }
            
            # Preserve any existing retrieval settings
            existing_settings = get_llm_settings()
            if "retrieval" in existing_settings:
                llm_settings["retrieval"] = existing_settings["retrieval"]
            
            # Save LLM settings
            save_llm_success = save_llm_settings(llm_settings)
            logger.info(f"LLM settings save result: {'SUCCESS' if save_llm_success else 'FAILED'}")
            
            # Determine overall success
            if save_welcome_success and save_llm_success:
                logger.info("Save operation successful")
                return True, "Success", "Welcome messages and LLM settings saved successfully"
            elif save_welcome_success:
                logger.info("Only welcome messages saved successfully")
                return True, "Partial Success", "Welcome messages saved, but LLM settings failed"
            elif save_llm_success:
                logger.info("Only LLM settings saved successfully")
                return True, "Partial Success", "LLM settings saved, but welcome messages failed"
            else:
                logger.info("Save operation failed completely")
                return True, "Error", "Failed to save both welcome messages and LLM settings"
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error saving settings: {error_msg}")
            import traceback
            logger.info(traceback.format_exc())
            return True, "Error", f"Error saving settings: {error_msg}"
        
    @app.callback(
        [Output("azure-settings-card", "style", allow_duplicate=True),
        Output("claude-settings-card", "style", allow_duplicate=True),
        Output("gemini-settings-card", "style", allow_duplicate=True),
        Output("llama-settings-card", "style", allow_duplicate=True)],
        [Input("llm-provider-select", "value")],
        prevent_initial_call=True
    )
    def toggle_provider_settings(provider):
        """Show/hide provider settings based on selection"""
        from dash.exceptions import PreventUpdate
        
        # If the components don't exist, don't update
        try:
            # Define styles
            hidden_style = {'display': 'none'}
            visible_style = {'display': 'block', 'marginBottom': '1rem', 'border': f'1px solid {COLORS["light_accent"]}'}
            
            # Default - all hidden
            azure_style = hidden_style
            claude_style = hidden_style
            gemini_style = hidden_style
            llama_style = hidden_style
            
            # Show selected provider
            if provider == "azure-gpt":
                azure_style = visible_style
            elif provider == "claude":
                claude_style = visible_style
            elif provider == "gemini":
                gemini_style = visible_style
            elif provider == "llama":
                llama_style = visible_style
            
            return azure_style, claude_style, gemini_style, llama_style
        except:
            raise PreventUpdate
    
    @app.callback(
        Output("standalone-welcome-table", "data"),
        [Input("url", "pathname")]
    )
    def refresh_welcome_table(pathname):
        """Refresh the welcome messages table when navigating to the settings page"""
        # Only update when on the settings page
        if pathname != '/settings':
            return no_update
        
        logger.info("Route_settings.py - register_standalone_callbacks - refresh_welcome_table : Refreshing welcome messages table")
        
        # Use our consistent function to create table data
        table_data = create_welcome_table_data()
        
        return table_data
    
    # Load LLM settings when navigating to settings page - SIMPLIFIED VERSION
    @app.callback(
        [Output("azure-api-key", "value"),
        Output("azure-endpoint", "value"),
        Output("azure-deployment-name", "value"),
        Output("azure-api-version", "value"),
        Output("claude-api-key", "value"),
        Output("claude-model", "value"),
        Output("claude-api-version", "value"),
        Output("gemini-api-key", "value"),
        Output("gemini-project-id", "value"),
        Output("gemini-model", "value"),
        Output("gemini-api-version", "value"),
        Output("llama-api-key", "value"),
        Output("llama-endpoint", "value"),
        Output("llama-model", "value"),
        Output("llama-api-version", "value"),
        Output("llm-provider-select", "value")],
        [Input("url", "pathname")]
    )
    def load_llm_settings(pathname):
        """Load LLM settings when settings page is opened"""
        from dash.exceptions import PreventUpdate
        
        if pathname != '/settings':
            # Return no_update for all outputs
            return [dash.no_update] * 16
            
        try:
            logger.info("\n--- LOADING LLM SETTINGS FOR SETTINGS PAGE ---")
            
            # Get the settings
            settings = get_llm_settings()
            
            logger.info(f"Settings loaded, provider is: {settings.get('provider', 'NOT FOUND')}")
            
            # Extract provider
            provider = settings.get("provider", "azure-gpt")
            
            # Extract Azure settings
            azure = settings.get("azure", {})
            azure_api_key = azure.get("api_key", "")
            azure_endpoint = azure.get("endpoint", "")
            azure_deployment = azure.get("deployment_name", "")
            azure_api_version = azure.get("api_version", "2023-05-15")
            
            # Extract Claude settings
            claude = settings.get("claude", {})
            claude_api_key = claude.get("api_key", "")
            claude_model = claude.get("model", "claude-3-opus-20240229")
            claude_api_version = claude.get("api_version", "2024-02-15")
            
            # Extract Gemini settings
            gemini = settings.get("gemini", {})
            gemini_api_key = gemini.get("api_key", "")
            gemini_project_id = gemini.get("project_id", "")
            gemini_model = gemini.get("model", "gemini-pro")
            gemini_api_version = gemini.get("api_version", "v1")
            
            # Extract Llama settings
            llama = settings.get("llama", {})
            llama_api_key = llama.get("api_key", "")
            llama_endpoint = llama.get("endpoint", "")
            llama_model = llama.get("model", "")
            llama_api_version = llama.get("api_version", "")
            
            # Debug values being returned
            logger.info(f"Returning provider: {provider}")
            logger.info(f"Returning Azure endpoint: {azure_endpoint}")
            logger.info(f"Returning Azure deployment: {azure_deployment}")
            logger.info(f"Returning Claude model: {claude_model}")
            
            return [
                azure_api_key,
                azure_endpoint,
                azure_deployment,
                azure_api_version,
                claude_api_key,
                claude_model,
                claude_api_version,
                gemini_api_key,
                gemini_project_id,
                gemini_model,
                gemini_api_version,
                llama_api_key,
                llama_endpoint,
                llama_model,
                llama_api_version,
                provider
            ]
            
        except Exception as e:
            logger.error(f"Error loading LLM settings: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return default values
            return [""] * 15 + ["azure-gpt"]
    
def verify_welcome_messages():
    """Verify that welcome messages file exists and can be read"""
    settings_dir = Path("storage/settings")
    welcome_messages_file = settings_dir / "group_welcome_messages.json"
    
    # Check if file exists
    if not welcome_messages_file.exists():
        logger.info(f"Route_settings.py - default - verify_welcome_messages : Welcome messages file does not exist: {welcome_messages_file.absolute()}")
        return False, "File does not exist"
    
    # Check if file is readable
    try:
        with open(welcome_messages_file, "r") as f:
            messages = json.load(f)
        logger.info(f"Route_settings.py - default - verify_welcome_messages : Successfully read {len(messages)} welcome messages from file")
        return True, messages
    except Exception as e:
        logger.error(f"Route_settings.py - default - verify_welcome_messages : Error reading welcome messages file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)

# Then add this line right after the save_welcome_messages function call in save_settings callback:

logger.info("Route_settings.py - default : Verifying saved messages can be read back")
verify_success, verify_result = verify_welcome_messages()
logger.info(f"Verification result: {verify_success}, {verify_result}")

# Add this function to verify LLM settings
def verify_llm_settings():
    """Verify that LLM settings file exists and can be read"""
    settings_dir = Path("storage/settings")
    llm_settings_file = settings_dir / "llm_settings.json"
    
    # Check if file exists
    if not llm_settings_file.exists():
        logger.info(f"Route_settings.py - default - verify_llm_settings : LLM settings file does not exist: {llm_settings_file.absolute()}")
        return False, "File does not exist"
    
    # Check if file is readable
    try:
        with open(llm_settings_file, "r") as f:
            settings = json.load(f)
        logger.info(f"Route_settings.py - default - verify_llm_settings : Successfully read LLM settings from file")
        #return True, settings #for debugging
        return True, None

    except Exception as e:
        logger.error(f"Route_settings.py - default - verify_llm_settings : Error reading LLM settings file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)
    
logger.info("Route_settings.py - default : Verifying llm setting can be read back")
verify_success, verify_result = verify_llm_settings()
logger.info(f"Route_settings.py - default : Verification result: {verify_success}, {verify_result}")