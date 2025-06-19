# components/prompt_settings_ui.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import logging
import json
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import colors from admin module
from auth.admin import COLORS

# This variable helps prevent duplicate registrations
CALLBACKS_REGISTERED = False

def create_prompt_settings_ui():
    """Create a UI component for prompt settings management"""
    return html.Div([
        html.H4("LLM Prompt Templates", className="mb-3"),
        html.P(
            "Configure the prompt templates used by all LLM providers. These prompts are used to interact with the AI models.",
            className="text-muted mb-4"
        ),
        
        # Prompt Type Selection
        dbc.Card([
            dbc.CardHeader("Prompt Template Editor"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Prompt Type", style={'fontWeight': '500'}),
                        dbc.Select(
                            id="prompt-type-select",
                            options=[
                                {"label": "Document Query Prompt", "value": "document_query"},
                                {"label": "System Instruction", "value": "system_instruction"},
                                {"label": "Welcome Message", "value": "welcome_message"},
                            ],
                            value="document_query",
                            style={
                                'borderColor': COLORS['light_accent'],
                                'marginBottom': '1rem',
                            }
                        ),
                    ], md=6),
                ]),
                
                # Prompt editor
                dbc.Label("Edit Prompt Template", style={'fontWeight': '500'}),
                dbc.Textarea(
                    id="prompt-template-textarea",
                    style={'height': '300px', 'fontFamily': 'monospace'},
                    className="mb-3"
                ),
                
                # Variable help text
                html.Div([
                    html.H6("Available Template Variables:", className="mb-2"),
                    html.Div([
                        dbc.Badge("{context}", color="primary", className="me-2 mb-2"),
                        dbc.Badge("{query}", color="primary", className="me-2 mb-2"),
                    ], id="prompt-variables-help"),
                    
                    html.Small([
                        "Variables will be replaced with actual values during use. ",
                        html.Br(),
                        "• {context} will be replaced with document chunks",
                        html.Br(),
                        "• {query} will be replaced with the user's question"
                    ], className="text-muted")
                ], className="mb-3 p-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                
                # Buttons
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            ["Reset to Default ", html.I(className="fas fa-undo")],
                            id="reset-prompt-button",
                            color="warning",
                            className="me-2"
                        ),
                        dbc.Button(
                            ["Save Prompt ", html.I(className="fas fa-save")],
                            id="save-prompt-button",
                            color="primary"
                        )
                    ], className="d-flex")
                ]),
                
                # Feedback message
                html.Div(id="prompt-feedback", className="mt-3")
            ])
        ], className="mb-4", style={
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["light_accent"]}',
            'boxShadow': COLORS['shadow'],
        }),
        
        # Prompt documentation and help
        dbc.Card([
            dbc.CardHeader("Prompt Documentation"),
            dbc.CardBody([
                html.H5("Understanding Prompt Types", className="mb-3"),
                
                html.Div([
                    html.H6("Document Query Prompt", className="mb-2"),
                    html.P(
                        "This is the main prompt template used when asking questions about documents. "
                        "It instructs the AI how to analyze document chunks and format responses.",
                        className="mb-3"
                    ),
                    
                    html.H6("System Instruction", className="mb-2"),
                    html.P(
                        "This prompt provides system-level instructions to the AI about its role "
                        "and how it should behave. It sets the overall context for the conversation.",
                        className="mb-3"
                    ),
                    
                    html.H6("Welcome Message", className="mb-2"),
                    html.P(
                        "This message is shown to users when they first start interacting with the system. "
                        "It introduces the capabilities and sets expectations."
                    )
                ])
            ])
        ], className="mb-4", style={
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["light_accent"]}',
            'boxShadow': COLORS['shadow'],
        }),
        
        # Hidden storage for prompt data
        dcc.Store(id="prompt-settings-store", storage_type="memory"),
        
        # Hidden div to track initialization state
        html.Div(id="prompt-settings-initialized", style={"display": "none"}),
    ])

def register_prompt_settings_callbacks(app):
    """Register callbacks for prompt settings UI"""
    global CALLBACKS_REGISTERED
    
    # Check if callbacks have already been registered to prevent duplicates
    if CALLBACKS_REGISTERED:
        logger.info("Prompt settings callbacks already registered, skipping")
        return
    
    try:
        from services.prompt_settings import PromptSettings
        
        # Multi-output callback for all prompt settings UI interactions
        @app.callback(
            [
                Output("prompt-template-textarea", "value"),
                Output("prompt-settings-store", "data"),
                Output("prompt-feedback", "children"),
                Output("prompt-feedback", "className"),
                Output("prompt-variables-help", "children")
            ],
            [
                Input("prompt-type-select", "value"),
                Input("reset-prompt-button", "n_clicks"),
                Input("save-prompt-button", "n_clicks"),
                Input("url", "pathname"),
            ],
            [
                State("prompt-template-textarea", "value"),
                State("prompt-settings-store", "data"),
            ]
        )
        def handle_prompt_settings(
            prompt_type, reset_clicks, save_clicks, pathname,
            current_text, current_prompts
        ):
            # Get the callback context to determine which input triggered the callback
            ctx = callback_context
            
            # Default return values
            textarea_value = current_text
            store_data = current_prompts
            feedback_message = ""
            feedback_class = ""
            variables_help = []
            
            # If no context or not on settings page, return no updates
            if not ctx.triggered or pathname != '/settings':
                raise PreventUpdate
            
            # Determine which input triggered the callback
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            try:
                # Handle URL change or prompt type change - load the prompt
                if trigger_id == "url" or trigger_id == "prompt-type-select":
                    logger.info(f"Loading prompt for type: {prompt_type}")
                    
                    # Get all prompts and store them
                    all_prompts = PromptSettings.get_prompt_settings()
                    store_data = all_prompts
                    
                    # Get the specific prompt
                    if prompt_type and prompt_type in all_prompts:
                        textarea_value = all_prompts[prompt_type]
                    else:
                        # Fallback to default if not found
                        textarea_value = PromptSettings.DEFAULT_PROMPTS.get(prompt_type, "")
                
                # Handle reset button click
                elif trigger_id == "reset-prompt-button" and reset_clicks:
                    logger.info(f"Resetting prompt for type: {prompt_type}")
                    
                    # Get the default prompt for this type
                    textarea_value = PromptSettings.DEFAULT_PROMPTS.get(prompt_type, "")
                    feedback_message = f"Prompt template for '{prompt_type}' reset to default."
                    feedback_class = "alert alert-warning p-2 mt-3"
                
                # Handle save button click
                elif trigger_id == "save-prompt-button" and save_clicks:
                    logger.info(f"Saving prompt for type: {prompt_type}")
                    
                    if not prompt_type or not current_text:
                        feedback_message = "Error: Missing prompt type or value"
                        feedback_class = "alert alert-danger p-2 mt-3"
                    else:
                        # Update the current prompts
                        if not current_prompts:
                            store_data = PromptSettings.get_prompt_settings()
                        else:
                            store_data = current_prompts.copy()
                            
                        # Update the selected prompt
                        store_data[prompt_type] = current_text
                        
                        # Save to file
                        success = PromptSettings.save_prompt_settings(store_data)
                        
                        if success:
                            feedback_message = f"Prompt template for '{prompt_type}' saved successfully!"
                            feedback_class = "alert alert-success p-2 mt-3"
                        else:
                            feedback_message = "Error: Failed to save prompt template"
                            feedback_class = "alert alert-danger p-2 mt-3"
                
                # Update variables help for the selected prompt type
                if prompt_type == "document_query":
                    variables_help = [
                        dbc.Badge("{context}", color="primary", className="me-2 mb-2"),
                        dbc.Badge("{query}", color="primary", className="me-2 mb-2"),
                    ]
                elif prompt_type == "system_instruction":
                    variables_help = [
                        dbc.Badge("No variables available", color="secondary", className="me-2 mb-2"),
                    ]
                elif prompt_type == "welcome_message":
                    variables_help = [
                        dbc.Badge("{username}", color="primary", className="me-2 mb-2"),
                        dbc.Badge("{group_name}", color="primary", className="me-2 mb-2"),
                    ]
                
                return textarea_value, store_data, feedback_message, feedback_class, variables_help
                
            except Exception as e:
                logger.error(f"Error in prompt settings callback: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return appropriate error feedback
                feedback_message = f"Error: {str(e)}"
                feedback_class = "alert alert-danger p-2 mt-3"
                
                # Try to preserve existing data where possible
                return textarea_value, store_data, feedback_message, feedback_class, variables_help
        
        # Mark callbacks as registered
        CALLBACKS_REGISTERED = True
        logger.info("Prompt settings callbacks registered successfully")
        
    except Exception as e:
        logger.error(f"Error registering prompt settings callbacks: {str(e)}")
        logger.error(traceback.format_exc())