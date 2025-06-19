# app/main.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"
__license__  = "MIT"

from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
from flask_session import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.layout import create_layout, create_document_item
from app.callbacks import (
    register_callbacks, 
    register_progress, 
    handle_folder_load, 
    handle_document_removal, 
    register_toggle_callback, 
    handle_url_load,
    register_upload_handlers, 
    register_feedback_callbacks,
    register_query_progress,
    register_auto_load_callback,
    register_chat_autoscroll,
    register_immediate_question_display,
    register_welcome_message_callbacks 
)
from app.document_callbacks import (
    register_document_view_callbacks,
    sync_documents_on_load,
    register_document_clientside_callbacks,
    register_auto_sync_callback,
    register_server_group_callbacks
)
from auth.models import Base
from auth.callbacks import register_auth_callbacks
from auth.layout import create_login_layout, create_register_layout, create_navbar
from auth.config import AUTH_CONFIG
from auth.admin import create_admin_layout, register_admin_callbacks  # Import admin functions
from flask import session
from app.document_view import create_document_list_layout, load_document_tags, save_document_tags
import traceback
import time  # Make sure to import time module
from auth.group_admin_callbacks import register_group_callbacks
from auth.group_management import GroupService
from auth.route_settings import create_standalone_settings, register_standalone_callbacks, register_group_llm_callbacks, register_prompt_settings_callbacks
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
def create_app():
    # Initialize Dash app
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
        ],
        assets_folder='../assets',
        assets_url_path='/assets',
        suppress_callback_exceptions=True,
        # IMPORTANT: Set this to 'initial_duplicate' instead of False
        prevent_initial_callbacks='initial_duplicate'
    )
    
    # Configure Flask server
    server = app.server
    server.config.update(AUTH_CONFIG)
    Session(server)
    
    # Setup database
    engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    db_session = DBSession()
    
    # Define the base layout with navbar and content area
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),  # Changed refresh to False
        html.Div(id='navbar-container'),
        html.Div(id='page-content'),
        html.Div(id='_dummy-output', style={'display': 'none'}),
        dcc.Store(id='auth-state', storage_type='session'),
        dcc.Store(id='vectorstore-state', storage_type='memory'),
        dcc.Store(id='document-state', storage_type='memory'),
        dcc.Store(id='chunk-mapping-state', storage_type='memory'),
        dcc.Store(id='upload-trigger', data=None),
        dcc.Store(id='document-state-view', storage_type='memory'),
        dcc.Store(id='document-tags-state', storage_type='memory'),
        # Add a trigger element for admin page
        dcc.Store(id='admin-init-trigger', data=None),
        dcc.Store(id='group-state', storage_type='memory'),
        dcc.Store(id='group-membership-state', storage_type='memory'),
        dcc.Store(id='page-loaded-trigger', data=None),
        
    ])
    
    # Callback to handle URL routing
    @app.callback(
        Output('page-content', 'children', allow_duplicate=True),
        Output('document-state-view', 'data', allow_duplicate=True),
        Output('admin-init-trigger', 'data', allow_duplicate=True),  # Added admin trigger
        [Input('url', 'pathname')],
        [State('auth-state', 'data'),
         State('document-state-view', 'data')],
        prevent_initial_call=True  
    )
    def display_page(pathname, auth_state, document_state):
        logger.info(f"Main.py - create_app - display_page : Current pathname: {pathname}")
        logger.info(f"Main.py - create_app - display_page : Auth state: {auth_state}")
        
        try:
            # Default route
            if pathname == '/' or not pathname:
                if not auth_state or not auth_state.get('authenticated'):
                    return create_login_layout(), None, None
                
                # Create main layout with error handling
                try:
                    main_layout = create_layout()
                    logger.info("Main.py - create_app - display_page : Main layout created successfully")
                    
                    # Check if user already has documents loaded
                    current_user_id = auth_state.get('user_id')
                    
                    # Only load documents if the document state is empty
                    if not document_state or len(document_state) == 0:
                        logger.info(f"Main.py - create_app - display_page : User {current_user_id} has no documents loaded, checking for persistent documents")
                        
                        # Get database session
                        engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
                        DBSession = sessionmaker(bind=engine)
                        db_session = DBSession()
                        
                        # Get persistent documents from document view
                        try:
                            # Sync documents from storage
                            document_state, vectorstore_state, chunk_mapping_state = sync_documents_on_load(pathname, auth_state, document_state)
                            
                            # Generate document selector options
                            if document_state:
                                selector_options = [
                                    {"label": info["filename"], "value": session_id}
                                    for session_id, info in document_state.items()
                                ]
                                logger.info(f"Main.py - create_app - display_page : Loaded {len(document_state)} persistent documents for user {current_user_id}")
                            else:
                                selector_options = []
                                logger.info(f"Main.py - create_app - display_page : No persistent documents found for user {current_user_id}")
                        
                        except Exception as doc_error:
                            logger.info(f"Error loading persistent documents: {str(doc_error)}")
                            traceback.print_exc()
                            document_state = document_state or {}
                            vectorstore_state = vectorstore_state or {}
                            chunk_mapping_state = chunk_mapping_state or {}
                            selector_options = selector_options or []
                    else:
                        logger.info(f"Main.py - create_app - display_page : User {current_user_id} already has {len(document_state)} documents loaded")
                        
                    return main_layout, None, None
                
                except Exception as e:
                    logger.error(f"Main.py - create_app - display_page : Error creating main layout ~ 1: {str(e)}")
                    traceback.print_exc()
                    return html.Div([
                        html.H3("Error loading application"),
                        html.P(f"Error: {str(e)}")
                    ]), None, None, dash.no_update
                
            # Auth routes
            if pathname == '/login':
                if auth_state and auth_state.get('authenticated'):
                    return dcc.Location(pathname='/', id='redirect'), None, None
                return create_login_layout(), None, None
                
            elif pathname == '/register':
                if auth_state and auth_state.get('authenticated'):
                    return dcc.Location(pathname='/', id='redirect'), None, None
                return create_register_layout(), None, None
                
            # Protected routes
            if not auth_state or not auth_state.get('authenticated'):
                return create_login_layout(), None, None
               
            if pathname == '/admin':
                if auth_state.get('role') != 'admin':
                    return html.Div([
                        html.H3("Access Denied"),
                        html.P("You don't have permission to access this page."),
                        html.A("Return Home", href='/')
                    ], className="container mt-5"), None, None
                
                # Create admin layout and pass a trigger to force data loading
                logger.info("Main.py - create_app - display_page : Creating admin layout and triggering data load")
                admin_layout = create_admin_layout()
                admin_trigger = {'timestamp': str(time.time())}
                return admin_layout, None, admin_trigger
            
            if pathname == '/documents':
                try:
                    document_state, vectorstore_state, chunk_state = sync_documents_on_load(pathname, auth_state, document_state)
                    tags_data = load_document_tags(auth_state.get('user_id'))
                    return create_document_list_layout(document_state, auth_state.get('user_id')), document_state, None
                except Exception as e:
                    logger.error(f"Main.py - create_app - display_page : Error creating document layout ~2 : {str(e)}")
                    return html.Div([
                        html.H3("Error loading documents"),
                        html.P(f"Error: {str(e)}")
                    ]), None, None
            elif pathname == '/settings':
                if auth_state.get('role') != 'admin':
                    return html.Div([
                        html.H3("Access Denied"),
                        html.P("You don't have permission to access this page."),
                        html.A("Return Home", href='/')
                    ], className="container mt-5"), None, None
                
                # Use the standalone settings implementation instead
                logger.info("Main.py - create_app - display_page : Creating standalone settings layout")
                settings_layout = create_standalone_settings()
                return settings_layout, None, None
                
            # 404 page
            return html.Div([
                html.H1('404 - Page Not Found'),
                html.A('Return Home', href='/')
            ], className="container mt-5"), None, None
            
        except Exception as e:
            logger.error(f"Main.py - create_app - display_page : Error in display_page ~ 3 : {str(e)}")
            logger.error(traceback.format_exc())  # Added full traceback
            return html.Div([
                html.H3("Application Error"),
                html.P(f"Error: {str(e)}")
            ]), None, None
    
    # Callback to update navbar
    @app.callback(
        Output('navbar-container', 'children'),
        [Input('auth-state', 'data')]
    )
    def update_navbar(auth_state):
        try:
            return create_navbar(auth_state)
        except Exception as e:
            logger.info(f"Main.py - create_app - update_navbar : Error updating navbar: {str(e)}")
            logger.info(traceback.format_exc())  # Added full traceback
            return html.Div() # Return empty div on error

    @app.callback(
        Output("document-list", "children"),
        [Input("document-state", "data")],
        prevent_initial_call=True
    )
    def update_document_list(doc_state):
        """Update document list when document state changes"""
        if not doc_state:
            return [
                html.Div(
                    "No documents uploaded. Click 'Add Documents' to get started.",
                    className="text-center text-muted py-4"
                )
            ]
        
        # Group documents by their group name
        grouped_docs = {}
        
        for session_id, info in doc_state.items():
            if not isinstance(info, dict) or not info.get('filename'):
                continue
            
            # Check if this is a persistent document
            is_persistent = info.get('source') in ['folder', 'group'] or 'path' in info
            group_name = info.get('group_name', 'Uploaded')
            
            # Group the documents
            if group_name not in grouped_docs:
                grouped_docs[group_name] = []
            
            grouped_docs[group_name].append({
                'session_id': session_id,
                'info': info,
                'is_persistent': is_persistent
            })
        
        # Prepare the document list
        document_list = []
        
        # Process groups and documents
        for group_name, group_docs in grouped_docs.items():
            # Separate persistent and non-persistent documents
            non_persistent_docs = [
                doc for doc in group_docs 
                if not doc['is_persistent']
            ]
            persistent_docs = [
                doc for doc in group_docs 
                if doc['is_persistent']
            ]
            
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
                document_list.append(group_header)
            
            # Add non-persistent documents
            for doc in non_persistent_docs:
                document_list.append(
                    create_document_item(
                        doc['info']['filename'], 
                        doc['session_id'], 
                        is_persistent=False
                    )
                )
        
        # If no documents at all, show a message
        if not document_list:
            document_list = [
                html.Div(
                    "No documents uploaded. Click 'Add Documents' to get started.",
                    className="text-center text-muted py-4"
                )
            ]
        
        return document_list
    
    @app.callback(
        Output('page-loaded-trigger', 'data'),
        [Input('page-content', 'children')],
        [State('auth-state', 'data')]
    )
    def set_page_loaded_trigger(content, auth_state):
        if auth_state and auth_state.get('authenticated'):
            # Generate a timestamp when page is loaded
            return {'timestamp': datetime.now().isoformat()}
        return None
        
    # Register all callbacks
    try:
        register_document_view_callbacks(app)
        register_auth_callbacks(app, db_session)
        register_admin_callbacks(app, db_session)  
        register_upload_handlers(app)
        register_toggle_callback(app)
        handle_url_load(app)
        handle_folder_load(app)
        handle_document_removal(app)
        register_callbacks(app)
        register_welcome_message_callbacks(app)
        register_progress(app)
        register_feedback_callbacks(app)
        register_query_progress(app)
        register_document_clientside_callbacks(app)
        register_group_callbacks(app, db_session)
        register_auto_sync_callback(app)
        register_auto_load_callback(app) 
        register_server_group_callbacks(app)
        register_chat_autoscroll(app)
        register_immediate_question_display(app)
        register_standalone_callbacks(app)
        register_prompt_settings_callbacks(app) 
        register_group_llm_callbacks(app)
        logger.info("All callbacks registered successfully")
    except Exception as e:
        logger.info(f"Error registering callbacks: {str(e)}")
        logger.info(traceback.format_exc())  # Added full traceback
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True, dev_tools_hot_reload=False)