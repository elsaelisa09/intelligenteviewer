# auth/callbacks.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from dash.dependencies import Input, Output, State
from dash import no_update, html, dcc
from flask import session
from .service import AuthService
from sqlalchemy.orm import Session
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def register_auth_callbacks(app, db_session: Session):
    auth_service = AuthService(db_session)
    
    @app.callback(
        [
            Output("url", "pathname",allow_duplicate=True),
            Output("login-error", "children",allow_duplicate=True),
            Output("auth-state", "data",allow_duplicate=True)
        ],
        [Input("login-button", "n_clicks")],
        [
            State("login-username", "value"),
            State("login-password", "value")
        ],
        prevent_initial_call=True
    )
    def handle_login(n_clicks, username, password):
        if not n_clicks or not username or not password:
            return no_update, no_update, no_update
            
        try:
            # Authenticate user
            user = auth_service.authenticate_user(username, password)
            
            if user:
                # Create session data
                session['user_id'] = user.id
                session['username'] = user.username
                session['role'] = user.role
                
                # Create auth state for Dash components
                auth_state = {
                    'authenticated': True,
                    'username': user.username,
                    'role': user.role,
                    'user_id': user.id
                }
                
                logger.info(f"auth~callbacks.py - register_auth_callbacks - handle_login : Login successful for user: {username}")
                logger.info(f"auth~callbacks.py - register_auth_callbacks - handle_login : Session data: {session}")
                logger.info(f"auth~callbacks.py - register_auth_callbacks - handle_login : Auth state: {auth_state}")
                
                return "/", "", auth_state
            else:
                logger.error(f"auth~callbacks.py - register_auth_callbacks - handle_login : Login failed for user: {username}")
                return no_update, "Invalid username or password", no_update
                
        except Exception as e:
            logger.error(f"auth~callbacks.py - register_auth_callbacks - handle_login : Login error: {str(e)}")
            return no_update, f"Login error: {str(e)}", no_update

    @app.callback(
        [
            Output("url", "pathname", allow_duplicate=True),
            Output("register-error", "children", allow_duplicate=True),
            Output("auth-state", "data", allow_duplicate=True)
        ],
        [Input("register-button", "n_clicks")],
        [
            State("register-username", "value"),
            State("register-email", "value"),
            State("register-password", "value"),
            State("register-confirm-password", "value")
        ],
        prevent_initial_call=True
    )
    def handle_registration(n_clicks, username, email, password, confirm_password):
        if not n_clicks:
            return no_update, no_update, no_update
            
        if not all([username, email, password, confirm_password]):
            return no_update, "Please fill in all fields", no_update
            
        if password != confirm_password:
            return no_update, "Passwords do not match", no_update
            
        try:
            user = auth_service.create_user(username, email, password)
            return "/login", "Registration successful! Please login.", no_update
        except Exception as e:
            return no_update, str(e), no_update

    @app.callback(
        [
            Output("url", "pathname", allow_duplicate=True),
            Output("auth-state", "data", allow_duplicate=True)
        ],
        [Input("logout-button", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_logout(n_clicks):
        if n_clicks:
            # Clear Flask session
            session.clear()
            # Return to login page and clear auth state
            return "/login", {'authenticated': False}
        return no_update, no_update