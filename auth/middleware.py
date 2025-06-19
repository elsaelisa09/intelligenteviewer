# auth/middleware.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from functools import wraps
from dash import no_update
from flask import session, redirect
import jwt
from auth.group_management import GroupService

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_role' not in session or session['user_role'] != 'admin':
            return no_update
        return f(*args, **kwargs)
    return decorated_function

def group_admin_required(group_name):
    """
    Decorator to check if the user is an admin of a specific group
    
    :param group_name: Name of the group to check admin status
    :return: Decorated function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if user is logged in
            if 'user_id' not in session:
                return redirect('/login')
            
            # Initialize group service
            group_service = GroupService()
            
            # Get the group
            groups = group_service.list_groups()
            group_id = next((
                g_id for g_id, group_info in groups.items() 
                if group_info.get('name', '').lower() == group_name.lower()
            ), None)
            
            # Check if user is a group admin
            if not group_id or not group_service.is_group_admin(str(session['user_id']), group_id):
                # Redirect or return unauthorized access message
                return redirect('/unauthorized')
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def group_member_required(group_name):
    """
    Decorator to check if the user is a member of a specific group
    
    :param group_name: Name of the group to check membership
    :return: Decorated function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if user is logged in
            if 'user_id' not in session:
                return redirect('/login')
            
            # Initialize group service
            group_service = GroupService()
            
            # Get the group
            groups = group_service.list_groups()
            group_id = next((
                g_id for g_id, group_info in groups.items() 
                if group_info.get('name', '').lower() == group_name.lower()
            ), None)
            
            # Check user group membership
            user_groups = group_service.get_user_groups(str(session['user_id']))
            
            if not group_id or group_id not in user_groups:
                # Redirect or return unauthorized access message
                return redirect('/unauthorized')
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

