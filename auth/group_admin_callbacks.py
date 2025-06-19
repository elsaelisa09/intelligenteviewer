# auth/group_admin_callbacks.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import traceback
import sys
from datetime import datetime
from dash.dependencies import Input, Output, State
from dash import no_update, callback_context, dcc, html
import dash_bootstrap_components as dbc
from .group_management import GroupService, save_groups, save_user_groups
from .models import User
from .admin import load_user_tags

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

def register_group_callbacks(app, db_session):
    """Register all group-related callbacks"""
    group_service = GroupService()

    def get_debug_info():
        """Collect comprehensive debug information"""
        debug_info = {
            "Python Version": sys.version,
            "Database Session": str(db_session),
            "Group Service": str(group_service),
            "Environment": {}
        }
        
        try:
            # Check database connection
            connection = db_session.bind.connect()
            debug_info["Database Connection"] = "Successful"
            connection.close()
        except Exception as e:
            debug_info["Database Connection"] = f"Failed: {str(e)}"
        
        return debug_info

    def get_user_options():
        """Helper function to fetch user options"""
        try:
            # Extensive logging
            print("=" * 50)
            print("DEBUG: Retrieving User Options")
            print("=" * 50)
            
            # Debug database session and environment
            debug_info = get_debug_info()
            for key, value in debug_info.items():
                print(f"{key}: {value}")
            
            # Use the same method as user management to load users
            # Fetch all users and user tags
            user_tags = load_user_tags()
            print(f"Loaded user tags: {user_tags}")
            
            # Fetch users from the database
            try:
                users = db_session.query(User).all()
                print(f"Query method: Standard SQLAlchemy query")
            except Exception as standard_query_error:
                print(f"Standard query failed: {standard_query_error}")
                try:
                    # Alternative query method
                    from sqlalchemy.orm import sessionmaker
                    Session = sessionmaker(bind=db_session.bind)
                    with Session() as new_session:
                        users = new_session.query(User).all()
                    print(f"Query method: Alternative session method")
                except Exception as alt_query_error:
                    print(f"Alternative query failed: {alt_query_error}")
                    users = []
            
            print(f"Total users found: {len(users)}")
            
            # Create user options with additional tag information
            user_options = []
            for user in users:
                # Get user tag if exists
                user_tag = user_tags.get(str(user.id), "")
                
                # Create user option with additional context
                label = f"{user.username}"
                if user_tag:
                    label += f" ({user_tag})"
                
                user_option = {
                    "label": label.strip(),
                    "value": str(user.id)
                }
                user_options.append(user_option)
                
                # Print detailed user information for debugging
                print(f"User Option - Label: {user_option['label']}, Value: {user_option['value']}")
            
            return user_options
        except Exception as e:
            print(f"CRITICAL ERROR fetching users: {e}")
            print(traceback.format_exc())
            return []

    @app.callback(
        [
            Output("groups-table", "data"),
            Output("group-members-dropdown", "options",allow_duplicate=True),
            Output("group-admins-dropdown", "options",allow_duplicate=True)
        ],
        [
            Input("url", "pathname"),
            Input("admin-refresh-trigger", "children"),
            Input("refresh-admin-data", "n_clicks")
        ],
        prevent_initial_call=True
    )
    def update_groups_table(pathname, refresh_trigger, refresh_clicks):
        """Load and display groups"""
        ctx = callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No triggers'
        print(f"Groups table update trigger: {trigger}")

        # Only update when on admin page or explicitly refreshed
        if pathname != "/admin" and trigger not in ["refresh-admin-data", "admin-refresh-trigger"]:
            return no_update

        # Fetch user options for dropdowns
        user_options = []
        for user in db_session.query(User).all():
            # Get user tags
            user_tags = load_user_tags()
            user_tag = user_tags.get(str(user.id), "")
            
            # Create label with tag if available
            label = user.username
            if user_tag:
                label += f" ({user_tag})"
            
            user_options.append({
                "label": label,
                "value": str(user.id)
            })

        # Get groups data
        groups_data = []
        groups = group_service.list_groups()
        
        for group_id, group_info in groups.items():
            # Fetch usernames for group members and admins
            group_members = group_service.get_group_members(group_id)
            member_usernames = []
            admin_usernames = []

            for member_id in group_members:
                member = db_session.query(User).filter(User.id == int(member_id)).first()
                if member:
                    member_usernames.append(member.username)
                    # Check if this member is an admin
                    if group_id in group_service.groups and member_id in group_service.groups[group_id].get('group_admins', []):
                        admin_usernames.append(member.username)

            groups_data.append({
                "group_id": group_id,
                "name": group_info.get("name", ""),
                "description": group_info.get("description", ""),
                "group_admins": ", ".join(admin_usernames),  # Display group admin usernames
                "members": ", ".join(member_usernames),
                "members_count": len(group_members),
                "created_at": group_info.get("created_at", "")
            })
        
        return groups_data, user_options, user_options
        
    @app.callback(
        [
            Output("admin-feedback-toast", "is_open", allow_duplicate=True),
            Output("admin-feedback-toast", "header", allow_duplicate=True),
            Output("admin-feedback-toast", "children", allow_duplicate=True),
            Output("admin-refresh-trigger", "children", allow_duplicate=True)
        ],
        [Input("remove-user-from-group-button", "n_clicks")],
        [
            State("groups-table", "selected_rows"),
            State("groups-table", "data"),
            State("group-members-select", "value")
        ],
        prevent_initial_call=True
    )
    def remove_user_from_group(n_clicks, selected_rows, table_data, selected_member):
        """Remove a selected user from a group"""
        if not n_clicks or not selected_rows or len(selected_rows) == 0 or not selected_member:
            return no_update, no_update, no_update, no_update
        
        try:
            selected_row_index = selected_rows[0]
            selected_group = table_data[selected_row_index]
            group_id = selected_group['group_id']
            
            print(f"Removing user {selected_member} from group {group_id}")
            
            # Remove the user from the group
            success = group_service.remove_user_from_group(selected_member, group_id)
            
            if success:
                # Get user information for the feedback message
                user = db_session.query(User).filter(User.id == int(selected_member)).first()
                username = user.username if user else "User"
                
                return (
                    True, 
                    "Success", 
                    f"User '{username}' removed from group '{selected_group['name']}'", 
                    f"refresh-{datetime.now().timestamp()}"
                )
            else:
                return (
                    True, 
                    "Error", 
                    "Failed to remove user from group", 
                    no_update
                )
            
        except Exception as e:
            print(f"Error removing user from group: {str(e)}")
            print(traceback.format_exc())
            return (
                True,
                "Error",
                f"Error: {str(e)}",
                no_update
            )

    @app.callback(
        [
            Output("manage-members-modal", "is_open", allow_duplicate=True),
            Output("group-members-select", "options", allow_duplicate=True),
            Output("group-members-select", "value", allow_duplicate=True),
            Output("manage-members-title", "children", allow_duplicate=True)
        ],
        [
            Input("manage-members-button", "n_clicks"),
            Input("close-manage-members-button", "n_clicks")
        ],
        [
            State("groups-table", "selected_rows"),
            State("groups-table", "data"),
            State("manage-members-modal", "is_open")
        ],
        prevent_initial_call=True
    )
    def toggle_manage_members_modal(manage_clicks, close_clicks, selected_rows, table_data, is_open):
        """Toggle the manage members modal and populate member options"""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "close-manage-members-button":
            return False, no_update, no_update, no_update
        
        if button_id == "manage-members-button" and manage_clicks:
            if not selected_rows or len(selected_rows) == 0:
                return no_update, no_update, no_update, no_update
            
            selected_row_index = selected_rows[0]
            selected_group = table_data[selected_row_index]
            group_id = selected_group['group_id']
            
            # Get current group members
            group_members = group_service.get_group_members(group_id)
            
            # Create options for the dropdown
            member_options = []
            for member_id in group_members:
                try:
                    user = db_session.query(User).filter(User.id == int(member_id)).first()
                    if user:
                        # Get user tag if exists
                        user_tags = load_user_tags()
                        user_tag = user_tags.get(str(user.id), "")
                        
                        label = user.username
                        if user_tag:
                            label += f" ({user_tag})"
                        
                        member_options.append({
                            "label": label,
                            "value": str(user.id)
                        })
                except Exception as e:
                    print(f"Error getting user info: {str(e)}")
            
            return True, member_options, None, f"Manage Members - {selected_group['name']}"
            
        return no_update, no_update, no_update, no_update

    @app.callback(
        [
            Output("group-members-dropdown", "options"),
            Output("group-admins-dropdown", "options")
        ],
        [
            Input("add-group-button", "n_clicks"),
            Input("edit-group-button", "n_clicks")
        ],
        [
            State("groups-table", "selected_rows"),
            State("groups-table", "data")
        ],
        prevent_initial_call=True
    )
    def populate_group_dropdowns(add_clicks, edit_clicks, selected_rows, table_data):
        """Populate group dropdowns with users not in any group"""
        try:
            # Fetch all users
            users = db_session.query(User).all()
            
            # Prepare user options
            user_options = []
            for user in users:
                # Check if user is already in a group
                user_groups = group_service.get_user_groups(str(user.id))
                
                # Only add users not in any group
                if not user_groups:
                    user_option = {
                        "label": user.username,
                        "value": str(user.id)
                    }
                    user_options.append(user_option)
            
            print("Available users for group membership:")
            for option in user_options:
                print(f"  - {option['label']} (ID: {option['value']})")
            
            return user_options, user_options
        
        except Exception as e:
            print(f"Error populating group dropdowns: {e}")
            print(traceback.format_exc())
            return [], []

    @app.callback(
        [
            Output("group-modal-title", "children", allow_duplicate=True),
            Output("group-modal", "is_open", allow_duplicate=True),
            Output("group-name-input", "value", allow_duplicate=True),
            Output("group-description-input", "value", allow_duplicate=True),
            Output("group-members-dropdown", "options", allow_duplicate=True),
            Output("group-members-dropdown", "value", allow_duplicate=True),
            Output("group-admins-dropdown", "options", allow_duplicate=True),
            Output("group-admins-dropdown", "value", allow_duplicate=True),
            Output("save-group-button", "children")
        ],
        [
            Input("add-group-button", "n_clicks"),
            Input("edit-group-button", "n_clicks"),
            Input("close-group-modal-button", "n_clicks")
        ],
        [
            State("groups-table", "selected_rows"),
            State("groups-table", "data")
        ],
        prevent_initial_call=True
    )
    def handle_group_modal(add_clicks, edit_clicks, close_clicks, selected_rows, table_data):
        """Handle opening and closing group modal"""
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Fetch user options for dropdowns
        user_options = get_user_options()
        
        if button_id == "close-group-modal-button":
            # Close modal
            return (
                no_update,  # modal title
                False,      # is_open
                no_update,  # group name
                no_update,  # group description
                no_update,  # members dropdown options
                no_update,  # selected members
                no_update,  # admins dropdown options
                no_update,  # selected admins
                no_update   # save button text
            )
        
        if button_id == "add-group-button":
            # Open modal for adding new group
            return (
                [html.I(className="fas fa-plus me-2", style={'color': COLORS['primary']}), "Add New Group"],
                True,   # is_open
                "",     # group name
                "",     # group description
                user_options,  # user options for members
                [],     # selected group members (empty)
                user_options,  # user options for admins
                [],     # selected group admins (empty)
                "Create Group"
            )
        
        elif button_id == "edit-group-button":
            # Open modal for editing existing group
            if not selected_rows or len(selected_rows) == 0:
                return no_update
            
            selected_row_index = selected_rows[0]
            selected_group = table_data[selected_row_index]
            group_id = selected_group['group_id']
            
            # Get existing group members and admins
            existing_members = group_service.get_group_members(group_id)
            existing_admins = group_service.groups.get(group_id, {}).get('group_admins', [])
            
            return (
                [html.I(className="fas fa-edit me-2", style={'color': COLORS['primary']}), "Edit Group"],
                True,   # is_open
                selected_group['name'],     # group name
                selected_group.get('description', ''),  # group description
                user_options,  # user options for members
                existing_members,  # selected group members
                user_options,  # user options for admins
                existing_admins,   # selected group admins
                "Update Group"
            )
        
        return no_update

    @app.callback(
        [
            Output("admin-feedback-toast", "is_open", allow_duplicate=True),
            Output("admin-feedback-toast", "header", allow_duplicate=True),
            Output("admin-feedback-toast", "children", allow_duplicate=True),
            Output("group-modal", "is_open", allow_duplicate=True),
            Output("admin-refresh-trigger", "children", allow_duplicate=True)
        ],
        [Input("save-group-button", "n_clicks")],
        [
            State("group-name-input", "value"),
            State("group-description-input", "value"),
            State("group-members-dropdown", "value"),
            State("group-admins-dropdown", "value"),
            State("groups-table", "selected_rows"),
            State("groups-table", "data")
        ],
        prevent_initial_call=True
    )
    def save_group(n_clicks, group_name, group_description, 
                selected_members, selected_admins, 
                selected_rows, table_data):
        """Save group with members and admins"""
        if not n_clicks:
            return no_update
        
        # Validate input
        if not group_name:
            return True, "Error", "Group name is required", no_update, ""
        
        try:
            # Normalize group name to create group_id
            group_id = group_name.lower().replace(' ', '_')
            
            # Determine if it's a new group or existing group
            is_new_group = group_id not in group_service.groups
            
            if is_new_group:
                # Create new group
                group_service.create_group(group_name, description=group_description or "")
            else:
                # Update existing group metadata
                group_service.groups[group_id] = {
                    "name": group_name,
                    "description": group_description or "",
                    "group_admins": [],  # Reset admins, will be set during member addition
                    "created_at": group_service.groups[group_id].get('created_at', str(datetime.now()))
                }
            
            # Validate member selection
            if not selected_members:
                return True, "Error", "At least one group member is required", no_update, ""
            
            # Add members with first user as default admin
            member_add_errors = []
            for i, user_id in enumerate(selected_members):
                is_first_user = (i == 0)
                success = group_service.add_user_to_group(user_id, group_id, is_first_user)
                
                if not success:
                    # Collect error details
                    member_add_errors.append(user_id)
            
            # Check if any member addition failed
            if member_add_errors:
                # Attempt to get usernames for error reporting
                error_usernames = []
                for user_id in member_add_errors:
                    try:
                        user = db_session.query(User).filter(User.id == int(user_id)).first()
                        error_usernames.append(user.username if user else user_id)
                    except:
                        error_usernames.append(user_id)
                
                return True, "Error", f"The following users could not be added to the group: {', '.join(error_usernames)}. They may already be members of another group.", no_update, ""
            
            # Validate and set group admins
            if selected_admins:
                # Ensure all admins are group members
                for admin_id in selected_admins:
                    if admin_id not in selected_members:
                        return True, "Error", "Group admins must be group members", no_update, ""
                
                # Update group admins
                group_service.groups[group_id]['group_admins'] = selected_admins
            
            # Save changes
            save_groups(group_service.groups)
            save_user_groups(group_service.user_groups)
            
            return (
                True, 
                "Success", 
                f"Group '{group_name}' {'created' if is_new_group else 'updated'} successfully", 
                False,  # Close modal
                f"refresh-{datetime.now().timestamp()}"
            )
        
        except Exception as e:
            return True, "Error", str(e), no_update, ""

    @app.callback(
        [
            Output("edit-group-button", "disabled"),
            Output("delete-group-button", "disabled"),
            Output("manage-members-button", "disabled")  # Add this line
        ],
        [Input("groups-table", "selected_rows")],
        [State("groups-table", "data")]
    )
    def update_group_action_buttons(selected_rows, table_data):
        """Enable/disable group action buttons based on selection"""
        if not selected_rows or not table_data or len(selected_rows) == 0:
            return True, True, True  # All buttons disabled
        
        return False, False, False  # All buttons enabled

    @app.callback(
        [
            Output("admin-feedback-toast", "is_open", allow_duplicate=True),
            Output("admin-feedback-toast", "header", allow_duplicate=True),
            Output("admin-feedback-toast", "children", allow_duplicate=True),
            Output("admin-refresh-trigger", "children", allow_duplicate=True)
        ],
        [Input("delete-group-button", "n_clicks")],
        [
            State("groups-table", "selected_rows"),
            State("groups-table", "data")
        ],
        prevent_initial_call=True
    )
    def delete_group(n_clicks, selected_rows, table_data):
        """Delete a selected group"""
        if not n_clicks or not selected_rows or len(selected_rows) == 0:
            return no_update
        
        selected_row_index = selected_rows[0]
        selected_group = table_data[selected_row_index]
        group_id = selected_group['group_id']
        
        try:
            # Remove group from groups
            groups = group_service.groups
            if group_id in groups:
                del groups[group_id]
                save_groups(groups)
            
            # Remove group from user group mappings
            user_groups = group_service.user_groups
            for user_id, user_group_list in user_groups.items():
                if group_id in user_group_list:
                    user_group_list.remove(group_id)
            save_user_groups(user_groups)
            
            return (
                True, 
                "Success", 
                f"Group '{selected_group['name']}' deleted successfully", 
                f"refresh-{datetime.now().timestamp()}"
            )
        
        except Exception as e:
            return True, "Error", str(e), ""

    # Return the service for use in other parts of the application
    return group_service