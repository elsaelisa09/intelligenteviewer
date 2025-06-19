# auth/group_management.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
# Base storage directory
BASE_STORAGE_DIR = Path('storage')
GROUPS_FILE = BASE_STORAGE_DIR / 'groups/user_groups.json'
USER_GROUPS_FILE = BASE_STORAGE_DIR / 'groups/user_group_mappings.json'

def ensure_storage_dirs():
    """Ensure storage directories exist"""
    BASE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_STORAGE_DIR / 'groups').mkdir(parents=True, exist_ok=True)

def load_groups() -> Dict[str, Dict]:
    """Load groups from file"""
    ensure_storage_dirs()
    try:
        if GROUPS_FILE.exists():
            with open(GROUPS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading groups: {e}")
        return {}

def save_groups(groups: Dict[str, Dict]):
    """Save groups to file"""
    ensure_storage_dirs()
    try:
        with open(GROUPS_FILE, 'w') as f:
            json.dump(groups, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving groups: {e}")
        return False

def load_user_groups() -> Dict[str, List[str]]:
    """Load user group mappings from file"""
    ensure_storage_dirs()
    try:
        if USER_GROUPS_FILE.exists():
            with open(USER_GROUPS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading user groups: {e}")
        return {}

def save_user_groups(user_groups: Dict[str, List[str]]):
    """Save user group mappings to file"""
    ensure_storage_dirs()
    try:
        with open(USER_GROUPS_FILE, 'w') as f:
            json.dump(user_groups, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving user groups: {e}")
        return False

class GroupService:
    def __init__(self):
        self.groups = load_groups()
        self.user_groups = load_user_groups()
    
    def create_group(self, group_name: str, description: str = "") -> bool:
        """Create a new group"""
        # Normalize group name
        group_id = group_name.lower().replace(' ', '_')
        
        # Check if group already exists
        if group_id in self.groups:
            return False
        
        # Create group
        self.groups[group_id] = {
            "name": group_name,
            "description": description,
            "group_admins": [],
            "created_at": str(datetime.now())
        }
        
        return save_groups(self.groups)
    
    def add_user_to_group(self, user_id: str, group_id: str, is_first_user: bool = False) -> bool:
        """
        Add a user to a group with strict single group membership
        
        Args:
        - user_id: ID of the user to add
        - group_id: ID of the group to add user to
        - is_first_user: Whether this is the first user in the group (becomes default admin)
        
        Returns:
        - Boolean indicating success of adding user to the group
        """
        # Convert user_id to string to ensure consistency
        user_id = str(user_id)
        group_id = str(group_id)
        
        # Validate group exists
        if group_id not in self.groups:
            print(f"Group {group_id} does not exist")
            return False
        
        # Check if user is already in ANY group
        for existing_group_id, group_members in self.user_groups.items():
            if user_id in group_members:
                if existing_group_id == group_id:
                    # User is already in this group
                    return True
                else:
                    # User is in a different group
                    print(f"User {user_id} is already a member of group {existing_group_id}")
                    return False
        
        # Add user to group mapping
        if user_id not in self.user_groups:
            self.user_groups[user_id] = []
        
        # Add to new group
        self.user_groups[user_id] = [group_id]
        
        # Set as group admin if first user
        if is_first_user:
            # Ensure group_admins key exists
            if 'group_admins' not in self.groups[group_id]:
                self.groups[group_id]['group_admins'] = []
            
            # Add as admin if not already there
            if user_id not in self.groups[group_id]['group_admins']:
                self.groups[group_id]['group_admins'] = [user_id]
        
        # Save changes
        save_user_groups(self.user_groups)
        save_groups(self.groups)
        
        return True
            
    
    def remove_user_from_group(self, user_id: str, group_id: str) -> bool:
        """Remove a user from a group"""
        user_id = str(user_id)
        group_id = str(group_id)
        
        # Debug information
        print(f"Attempting to remove user {user_id} from group {group_id}")
        print(f"Current user groups: {self.user_groups}")
        
        # Remove from user groups
        if user_id in self.user_groups:
            if group_id in self.user_groups[user_id]:
                self.user_groups[user_id].remove(group_id)
                print(f"Removed group {group_id} from user {user_id}'s group list")
                # If user has no more groups, remove the user entry completely
                if not self.user_groups[user_id]:
                    del self.user_groups[user_id]
                    print(f"User {user_id} has no more groups, removing entry")
                # Save the changes
                save_user_groups(self.user_groups)
            else:
                print(f"Group {group_id} not found in user {user_id}'s group list")
        else:
            print(f"User {user_id} not found in user groups")
        
        # Remove from group admins if applicable
        if group_id in self.groups and 'group_admins' in self.groups[group_id]:
            if user_id in self.groups[group_id]['group_admins']:
                self.groups[group_id]['group_admins'].remove(user_id)
                print(f"Removed user {user_id} from group {group_id}'s admin list")
                
                # If removing the last admin, clear admin list
                if not self.groups[group_id]['group_admins']:
                    self.groups[group_id]['group_admins'] = []
                    print(f"Group {group_id} has no more admins")
                
                # Save the changes
                save_groups(self.groups)
        
        # Debugging the state after removal
        print(f"User groups after removal: {self.user_groups}")
        
        return True
    
    def get_user_groups(self, user_id: str) -> List[str]:
        """Get groups for a user"""
        user_id = str(user_id)
        return self.user_groups.get(user_id, [])
    
    def is_group_admin(self, user_id: str, group_id: str) -> bool:
        """Check if user is a group admin"""
        user_id = str(user_id)
        return (group_id in self.groups and 
                user_id in self.groups[group_id]['group_admins'])
    
    def get_group_members(self, group_id: str) -> List[str]:
        """Get all members of a group"""
        members = []
        for user_id, groups in self.user_groups.items():
            if group_id in groups:
                members.append(user_id)
        return members
    
    def list_groups(self) -> Dict[str, Dict]:
        """List all groups"""
        return self.groups
    
    def get_group_names_for_user(self, user_id: str) -> List[str]:
        """Get group names for a user"""
        user_id = str(user_id)
        group_names = []
        
        # Get group IDs for the user
        user_group_ids = self.user_groups.get(user_id, [])
        
        # Convert group IDs to names
        for group_id in user_group_ids:
            if group_id in self.groups:
                group_names.append(self.groups[group_id].get('name', group_id))
        
        return group_names
    
    def get_group_admin_groups(self, user_id: str) -> List[str]:
        """Get groups where the user is an admin"""
        user_id = str(user_id)
        admin_groups = []
        
        # Check each group
        for group_id, group_info in self.groups.items():
            if 'group_admins' in group_info and user_id in group_info['group_admins']:
                admin_groups.append(group_info.get('name', group_id))
        
        return admin_groups

    def ensure_default_group_for_user(self, user_id):
        """
        Ensure the user has at least one group.
        If no groups exist, create a default group and add the user to it.
        
        Args:
            user_id (int): ID of the user
        
        Returns:
            Group: The user's default group
        """
        from auth.models import User, Group, GroupMembership
        
        # Check if user already has any groups
        existing_groups = (
            self.db_session.query(GroupMembership)
            .filter(GroupMembership.user_id == user_id)
            .first()
        )
        
        if existing_groups:
            # Return the first group (default)
            return existing_groups.group
        
        # Create a default group if no groups exist
        default_group_name = f"default_group_{user_id}"
        
        # Check if default group already exists
        default_group = (
            self.db_session.query(Group)
            .filter(Group.name == default_group_name)
            .first()
        )
        
        if not default_group:
            # Create new default group
            default_group = Group(
                name=default_group_name,
                description=f"Default group for user {user_id}",
                is_default=True
            )
            self.db_session.add(default_group)
        
        # Create group membership
        membership = GroupMembership(
            user_id=user_id,
            group=default_group,
            role='owner'  # Assuming the user is the owner of their default group
        )
        self.db_session.add(membership)
        
        # Commit changes
        try:
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            raise e
        
        return default_group
    
    def get_user_default_group(self, user_id):
        """
        Get the user's default group.
        
        Args:
            user_id (int): ID of the user
        
        Returns:
            Group: The user's default group, or creates one if not exists
        """
        # Reuse the ensure method to guarantee a group exists
        return self.ensure_default_group_for_user(user_id)
    
