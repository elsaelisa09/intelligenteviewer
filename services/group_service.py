#services/group_service.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import json
from typing import List, Dict
from app.storage_config import BASE_STORAGE_DIR
from pathlib import Path

class GroupService:
    def __init__(self):
        # Define the base directory for group storage
        self.GROUPS_DIR = Path(BASE_STORAGE_DIR) / "groups"
        self.GROUPS_DIR.mkdir(parents=True, exist_ok=True)

    def get_available_groups(self, user_id: int) -> List[Dict]:
        """
        Fetch available groups for a user
        """
        print("=" * 50)
        print(f"Retrieving groups for User ID: {user_id}")
        
        try:
            # Path to user group mappings
            mapping_file = self.GROUPS_DIR / 'user_group_mappings.json'
            
            if not mapping_file.exists():
                print("ERROR: Group mapping file does not exist!")
                return []
            
            # Read user-group mappings
            with open(mapping_file, 'r') as f:
                user_group_mapping = json.load(f)
            
            # Convert user ID to string (for JSON compatibility)
            user_id_str = str(user_id)
            
            print(f"User ID (str): {user_id_str}")
            print(f"Available mappings: {json.dumps(user_group_mapping, indent=2)}")
            
            # Get groups for this user
            user_groups = user_group_mapping.get(user_id_str, [])
            
            print(f"Groups for user {user_id_str}: {user_groups}")
            
            # If no groups, return empty list
            if not user_groups:
                print(f"WARNING: No groups found for user {user_id_str}")
                return []
            
            # Convert to list of dictionaries with group details
            groups = [
                {"id": group, "name": group} 
                for group in user_groups
            ]
            
            print(f"Processed groups: {groups}")
            print("=" * 50)
            
            return groups
        
        except Exception as e:
            print(f"ERROR in get_available_groups: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def get_group_documents(self, user_id: int, group_name: str) -> List[Dict]:
        """
        Fetch documents for a specific group
        
        Args:
            user_id (int): ID of the current user
            group_name (str): Name of the group
        
        Returns:
            List of document information
        """
        try:
            from app.storage_config import ORIGINAL_FILES_DIR, get_group_for_user
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from auth.config import AUTH_CONFIG

            # Create database session
            engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
            DBSession = sessionmaker(bind=engine)
            db_session = DBSession()

            # Get group directory
            group_dir = ORIGINAL_FILES_DIR / str(group_name)
            
            # Verify the group belongs to the user
            user_group = get_group_for_user(user_id, db_session)
            if user_group != group_name:
                print(f"User {user_id} does not have access to group {group_name}")
                return []

            # Check if group directory exists
            if not group_dir.exists():
                print(f"Group directory not found: {group_dir}")
                return []

            # Collect documents
            documents = []
            for file_path in group_dir.glob('*'):
                if file_path.is_file():
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                        
                        # Encode content to base64
                        import base64
                        content_b64 = base64.b64encode(content).decode('utf-8')
                        
                        documents.append({
                            'filename': file_path.name,
                            'content': content_b64,
                            'group_name': group_name
                        })
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}")
            
            return documents
        
        except Exception as e:
            print(f"Error fetching group documents: {str(e)}")
            return []