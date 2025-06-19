# storage_config.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from auth.config import AUTH_CONFIG 
import traceback
# Define base storage directory relative to project root
BASE_STORAGE_DIR = Path('storage')

# Create specific directories for different types of data
ORIGINAL_FILES_DIR = BASE_STORAGE_DIR / 'original_files'
CHUNK_MAPS_DIR = BASE_STORAGE_DIR / 'chunk_maps'
VECTOR_STORE_DIR = BASE_STORAGE_DIR / 'vector_stores'
DOC_STATUS_DIR = BASE_STORAGE_DIR / 'doc_status'


# Create directories if they don't exist
def init_storage():
    """Initialize storage directories"""
    directories = [
        ORIGINAL_FILES_DIR,
        CHUNK_MAPS_DIR,
        VECTOR_STORE_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    print(f"Storage directories initialized at {BASE_STORAGE_DIR}")

def get_user_directory(user_id: str, dir_type: str) -> Path:
    """Get user-specific directory path"""
    base_dir = {
        'original': ORIGINAL_FILES_DIR,
        'chunks': CHUNK_MAPS_DIR,
        'vectors': VECTOR_STORE_DIR
    }.get(dir_type)
    
    if not base_dir:
        raise ValueError(f"Invalid directory type: {dir_type}")
    
    user_dir = base_dir / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def get_group_directory(group_name: str, dir_type: str) -> Path:
    """Get group-specific directory path"""
    base_dir = {
        'original': ORIGINAL_FILES_DIR,
        'chunks': CHUNK_MAPS_DIR,
        'vectors': VECTOR_STORE_DIR
    }.get(dir_type)
    
    if not base_dir:
        raise ValueError(f"Invalid directory type: {dir_type}")
    
    group_dir = base_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir

def save_file(content: bytes, filename: str, user_id: str, dir_type: str, group_name: str) -> Path:
    """Save file to appropriate directory"""
    group_dir = get_group_directory(group_name, dir_type)
    user_dir = get_user_directory(user_id, dir_type)
    file_path = group_dir / filename
    
    # Save the file
    with open(file_path, 'wb') as f:
        f.write(content)
    
    return file_path


def delete_file(filename: str, user_id: str, dir_type: str, group_name: str) -> bool:
    """Delete file from storage"""
    group_dir = get_group_directory(group_name, dir_type)
    user_dir = get_user_directory(user_id, dir_type)
    file_path = user_dir / filename
    
    try:
        if file_path.exists():
            file_path.unlink()
        return True
    except Exception as e:
        print(f"Storage_config.py - delete_file : Error deleting file {filename}: {str(e)}")
        return False
    
def get_group_for_user(user_id: int, db_session=None) -> str:
    """
    Get the first group for a user from custom JSON mapping.
    
    Args:
        user_id (int): ID of the user
        db_session: Optional database session (not used in this implementation)
    
    Returns:
        str: Name of the first group, or 'public' if no groups found
    """
    import json
    from pathlib import Path
    import os

    # Define the path to the user-group mapping file
    GROUPS_DIR = Path('storage') / 'groups'
    mapping_file = GROUPS_DIR / 'user_group_mappings.json'

    try:
        # Ensure the directory exists
        GROUPS_DIR.mkdir(parents=True, exist_ok=True)

        # Convert user_id to string for JSON key
        user_id_str = str(user_id)
        #print(f"user id of current user is {user_id_str}")

        # Read the existing mapping
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                user_group_mapping = json.load(f)
            
            # Get groups for this user
            user_groups = user_group_mapping.get(user_id_str, [])
            print(user_groups)

            # Return the first group if exists, otherwise 'public'
            if user_groups:
                return user_groups[0]
        
        # If no groups found, return 'public'
        return 'public'

    except Exception as e:
        print(f"Storage_config.py - get_group_for_user : Error getting group for user {user_id}: {str(e)}")
        return 'public'