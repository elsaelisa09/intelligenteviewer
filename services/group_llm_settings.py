# services/group_llm_settings.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroupLLMSettings:
    """Class to manage LLM settings per group"""
    
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
                    "groups": {}             # Map of group_id to provider
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
                "groups": {}
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