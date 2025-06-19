# auth/settings_service.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import json
from pathlib import Path
from .settings_layout import (
    load_group_messages, 
    load_llm_settings, 
    load_retrieval_settings
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SettingsService:
    """Service for accessing application settings"""
    
    def __init__(self):
        """Initialize settings service"""
        self.refresh_settings()
    
    def refresh_settings(self):
        """Reload all settings from storage"""
        self.group_messages = load_group_messages()
        self.llm_settings = load_llm_settings()
        self.retrieval_settings = load_retrieval_settings()
    
    def get_group_welcome_message(self, group_id):
        """
        Get welcome message for a specific group
        
        Args:
            group_id (str): ID of the group
            
        Returns:
            str: Welcome message for the group, or default message if not found
        """
        return self.group_messages.get(
            group_id, 
            "Welcome! Ask questions about the documents in this group."
        )
    
    def get_active_llm_provider(self):
        """
        Get the currently active LLM provider configuration
        
        Returns:
            tuple: (provider_name, provider_config)
        """
        provider_name = self.llm_settings.get("provider", "azure-gpt")
        provider_config = self.llm_settings.get(provider_name, {})
        return provider_name, provider_config
    
    def get_relevant_chunks_per_doc(self):
        """Get the number of relevant chunks to retrieve per document"""
        return self.retrieval_settings.get("relevant_chunks_per_doc", 3)
    
    def get_similarity_threshold(self):
        """Get the similarity threshold for retrieval"""
        return self.retrieval_settings.get("similarity_threshold", 0.75)
    
    def get_max_total_chunks(self):
        """Get the maximum total chunks to include in context"""
        return self.retrieval_settings.get("max_total_chunks", 10)
    
    def get_system_message(self):
        """Get the default system message for LLM interactions"""
        return self.retrieval_settings.get(
            "system_message",
            "You are a helpful assistant that answers questions based on the provided documents."
        )
    
    def get_temperature(self):
        """Get the temperature setting for LLM responses"""
        return self.retrieval_settings.get("temperature", 0.7)
    
    def create_llm_client(self):
        """
        Create an LLM client based on current settings
        
        Returns:
            object: LLM client for the current provider
        """
        provider_name, provider_config = self.get_active_llm_provider()
        
        if provider_name == "azure-gpt":
            # Example: Create Azure OpenAI client
            return self._create_azure_client(provider_config)
        elif provider_name == "claude":
            # Example: Create Claude client
            return self._create_claude_client(provider_config)
        elif provider_name == "gemini":
            # Example: Create Gemini client
            return self._create_gemini_client(provider_config)
        elif provider_name == "llama":
            # Example: Create LLAMA client
            return self._create_llama_client(provider_config)
        else:
            # Default to Azure if no valid provider is set
            return self._create_azure_client(self.llm_settings.get("azure-gpt", {}))
    
    def _create_azure_client(self, config):
        """
        Create an Azure OpenAI client
        
        Args:
            config (dict): Azure configuration
            
        Returns:
            object: Azure OpenAI client (Mock example)
        """
        # In a real implementation, you would create an actual Azure OpenAI client
        logger.info("settings_service.py - SettingService - _create_azure_client : Creating Azure OpenAI client with config:", config)
        return {"type": "azure-gpt", "config": config}
    
    def _create_claude_client(self, config):
        """
        Create a Claude client
        
        Args:
            config (dict): Claude configuration
            
        Returns:
            object: Claude client (Mock example)
        """
        # In a real implementation, you would create an actual Claude client
        logger.info("settings_service.py - SettingService - _create_claude_client : Creating Claude client with config:", config)
        logger.info("Creating Claude client with config:", config)
        return {"type": "claude", "config": config}
    
    def _create_gemini_client(self, config):
        """
        Create a Gemini client
        
        Args:
            config (dict): Gemini configuration
            
        Returns:
            object: Gemini client (Mock example)
        """
        # In a real implementation, you would create an actual Gemini client
        logger.info("settings_service.py - SettingService - _create_gemini_client : Creating Gemini client with config:", config)
        return {"type": "gemini", "config": config}
    
    def _create_llama_client(self, config):
        """
        Create a LLAMA client
        
        Args:
            config (dict): LLAMA configuration
            
        Returns:
            object: LLAMA client (Mock example)
        """
        # In a real implementation, you would create an actual LLAMA client
        logger.info("settings_service.py - SettingService - _create_llama_client : Creating LLAMA client with config:", config)
        return {"type": "llama", "config": config}
    
    def apply_retrieval_settings(self, retriever):
        """
        Apply current retrieval settings to a retriever object
        
        Args:
            retriever: The retriever object to configure
            
        Returns:
            object: Configured retriever
        """
        # In a real implementation, this would update the retriever's parameters
        # based on the current settings
        retriever.k = self.get_relevant_chunks_per_doc()
        retriever.similarity_threshold = self.get_similarity_threshold()
        retriever.max_chunks = self.get_max_total_chunks()
        return retriever