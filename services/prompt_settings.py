# services/prompt_settings.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptSettings:
    """Class to manage prompt templates for LLM services"""
    
    # Default prompts for different use cases
    DEFAULT_PROMPTS = {
        "document_query": """Answer the question based on the provided chunks of text. You must answer using the same language used mostly in the context. 

VERY IMPORTANT: After your answer, you MUST specify the most relevant chunk numbers that used for your short answer, in order of relevance, WITH their filenames. Format it exactly as [X from FILENAME, Y from FILENAME] where X and Y are the chunk numbers.

Context:
{context}

Question: {query}

Format your response exactly like this:
Answer: <your short answer using same language as context>
Reason: <your reason (if any) using same language as context>
Used chunks: [X from FILENAME, Y from FILENAME, Z from FILENAME] where X, Y, Z are the chunk numbers that are most relevant to the answer, in order of relevance.""",
        
        "system_instruction": """You are an expert that answers questions based on provided context. 
ALWAYS specify the most relevant chunks you used in your answer (maximum 3) and ALWAYS include the filename for each chunk.
Format the chunk references EXACTLY as: [X from FILENAME, Y from FILENAME] where X and Y are the chunk numbers.
Your chunk references MUST appear in a separate line starting with 'Used chunks:' at the end of your response.""",
        
        "welcome_message": """Welcome to the document assistant! I can help you find information in your documents.
Ask me any question about your uploaded documents, and I'll do my best to provide relevant answers."""
    }
    
    @staticmethod
    def get_prompt_settings():
        """Get prompt settings from storage"""
        try:
            # Define paths
            storage_dir = Path("storage")
            settings_dir = storage_dir / "settings"
            prompts_file = settings_dir / "prompt_templates.json"
            
            # Ensure settings directory exists
            settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if settings file exists
            if prompts_file.exists():
                try:
                    with open(prompts_file, "r") as f:
                        prompts = json.load(f)
                    logger.info(f"Loaded prompt templates from {prompts_file}")
                    
                    # Ensure all default prompt types exist
                    for prompt_type, default_prompt in PromptSettings.DEFAULT_PROMPTS.items():
                        if prompt_type not in prompts:
                            prompts[prompt_type] = default_prompt
                            logger.info(f"Added missing default prompt type: {prompt_type}")
                    
                    return prompts
                except Exception as e:
                    logger.error(f"Error loading prompt templates file: {str(e)}")
            
            # If no settings file or error loading, return defaults
            logger.info("Using default prompt templates")
            return PromptSettings.DEFAULT_PROMPTS.copy()
            
        except Exception as e:
            logger.error(f"Error in get_prompt_settings: {str(e)}")
            import traceback
            traceback.print_exc()
            return PromptSettings.DEFAULT_PROMPTS.copy()
    
    @staticmethod
    def save_prompt_settings(prompts):
        """Save prompt settings to storage"""
        try:
            storage_dir = Path("storage")
            settings_dir = storage_dir / "settings"
            prompts_file = settings_dir / "prompt_templates.json"
            
            # Ensure settings directory exists
            settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure all default prompt types exist
            for prompt_type, default_prompt in PromptSettings.DEFAULT_PROMPTS.items():
                if prompt_type not in prompts:
                    prompts[prompt_type] = default_prompt
                    logger.info(f"Added missing default prompt type: {prompt_type}")
            
            # Save settings
            with open(prompts_file, "w") as f:
                json.dump(prompts, f, indent=4)
            
            logger.info(f"Saved prompt templates to {prompts_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving prompt templates: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def get_prompt(prompt_type):
        """Get a specific prompt by type"""
        prompts = PromptSettings.get_prompt_settings()
        
        if prompt_type in prompts:
            return prompts[prompt_type]
        elif prompt_type in PromptSettings.DEFAULT_PROMPTS:
            # If not in saved prompts but exists in defaults
            return PromptSettings.DEFAULT_PROMPTS[prompt_type]
        else:
            # Unknown prompt type
            logger.warning(f"Unknown prompt type requested: {prompt_type}")
            return ""
    
    @staticmethod
    def format_prompt(prompt_type, **kwargs):
        """Get and format a prompt with the provided variables"""
        prompt_template = PromptSettings.get_prompt(prompt_type)
        
        try:
            # Format the prompt template with provided variables
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing key in prompt formatting: {str(e)}")
            return prompt_template  # Return unformatted template
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return prompt_template  # Return unformatted template