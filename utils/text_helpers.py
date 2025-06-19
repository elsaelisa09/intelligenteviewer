# utils/text_helpers.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from typing import List, Dict, Union, Optional
import re
from dataclasses import dataclass

@dataclass
class TextSection:
    """Data class for storing processed text sections"""
    type: str  # 'paragraph', 'code', or 'heading'
    text: str
    id: int

class TextProcessor:
    @staticmethod
    def process_content(content: str):
        """Process document content while preserving code blocks and structure"""
        if not content:
            return []

        def is_markdown_heading(line):
            if not line.strip().startswith('#'):
                return False
            heading_level = 0
            for char in line.strip():
                if char != '#':
                    break
                heading_level += 1
            remaining = line.strip()[heading_level:]
            return remaining.startswith(' ') and len(remaining.strip()) > 1

        def looks_like_code(text):
            code_indicators = [
                'def ',
                'class ',
                'import ',
                #'from ',
                'return ',
                #'@',
                'print(',
                'if __name__',
                '.py',
                'except:',
                'try:',
                'else:',
                'elif ',
            # '##  ',
                '    ',  # Indentation (4 spaces)
                '\t',    # Tab character
            ]
            return any(indicator in text for indicator in code_indicators)

        sections = []
        current_section = []
        in_code_block = False
        code_block_indent = 0
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Detect code block start
            if not in_code_block and looks_like_code(line):
                if current_section:
                    sections.append({
                        "type": "paragraph",
                        "text": '\n'.join(current_section)
                    })
                    current_section = []
                in_code_block = True
                code_block_indent = len(line) - len(line.lstrip())
                current_section = [line]
                continue

            # Inside code block
            if in_code_block:
                # Check if we're leaving the code block
                if not line.strip():
                    # Empty line in code block
                    current_section.append(line)
                elif len(line) - len(line.lstrip()) < code_block_indent and not looks_like_code(line):
                    # End of code block
                    sections.append({
                        "type": "code",
                        "text": '\n'.join(current_section)
                    })
                    current_section = [line]
                    in_code_block = False
                else:
                    # Continue code block
                    current_section.append(line)
                continue

            # Regular text processing
            if is_markdown_heading(line):
                if current_section:
                    sections.append({
                        "type": "paragraph",
                        "text": '\n'.join(current_section)
                    })
                    current_section = []
                sections.append({
                    "type": "heading",
                    "text": line
                })
            else:
                if not line.strip() and current_section:
                    sections.append({
                        "type": "paragraph",
                        "text": '\n'.join(current_section)
                    })
                    current_section = []
                elif line.strip():
                    current_section.append(line)

        # Add the last section if exists
        if current_section:
            if in_code_block:
                sections.append({
                    "type": "code",
                    "text": '\n'.join(current_section)
                })
            else:
                sections.append({
                    "type": "paragraph",
                    "text": '\n'.join(current_section)
                })

        return [{"id": i, **section} for i, section in enumerate(sections)]

    @staticmethod
    def extract_text_from_component(component: Union[str, List, Dict, None]) -> str:
        """
        Extract text from a Dash component recursively.
        
        Args:
            component: Dash component or nested structure
            
        Returns:
            str: Extracted text content
        """
        if component is None:
            return ""
            
        if isinstance(component, str):
            return component
            
        if isinstance(component, list):
            return " ".join(TextProcessor.extract_text_from_component(item) 
                          for item in component if item is not None)
            
        if isinstance(component, dict):
            if "props" in component:
                return TextProcessor.extract_text_from_component(
                    component["props"].get("children", "")
                )
            return ""
            
        return str(component)

    @staticmethod
    def analyze_llm_response(response_text: str) -> bool:
        """
        Analyze if LLM response indicates no relevant information.
        
        Args:
            response_text: Response text from LLM
            
        Returns:
            bool: True if response indicates no relevant information
        """
        no_info_phrases = [
            "no relevant",
            "cannot find",
            "no information",
            "don't have enough information",
            "no specific",
            "not mentioned",
            "not found",
            "cannot answer",
            "no answer",
            "not provided",
            "no context",
            "insufficient information"
        ]
        
        response_lower = response_text.lower()
        return any(phrase in response_lower for phrase in no_info_phrases)