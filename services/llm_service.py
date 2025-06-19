#services/llm_service.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import openai
import re
import json
import os
from pathlib import Path
from services.language_service import LanguageService
from services.prompt_settings import PromptSettings
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMServiceFactory:
    """Factory class to create LLM service instances based on settings"""
    
    @staticmethod
    def get_llm_settings():
        """Get LLM settings from storage/settings directory"""
        try:
            # Define paths
            storage_dir = Path("storage")
            settings_dir = storage_dir / "settings"
            llm_settings_file = settings_dir / "llm_settings.json"
            
            # Ensure settings directory exists
            settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if settings file exists
            if llm_settings_file.exists():
                try:
                    with open(llm_settings_file, "r") as f:
                        settings = json.load(f)
                    logger.info(f"Llm_service.py - LLMServiceFactory - get_llm_settings : Loaded LLM settings from {llm_settings_file}")
                    return settings
                except Exception as e:
                    logger.error(f"Llm_service.py - LLMServiceFactory - get_llm_settings ~1 : Error loading LLM settings file: {str(e)}")
            
            # If no settings file or error loading, check route_settings
            try:
                from auth.route_settings import get_llm_settings
                settings = get_llm_settings()
                if settings:
                    logger.info("Llm_service.py - LLMServiceFactory - get_llm_settings : Loaded LLM settings from route_settings")
                    return settings
            except Exception as e:
                logger.info(f"Llm_service.py - LLMServiceFactory - get_llm_settings ~ 2 : Error loading LLM settings from route_settings: {str(e)}")
            
            # Fall back to environment variables and defaults
            return {
                "provider": os.environ.get("LLM_PROVIDER", "azure-gpt"),
                "azure": {
                    "api_key": os.environ.get("AZURE_API_KEY", ""),
                    "endpoint": os.environ.get("AZURE_ENDPOINT", ""),
                    "deployment_name": os.environ.get("AZURE_DEPLOYMENT_NAME", ""),
                    "api_version": os.environ.get("AZURE_API_VERSION", "2023-05-15"),
                },
                "claude": {
                    "api_key": os.environ.get("CLAUDE_API_KEY", ""),
                    "model": os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229"),
                    "api_version": os.environ.get("CLAUDE_API_VERSION", "2024-02-15"),
                },
                "gemini": {
                    "api_key": os.environ.get("GEMINI_API_KEY", ""),
                    "project_id": os.environ.get("GEMINI_PROJECT_ID", ""),
                    "model": os.environ.get("GEMINI_MODEL", "gemini-pro"),
                    "api_version": os.environ.get("GEMINI_API_VERSION", "v1"),
                },
                "llama": {
                    "api_key": os.environ.get("LLAMA_API_KEY", ""),
                    "endpoint": os.environ.get("LLAMA_ENDPOINT", ""),
                    "model": os.environ.get("LLAMA_MODEL", ""),
                    "api_version": os.environ.get("LLAMA_API_VERSION", ""),
                }
            }
        except Exception as e:
            logger.info(f"Llm_service.py - LLMServiceFactory - get_llm_settings ~ 3: Error in get_llm_settings: {str(e)}")
            import traceback
            traceback.print_exc()
            # Absolute fallback to defaults
            return {
                "provider": "azure-gpt",
                "azure": {
                    "api_key": "",
                    "endpoint": "",
                    "deployment_name": "",
                    "api_version": "2023-05-15",
                }
            }
    
    @staticmethod
    def save_llm_settings(settings):
        """Save LLM settings to storage/settings directory"""
        try:
            storage_dir = Path("storage")
            settings_dir = storage_dir / "settings"
            llm_settings_file = settings_dir / "llm_settings.json"
            
            # Ensure settings directory exists
            settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Save settings
            with open(llm_settings_file, "w") as f:
                json.dump(settings, f, indent=4)
            
            logger.info(f"Saved LLM settings to {llm_settings_file}")
            return True
        except Exception as e:
            logger.info(f"Error saving LLM settings: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def create_llm_service():
        """Create an appropriate LLM service based on settings"""
        settings = LLMServiceFactory.get_llm_settings()
        provider = settings.get("provider", "azure-gpt")
        
        logger.info(f"Llm_service.py - LLMServiceFactory - create_llm_service : Creating LLM service for provider: {provider}")
        
        if provider == "azure-gpt":
            azure_settings = settings.get("azure", {})
            return AzureOpenAIService(
                api_key=azure_settings.get("api_key", ""),
                api_base=azure_settings.get("endpoint", ""),
                api_version=azure_settings.get("api_version", "2023-05-15"),
                deployment_name=azure_settings.get("deployment_name", "")
            )
        elif provider == "claude":
            claude_settings = settings.get("claude", {})
            return ClaudeService(
                api_key=claude_settings.get("api_key", ""),
                model=claude_settings.get("model", "claude-3-opus-20240229"),
                api_version=claude_settings.get("api_version", "2024-02-15")
            )
        elif provider == "gemini":
            gemini_settings = settings.get("gemini", {})
            return GeminiService(
                api_key=gemini_settings.get("api_key", ""),
                project_id=gemini_settings.get("project_id", ""),
                model=gemini_settings.get("model", "gemini-pro"),
                api_version=gemini_settings.get("api_version", "v1")
            )
        elif provider == "llama":
            llama_settings = settings.get("llama", {})
            return LlamaService(
                api_key=llama_settings.get("api_key", ""),
                api_base=llama_settings.get("endpoint", ""),
                model=llama_settings.get("model", ""),
                api_version=llama_settings.get("api_version", "")
            )
        else:
            # Default to Azure OpenAI
            azure_settings = settings.get("azure", {})
            return AzureOpenAIService(
                api_key=azure_settings.get("api_key", ""),
                api_base=azure_settings.get("endpoint", ""),
                api_version=azure_settings.get("api_version", "2023-05-15"),
                deployment_name=azure_settings.get("deployment_name", "")
            )

    @staticmethod
    def create_llm_service_for_group(group_id=None):
        """
        Create an LLM service for a specific group
        
        Args:
            group_id (str): The group ID to create the service for
            
        Returns:
            LLMService: An instance of the appropriate LLM service for the group
        """
        try:
            # Import the GroupLLMSettings class
            from services.group_llm_settings import GroupLLMSettings
            
            # Get the provider for this group
            if group_id:
                provider = GroupLLMSettings.get_provider_for_group(group_id)
            else:
                # If no group ID provided, get the default settings
                settings = LLMServiceFactory.get_llm_settings()
                provider = settings.get("provider", "azure-gpt")
            
            logger.info(f"Creating LLM service for group '{group_id}' using provider: {provider}")
            
            # Create the appropriate service based on provider
            if provider == "azure-gpt":
                azure_settings = LLMServiceFactory.get_llm_settings().get("azure", {})
                service = AzureOpenAIService(
                    api_key=azure_settings.get("api_key", ""),
                    api_base=azure_settings.get("endpoint", ""),
                    api_version=azure_settings.get("api_version", "2023-05-15"),
                    deployment_name=azure_settings.get("deployment_name", "")
                )
            elif provider == "claude":
                claude_settings = LLMServiceFactory.get_llm_settings().get("claude", {})
                service = ClaudeService(
                    api_key=claude_settings.get("api_key", ""),
                    model=claude_settings.get("model", "claude-3-opus-20240229"),
                    api_version=claude_settings.get("api_version", "2024-02-15")
                )
            elif provider == "gemini":
                gemini_settings = LLMServiceFactory.get_llm_settings().get("gemini", {})
                service = GeminiService(
                    api_key=gemini_settings.get("api_key", ""),
                    project_id=gemini_settings.get("project_id", ""),
                    model=gemini_settings.get("model", "gemini-pro"),
                    api_version=gemini_settings.get("api_version", "v1")
                )
            elif provider == "llama":
                llama_settings = LLMServiceFactory.get_llm_settings().get("llama", {})
                service = LlamaService(
                    api_key=llama_settings.get("api_key", ""),
                    api_base=llama_settings.get("endpoint", ""),
                    model=llama_settings.get("model", ""),
                    api_version=llama_settings.get("api_version", "")
                )
            else:
                # Default to Azure OpenAI
                azure_settings = LLMServiceFactory.get_llm_settings().get("azure", {})
                service = AzureOpenAIService(
                    api_key=azure_settings.get("api_key", ""),
                    api_base=azure_settings.get("endpoint", ""),
                    api_version=azure_settings.get("api_version", "2023-05-15"),
                    deployment_name=azure_settings.get("deployment_name", "")
                )
            
            # Set group-specific retrieval parameters if available
            if group_id:
                retrieval_params = GroupLLMSettings.get_retrieval_params_for_group(group_id)
                if retrieval_params:
                    logger.info(f"Setting group-specific retrieval parameters for group '{group_id}': {retrieval_params}")
                    service.chunks_per_doc = retrieval_params.get("chunks_per_doc", 3)
                    service.max_total_chunks = retrieval_params.get("max_total_chunks", 10)
                    service.similarity_threshold = retrieval_params.get("similarity_threshold", 0.75)
                    service.default_language = retrieval_params.get("default_language", "en")
            
            return service
            
        except Exception as e:
            logger.error(f"Error creating LLM service for group '{group_id}': {str(e)}")
            # Fallback to default service
            return LLMServiceFactory.create_llm_service()

# Base LLM service class (abstract)
class LLMService:
    def __init__(self):
        self.language_service = LanguageService()
        
        # Default retrieval parameters
        self.chunks_per_doc = 3
        self.max_total_chunks = 10
        self.similarity_threshold = 0.75
        self.default_language = "en"
    
    def get_response(self, context, query):
        """Get response from LLM with improved chunk tracking and language handling"""
        raise NotImplementedError("Subclasses must implement get_response method")
    
    def _extract_relevant_documents(self, analysis):
        """Extract which documents were actually used as sources with improved pattern matching"""
        relevant_docs = set()
        
        # Debug the input
        logger.info("Llm_service.py - LLMService - _extract_relevant_document : Debug - Full analysis for document extraction:")
        logger.info(f"Llm_service.py - LLMServiceFactory - _extract_relevant_document : {analysis}")
        
        # Enhanced patterns to catch more variations (fixed syntax)
        doc_patterns = [
            r"Document:\s*([^:\n]+?)(?:\n|$)",  # Match document name at start of line
            r"Document:\s*([^:\n]+?)(?=\s*Used as|$)",  # Match document with "Used as" following
            r"Document:\s*([^:\n]+?)(?=\s*Score|$)",  # Match document with "Score" following
            r"([^:\n]+?\.(?:pdf|txt|docx))\s*(?:was used|is relevant|Used as)",  # Match filenames with extensions
            r'From\s+"([^"\n]+?)"',  # Match double-quoted document names
            r"From\s+'([^'\n]+?)'",  # Match single-quoted document names
            r"Source:\s*([^:\n]+?)(?:\n|$)"  # Match source declarations
        ]
        
        # First pass: look for explicit usage indicators
        for pattern in doc_patterns:
            matches = re.finditer(pattern, analysis, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                doc_name = match.group(1).strip()
                if doc_name:
                    # Clean up the document name
                    doc_name = re.sub(r'\s+', ' ', doc_name)  # Normalize whitespace
                    doc_name = doc_name.strip('" ')  # Remove quotes and spaces
                    logger.debug(f"Llm_service.py - LLMServiceFactory - _extract_relevant_document: Debug - Found document: {doc_name}")
                    
                    # Check surrounding context for relevance
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(analysis), match.end() + 50)
                    context = analysis[context_start:context_end]
                    
                    # Look for positive indicators in context
                    positive_indicators = [
                        'used', 'relevant', 'source', 'referenced', 'cited',
                        'score', 'chunk', 'provides', 'contains', 'supports', 'aligned'
                    ]
                    
                    # Look for negative indicators in context
                    negative_indicators = [
                        'not used', 'irrelevant', 'unrelated', 'not relevant',
                        'not a source', 'not referenced', 'not cited'
                    ]
                    
                    # Check if context contains positive indicators but not negative ones
                    has_positive = any(indicator in context.lower() for indicator in positive_indicators)
                    has_negative = any(indicator in context.lower() for indicator in negative_indicators)
                    
                    if has_positive and not has_negative:
                        relevant_docs.add(doc_name)
                        logger.debug(f"Llm_service.py - LLMService - _extract_relevant_document : Added positive relevant document: {doc_name}")
                    elif not has_positive and not has_negative:
                        relevant_docs.add(doc_name)
                        logger.debug(f"Llm_service.py - LLMService - _extract_relevant_document : Added neutral relevant document: {doc_name}")

        
        # Second pass: look for text chunks with high scores
        chunk_score_pattern = r"Chunk\s+\d+:\s*(?:Score\s*)?(\d+)/10"
        doc_chunk_pattern = r"Document:\s*([^:\n]+?)(?:\n|$)(.*?)(?=Document:|$)"
        
        for match in re.finditer(doc_chunk_pattern, analysis, re.DOTALL):
            doc_name = match.group(1).strip()
            chunk_section = match.group(2)
            
            # Find scores in this document's chunks
            scores = [int(score) for score in re.findall(chunk_score_pattern, chunk_section)]
            if scores and max(scores) >= 7:  # If any chunk has a high score
                doc_name = re.sub(r'\s+', ' ', doc_name).strip('" ')
                relevant_docs.add(doc_name)
                logger.debug(f"Llm_service.py - LLMService - _extract_relevant_document : Added document due to high chunk score: {doc_name}")
        
        logger.info(f"Debug - Final relevant documents: {relevant_docs}")
        return relevant_docs

    def _extract_scores(self, analysis, chunks):
        """Extract scores from LLM analysis with improved parsing"""
        scores = {}
        logger.info("Debug - Extracting scores for chunks:")
        
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            content_preview = chunk['content'][:50] if len(chunk['content']) > 50 else chunk['content']
            logger.debug(f"Llm_service.py - LLMService - _extract_scores : Processing chunk {chunk_id}: {content_preview}...")
            
            score_found = False
            
            # Enhanced scoring patterns
            score_patterns = [
                (r"Chunk.*?(\d+)/10", 10),  # Divide by 10
                (r"Score:?\s*(\d+)\s*/?\s*10?", 10),  # Handle "Score: X" or "Score: X/10"
                (r"Rating:?\s*(\d+)", 10),
                (r"Relevance:?\s*(\d+)", 10),
                (r"Chunk.*?score.*?(\d+)", 10),  # More general pattern
                (r"(\d+)\s*out of\s*10", 10),  # "X out of 10" format
                (r"(\d+)\s*points", 10),  # "X points" format
            ]
            
            # First try to find an exact match for the chunk content
            chunk_context = self._find_chunk_context(analysis, content_preview)
            if chunk_context:
                logger.debug(f"Llm_service.py - LLMService - _extract_scores : Found context for chunk: {chunk_context[:100]}...")
                
                for pattern, scale in score_patterns:
                    match = re.search(pattern, chunk_context, re.IGNORECASE)
                    if match:
                        try:
                            score = int(match.group(1))
                            normalized_score = score / scale
                            scores[chunk_id] = normalized_score
                            score_found = True
                            logger.info(f"Llm_service.py - LLMService - _extract_scores : Found score {score}/{scale} = {normalized_score} for chunk {chunk_id}")
                            break
                        except ValueError:
                            continue
            
            # If no score found but chunk is from a relevant document, use initial score
            if not score_found:
                initial_score = chunk.get('initial_score', 0.5)  # Default to 0.5 if no initial score
                scores[chunk_id] = initial_score
                logger.debug(f"Llm_service.py - LLMService - _extract_scores : Using initial score {initial_score} for chunk {chunk_id}")
        
        logger.info(f"Debug - Final scores: {scores}")
        return scores

    def _combine_and_rank_scores(self, chunks_to_rank, scores):
        """Combine LLM scores with initial similarity scores and rank chunks"""
        for chunk in chunks_to_rank:
            chunk['final_score'] = (
                0.89 * scores[chunk['chunk_id']] + 
                0.11 * chunk['initial_score'])
            logger.info(chunk)
            logger.info(chunk['final_score'])
            
        sorted_chunk = sorted(
            chunks_to_rank, 
            key=lambda x: x['final_score'], 
            reverse=True
        )
        
        # Use class retrieval parameters instead of hardcoded values
        # Return up to max_total_chunks chunks, but also apply the modified condition
        if len(sorted_chunk) >= 2 and sorted_chunk[0]['final_score'] - sorted_chunk[1]['final_score'] <= 0.1:
            return sorted_chunk[:min(self.max_total_chunks, 2)]
        else:
            return [sorted_chunk[0]]
        
    def _find_chunk_context(self, analysis, chunk_start):
        """Find the analysis context for a specific chunk"""
        # Look for chunk content with surrounding context
        chunk_loc = analysis.find(chunk_start)
        if chunk_loc >= 0:
            # Get surrounding 200 characters
            start = max(0, chunk_loc - 100)
            end = min(len(analysis), chunk_loc + 300)
            return analysis[start:end]
        return None
    
    def get_default_language(self):
        """Get the default language for this service"""
        return self.default_language
    
# Azure OpenAI Service Implementation
class AzureOpenAIService(LLMService):
    def __init__(self, api_key, api_base, api_version, deployment_name):
        super().__init__()
        
        # Fix for empty or incomplete API base URL
        if not api_base or not (api_base.startswith('http://') or api_base.startswith('https://')):
            if api_base:
                api_base = f"https://{api_base}"
            else:
                logger.info("WARNING: Empty API base URL. Using placeholder that will likely fail.")
                api_base = "https://example.openai.azure.com"
        
        logger.info(f"Llm_service.py - AzureOpenAIService(LLMService) - _init : Initializing Azure OpenAI with base URL: {api_base}")
        
        openai.api_type = "azure"
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_key = api_key
        self.deployment_name = deployment_name
        logger.info(f"Llm_service.py - AzureOpenAIService(LLMService) - __init__ : Initialized Azure OpenAI service with deployment: {deployment_name}")
        logger.info(f"Llm_service.py - AzureOpenAIService(LLMService) - __init__ : Using API version: {api_version}")

    def get_response(self, context, query):
        """Get response from Azure OpenAI with improved chunk tracking and language handling"""
        try:
            # Detect query language
            query_lang, _ = self.language_service.detect_language(query)
            logger.info(f"Query language detected: {query_lang}")
            
            # Split context while preserving document boundaries
            document_chunks = {}
            current_doc = None
            current_chunks = []
            doc_languages = {}  # Keep track of document languages
            
            for line in context.split('\n'):
                if line.startswith('=== From Document:'):
                    if current_doc and current_chunks:
                        document_chunks[current_doc] = current_chunks
                    current_doc = line.replace('=== From Document:', '').strip()
                    current_chunks = []
                elif line.startswith('[chunk_'):
                    # Extract language information if present
                    lang_info = None
                    if '(lang:' in line and ')' in line.split('(lang:')[1]:
                        lang_part = line.split('(lang:')[1].split(')')[0]
                        lang_info = lang_part.strip()
                        
                        # Keep track of document languages
                        if current_doc and lang_info:
                            if current_doc not in doc_languages:
                                doc_languages[current_doc] = []
                            if lang_info not in doc_languages[current_doc]:
                                doc_languages[current_doc].append(lang_info)
                    
                    current_chunks.append(line)
            
            if current_doc and current_chunks:
                document_chunks[current_doc] = current_chunks

            # Prepare numbered chunks with document context
            numbered_chunks = []
            chunk_mapping = {}  # Map numeric index to chunk_id and filename
            counter = 1
            
            # Determine primary language for the context
            all_langs = []
            for doc, langs in doc_languages.items():
                all_langs.extend(langs)
            
            # Count language occurrences to find dominant language
            lang_counts = {}
            for lang in all_langs:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            # Determine dominant language in context
            context_lang = max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else 'id'
            logger.info(f"Primary context language detected: {context_lang}")
            
            # Process each document and its chunks
            for doc, chunks in document_chunks.items():
                for chunk in chunks:
                    # First extract the chunk ID properly
                    chunk_id_match = re.search(r'\[chunk_([^\s\]]+)', chunk)
                    if not chunk_id_match:
                        continue
                    
                    # Extract the chunk ID
                    chunk_id = chunk_id_match.group(1)
                    # Remove any closing bracket if it got included
                    if ']' in chunk_id:
                        chunk_id = chunk_id.split(']')[0]
                    
                    # Extract the filename - look for quoted filenames first
                    filename = doc  # Default to document name
                    # Look for "from "FILENAME"" pattern (with quotes)
                    quoted_match = re.search(r'from\s+"([^"]+)"', chunk)
                    if quoted_match:
                        filename = quoted_match.group(1).strip()
                    else:
                        # Fall back to standard pattern
                        from_match = re.search(r'from\s+([^\s(]+)', chunk)
                        if from_match:
                            filename = from_match.group(1).strip()
                    
                    # Clean the filename by removing any trailing punctuation
                    filename = re.sub(r'[,\.\]\)\s]+$', '', filename)
                    
                    # Extract text part (everything after the chunk identifier)
                    text_start = chunk.find(']')
                    if text_start > 0:
                        chunk_text = chunk[text_start + 1:].strip()
                    else:
                        chunk_text = chunk  # Fallback
                    
                    # Format the numbered chunk with both chunk number and filename
                    # Use quotes around the filename to preserve spaces and special characters
                    numbered_chunks.append(f'[{counter}] Chunk_{chunk_id} from "{filename}":\n{chunk_text}\n')
                    
                    # Store mapping for later retrieval
                    # This is the key change: map the context position number to the actual chunk ID
                    chunk_mapping[counter] = {
                        'chunk_id': chunk_id,  # Store the original chunk ID 
                        'filename': filename
                    }
                    counter += 1

            print("###################################################################")
            logger.info("LLM service numbered chunks")
            for i, chunk in enumerate(numbered_chunks[:5]):  # Print first 5 chunks for debug
                logger.info(f"{i+1}. {chunk[:200]}...")
            logger.info(f"Total chunks: {len(numbered_chunks)}")
            print("###################################################################")
            
            # Print the chunk mapping for debugging
            logger.info("Chunk mapping (first 5 entries):")
            for i in range(1, min(6, len(chunk_mapping) + 1)):
                logger.info(f"Position {i} -> Chunk ID: {chunk_mapping[i]['chunk_id']}, Filename: {chunk_mapping[i]['filename']}")
            print("###################################################################")

            # Determine if translation is needed for query
            translated_query = query
            if query_lang != context_lang.replace('"',''):
                translated_query = self.language_service.translate_text(query, query_lang, context_lang)
                logger.info(f"Translated query from {query_lang} to {context_lang}: {query} -> {translated_query}")
            
            # Use translated query for prompting if languages differ
            prompt_query = translated_query if query_lang != context_lang else query

            # Get prompt from centralized prompt settings
            prompt_context = '\n'.join(numbered_chunks)
            formatted_prompt = PromptSettings.format_prompt(
                "document_query",
                context=prompt_context,
                query=prompt_query
            )
            
            # Get system prompt from centralized settings
            system_prompt = PromptSettings.get_prompt("system_instruction")

            # Debug info before making the API call
            logger.info(f"Making Azure OpenAI API request with:")
            logger.info(f"- API Base: {openai.api_base}")
            logger.info(f"- API Version: {openai.api_version}")
            logger.info(f"- Deployment: {self.deployment_name}")

            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=800,
            )

            answer_text = response.choices[0].message.content
            logger.info(f"Raw LLM response: {answer_text}")
            
            # Extract chunks used with improved parsing that captures both chunk numbers and filenames
            chunks_match = re.search(r"Used chunks:\s*\[(.*?)\]", answer_text, re.IGNORECASE | re.DOTALL)
            used_chunks = []
            
            if chunks_match:
                # Parse the chunks list to extract both chunk numbers and filenames
                chunks_list = chunks_match.group(1).strip()
                logger.info(f"Extracted chunk list: {chunks_list}")
                
                # Look for patterns with quotes around filenames: X from "FILENAME"
                chunk_entries = re.findall(r'(\d+)(?:\s+from\s+|\s+in\s+|\s+of\s+)["\'"]?([^"\',]+)["\']?', chunks_list)
                
                if chunk_entries:
                    for chunk_num_str, filename in chunk_entries:
                        try:
                            chunk_num = int(chunk_num_str.strip())
                            if chunk_num in chunk_mapping:
                                # IMPORTANT FIX: Get the ORIGINAL chunk_id from the mapping
                                original_chunk_id = chunk_mapping[chunk_num]['chunk_id']
                                original_filename = chunk_mapping[chunk_num]['filename']
                                
                                # Create chunk info string with the ORIGINAL chunk ID, not the position number
                                chunk_info = f"chunk_{original_chunk_id} from {original_filename}"
                                used_chunks.append(chunk_info)
                                logger.info(f"Added chunk: {chunk_info} (from position {chunk_num})")
                            else:
                                logger.info(f"Warning: Chunk position {chunk_num} not found in mapping")
                        except (ValueError, KeyError) as e:
                            logger.error(f"Error processing chunk reference: {e}")
                else:
                    # Fallback: If no entries found with the pattern, try just extracting numbers
                    # and matching them to the chunk mapping
                    number_matches = re.findall(r'(\d+)', chunks_list)
                    for num_str in number_matches:
                        try:
                            chunk_num = int(num_str.strip())
                            if chunk_num in chunk_mapping:
                                # Get the ORIGINAL chunk_id, not the position number
                                original_chunk_id = chunk_mapping[chunk_num]['chunk_id']
                                original_filename = chunk_mapping[chunk_num]['filename']
                                
                                chunk_info = f"chunk_{original_chunk_id} from {original_filename}"
                                used_chunks.append(chunk_info)
                                logger.info(f"Added chunk (fallback): {chunk_info} (from position {chunk_num})")
                            else:
                                logger.info(f"Warning: Chunk position {chunk_num} not found in mapping")
                        except (ValueError, KeyError) as e:
                            logger.info(f"Error in fallback chunk extraction: {e}")
                
                # Remove the chunks list from the answer
                answer_only = answer_text.split("Used chunks:")[0].strip()
            else:
                # If no explicit "Used chunks:" section, look for any numbers in the response
                # and try to match them to chunks
                answer_only = answer_text
                logger.warning("Warning: No 'Used chunks:' section found in LLM response")
                
                # Try to find any numbers that might be chunk references
                all_numbers = re.findall(r'(?<!\d)(\d+)(?!\d)', answer_text)
                for num_str in all_numbers:
                    try:
                        chunk_num = int(num_str)
                        if chunk_num in chunk_mapping:
                            # Get the ORIGINAL chunk_id, not the position number
                            original_chunk_id = chunk_mapping[chunk_num]['chunk_id']
                            original_filename = chunk_mapping[chunk_num]['filename']
                            
                            chunk_info = f"chunk_{original_chunk_id} from {original_filename}"
                            if chunk_info not in used_chunks:
                                used_chunks.append(chunk_info)
                                logger.info(f"Found potential chunk reference: {chunk_info} (from position {chunk_num})")
                        else:
                            logger.info(f"Warning: Chunk position {chunk_num} not found in mapping")
                    except (ValueError, KeyError):
                        pass
                        
                # If still no chunks found, use the first chunk as fallback
                if not used_chunks and chunk_mapping:
                    first_chunk = list(chunk_mapping.keys())[0]
                    original_chunk_id = chunk_mapping[first_chunk]['chunk_id']
                    original_filename = chunk_mapping[first_chunk]['filename']
                    
                    chunk_info = f"chunk_{original_chunk_id} from {original_filename}"
                    used_chunks.append(chunk_info)
                    logger.info(f"Using fallback first chunk: {chunk_info} (from position {first_chunk})")

            # Translate response back to original query language if needed
            if query_lang != context_lang:
                logger.info(f"Translating response back to {query_lang} from {context_lang}")
                translated_answer = self.language_service.translate_text(answer_only, context_lang, query_lang)
                logger.info(f"Original response: {answer_only[:100]}...")
                logger.info(f"Translated response: {translated_answer[:100]}...")
                return answer_only, translated_answer, used_chunks
            
            return answer_only, answer_only, used_chunks

        except Exception as e:
            logger.info(f"Error getting Azure OpenAI response: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []
        
# Google Gemini Service Implementation
class GeminiService(LLMService):
    def __init__(self, api_key, project_id="", model="gemini-pro", api_version="v1"):
        super().__init__()
        self.api_key = api_key
        self.project_id = project_id
        self.model = model
        self.api_version = api_version
        logger.info(f"Llm_service.py - GeminiService(LLMService) - init : Initialized Gemini service with model: {model}")
    
    def get_response(self, context, query):
        try:
            # Import Gemini API
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key, http_options={'api_version': self.api_version})
            
            # Detect query language
            query_lang, _ = self.language_service.detect_language(query)
            logger.info(f"Query language detected: {query_lang}")
            
            # Process the context (same as in Claude implementation)
            document_chunks, numbered_chunks, chunk_mapping, context_lang = self._process_context(context)
            
            # Translate query if needed
            prompt_query = query
            if query_lang != context_lang:
                prompt_query = self.language_service.translate_text(query, query_lang, context_lang)
                logger.info(f"Translated query from {query_lang} to {context_lang}: {query} -> {prompt_query}")
            
            # Get prompt from centralized prompt settings
            prompt_context = '\n'.join(numbered_chunks)
            formatted_prompt = PromptSettings.format_prompt(
                "document_query",
                context=prompt_context,
                query=prompt_query
            )
            
            # Get system prompt from centralized settings
            system_prompt = PromptSettings.get_prompt("system_instruction")
            combined_prompt = f"System instructions: {system_prompt}\n\n{formatted_prompt}"
            
            # Get Gemini model
            model = self.model
            
            
            # Generate content  
            response = client.models.generate_content(
                model = model,
                contents = combined_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=800,
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                        }
                    ]
                )
            )
            
            # Extract text response
            answer_text = response.text
            logger.info(f"Raw Gemini response: {answer_text}")
            
            # Extract used chunks using the same logic as other implementations
            used_chunks = self._extract_used_chunks(answer_text, chunk_mapping)
            
            # Get final answer (without the chunk list)
            answer_only = answer_text.split("Used chunks:")[0].strip() if "Used chunks:" in answer_text else answer_text
            
            # Translate response back to original query language if needed
            if query_lang != context_lang:
                logger.info(f"Translating response back to {query_lang} from {context_lang}")
                translated_answer = self.language_service.translate_text(answer_only, context_lang, query_lang)
                logger.info(f"Original response: {answer_only[:100]}...")
                logger.info(f"Translated response: {translated_answer[:100]}...")
                return answer_only, translated_answer, used_chunks
            
            return answer_only, answer_only, used_chunks
            
        except Exception as e:
            logger.error(f"Error getting Gemini response: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []
    
    # Reuse the _process_context and _extract_used_chunks methods from ClaudeService
    def _process_context(self, context):
        """Process context - reusing implementation from Claude Service"""
        return ClaudeService._process_context(self, context)
    
    def _extract_used_chunks(self, answer_text, chunk_mapping):
        """Extract referenced chunks - reusing implementation from Claude Service"""
        return ClaudeService._extract_used_chunks(self, answer_text, chunk_mapping)


# Llama Service Implementation
class LlamaService(LLMService):
    def __init__(self, api_key, api_base, model="llama-70b", api_version=""):
        super().__init__()
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.api_version = api_version
        logger.info(f"Llm_service.py - LlamaService(LLMService) - init : Initialized Llama service with model: {model}")
    
    def get_response(self, context, query):
        try:
            # Llama often uses a simple REST API approach similar to OpenAI
            import requests
            import json
            
            # Detect query language
            query_lang, _ = self.language_service.detect_language(query)
            logger.info(f"Query language detected: {query_lang}")
            
            # Process the context (same as in Claude implementation)
            document_chunks, numbered_chunks, chunk_mapping, context_lang = self._process_context(context)
            
            # Translate query if needed
            prompt_query = query
            if query_lang != context_lang:
                prompt_query = self.language_service.translate_text(query, query_lang, context_lang)
                logger.info(f"Translated query from {query_lang} to {context_lang}: {query} -> {prompt_query}")
            
            # Get prompt from centralized prompt settings
            prompt_context = '\n'.join(numbered_chunks)
            formatted_prompt = PromptSettings.format_prompt(
                "document_query",
                context=prompt_context,
                query=prompt_query
            )
            
            # Get system prompt from centralized settings
            system_prompt = PromptSettings.get_prompt("system_instruction")

            # Create headers with API key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Create payload
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 800
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse response
            resp_json = response.json()
            answer_text = resp_json['choices'][0]['message']['content']
            logger.info(f"Raw Llama response: {answer_text}")
            
            # Extract used chunks using the same logic as other implementations
            used_chunks = self._extract_used_chunks(answer_text, chunk_mapping)
            
            # Get final answer (without the chunk list)
            answer_only = answer_text.split("Used chunks:")[0].strip() if "Used chunks:" in answer_text else answer_text
            
            # Translate response back to original query language if needed
            if query_lang != context_lang:
                logger.info(f"Translating response back to {query_lang} from {context_lang}")
                translated_answer = self.language_service.translate_text(answer_only, context_lang, query_lang)
                logger.info(f"Original response: {answer_only[:100]}...")
                logger.info(f"Translated response: {translated_answer[:100]}...")
                return answer_only, translated_answer, used_chunks
            
            return answer_only, answer_only, used_chunks
            
        except Exception as e:
            logger.error(f"Error getting Llama response: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []
    
    # Reuse the _process_context and _extract_used_chunks methods from ClaudeService
    def _process_context(self, context):
        """Process context - reusing implementation from Claude Service"""
        return ClaudeService._process_context(self, context)
    
    def _extract_used_chunks(self, answer_text, chunk_mapping):
        """Extract referenced chunks - reusing implementation from Claude Service"""
        return ClaudeService._extract_used_chunks(self, answer_text, chunk_mapping)

# Anthropic Claude Service Implementation
class ClaudeService(LLMService):
    def __init__(self, api_key, model="claude-3-opus-20240229", api_version="2024-02-15"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.api_version = api_version
        logger.info(f"Llm_service.py - ClaudeService(LLMService) - init : Initialized Claude service with model: {model}")
    
    def get_response(self, context, query):
        try:
            import anthropic
            
            # Initialize Anthropic client
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Detect query language
            query_lang, _ = self.language_service.detect_language(query)
            logger.info(f"Query language detected: {query_lang}")
            
            # Process context and create numbered chunks (similar to Azure implementation)
            document_chunks, numbered_chunks, chunk_mapping, context_lang = self._process_context(context)
            
            # Translate query if needed
            prompt_query = query
            if query_lang != context_lang:
                prompt_query = self.language_service.translate_text(query, query_lang, context_lang)
                logger.info(f"Translated query from {query_lang} to {context_lang}: {query} -> {prompt_query}")
            
            # Get prompt from centralized prompt settings
            prompt_context = '\n'.join(numbered_chunks)
            formatted_prompt = PromptSettings.format_prompt(
                "document_query",
                context=prompt_context,
                query=prompt_query
            )
            
            # Get system prompt from centralized settings
            system_prompt = PromptSettings.get_prompt("system_instruction")

            # Make the API call to Claude
            response = client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.2,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ]
            )

            # Get the response text
            answer_text = response.content[0].text
            logger.info(f"Raw Claude response: {answer_text}")
            
            # Extract used chunks with the same logic as Azure implementation
            used_chunks = self._extract_used_chunks(answer_text, chunk_mapping)
            
            # Get final answer (without the chunk list)
            answer_only = answer_text.split("Used chunks:")[0].strip() if "Used chunks:" in answer_text else answer_text
            
            # Translate response back to original query language if needed
            if query_lang != context_lang:
                logger.info(f"Translating response back to {query_lang} from {context_lang}")
                translated_answer = self.language_service.translate_text(answer_only, context_lang, query_lang)
                logger.info(f"Original response: {answer_only[:100]}...")
                logger.info(f"Translated response: {translated_answer[:100]}...")
                return answer_only, translated_answer, used_chunks
            
            return answer_only, answer_only, used_chunks
            
        except Exception as e:
            logger.info(f"Error getting Claude response: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []
    
    def _process_context(self, context):
        """Process context similar to Azure implementation"""
        # Split context while preserving document boundaries
        document_chunks = {}
        current_doc = None
        current_chunks = []
        doc_languages = {}  # Keep track of document languages
        
        for line in context.split('\n'):
            if line.startswith('=== From Document:'):
                if current_doc and current_chunks:
                    document_chunks[current_doc] = current_chunks
                current_doc = line.replace('=== From Document:', '').strip()
                current_chunks = []
            elif line.startswith('[chunk_'):
                # Extract language information if present
                lang_info = None
                if '(lang:' in line and ')' in line.split('(lang:')[1]:
                    lang_part = line.split('(lang:')[1].split(')')[0]
                    lang_info = lang_part.strip()
                    
                    # Keep track of document languages
                    if current_doc and lang_info:
                        if current_doc not in doc_languages:
                            doc_languages[current_doc] = []
                        if lang_info not in doc_languages[current_doc]:
                            doc_languages[current_doc].append(lang_info)
                
                current_chunks.append(line)
        
        if current_doc and current_chunks:
            document_chunks[current_doc] = current_chunks

        # Prepare numbered chunks with document context
        numbered_chunks = []
        chunk_mapping = {}  # Map numeric index to chunk_id and filename
        counter = 1
        
        # Determine primary language for the context
        all_langs = []
        for doc, langs in doc_languages.items():
            all_langs.extend(langs)
        
        # Count language occurrences to find dominant language
        lang_counts = {}
        for lang in all_langs:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Determine dominant language in context
        context_lang = max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else 'en'
        logger.info(f"Llm_service.py - ClaudeService(LLMService) - get_response : Primary context language detected: {context_lang}")
        
        # Process each document and its chunks
        for doc, chunks in document_chunks.items():
            for chunk in chunks:
                # First extract the chunk ID properly
                chunk_id_match = re.search(r'\[chunk_([^\s\]]+)', chunk)
                if not chunk_id_match:
                    continue
                
                # Extract the chunk ID
                chunk_id = chunk_id_match.group(1)
                # Remove any closing bracket if it got included
                if ']' in chunk_id:
                    chunk_id = chunk_id.split(']')[0]
                
                # Extract the filename - look for quoted filenames first
                filename = doc  # Default to document name
                # Look for "from "FILENAME"" pattern (with quotes)
                quoted_match = re.search(r'from\s+"([^"]+)"', chunk)
                if quoted_match:
                    filename = quoted_match.group(1).strip()
                else:
                    # Fall back to standard pattern
                    from_match = re.search(r'from\s+([^\s(]+)', chunk)
                    if from_match:
                        filename = from_match.group(1).strip()
                
                # Clean the filename by removing any trailing punctuation
                filename = re.sub(r'[,\.\]\)\s]+$', '', filename)
                
                # Extract text part (everything after the chunk identifier)
                text_start = chunk.find(']')
                if text_start > 0:
                    chunk_text = chunk[text_start + 1:].strip()
                else:
                    chunk_text = chunk  # Fallback
                
                # Format the numbered chunk with both chunk number and filename
                # Use quotes around the filename to preserve spaces and special characters
                numbered_chunks.append(f'[{counter}] Chunk_{chunk_id} from "{filename}":\n{chunk_text}\n')
                
                # Store mapping for later retrieval
                chunk_mapping[counter] = {
                    'chunk_id': chunk_id,
                    'filename': filename
                }
                counter += 1
        
        return document_chunks, numbered_chunks, chunk_mapping, context_lang
    
    def _extract_used_chunks(self, answer_text, chunk_mapping):
        """Extract referenced chunks from the response"""
        used_chunks = []
        
        # Extract chunks used with improved parsing that captures both chunk numbers and filenames
        chunks_match = re.search(r"Used chunks:\s*\[(.*?)\]", answer_text, re.IGNORECASE | re.DOTALL)
        
        if chunks_match:
            # Parse the chunks list to extract both chunk numbers and filenames
            chunks_list = chunks_match.group(1).strip()
            logger.info(f"Llm_service.py - ClaudeService(LLMService) - get_response : Extracted chunk list: {chunks_list}")
            
            # Look for patterns with quotes around filenames: X from "FILENAME"
            chunk_entries = re.findall(r'(\d+)(?:\s+from\s+|\s+in\s+|\s+of\s+)["\'"]?([^"\',]+)["\']?', chunks_list)
            
            if chunk_entries:
                for chunk_num_str, filename in chunk_entries:
                    try:
                        chunk_num = int(chunk_num_str.strip())
                        if chunk_num in chunk_mapping:
                            # Get the ORIGINAL chunk_id from the mapping
                            original_chunk_id = chunk_mapping[chunk_num]['chunk_id']
                            original_filename = chunk_mapping[chunk_num]['filename']
                            
                            # Create chunk info string with the ORIGINAL chunk ID, not the position number
                            chunk_info = f"chunk_{original_chunk_id} from {original_filename}"
                            used_chunks.append(chunk_info)
                            logger.info(f"Llm_service.py - ClaudeService(LLMService) - get_response : Added chunk: {chunk_info} (from position {chunk_num})")
                        else:
                            logger.warning(f"Llm_service.py - ClaudeService(LLMService) - get_response : Warning: Chunk position {chunk_num} not found in mapping")
                    except (ValueError, KeyError) as e:
                        logger.info(f"Error processing chunk reference: {e}")
            else:
                # Fallback extraction logic
                number_matches = re.findall(r'(\d+)', chunks_list)
                for num_str in number_matches:
                    try:
                        chunk_num = int(num_str.strip())
                        if chunk_num in chunk_mapping:
                            original_chunk_id = chunk_mapping[chunk_num]['chunk_id']
                            original_filename = chunk_mapping[chunk_num]['filename']
                            
                            chunk_info = f"chunk_{original_chunk_id} from {original_filename}"
                            used_chunks.append(chunk_info)
                            logger.info(f"Llm_service.py - ClaudeService(LLMService) - get_response : Added chunk (fallback): {chunk_info} (from position {chunk_num})")
                    except (ValueError, KeyError) as e:
                        logger.info(f"Error in fallback chunk extraction: {e}")
        else:
            # If no explicit chunks section, fallback to numbers in text
            logger.info("Llm_service.py - ClaudeService(LLMService) - get_response : No 'Used chunks:' section found in response")
            
            # Fallback mechanism
            all_numbers = re.findall(r'(?<!\d)(\d+)(?!\d)', answer_text)
            for num_str in all_numbers:
                try:
                    chunk_num = int(num_str)
                    if chunk_num in chunk_mapping:
                        original_chunk_id = chunk_mapping[chunk_num]['chunk_id']
                        original_filename = chunk_mapping[chunk_num]['filename']
                        
                        chunk_info = f"chunk_{original_chunk_id} from {original_filename}"
                        if chunk_info not in used_chunks:
                            used_chunks.append(chunk_info)
                    else:
                        logger.warning(f"Llm_service.py - ClaudeService(LLMService) - get_response : Warning: Chunk position {chunk_num} not found in mapping")
                except (ValueError, KeyError):
                    pass
            
            # If still no chunks found, use the first chunk as fallback
            if not used_chunks and chunk_mapping:
                first_chunk = list(chunk_mapping.keys())[0]
                original_chunk_id = chunk_mapping[first_chunk]['chunk_id']
                original_filename = chunk_mapping[first_chunk]['filename']
                
                chunk_info = f"chunk_{original_chunk_id} from {original_filename}"
                used_chunks.append(chunk_info)
                logger.info(f"Llm_service.py - ClaudeService(LLMService) - get_response : Using fallback first chunk: {chunk_info}")
        
        return used_chunks

    # Function for backwards compatibility
@staticmethod
def create_llm_service():
    """Create an appropriate LLM service based on settings"""
    settings = LLMServiceFactory.get_llm_settings()
    provider = settings.get("provider", "azure-gpt")
    
    logger.info(f"Llm_service.py - create_llm_service : Creating LLM service for provider: {provider}")
    
    if provider == "azure-gpt":
        azure_settings = settings.get("azure", {})
        
        # Extract settings with more detailed debug info
        api_key = azure_settings.get("api_key", "")
        api_base = azure_settings.get("endpoint", "")
        api_version = azure_settings.get("api_version", "2023-05-15")
        deployment_name = azure_settings.get("deployment_name", "")
        
        # Fix for empty or malformed API base URL
        if not api_base or not (api_base.startswith('http://') or api_base.startswith('https://')):
            if api_base:
                api_base = f"https://{api_base}"
                logger.info(f"Llm_service.py - create_llm_service : Fixed API base URL: {api_base}")
            else:
                logger.warning("Llm_service.py - create_llm_service : WARNING: Empty API base URL. Using placeholder that will likely fail.")
                api_base = "https://example.openai.azure.com"
        
        # Print debug info
        logger.info(f"Llm_service.py - create_llm_service : Azure OpenAI Settings:")
        logger.info(f"Llm_service.py - create_llm_service : - API Base: {api_base}")
        logger.info(f"Llm_service.py - create_llm_service : - API Version: {api_version}")
        logger.info(f"Llm_service.py - create_llm_service : - Deployment Name: {deployment_name}")
        logger.info(f"Llm_service.py - create_llm_service : - API Key length: {len(api_key)} chars")
        
        return AzureOpenAIService(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            deployment_name=deployment_name
        )
    elif provider == "claude":
        claude_settings = settings.get("claude", {})
        return ClaudeService(
            api_key=claude_settings.get("api_key", ""),
            model=claude_settings.get("model", "claude-3-opus-20240229"),
            api_version=claude_settings.get("api_version", "2024-02-15")
        )
    elif provider == "gemini":
        gemini_settings = settings.get("gemini", {})
        return GeminiService(
            api_key=gemini_settings.get("api_key", ""),
            project_id=gemini_settings.get("project_id", ""),
            model=gemini_settings.get("model", "gemini-pro"),
            api_version=gemini_settings.get("api_version", "v1")
        )
    elif provider == "llama":
        llama_settings = settings.get("llama", {})
        return LlamaService(
            api_key=llama_settings.get("api_key", ""),
            api_base=llama_settings.get("endpoint", ""),
            model=llama_settings.get("model", ""),
            api_version=llama_settings.get("api_version", "")
        )
    else:
        # Default to Azure OpenAI
        azure_settings = settings.get("azure", {})
        return AzureOpenAIService(
            api_key=azure_settings.get("api_key", ""),
            api_base=azure_settings.get("endpoint", ""),
            api_version=azure_settings.get("api_version", "2023-05-15"),
            deployment_name=azure_settings.get("deployment_name", "")
        )