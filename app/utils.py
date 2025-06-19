__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

def get_user_primary_group(user_id):
    """
    Get a user's primary (first) group
    
    Args:
        user_id (str): The user ID
        
    Returns:
        tuple: (group_id, group_name) or (user_id, "user_{user_id}") if no group
    """
    # Handle None user_id
    if not user_id:
        print("Warning: user_id is None in get_user_primary_group")
        # Create a fallback ID
        fallback_id = "default_user"
        return fallback_id, "Default User"
        
    # Import GroupService
    from auth.group_management import GroupService
    
    try:
        # Initialize group service
        group_service = GroupService()
        
        # Get user's groups
        user_groups = group_service.get_user_groups(str(user_id))
        
        # If user has at least one group, return the first one
        if user_groups and len(user_groups) > 0:
            primary_group_id = user_groups[0]
            # Get the group name
            group_info = group_service.groups.get(primary_group_id, {})
            group_name = group_info.get('name', primary_group_id)
            print(f"User {user_id} has primary group: {primary_group_id} ({group_name})")
            return primary_group_id, group_name
            
    except Exception as e:
        print(f"utils.py - get_user_primary_group : Error getting user's primary group: {str(e)}")

# app/utils/document_utils.py
import base64
import docx2txt
import io
import traceback

def parse_contents(contents, filename, MAX_DOC_SIZE=50*1024*1024):
    """Parse document contents based on file type"""
    try:
        print(f"Parsing Contents for {filename}")
        
        if not contents:
            print("WARNING: No contents received!")
            return None, [], [], ''

        try:
            content_type, content_string = contents.split(',', 1)
            decoded = base64.b64decode(content_string)
        except Exception as e:
            print(f"Error processing content: {str(e)}")
            return None, [], [], ''

        if len(decoded) > MAX_DOC_SIZE:
            raise Exception("File too large (max 50MB)")

        if filename.lower().endswith('.pdf'):
            # Import here to avoid circular imports
            from services.document_processor import DocumentProcessor
            DocProc = DocumentProcessor()
            content, images, tables, plain_text = DocProc.process_pdf(decoded)
            return content, images, tables, plain_text

        elif filename.lower().endswith(('.txt', '.md')):
            content = decoded.decode("utf-8")
            return content, [], [], content

        elif filename.lower().endswith('.docx'):
            content = docx2txt.process(io.BytesIO(decoded))
            return content, [], [], content

        else:
            raise Exception("Unsupported file type")

    except Exception as e:
        print(f"utils.py - parse_contents : Error in parse_contents: {str(e)}")
        traceback.print_exc()
        raise