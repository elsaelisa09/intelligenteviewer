# DokuAI: Intelligent Document Viewer with AI-Powered Question Answering

An intelligent document viewer system that makes your documents searchable and answerable with state-of-the-art AI.

## Overview

DokuAI is a robust document viewer solution based on Python Dash and PDF.js that combines traditional document storage with advanced AI capabilities. The system allows users to:

- Upload, manage, and organize documents (PDF, DOCX, TXT, MD)
- Ask natural language questions about document content
- Receive AI-generated answers with highlighted source references
- Group documents and manage permissions for team-based access
- Support for multiple languages (with specialized handling for English and Indonesian)

The application uses a modern vector search architecture to analyze documents, store their semantic meaning, and retrieve relevant information when questions are asked.

There are 2 modes of documents upload : persistent modes and on-the-fly modes. For persistent modes you need to upload it in document tab ("/documents), wherer as for on-the-fly modes, you directly upload it on the main page and will be forgetted every time you refresh or logout and login.

Each user will be assigned to a group. An admin can assign an admin group. An admin group can upload documents for persistent mode. If the group have persisten documents, this documents will be load automatically each time the user login, and the user can directly ask questions. User can opt to remove the document group temporarily and upload their own document if needed.

I provide some default samples and groups for audit/testing purpose.


<img src="images\querying.png" alt="Query" width="1000"/>

## Key Features

### Document Management
- **Multi-format support**: Handle PDFs, Word documents, Markdown, and plain text
- **Persistent storage**: Documents are saved to disk and can be reloaded
- **Document grouping**: Organize documents into logical groups
- **Permission controls**: Admin and group-based access management
- **Document tagging**: Add custom tags to documents for easier organization

### AI Question Answering
- **Natural language queries**: Ask questions in plain language
- **Source highlighting**: See exactly which parts of which documents were used to generate answers
- **Multi-document queries**: Ask questions that span multiple documents
- **Multilingual support**: Process documents and queries in multiple languages (English/Indonesian focus)
- **Language auto-detection**: Automatically detect document and query language

### User Management
- **User authentication**: Secure login system
- **Role-based access**: Admin, group admin, and regular user roles
- **Group management**: Create and manage document groups with designated admins
- **Custom welcome messages**: Set group-specific welcome messages

### AI Configuration
- **Multiple LLM providers**: Support for Azure OpenAI, Claude, Gemini, and Llama models
- **Customizable retrieval parameters**: Configure document chunking and similarity thresholds
- **Per-group LLM settings**: Set different AI models for different groups

## Technical Architecture

The system is built on a modern stack with several key components:

- **Frontend**: Dash (Python-based reactive web framework) + PDF-JS
- **Document Processing**: PyMuPDF, docx2txt, structure-aware chunking
- **Vector Search**: FAISS for efficient similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM Integration**: Azure OpenAI, Claude, Gemini, and local Llama
- **Language Support**: langid, deep_translator for language detection and translation
- **Authentication**: Custom Flask-based authentication system
- **Storage**: File-based storage with SQLAlchemy database for user management

## System Requirements

- Python 3.8+
- 4GB+ RAM recommended
- SSD storage for better performance
- CUDA-compatible GPU (optional, for faster embeddings)

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/arisukma/IntelligentViewer
   cd docsai
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the database
   ```bash
   python init_db.py
   ```

5. Start the application
   ```bash
   python -m app.main
   ```

6. Open your browser to `http://localhost:8050` and login. 

7. Configure your group and llm connection and parameters in settings ('http://localhost:8050/settings'). Assign group membership to user, setup your LLM keys and parameters.

## Application Snapshot

<img src="images\AdminDashboard.png" alt="Admin" width="1000"/>
<img src="images\system_settings.png" alt="Settings" width="1000"/>
<img src="images\upload_persistent_documents.png" alt="Persistent" width="1000"/>
## Usage

### First-time Setup

1. Log in with the default admin account (admin/admin123)
2. Create user accounts and groups
3. Configure LLM settings in the Settings page
4. Upload your first documents

### Basic Workflow

1. **Upload documents**: Add documents through the upload interface
2. **Organize**: Tag documents and arrange them in groups if needed
3. **Ask questions**: Type natural language questions in the query box
4. **Review answers**: See AI-generated answers with source highlights
5. **Explore documents**: Click on documents to view their content with highlighted sections

### Admin Tasks

1. **User management**: Create, edit, and manage users
2. **Group configuration**: Set up document groups and assign group admins
3. **LLM settings**: Configure AI providers and retrieval parameters
4. **Welcome messages**: Customize group-specific welcome messages

## Configuration

### LLM Providers

The system supports multiple LLM providers that can be configured in the Settings page:

- **Azure OpenAI**: Microsoft's hosted OpenAI models
- **Claude**: Anthropic's Claude models
- **Gemini**: Google's Gemini AI models
- **Llama**: Self-hosted Llama models

### Retrieval Parameters

For each group, you can customize:
- **Chunks per document**: How many chunks to retrieve from each document
- **Maximum total chunks**: Total chunk limit across all documents
- **Similarity threshold**: Minimum relevance score (0-1) for chunks
- **Default language**: Preferred language for the group

## Tips
For better results you can frame your document in form .txt. In current version I use standard way to extract text from .pdf document/s. Pre processing your pdf documents using commercial software before uploading to this platfrom will be a better approach.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License 

## Acknowledgments

- Developed by Ari S Negara (ari.sukmanegara@gmail.com)
- Built with various open-source technologies and libraries
- Special thanks to the teams behind FAISS, Sentence Transformers, and PyMuPDF

---

© 2025 Ari S. Negara
