# app/callbacks.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.2" # Versi disesuaikan dengan arsitektur baru
__license__  = "MIT"

# Impor library standar dan pihak ketiga
import logging
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dash import Input, Output, State, ctx, html, dcc, PreventUpdate
import traceback
import os
import pandas as pd
import requests
from urllib.parse import urlparse
import shutil

# --- [DIUBAH] Impor terpusat untuk layanan-layanan baru ---
from services.structure_aware_processor import StructureAwareProcessor
from services.vector_store import VectorStoreService
from services.structure.retrieval import HybridRetriever
from services.llm_service import LLMServiceFactory

# Impor lain dari aplikasi
from utils.visualization import create_system_notification
from app import layout
from app.storage_config import get_group_for_user, BASE_STORAGE_DIR
from app.document_callbacks import sync_documents_on_load

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- [BARU] Inisialisasi semua layanan di satu tempat ---
try:
    logger.info("Menginisialisasi layanan-layanan utama...")
    structure_aware_processor = StructureAwareProcessor()
    vector_store_service = VectorStoreService()
    hybrid_retriever = HybridRetriever(vector_store_service)
    logger.info("Semua layanan berhasil diinisialisasi.")
except Exception as e:
    logger.error(f"FATAL: Gagal menginisialisasi layanan inti: {e}", exc_info=True)
    structure_aware_processor = None
    vector_store_service = None
    hybrid_retriever = None

# --- Fungsi Helper ---

def create_feedback_buttons(index: str) -> html.Div:
    """Membuat tombol feedback (Suka/Tidak Suka)."""
    return html.Div([
        html.Button([html.I(className="fas fa-thumbs-up me-1"), "Like"],
                    id={'type': 'like-button', 'index': index},
                    className="btn btn-outline-success btn-sm me-2"),
        html.Button([html.I(className="fas fa-thumbs-down me-1"), "Dislike"],
                    id={'type': 'dislike-button', 'index': index},
                    className="btn btn-outline-danger btn-sm"),
        html.Div("Thank you for your feedback!",
                 id={'type': 'feedback-message', 'index': index},
                 style={'display': 'none'}, className="text-muted mt-2 small")
    ], className="mt-2")

def get_retrieval_settings(group_name: str) -> Dict:
    """Mendapatkan pengaturan retrieval per grup."""
    from auth.route_settings import get_group_llm_settings
    try:
        settings = get_group_llm_settings(group_name)
        return {
            "chunks_per_doc": settings.get("chunks_per_doc", 3),
            "max_total_chunks": settings.get("max_total_chunks", 10),
            "similarity_threshold": settings.get("similarity_threshold", 0.75),
            "lang": settings.get("default_language", "en")
        }
    except Exception as e:
        logger.warning(f"Gagal mendapatkan pengaturan untuk grup '{group_name}', menggunakan default. Error: {e}")
        return {"chunks_per_doc": 3, "max_total_chunks": 10, "similarity_threshold": 0.75, "lang": "en"}

def save_feedback_to_csv(query, response, feedback_type, document_name):
    """Menyimpan feedback ke file CSV."""
    feedback_file = "feedback_data.csv"
    timestamp = datetime.now().isoformat()
    new_data = {'timestamp': [timestamp], 'query': [query], 'response': [response], 
                'feedback': [feedback_type], 'document': [document_name]}
    try:
        df = pd.read_csv(feedback_file) if os.path.exists(feedback_file) else pd.DataFrame(columns=new_data.keys())
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(feedback_file, index=False)
        return True
    except Exception as e:
        logger.error(f"Gagal menyimpan feedback: {e}")
        return False

# --- Registrasi Callback ---

def register_callbacks(app):
    """Mendaftarkan semua callback aplikasi."""
    register_main_callbacks(app)
    register_ui_callbacks(app)
    register_viewer_callbacks(app)
    register_feedback_callbacks(app)
    register_auto_load_callback(app) # Tambahkan ini

def register_main_callbacks(app):
    """Mendaftarkan callback utama untuk upload, query, dan penghapusan."""

    @app.callback(
        [
            Output("document-list", "children"),
            Output("document-state", "data"),
            Output("vectorstore-state", "data"),
            Output("chunk-mapping-state", "data"),
            Output("chat-history", "children"),
            Output("document-selector", "options"),
            Output("upload-error", "children"),
            Output("upload-error", "style"),
            Output("upload-status", "data", allow_duplicate=True)
        ],
        Input("upload-document", "contents"),
        [
            State("upload-document", "filename"),
            State("document-list", "children"),
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("auth-state", "data"),
        ],
        prevent_initial_call=True
    )
    def handle_upload(list_of_contents, list_of_names, doc_list, doc_state, vstore_state, chunk_state, chat_history, auth_state):
        if not list_of_contents:
            raise PreventUpdate

        doc_state, vstore_state, chunk_state, chat_history = doc_state or {}, vstore_state or {}, chunk_state or {}, chat_history or []
        new_doc_list = doc_list or []

        if not auth_state or not auth_state.get('authenticated'):
            return (dash.no_update,) * 6 + (html.Div("Authentication required."), {"display": "block"}, dash.no_update)

        user_id = auth_state.get('user_id')
        group_id = get_group_for_user(user_id)

        for contents, filename in zip(list_of_contents, list_of_names):
            try:
                logger.info(f"Memulai pipeline upload untuk: {filename}")
                # 1. Proses dokumen untuk mendapatkan chunk terstruktur
                structured_chunks, pdf_blob = structure_aware_processor.process_document(contents, filename)
                if not structured_chunks:
                    logger.warning(f"Tidak ada chunk yang dihasilkan dari {filename}")
                    continue

                # 2. Tambahkan dokumen ke VectorStoreService (FAISS + Elasticsearch)
                session_id, chunk_mapping = vector_store_service.add_document(
                    chunks=structured_chunks, filename=filename, user_id=user_id, group_id=group_id, is_temporary=True
                )

                # 3. Update state aplikasi
                doc_state[session_id] = {
                    "filename": filename, "content": pdf_blob or contents, "content_loaded": True,
                    "source": "upload", "timestamp": datetime.now().isoformat(), "user_id": user_id, "group_id": group_id
                }
                vstore_state[session_id] = session_id
                chunk_state[session_id] = json.dumps(chunk_mapping)
                new_doc_list.append(layout.create_document_item(filename, session_id))
                chat_history.append(create_system_notification(f"Added document: {filename}", type="success"))
            
            except Exception as e:
                logger.error(f"Gagal memproses upload untuk {filename}: {e}", exc_info=True)
                return (dash.no_update,) * 6 + (html.Div(f"Error processing {filename}: {e}"), {"display": "block"}, dash.no_update)
        
        selector_options = [{"label": info["filename"], "value": sid} for sid, info in doc_state.items()]
        status_data = {"progress": 100, "status": "Upload complete!"}

        return (new_doc_list, doc_state, vstore_state, chunk_state, chat_history, 
                selector_options, "", {"display": "none"}, status_data)

    @app.callback(
        [
            Output("chat-history", "children", allow_duplicate=True),
            Output("query-input", "value", allow_duplicate=True),
            Output("document-state", "data", allow_duplicate=True),
            Output("document-selector", "value", allow_duplicate=True),
            Output("pdf-highlights", "data", allow_duplicate=True)
        ],
        [Input("submit-btn", "n_clicks"), Input("query-input", "n_submit")],
        [
            State("query-input", "value"),
            State("chat-history", "children"),
            State("document-state", "data"),
            State("chunk-mapping-state", "data"),
            State("auth-state", "data"),
        ],
        prevent_initial_call=True
    )
    def handle_query(n_clicks, n_submit, query, chat_history, doc_state, chunk_state, auth_state):
        if not query or (not n_clicks and not n_submit):
            raise PreventUpdate

        chat_history, doc_state = chat_history or [], doc_state or {}
        chat_history.append(html.Div(f"{query}", className="user-message"))

        if not auth_state or not auth_state.get('authenticated'):
            chat_history.append(html.Div("Error: Anda harus login untuk bertanya.", className="assistant-message error-message"))
            return chat_history, "", doc_state, None, None

        user_id = auth_state.get('user_id')
        group_id = get_group_for_user(user_id)
        llm_serv = LLMServiceFactory.create_llm_service_for_group(group_id)
        retrieval_settings = get_retrieval_settings(group_id)

        try:
            target_filename, query_text = None, query
            if query.startswith("@"):
                parts = query.split(" ", 1)
                doc_name_prefix = parts[0][1:]
                for sid, info in doc_state.items():
                    if info['filename'].lower().startswith(doc_name_prefix.lower()):
                        target_filename, query_text = info['filename'], parts[1] if len(parts) > 1 else ""
                        break
                if not target_filename: raise ValueError(f"Dokumen '{doc_name_prefix}' tidak ditemukan.")
            
            if not target_filename and doc_state:
                # Default ke dokumen pertama jika tidak ada yang spesifik
                target_filename = next(iter(doc_state.values()))['filename']

            if not target_filename: raise ValueError("Tidak ada dokumen yang aktif untuk dicari.")

            session_id = next((sid for sid, info in doc_state.items() if info['filename'] == target_filename), None)
            if not session_id: raise ValueError(f"Sesi aktif untuk '{target_filename}' tidak ditemukan.")

            logger.info(f"Menjalankan Hybrid Retriever pada file '{target_filename}'")
            relevant_chunks = hybrid_retriever.retrieve(
                query=query_text, session_id=session_id, filename=target_filename,
                group_id=group_id, top_k=retrieval_settings.get("max_total_chunks", 5)
            )

            if not relevant_chunks:
                response = "Saya tidak dapat menemukan informasi yang relevan di dalam dokumen yang dipilih."
                chat_history.append(html.Div(response, className="assistant-message"))
                return chat_history, "", doc_state, session_id, None

            context_for_llm = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
            assistant_reply, _, used_chunk_ids_from_llm = llm_serv.get_response(context_for_llm, query_text)
            
            # Proses highlight
            highlights = []
            all_used_chunk_ids = {chunk['chunk_id'] for chunk in relevant_chunks}
            for chunk in relevant_chunks:
                if chunk['chunk_id'] in all_used_chunk_ids:
                     meta = chunk.get('metadata', {})
                     coords = meta.get('coordinates')
                     page_num = meta.get('page_number')
                     if coords and page_num is not None:
                         highlights.append({'page': page_num, 'coords': coords, 'chunk_id': chunk['chunk_id']})

            doc_state[session_id]['highlights'] = highlights
            
            response_id = f"response-{len(chat_history)}"
            source_citation = html.Div(f"Source: {target_filename}", className="source-citation")
            chat_history.append(html.Div([
                dcc.Markdown(assistant_reply), source_citation, create_feedback_buttons(response_id)
            ], className="chat-message llm-response", id={'type': 'response-container', 'index': response_id}))

            return chat_history, "", doc_state, session_id, json.dumps(highlights)

        except Exception as e:
            logger.error(f"Error saat menangani query: {e}", exc_info=True)
            chat_history.append(html.Div(f"Error: {e}", className="assistant-message error-message"))
            return chat_history, query, doc_state, None, None

    @app.callback(
        [
            Output("document-list", "children", allow_duplicate=True),
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Output("document-selector", "options", allow_duplicate=True),
            Output("document-selector", "value", allow_duplicate=True),
        ],
        Input({"type": "remove-document", "index": ALL}, "n_clicks"),
        [
            State("document-list", "children"),
            State("document-state", "data"),
            State("vectorstore-state", "data"),
            State("chunk-mapping-state", "data"),
            State("chat-history", "children"),
            State("document-selector", "value"),
        ],
        prevent_initial_call=True
    )
    def remove_document(n_clicks, doc_list, doc_state, vstore_state, chunk_state, chat_history, current_selected_doc):
        if not any(n_clicks):
            raise PreventUpdate
        
        triggered_id = ctx.triggered_id['index']
        
        if triggered_id in doc_state:
            filename = doc_state[triggered_id].get("filename", "Unknown")
            group_id = doc_state[triggered_id].get("group_id", "Unknown")
            
            logger.info(f"Menghapus dokumen: {filename} (session: {triggered_id})")
            vector_store_service.cleanup_storage(triggered_id, filename, group_id)
            
            # Hapus dari state
            doc_state.pop(triggered_id, None)
            vstore_state.pop(triggered_id, None)
            chunk_state.pop(triggered_id, None)

            # Update UI
            new_doc_list = [item for item in doc_list if item['props']['id']['index'] != triggered_id]
            new_options = [{"label": info["filename"], "value": sid} for sid, info in doc_state.items()]
            new_selected_doc = None if current_selected_doc == triggered_id else current_selected_doc
            chat_history.append(create_system_notification(f"Removed document: {filename}", type="info"))

            return new_doc_list, doc_state, vstore_state, chunk_state, chat_history, new_options, new_selected_doc
        
        return (dash.no_update,) * 7

def register_feedback_callbacks(app):
    @app.callback(
        [Output({'type': 'feedback-message', 'index': MATCH}, 'style'),
         Output({'type': 'like-button', 'index': MATCH}, 'disabled'),
         Output({'type': 'dislike-button', 'index': MATCH}, 'disabled')],
        [Input({'type': 'like-button', 'index': MATCH}, 'n_clicks'),
         Input({'type': 'dislike-button', 'index': MATCH}, 'n_clicks')],
        [State('chat-history', 'children'),
         State('document-selector', 'value'),
         State('document-state', 'data')]
    )
    def handle_feedback(like_clicks, dislike_clicks, chat_history, selected_doc, doc_state):
        if not like_clicks and not dislike_clicks:
            raise PreventUpdate
            
        triggered = ctx.triggered_id
        feedback_type = 'like' if triggered['type'] == 'like-button' else 'dislike'
        
        try:
            response_index = next((i for i, msg in enumerate(chat_history) if isinstance(msg, dict) and msg.get('props', {}).get('id', {}).get('index') == triggered['index']), -1)
            if response_index > 0:
                query = chat_history[response_index-1]['props']['children']
                response_content = chat_history[response_index]['props']['children']
                response = response_content[0]['props']['children'] if isinstance(response_content, list) and isinstance(response_content[0], dict) else str(response_content)
                doc_name = doc_state[selected_doc]['filename'] if selected_doc and selected_doc in doc_state else 'Unknown'
                save_feedback_to_csv(query, response, feedback_type, doc_name)
                return {'display': 'block'}, True, True
        except Exception as e:
            logger.error(f"Error saat menangani feedback: {e}", exc_info=True)
        
        return {'display': 'none'}, False, False

def register_viewer_callbacks(app):
    @app.callback(
        Output("document-viewer-container", "children"),
        Input("document-selector", "value"),
        State("document-state", "data"),
        prevent_initial_call=True
    )
    def update_document_viewer(selected_doc, doc_state):
        if not selected_doc or not doc_state or selected_doc not in doc_state:
            return layout.create_empty_viewer()

        doc_info = doc_state[selected_doc]
        filename = doc_info["filename"]
        content = doc_info.get("content")
        highlights = doc_info.get("highlights", [])

        if filename.lower().endswith('.pdf'):
            return layout.create_pdf_viewer(content)
        else:
            return layout.create_text_viewer(content, highlights)
            
    # Callback untuk memperbarui highlight pada PDF viewer
    app.clientside_callback(
        """
        function(highlights_json, selected_doc, doc_info) {
            if (!selected_doc || !highlights_json || !doc_info || !doc_info.filename.toLowerCase().endsWith('.pdf')) {
                return window.dash_clientside.no_update;
            }
            const highlights = JSON.parse(highlights_json);
            if (window.pdfViewer && highlights) {
                console.log("Applying highlights to PDF:", highlights);
                window.pdfViewer.setHighlights(highlights);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('document-viewer-container', 'data-dummy-output'), # Dummy output
        Input('pdf-highlights', 'data'),
        State('document-selector', 'value'),
        State('document-state', 'data')
    )

def register_ui_callbacks(app):
    # Callback untuk auto-scroll chat
    app.clientside_callback(
        """
        function(children) {
            setTimeout(function() {
                const chatHistory = document.getElementById('chat-history');
                if (chatHistory) {
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
            }, 100);
            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-history', 'data-scroll'),
        Input('chat-history', 'children')
    )
    
def register_auto_load_callback(app):
    """Mendaftarkan callback untuk memuat dokumen persisten saat aplikasi dimulai."""
    @app.callback(
        [
            Output("document-state", "data", allow_duplicate=True),
            Output("vectorstore-state", "data", allow_duplicate=True),
            Output("chunk-mapping-state", "data", allow_duplicate=True),
            Output("document-list", "children", allow_duplicate=True),
            Output("document-selector", "options", allow_duplicate=True)
        ],
        Input("page-load-trigger", "data"), # Ganti nama ID jika berbeda di layout
        [
            State("auth-state", "data"),
            State("document-state", "data"),
        ],
        prevent_initial_call=True
    )
    def auto_load_documents(trigger, auth_state, doc_state):
        if not trigger or not auth_state or not auth_state.get('authenticated'):
            raise PreventUpdate
        
        # Jangan muat ulang jika sudah ada dokumen
        if doc_state:
            raise PreventUpdate

        logger.info("Memuat dokumen persisten untuk pengguna...")
        user_id = auth_state.get('user_id')
        group_id = get_group_for_user(user_id)
        
        new_doc_state = {}
        new_vstore_state = {}
        new_chunk_state = {}
        
        group_path = BASE_STORAGE_DIR / 'vector_stores' / group_id
        if not group_path.exists():
            return {}, {}, {}, [], []

        for file_dir in group_path.iterdir():
            if file_dir.is_dir():
                filename = file_dir.name
                session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{group_id}/{filename}")) # ID konsisten
                
                new_doc_state[session_id] = {
                    "filename": filename, "content": None, "content_loaded": False,
                    "source": "persistent", "timestamp": datetime.now().isoformat(),
                    "user_id": user_id, "group_id": group_id, 'file_path': str(file_dir)
                }
                new_vstore_state[session_id] = session_id
                
                # Muat chunk mapping jika ada
                # (Logika ini mungkin perlu disesuaikan dengan cara Anda menyimpan mapping)
        
        doc_list = [layout.create_document_item(info['filename'], sid) for sid, info in new_doc_state.items()]
        options = [{"label": info['filename'], "value": sid} for sid, info in new_doc_state.items()]

        logger.info(f"Berhasil memuat {len(new_doc_state)} dokumen persisten.")
        return new_doc_state, new_vstore_state, new_chunk_state, doc_list, options
