# services/vector_store.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.3" # Versi diperbarui dengan fungsi load
__license__  = "MIT"

import os
import json
import shutil
from pathlib import Path
import tempfile
import uuid
import logging
import traceback
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# Impor library pihak ketiga
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Impor dari proyek Anda
from services.models.structured_chunk import StructuredChunk
from app.storage_config import BASE_STORAGE_DIR

# Impor untuk Elasticsearch
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk
from app.config import ELASTICSEARCH_HOST, ELASTICSEARCH_INDEX_NAME

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi model embedding
try:
    os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        cache_folder="./model_cache"
    )
except Exception as e:
    logger.error(f"Gagal memuat model embedding: {e}")
    embeddings_model = None

class VectorStoreService:
    """
    Mengelola penyimpanan dan pengambilan chunk dokumen.
    Mendukung penyimpanan ganda (FAISS + Elasticsearch) dan pemuatan (load).
    """
    def __init__(self):
        if not embeddings_model:
            raise RuntimeError("Model embedding tidak berhasil dimuat. VectorStoreService tidak dapat berfungsi.")
            
        self.embeddings = embeddings_model
        self.TEMP_DIR = Path(tempfile.gettempdir()) / "faiss_indices_temp"
        self.TEMP_DIR.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # [DIUBAH] Menambahkan cache untuk vector store dan chunk mapping
        self._vectorstore_cache = {}
        self._chunk_mapping_cache = {}

        # Inisialisasi Klien Elasticsearch
        self.es_client = None
        self.es_index_name = ELASTICSEARCH_INDEX_NAME
        try:
            logger.info(f"Menghubungkan ke Elasticsearch di: {ELASTICSEARCH_HOST}")
            self.es_client = Elasticsearch(
                hosts=[ELASTICSEARCH_HOST], 
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3
            )
            if not self.es_client.ping():
                raise ConnectionError("Koneksi ke server Elasticsearch gagal.")
            logger.info("Berhasil terhubung ke Elasticsearch.")
            self._initialize_elasticsearch_index()
        except Exception as e:
            self.es_client = None
            logger.error(f"Inisialisasi Elasticsearch gagal: {e}. Fitur pencarian keyword (BM25) akan nonaktif.")

    def _initialize_elasticsearch_index(self):
        """[BARU] Memeriksa apakah index sudah ada, jika tidak, maka akan dibuat."""
        if self.es_client and not self.es_client.indices.exists(index=self.es_index_name):
            try:
                self.es_client.indices.create(index=self.es_index_name)
                logger.info(f"Index Elasticsearch '{self.es_index_name}' berhasil dibuat.")
            except Exception as e:
                logger.error(f"Gagal membuat index Elasticsearch '{self.es_index_name}': {e}")

    def add_document(self, chunks: List[StructuredChunk], filename: str, user_id: str, group_id: str, is_temporary: bool = False) -> Tuple[str, Dict]:
        if not chunks:
            self.logger.warning(f"Tidak ada chunk yang diberikan untuk file {filename}. Proses penyimpanan dibatalkan.")
            return None, {}

        session_id = str(uuid.uuid4())
        self.logger.info(f"Memulai pipeline penyimpanan ganda untuk file: {filename}, session: {session_id}")

        try:
            documents_for_faiss = []
            es_actions = []
            chunk_mapping = {}

            for chunk in chunks:
                doc = Document(page_content=chunk.text, metadata=chunk.metadata)
                documents_for_faiss.append(doc)
                
                if self.es_client:
                    es_doc = {
                        'content': chunk.text, 'filename': filename, 'group_id': group_id,
                        'chunk_id': chunk.id, 'chunk_index': chunk.chunk_index, 'page_number': chunk.page_number,
                        'section': chunk.section_title, 'keywords': chunk.keywords, 'topics': chunk.topics
                    }
                    action = {"_index": self.es_index_name, "_id": chunk.id, "_source": es_doc}
                    es_actions.append(action)
                chunk_mapping[chunk.id] = chunk.metadata

            vectorstore = FAISS.from_documents(documents_for_faiss, self.embeddings)
            self._save_faiss_store(session_id, vectorstore, filename, group_id, is_temporary)

            if self.es_client and es_actions:
                try:
                    success, failed = bulk(self.es_client, es_actions, raise_on_error=True)
                    logger.info(f"Bulk indexing ke Elasticsearch berhasil: {success} dokumen.")
                    if failed: logger.error(f"Gagal mengindeks {len(failed)} dokumen: {failed}")
                except Exception as e:
                    logger.error(f"Gagal melakukan bulk indexing ke Elasticsearch: {e}", exc_info=True)
            
            self._chunk_mapping_cache[session_id] = chunk_mapping
            self._vectorstore_cache[session_id] = vectorstore # Simpan ke cache setelah dibuat
            return session_id, chunk_mapping

        except Exception as e:
            self.logger.error(f"Error besar saat `add_document` untuk {filename}: {str(e)}", exc_info=True)
            self.cleanup_storage(session_id, filename, group_id)
            raise
            
    def load_vectorstore(self, session_id: str, filename: str, group_id: str) -> Optional[FAISS]:
        """
        [BARU] Memuat vector store FAISS dari disk.
        Mencari di cache terlebih dahulu, lalu di direktori permanen, lalu temporer.
        """
        if session_id in self._vectorstore_cache:
            self.logger.debug(f"Mengambil vector store untuk session '{session_id}' dari cache.")
            return self._vectorstore_cache[session_id]

        # Cek penyimpanan permanen terlebih dahulu
        perm_path = BASE_STORAGE_DIR / 'vector_stores' / group_id / filename
        if perm_path.exists():
            try:
                self.logger.info(f"Memuat vector store dari penyimpanan permanen: {perm_path}")
                vs = FAISS.load_local(str(perm_path), self.embeddings, allow_dangerous_deserialization=True)
                self._vectorstore_cache[session_id] = vs
                return vs
            except Exception as e:
                self.logger.error(f"Gagal memuat vector store permanen dari {perm_path}: {e}")
                return None
        
        # Cek penyimpanan temporer jika tidak ditemukan di permanen
        temp_path = self.TEMP_DIR / session_id
        if temp_path.exists():
            try:
                self.logger.info(f"Memuat vector store dari penyimpanan temporer: {temp_path}")
                vs = FAISS.load_local(str(temp_path), self.embeddings, allow_dangerous_deserialization=True)
                self._vectorstore_cache[session_id] = vs
                return vs
            except Exception as e:
                self.logger.error(f"Gagal memuat vector store temporer dari {temp_path}: {e}")
                return None

        self.logger.warning(f"Vector store untuk session '{session_id}' atau file '{filename}' tidak ditemukan.")
        return None

    def cleanup_storage(self, session_id: str, filename: str, group_id: str):
        self.logger.info(f"Memulai cleanup untuk file '{filename}' (session: {session_id}, group: {group_id})")
        
        self._chunk_mapping_cache.pop(session_id, None)
        self._vectorstore_cache.pop(session_id, None) # Hapus juga dari cache vectorstore

        perm_dir = BASE_STORAGE_DIR / 'vector_stores' / group_id / filename
        temp_dir = self.TEMP_DIR / session_id
        
        for dir_path in [perm_dir, temp_dir]:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    self.logger.info(f"Direktori penyimpanan lokal '{dir_path}' berhasil dihapus.")
                except Exception as e:
                    self.logger.error(f"Gagal menghapus direktori '{dir_path}': {e}")
        
        if self.es_client:
            try:
                query = {"query": {"bool": {"must": [{"term": {"group_id.keyword": group_id}}, {"term": {"filename.keyword": filename}}]}}}
                self.es_client.delete_by_query(index=self.es_index_name, body=query, refresh=True)
                self.logger.info(f"Berhasil menghapus dokumen dari Elasticsearch untuk '{filename}'.")
            except Exception as e:
                self.logger.error(f"Gagal menghapus data dari Elasticsearch untuk '{filename}': {e}", exc_info=True)
    
    def _save_faiss_store(self, session_id: str, vectorstore: FAISS, filename: str, group_id: str, is_temporary: bool):
        save_path = self.TEMP_DIR / session_id if is_temporary else BASE_STORAGE_DIR / 'vector_stores' / group_id / filename
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(folder_path=str(save_path))
            self.logger.info(f"Vector store FAISS untuk '{filename}' berhasil disimpan di: {save_path}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan FAISS store di '{save_path}': {e}", exc_info=True)
