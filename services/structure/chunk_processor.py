# services/structure/chunk_processor.py (Direvisi sebagai Transformer)

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.2" # Versi diubah untuk mencerminkan peran sebagai transformer

import uuid
import logging
from typing import List, Dict, Tuple, Any
from datetime import datetime

# Impor elemen dari unstructured
from unstructured.documents.elements import Element, CompositeElement

# Impor model StructuredChunk dari aplikasi Anda
from ..models.structured_chunk import StructuredChunk

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Coba impor spacy untuk pengayaan (enrichment)
try:
    import spacy
    # [DIUBAH] Muat model hanya sekali saat modul diimpor
    NLP_MODEL = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("Model Spacy 'en_core_web_sm' berhasil dimuat.")
except (ImportError, OSError):
    NLP_MODEL = None
    SPACY_AVAILABLE = False
    logger.warning("Spacy tidak ditemukan atau model tidak bisa dimuat. Fitur ekstraksi keyword/topik akan dinonaktifkan.")


class ChunkTransformer:
    """
    [NAMA DIUBAH] Kelas ini bertugas mengubah (transform) daftar elemen yang sudah di-chunk
    dari unstructured.io menjadi daftar objek `StructuredChunk` internal aplikasi.

    Kelas ini TIDAK LAGI melakukan chunking. Chunking sudah dilakukan di DocumentProcessor.
    """
    def __init__(self):
        """
        Inisialisasi transformer. Tidak memerlukan parameter chunking lagi.
        """
        self.logger = logging.getLogger(__name__)

    def transform_elements(self, chunks_from_unstructured: List[Element], filename: str) -> List[StructuredChunk]:
        """
        [DIUBAH] Fungsi utama untuk mengubah daftar elemen menjadi StructuredChunk.

        Args:
            chunks_from_unstructured (List[Element]): Daftar elemen yang SUDAH di-chunk
                                                      oleh DocumentProcessor.
            filename (str): Nama file asli untuk pencatatan metadata.

        Returns:
            List[StructuredChunk]: Daftar objek StructuredChunk yang siap disimpan.
        """
        if not chunks_from_unstructured:
            self.logger.warning(f"Tidak ada chunk yang diterima dari {filename} untuk ditransformasi.")
            return []
        
        self.logger.info(f"Memulai transformasi {len(chunks_from_unstructured)} chunk dari file {filename}.")
        
        final_structured_chunks = []
        for i, chunk_element in enumerate(chunks_from_unstructured):
            # Mengonversi setiap elemen chunk menjadi objek StructuredChunk
            structured_chunk = self._convert_to_structured_chunk(chunk_element, filename, i)
            if structured_chunk:
                final_structured_chunks.append(structured_chunk)
            
        self.logger.info(f"Berhasil mentransformasi {len(final_structured_chunks)} chunk dari {filename}.")
        return final_structured_chunks

    def _convert_to_structured_chunk(self, chunk: Element, filename: str, index: int) -> StructuredChunk:
        """
        Mengonversi sebuah elemen chunk (biasanya CompositeElement) dari unstructured
        menjadi objek StructuredChunk yang kaya metadata.
        """
        chunk_text = str(chunk)
        if not chunk_text.strip():
            return None # Lewati chunk kosong

        chunk_id = f"{filename}_{index}_{str(uuid.uuid4())[:8]}"
        
        # Ekstrak metadata dari elemen unstructured
        metadata = self._extract_metadata(chunk)
        metadata['filename'] = filename
        metadata['creation_time'] = datetime.now().isoformat()
        
        # Tambahkan informasi semantik (keywords, topics) jika Spacy tersedia
        keywords, topics = self._extract_semantic_info(chunk_text)
        metadata['keywords'] = keywords
        metadata['topics'] = topics
        
        # Ambil path hierarki dari metadata untuk struktur
        hierarchical_path = metadata.get('path_parts', [])
        section_title = hierarchical_path[-1] if hierarchical_path else "Uncategorized"

        return StructuredChunk(
            id=chunk_id,
            text=chunk_text,
            metadata=metadata,
            section_title=section_title,
            parent_section=hierarchical_path[-2] if len(hierarchical_path) > 1 else None,
            hierarchical_path=hierarchical_path,
            semantic_type=metadata.get('category', 'general'),
            topics=topics,
            keywords=keywords,
            chunk_index=index,
            page_number=metadata.get('page_number')
        )

    def _extract_metadata(self, element: Element) -> Dict[str, Any]:
        """
        [DISEMPURNAKAN] Mengekstrak dan menormalkan metadata yang relevan dari elemen unstructured.
        Ini adalah fungsi inti dari transformer.
        """
        meta = element.metadata.to_dict()
        
        # Jika ini adalah CompositeElement, ia memiliki sub-elemen.
        # Metadata yang paling akurat (seperti page_number) ada di sub-elemen tersebut.
        orig_elements_meta = meta.get('orig_elements', [])
        
        page_numbers = []
        path_parts = []
        
        # Ambil metadata dari elemen-elemen asli yang membentuk chunk ini
        if orig_elements_meta:
            # Ambil path dari elemen pertama sebagai representasi hirarki
            first_el_meta = orig_elements_meta[0]
            path_str = first_el_meta.get('path', '')
            path_parts = [p for p in path_str.split('/') if p]

            # Kumpulkan semua nomor halaman dari sub-elemen
            for sub_meta in orig_elements_meta:
                if sub_meta.get('page_number') is not None:
                    page_numbers.append(sub_meta['page_number'])

        # Ambil page number pertama sebagai acuan, atau dari metadata utama jika tidak ada
        page_number = page_numbers[0] if page_numbers else meta.get('page_number')
        
        return {
            "page_number": page_number,
            "category": meta.get('category', 'Unknown'),
            "path_parts": path_parts
        }

    def _extract_semantic_info(self, text: str) -> Tuple[List[str], List[str]]:
        """
        [TIDAK DIUBAH] Mengekstrak keyword dan topik dari teks menggunakan Spacy.
        """
        if not SPACY_AVAILABLE or not NLP_MODEL or not text:
            return [], []

        doc = NLP_MODEL(text)
        
        keywords = {ent.text.strip() for ent in doc.ents}
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and len(chunk.text.strip()) > 3:
                keywords.add(chunk.text.strip())
        
        topics = set()
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        topics.add(f"{token.text} {token.head.text} {child.text}")
                        break
                        
        return sorted(list(keywords), key=len, reverse=True)[:10], list(topics)[:5]
