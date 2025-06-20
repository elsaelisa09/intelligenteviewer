# services/structure_aware_processor.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.2" # Versi disesuaikan dengan arsitektur baru

import logging
from typing import List, Tuple, Optional

# [DIUBAH] Impor prosesor dan transformer yang telah kita refactor
from services.document_processor import DocumentProcessor
from services.structure.chunk_processor import ChunkTransformer

# Impor model data chunk
from services.models.structured_chunk import StructuredChunk

# Konfigurasi logging
logger = logging.getLogger(__name__)

class StructureAwareProcessor:
    """
    Orkestrator utama untuk pipeline pemrosesan dokumen.
    Kelas ini mengoordinasikan DocumentProcessor (untuk ekstraksi & chunking) dan
    ChunkTransformer (untuk konversi ke format internal) untuk mengubah file mentah 
    menjadi daftar StructuredChunk yang siap digunakan.
    """
    def __init__(self):
        """
        Inisialisasi prosesor dengan membuat instance dari sub-komponennya.
        """
        self.doc_processor = DocumentProcessor()
        # [DIUBAH] Menggunakan ChunkTransformer yang baru
        self.chunk_transformer = ChunkTransformer()
        self.logger = logging.getLogger(__name__)
        self.logger.info("StructureAwareProcessor siap dengan pipeline baru (DocumentProcessor -> ChunkTransformer).")

    def process_document(self, contents: str, filename: str) -> Tuple[List[StructuredChunk], Optional[str]]:
        """
        [DIUBAH] Memproses dokumen secara keseluruhan, dari konten base64 hingga menjadi chunk terstruktur.

        Args:
            contents (str): String base64 dari konten file.
            filename (str): Nama file yang diunggah.

        Returns:
            Tuple[List[StructuredChunk], Optional[str]]:
            - List[StructuredChunk]: Daftar chunk terstruktur hasil pemrosesan.
            - Optional[str]: Blob PDF dalam format base64 untuk rendering (jika file adalah PDF).
        """
        self.logger.info(f"Memulai pipeline pemrosesan untuk: {filename}")

        try:
            # --- Langkah 1: Ekstraksi & Chunking dengan DocumentProcessor ---
            # Mengubah file mentah menjadi daftar elemen yang sudah di-chunk oleh unstructured.
            self.logger.info("Langkah 1: Mengekstrak dan membuat chunk dokumen dengan DocumentProcessor...")
            
            # [DIUBAH] Menangani return value baru dari doc_processor (hanya 2 item)
            unstructured_chunks, pdf_blob = self.doc_processor.process_document(contents, filename)

            if not unstructured_chunks:
                self.logger.warning(f"Tidak ada chunk yang dapat diekstrak dari {filename}.")
                return [], pdf_blob

            self.logger.info(f"Langkah 1 Selesai: {len(unstructured_chunks)} chunk berhasil dibuat oleh DocumentProcessor.")

            # --- Langkah 2: Transformasi dengan ChunkTransformer ---
            # Mengubah daftar elemen chunk menjadi format StructuredChunk internal kita.
            self.logger.info("Langkah 2: Mengubah chunk menjadi format StructuredChunk...")
            
            # [DIUBAH] Memanggil metode yang benar dari ChunkTransformer
            structured_chunks = self.chunk_transformer.transform_elements(unstructured_chunks, filename)
            
            self.logger.info(f"Langkah 2 Selesai: {len(structured_chunks)} StructuredChunk berhasil ditransformasi.")

            # --- Langkah 3: Finalisasi ---
            # Mengembalikan hasil akhir yang siap untuk di-embed dan disimpan.
            self.logger.info(f"Pipeline pemrosesan untuk {filename} selesai dengan sukses.")
            return structured_chunks, pdf_blob

        except Exception as e:
            self.logger.error(f"Terjadi kesalahan dalam pipeline StructureAwareProcessor untuk file {filename}: {e}", exc_info=True)
            # Mengembalikan list kosong jika terjadi error agar aplikasi tidak crash
            return [], None
