# services/document_processor.py (Direvisi dengan Ekstraksi & Chunking Semantik)

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.2" # Versi ditingkatkan untuk chunking semantik

# Impor library standar dan pihak ketiga
from typing import Tuple, List, Optional, Dict, Any
import base64
import io
import pandas as pd
import logging
import traceback
import tempfile
import os

# [DIUBAH] Impor yang diperlukan dari unstructured
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Table

# [DIHAPUS] Fitz tidak lagi dibutuhkan untuk proses ingestion utama, hanya untuk highlighting
# import fitz 

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Kelas ini bertanggung jawab untuk memproses dokumen yang diunggah.
    Versi ini menggunakan unstructured.io untuk Ekstraksi DAN Chunking Semantik.
    """
    def __init__(self, max_doc_size: int = 50 * 1024 * 1024):
        self.max_doc_size = max_doc_size
        self.logger = logging.getLogger(__name__)

    def process_document(self, contents: str, filename: str) -> Tuple[List[Element], Optional[str]]:
        """
        [DIUBAH TOTAL] Memproses, mengekstrak, DAN membuat chunk dokumen dari content string.

        Alur kerja:
        1. Dekode konten base64.
        2. Simpan ke file sementara (lebih robust untuk semua strategi).
        3. Gunakan `unstructured.partition` untuk mengekstrak elemen mentah.
        4. Gunakan `unstructured.chunking.chunk_by_title` untuk menggabungkan elemen
           menjadi chunk-chunk yang bermakna secara semantik.
        5. Kembalikan chunk yang sudah jadi dan blob PDF untuk rendering.

        Args:
            contents (str): String base64 dari konten file.
            filename (str): Nama file yang diunggah.

        Returns:
            Tuple[List[Element], Optional[str]]:
            - List[Element]: Daftar CHUNK terstruktur (siap untuk diproses lebih lanjut).
            - Optional[str]: Blob PDF dalam format base64 untuk rendering.
        """
        if not contents:
            return [], None

        try:
            content_type, content_string = contents.split(',', 1)
            decoded_bytes = base64.b64decode(content_string)

            if len(decoded_bytes) > self.max_doc_size:
                raise ValueError("Ukuran file melebihi batas maksimum 50MB")

            # [BARU] Menggunakan file sementara untuk kompatibilitas terbaik
            file_suffix = os.path.splitext(filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp:
                tmp.write(decoded_bytes)
                tmp_path = tmp.name
            
            self.logger.info(f"Memulai partisi untuk {filename}...")
            
            # [DIUBAH] Menggunakan strategi "hi_res" untuk PDF demi kualitas terbaik
            strategy = "hi_res" if filename.lower().endswith('.pdf') else "auto"
            
            elements = partition(filename=tmp_path, strategy=strategy)
            
            self.logger.info(f"Partisi selesai, ditemukan {len(elements)} elemen. Memulai chunking...")
            
            # [BARU] Ini adalah langkah CHUNKING SEMANTIK yang hilang sebelumnya.
            # Mengelompokkan elemen berdasarkan judulnya.
            chunks = chunk_by_title(
                elements,
                max_characters=1000,      # Maksimum karakter per chunk
                new_after_n_chars=800,   # Buat chunk baru jika sudah mendekati maks
                combine_text_under_n_chars=500 # Gabungkan teks pendek
            )
            
            self.logger.info(f"Chunking semantik selesai, menghasilkan {len(chunks)} chunk.")

            pdf_blob = None
            if filename.lower().endswith('.pdf'):
                pdf_blob = f"data:application/pdf;base64,{content_string}"

            # [DIUBAH] Return value sekarang jauh lebih sederhana dan fokus
            return chunks, pdf_blob

        except Exception as e:
            self.logger.error(f"Error di document_processor.py saat memproses {filename}: {str(e)}")
            traceback.print_exc()
            raise Exception(f"Error memproses file '{filename}': {str(e)}")
        finally:
            # [BARU] Selalu pastikan file sementara dihapus setelah selesai
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    # ===================================================================================
    # Metode-metode di bawah ini (untuk highlighting) dipertahankan apa adanya.
    # Nanti, kita bisa mengadaptasinya untuk menggunakan metadata dari `chunks`
    # yang baru (misalnya `chunk.metadata.page_number` dan `chunk.metadata.bbox`).
    # ===================================================================================

    def get_pdf_highlights(self, pdf_content: str, highlight_chunks: List[str], chunk_mapping: Dict, llm_answer: str = None) -> List[Dict]:
        """Dapatkan sorotan untuk ditampilkan di UI. Dipertahankan untuk kompatibilitas."""
        if not highlight_chunks or not chunk_mapping:
            logger.info("Tidak ada chunk atau pemetaan untuk highlighting.")
            return []
        try:
            import fitz 
            import hashlib
            if not pdf_content or not pdf_content.startswith('data:application/pdf;base64,'):
                return []
            content_type, content_string = pdf_content.split(',', 1)
            pdf_bytes = base64.b64decode(content_string)
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            highlights, seen_highlights = [], set()
            first_page_text = doc[0].get_text("text") if doc.page_count > 0 else ""
            doc_source_id = hashlib.md5(first_page_text.encode()).hexdigest()
            for chunk_id in highlight_chunks:
                if chunk_id in chunk_mapping:
                    chunk_data = chunk_mapping[chunk_id]
                    chunk_text = chunk_data.get('text', '').strip()
                    if not chunk_text: continue
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        clean_sentence = ' '.join(chunk_text.split())
                        for rect in page.search_for(clean_sentence):
                            rect_key = f"{page_num}_{rect.x0:.1f}_{rect.y0:.1f}"
                            if rect_key not in seen_highlights:
                                highlights.append({'page': page_num, 'coords': rect.to_dict(), 'text': clean_sentence[:100], 'chunk_id': chunk_id, 'source_id': doc_source_id})
                                seen_highlights.add(rect_key)
            if highlights:
                highlights = self._merge_overlapping_highlights(highlights)
            doc.close()
            return highlights
        except Exception as e:
            logger.error(f"Error di get_pdf_highlights: {str(e)}", exc_info=True)
            return []
            
    def highlight_pdf(self, pdf_content: str, highlights: List[Dict]) -> Optional[str]:
        """Terapkan sorotan ke PDF. Dipertahankan untuk kompatibilitas."""
        if not highlights or not pdf_content: return pdf_content
        try:
            import fitz
            content_type, content_string = pdf_content.split(',', 1)
            pdf_bytes = base64.b64decode(content_string)
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            for hl in highlights:
                if hl['page'] < len(doc):
                    page = doc[hl['page']]
                    rect = fitz.Rect(hl['coords']['x0'], hl['coords']['y0'], hl['coords']['x1'], hl['coords']['y1'])
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0.8, 0)); annot.set_opacity(0.4); annot.update()
            output_buffer = io.BytesIO()
            doc.save(output_buffer)
            doc.close()
            highlighted_content = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            return f"data:application/pdf;base64,{highlighted_content}"
        except Exception as e:
            logger.error(f"Error di highlight_pdf: {str(e)}", exc_info=True)
            return pdf_content

    def _merge_overlapping_highlights(self, highlights: List[Dict]) -> List[Dict]:
        """Gabungkan sorotan yang tumpang tindih. Dipertahankan untuk kompatibilitas."""
        if not highlights: return []
        def overlap(r1, r2): return not (r1['x2'] < r2['x1'] or r1['x1'] > r2['x2'] or r1['y2'] < r2['y1'] or r1['y1'] > r2['y2'])
        def merge_rects(r1, r2): return {'x0': min(r1['x0'], r2['x0']), 'y0': min(r1['y0'], r2['y0']), 'x1': max(r1['x1'], r2['x1']), 'y1': max(r1['y1'], r2['y1'])}
        h_by_page = {}; [h_by_page.setdefault(h['page'], []).append(h) for h in highlights]
        final = []
        for page, hls in h_by_page.items():
            if not hls: continue
            hls.sort(key=lambda x: (x['coords']['y0'], x['coords']['x0']))
            merged = [hls[0]]
            for i in range(1, len(hls)):
                last = merged[-1]; current = hls[i]
                # Menggunakan Rect object untuk merge
                last_rect = fitz.Rect(last['coords'])
                current_rect = fitz.Rect(current['coords'])
                if last_rect.intersects(current_rect):
                    last_rect.include_rect(current_rect)
                    last['coords'] = last_rect.to_dict()
                else:
                    merged.append(current)
            final.extend(merged)
        return final