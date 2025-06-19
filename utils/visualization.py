#utils/visualization.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import pandas as pd
from dash import html
from PIL import Image
import base64
import io
from utils.text_helpers import TextProcessor
from services.text_analysis import TextAnalyzer
from dash import dcc, html, Input, Output, State, ctx
import fitz  # PyMuPDF
import re
import logging

text_analyzer = TextAnalyzer()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_system_notification(message, type="success", action=None):
    """Create a system notification message component
    
    Args:
        message (str): The notification message
        type (str): The type of notification ('success', 'error', 'info')
        action (str): The type of action ('add', 'remove', 'welcome', 'load', or None)
    """
    # Determine icon based on action and type
    icon_class = ""
    icon_style = {}
    
    if action == "remove":
        icon_class = "fas fa-times"
        icon_style = {"color": "#dc3545"}  # Red color for remove
    elif action == "add":
        icon_class = "fas fa-check-circle"
        icon_style = {"color": "#28a745"}  # Green color for add
    elif action == "welcome":
        icon_class = "fas fa-comments"
        icon_style = {"color": "#17a2b8"}  # Blue color for welcome messages
    elif action == "load":
        icon_class = "fas fa-sync"
        icon_style = {"color": "#6610f2"}  # Purple color for load
    else:
        # Default icons based on type
        if type == "success":
            icon_class = "fas fa-check-circle"
            icon_style = {"color": "#28a745"}  # Green for success
        elif type == "error":
            icon_class = "fas fa-exclamation-circle"
            icon_style = {"color": "#dc3545"}  # Red for error
        elif type == "info":
            icon_class = "fas fa-info-circle"
            icon_style = {"color": "#17a2b8"}  # Blue for info
        else:
            icon_class = "fas fa-exclamation-circle"
            icon_style = {}

    if action == "welcome":
        from dash import html
        return html.Div(
            html.Div([
                html.I(
                    className=f"{icon_class} me-2",
                    style=icon_style
                ),
                message
            ], className="p-3 border rounded bg-light"),
            className=f"system-notification welcome-notification mb-3"
        )
    else:
        # Default small notification style
        from dash import html
        return html.Div(
            html.Small([
                html.I(
                    className=f"{icon_class} me-2",
                    style=icon_style
                ),
                message
            ]),
            className=f"system-notification {type}"
        )

def highlight_pdf(input_pdf_path, output_pdf_path, phrases):
    """
    Highlights phrases/words in a PDF and saves the modified PDF.

    Parameters:
        input_pdf_path (str): Path to the original PDF.
        output_pdf_path (str): Path to save the highlighted PDF.
        phrases (list): List of phrases/words to highlight.
    """
    # Open the PDF
    pdf_document = fitz.open(input_pdf_path)

    # Compile the phrases into a regex pattern for case-insensitive matching
    phrase_pattern = re.compile('|'.join(re.escape(p) for p in phrases), re.IGNORECASE)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text_instances = []

        # Search for phrases on the page
        for match in phrase_pattern.finditer(page.get_text()):
            # Find positions of each match
            text_instances.extend(page.search_for(match.group()))

        # Add highlight annotations for each match
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()  # Apply the highlight

    # Save the modified PDF
    pdf_document.save(output_pdf_path, deflate=True)
    pdf_document.close()

def should_highlight(text, chunk_mapping, highlighted_chunk_ids, assistant_reply):
    """
    Enhanced version of highlighting function with improved semantic similarity detection
    and better text matching techniques.
    """
    if not highlighted_chunk_ids or not assistant_reply:
        return False

    text_normalized = text_analyzer._normalize_text(text)
    assistant_normalized = text_analyzer._normalize_text(assistant_reply)
    
    # Calculate multiple similarity metrics
    similarity_score = text_analyzer.calculate_semantic_similarity(text_normalized, assistant_normalized)
    has_overlap = text_analyzer.has_significant_overlap(text_normalized, assistant_normalized)
    
    # Dynamic thresholding based on text characteristics
    base_threshold = 0.3
    
    # Adjust threshold based on text length
    text_length = len(text_normalized.split())
    if text_length < 30:
        threshold = base_threshold * 1.2  # Higher threshold for very short texts
    elif text_length > 200:
        threshold = base_threshold * 0.8  # Lower threshold for long texts
    else:
        threshold = base_threshold
        
    # Boost score if there's significant overlap
    if has_overlap:
        similarity_score *= 1.25
        
    # Debug information
    logger.debug(f"visualization.py - should_highlight : \nHighlighting Analysis:")
    logger.debug(f"visualization.py - should_highlight : Text length: {text_length} words")
    logger.debug(f"visualization.py - should_highlight : Similarity score: {similarity_score:.3f}")
    logger.debug(f"visualization.py - should_highlight : Has significant overlap: {has_overlap}")
    logger.debug(f"visualization.py - should_highlight : Applied threshold: {threshold:.3f}")
    
    # Return True if either similarity score exceeds threshold or significant overlap is found
    should_highlight = similarity_score > threshold or has_overlap
    logger.debug(f"visualization.py - should_highlight : Highlighting decision: {'✓' if should_highlight else '✗'}")
    
    return should_highlight

def format_text_block(text_block):
    """
    Format text block with proper styling based on extracted properties
    """
    style = {
        "margin": "0",
        "padding": "2px 0",
        "lineHeight": "1.5"
    }
    
    # Add styling based on text properties
    if text_block.get("is_bold"):
        style["fontWeight"] = "bold"
    if text_block.get("is_italic"):
        style["fontStyle"] = "italic"
    
    # Scale font size relative to base size
    base_size = 16  # Base font size in pixels
    font_scale = text_block.get("font_size", base_size) / base_size
    style["fontSize"] = f"{max(11, min(24, base_size * font_scale))}px"
    
    return html.Span(text_block["text"], style=style)

class DocumentVisualizer:
    @staticmethod
    def format_text_block(text_block):
        """Format text block with proper styling based on extracted properties"""
        style = {
            "margin": "0",
            "padding": "2px 0",
            "lineHeight": "1.5"
        }
        
        if text_block.get("is_bold"):
            style["fontWeight"] = "bold"
        if text_block.get("is_italic"):
            style["fontStyle"] = "italic"
        
        base_size = 16
        font_scale = text_block.get("font_size", base_size) / base_size
        style["fontSize"] = f"{max(11, min(24, base_size * font_scale))}px"
        
        return html.Span(text_block["text"], style=style)

    @staticmethod
    def table_to_html(df):
        """Convert DataFrame to a properly rendered Dash table"""
        if df is None or df.empty:
            return None
        
        try:
            df = df.fillna('')
            df = df.applymap(lambda x: str(x).strip())
            
            header = [
                html.Th(col, style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '12px',
                    'border': '1px solid #dee2e6',
                    'textAlign': 'left'
                }) for col in df.columns
            ]
            header_row = [html.Tr(header)]
            
            rows = []
            for idx, row in df.iterrows():
                cells = [
                    html.Td(cell, style={
                        'padding': '8px',
                        'border': '1px solid #dee2e6',
                        'textAlign': 'left'
                    }) for cell in row
                ]
                rows.append(html.Tr(cells))
            
            table = html.Div([
                html.Table(
                    header_row + rows,
                    style={
                        'width': '100%',
                        'borderCollapse': 'collapse',
                        'marginBottom': '1rem',
                        'backgroundColor': 'white'
                    }
                )
            ], style={
                'overflowX': 'auto',
                'marginTop': '20px',
                'marginBottom': '20px'
            })
            
            return table
            
        except Exception as e:
            logger.error(f"visualization.py - DocumentVisualizer - table_to_html : Error converting table to HTML: {e}")
            return None

    @staticmethod
    def create_highlighted_content(content, chunk_mapping, highlighted_chunk_ids, assistant_reply, images=None, tables=None):
        """Modified content creator with improved scrolling to highlighted content"""
        content_container = []
        most_relevant_id = None
        highest_similarity = 0
        TextAn = TextAnalyzer()
        if isinstance(content, list):  # PDF content with layout information
            for page_num, page_content in enumerate(content):
                page_container = []
                current_line = []
                current_y = None
                
                phrases = []
                for line in page_content:
                    for span in line:
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                            
                        bbox = span.get("bbox", None)
                        y_pos = bbox[1] if bbox else None
                        
                        if current_y is not None and y_pos is not None and abs(y_pos - current_y) > 5:
                            if current_line:
                                line_text = " ".join(span["text"] for span in current_line)
                                should_highlight_line = should_highlight(
                                    line_text,
                                    chunk_mapping,
                                    highlighted_chunk_ids,
                                    assistant_reply
                                )
                                
                                # Calculate similarity score for auto-scrolling
                                if should_highlight_line:
                                    similarity = text_analyzer.calculate_semantic_similarity(line_text, assistant_reply)
                                    if similarity > highest_similarity:
                                        highest_similarity = similarity
                                        most_relevant_id = f"highlight-{len(content_container)}"
                                
                                current_line = []
                        
                        current_line.append(span)
                        current_y = y_pos
                
                # Process last line
                if current_line:
                    line_text = " ".join(span["text"] for span in current_line)
                    should_highlight_line = should_highlight(
                        line_text,
                        chunk_mapping,
                        highlighted_chunk_ids,
                        assistant_reply
                    )
                    
                    if should_highlight_line:
                        similarity = text_analyzer.calculate_semantic_similarity(line_text, assistant_reply)
                        if similarity > highest_similarity:
                            highest_similarity = similarity
                            most_relevant_id = f"highlight-{len(content_container)}"
                    
                    line_container = []
                    for span_data in current_line:
                        formatted_span = format_text_block(span_data)
                        if should_highlight_line:
                            line_container.append(
                                html.Div(
                                    formatted_span,
                                    id=f"highlight-{len(content_container)}",
                                    style={
                                        "backgroundColor": "#fff3cd",
                                        "padding": "2px 4px",
                                        "borderRadius": "2px",
                                        "border": "1px solid #ffeeba"
                                    },
                                    className="highlighted-text"
                                )
                            )
                        else:
                            line_container.append(formatted_span)
                    
                    page_container.extend([
                        html.Span(line_container, style={"display": "block"}),
                        html.Br()
                    ])
                
                content_container.append(
                    html.Div(
                        page_container,
                        style={
                            "margin": "20px 0",
                            "padding": "20px",
                            "backgroundColor": "white",
                            "borderRadius": "4px",
                            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"
                        }
                    )
                )
                
                
        else:
            # Handle non-PDF content with similar highlighting and ID assignment...
            text_proc = TextProcessor()
            processed_content = text_proc.process_content(content)
            
            for section in processed_content:
                if section["type"] == "heading":
                    content_container.append(
                        html.H3(
                            section["text"].lstrip('#').strip(),
                            style={
                                'marginTop': '2rem',
                                'marginBottom': '1rem',
                                'fontWeight': 'bold'
                            }
                        )
                    )
                else:
                    text = section["text"].strip()
                    if text:
                        should_highlight_result = should_highlight(
                            text,
                            chunk_mapping,
                            highlighted_chunk_ids,
                            assistant_reply
                        )
                        
                        if should_highlight_result:
                            similarity = text_analyzer.calculate_semantic_similarity(text, assistant_reply)
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                most_relevant_id = f"highlight-{len(content_container)}"
                        
                        style = {
                            'marginBottom': '1.5rem',
                            'lineHeight': '1.6',
                            'padding': '0.75rem',
                        }
                        
                        if should_highlight_result:
                            style.update({
                                'backgroundColor': '#fff3cd',
                                'borderRadius': '4px',
                                'border': '2px solid #ffeeba',
                                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                            })
                            content_container.append(
                                html.Div(
                                    text,
                                    id=f"highlight-{len(content_container)}",
                                    style=style,
                                    className="highlighted-text"
                                )
                            )
                        else:
                            content_container.append(html.Div(text, style=style))
        
        # Add a hidden input to store the most relevant highlight ID
        content_container.append(
            dcc.Input(
                id='most-relevant-highlight',
                type='hidden',
                value=most_relevant_id
            )
        )
        
        return html.Div(
            content_container,
            style={
                "padding": "20px",
                "backgroundColor": "#f8f9fa"
            }
        )

    @staticmethod
    def _create_page_container(page_content, chunk_mapping, highlighted_chunk_ids, assistant_reply, content_container, update_tracking_fn):
        """Helper method to create page container for PDF content"""
        page_container = []
        current_line = []
        current_y = None
        
        for line in page_content:
            DocumentVisualizer._process_line(
                line,
                current_line,
                current_y,
                page_container,
                chunk_mapping,
                highlighted_chunk_ids,
                assistant_reply,
                content_container,
                update_tracking_fn
            )
        
        return html.Div(
            page_container,
            style={
                "margin": "20px 0",
                "padding": "20px",
                "backgroundColor": "white",
                "borderRadius": "4px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"
            }
        )

    @staticmethod
    def _create_regular_content(content, chunk_mapping, highlighted_chunk_ids, assistant_reply, update_tracking_fn):
        """Helper method to create regular content container"""
        text_proc = TextProcessor()
        regular_container = []
        processed_content = text_proc.process_content(content)
        
        for section in processed_content:
            if section["type"] == "heading":
                regular_container.append(
                    html.H3(
                        section["text"].lstrip('#').strip(),
                        style={
                            'marginTop': '2rem',
                            'marginBottom': '1rem',
                            'fontWeight': 'bold'
                        }
                    )
                )
            else:
                DocumentVisualizer._process_section(
                    section,
                    regular_container,
                    chunk_mapping,
                    highlighted_chunk_ids,
                    assistant_reply,
                    update_tracking_fn
                )
        
        return regular_container

    