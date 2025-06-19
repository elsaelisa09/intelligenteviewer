__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class StructuredChunk:
    """Data model for structure-aware document chunks."""
    
    id: str
    text: str
    metadata: Dict
    
    # Document structure information
    section_title: Optional[str] = None
    parent_section: Optional[str] = None
    hierarchical_path: List[str] = field(default_factory=list)
    
    # Semantic information
    semantic_type: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Relationships
    references: List[str] = field(default_factory=list)
    referenced_by: List[str] = field(default_factory=list)
    related_chunks: List[str] = field(default_factory=list)
    
    # Document positions and layout
    positions: List[Dict] = field(default_factory=list)
    page_number: Optional[int] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    importance_score: float = 1.0
    chunk_index: int = 0
    
    def to_dict(self) -> Dict:
        """Convert the chunk to a dictionary representation."""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata,
            'section_title': self.section_title,
            'parent_section': self.parent_section,
            'hierarchical_path': self.hierarchical_path,
            'semantic_type': self.semantic_type,
            'topics': self.topics,
            'keywords': self.keywords,
            'references': self.references,
            'referenced_by': self.referenced_by,
            'related_chunks': self.related_chunks,
            'positions': self.positions,
            'page_number': self.page_number,
            'created_at': self.created_at.isoformat(),
            'importance_score': self.importance_score,
            'chunk_index': self.chunk_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuredChunk':
        """Create a StructuredChunk instance from a dictionary."""
        # Convert ISO format string back to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def update_importance_score(self, references_weight: float = 0.3, 
                              position_weight: float = 0.2,
                              semantic_weight: float = 0.5) -> None:
        """Update the chunk's importance score based on various factors."""
        # Reference score based on incoming and outgoing references
        reference_score = len(self.references) + len(self.referenced_by)
        normalized_ref_score = min(reference_score / 10, 1.0)
        
        # Position score (earlier chunks in sections might be more important)
        position_score = 1.0 - (self.chunk_index / 100)  # Normalize to 0-1
        
        # Semantic score based on type
        semantic_scores = {
            'definition': 1.0,
            'procedure': 0.9,
            'example': 0.7,
            'table_reference': 0.8,
            'general': 0.5
        }
        semantic_score = semantic_scores.get(self.semantic_type, 0.5)
        
        # Calculate weighted score
        self.importance_score = (
            references_weight * normalized_ref_score +
            position_weight * position_score +
            semantic_weight * semantic_score
        )
    
    def add_reference(self, chunk_id: str) -> None:
        """Add a reference to another chunk."""
        if chunk_id not in self.references:
            self.references.append(chunk_id)
            self.update_importance_score()
    
    def add_referenced_by(self, chunk_id: str) -> None:
        """Add a chunk that references this chunk."""
        if chunk_id not in self.referenced_by:
            self.referenced_by.append(chunk_id)
            self.update_importance_score()
    
    def add_related_chunk(self, chunk_id: str) -> None:
        """Add a related chunk."""
        if chunk_id not in self.related_chunks:
            self.related_chunks.append(chunk_id)
            
    def get_context(self) -> str:
        """Get the chunk's context string including hierarchical path."""
        path_str = " > ".join(self.hierarchical_path)
        return f"{path_str} | {self.semantic_type or 'general'}"