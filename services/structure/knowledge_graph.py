import networkx as nx
import json
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
import numpy as np
from pathlib import Path

from ..models.structured_chunk import StructuredChunk

class DocumentKnowledgeGraph:
    """
    Manages a knowledge graph of document chunks and their relationships.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        # Initialize main graph
        self.graph = nx.DiGraph()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Storage configuration
        self.storage_path = storage_path
        
        # Edge types and weights
        self.edge_types = {
            'hierarchical': 1.0,    # Section hierarchy relationships
            'reference': 0.8,       # Explicit references between chunks
            'semantic': 0.6,        # Semantic similarity relationships
            'adjacent': 0.4,        # Adjacent chunks in document
            'keyword': 0.3,         # Shared keyword relationships
        }
        
        # Cache for computed similarities
        self.similarity_cache = {}
        
        # Statistics tracking
        self.stats = defaultdict(int)

    def add_document(self, chunks: List[StructuredChunk], document_id: str) -> None:
        """
        Add a document's chunks to the knowledge graph.
        
        Args:
            chunks (List[StructuredChunk]): List of document chunks
            document_id (str): Unique identifier for the document
        """
        try:
            self.logger.info(f"Adding document {document_id} with {len(chunks)} chunks")
            
            # Add document node
            self.graph.add_node(document_id, 
                              type='document',
                              timestamp=datetime.now().isoformat())
            
            # Process chunks
            for chunk in chunks:
                self.add_chunk(chunk, document_id)
            
            # Build relationships between chunks
            self._build_chunk_relationships(chunks)
            
            # Update statistics
            self.stats['documents'] += 1
            self.stats['chunks'] += len(chunks)
            
            self.logger.info(f"Successfully added document {document_id} to knowledge graph")
            
        except Exception as e:
            self.logger.error(f"Error adding document {document_id}: {str(e)}")
            raise

    def add_chunk(self, chunk: StructuredChunk, document_id: str) -> None:
        """
        Add a single chunk to the knowledge graph.
        
        Args:
            chunk (StructuredChunk): Chunk to add
            document_id (str): ID of the document this chunk belongs to
        """
        # Add chunk node
        self.graph.add_node(
            chunk.id,
            type='chunk',
            text=chunk.text,
            semantic_type=chunk.semantic_type,
            section_title=chunk.section_title,
            metadata=chunk.metadata,
            document_id=document_id
        )
        
        # Add hierarchical relationships
        if chunk.hierarchical_path:
            prev_node = document_id
            for section in chunk.hierarchical_path:
                section_id = f"{document_id}_{section}"
                
                # Add section node if it doesn't exist
                if not self.graph.has_node(section_id):
                    self.graph.add_node(
                        section_id,
                        type='section',
                        title=section,
                        document_id=document_id
                    )
                
                # Add hierarchical edge
                self.graph.add_edge(
                    prev_node,
                    section_id,
                    type='hierarchical',
                    weight=self.edge_types['hierarchical']
                )
                prev_node = section_id
            
            # Connect chunk to its section
            self.graph.add_edge(
                prev_node,
                chunk.id,
                type='hierarchical',
                weight=self.edge_types['hierarchical']
            )
        
        # Add document-chunk edge
        self.graph.add_edge(
            document_id,
            chunk.id,
            type='contains',
            weight=1.0
        )

    def _build_chunk_relationships(self, chunks: List[StructuredChunk]) -> None:
        """
        Build relationships between chunks based on various criteria.
        
        Args:
            chunks (List[StructuredChunk]): List of chunks to process
        """
        # Build indices for efficient lookup
        section_chunks = defaultdict(list)
        keyword_chunks = defaultdict(set)
        
        for chunk in chunks:
            section_chunks[chunk.section_title].append(chunk)
            for keyword in chunk.keywords:
                keyword_chunks[keyword].add(chunk.id)
        
        # Process each chunk
        for chunk in chunks:
            # Add reference relationships
            for ref_id in chunk.references:
                if self.graph.has_node(ref_id):
                    self.graph.add_edge(
                        chunk.id,
                        ref_id,
                        type='reference',
                        weight=self.edge_types['reference']
                    )
            
            # Add adjacent chunk relationships
            same_section = section_chunks[chunk.section_title]
            chunk_idx = same_section.index(chunk)
            
            if chunk_idx > 0:
                prev_chunk = same_section[chunk_idx - 1]
                self.graph.add_edge(
                    chunk.id,
                    prev_chunk.id,
                    type='adjacent',
                    weight=self.edge_types['adjacent']
                )
            
            if chunk_idx < len(same_section) - 1:
                next_chunk = same_section[chunk_idx + 1]
                self.graph.add_edge(
                    chunk.id,
                    next_chunk.id,
                    type='adjacent',
                    weight=self.edge_types['adjacent']
                )
            
            # Add keyword-based relationships
            for keyword in chunk.keywords:
                related_chunks = keyword_chunks[keyword]
                for related_id in related_chunks:
                    if related_id != chunk.id:
                        self.graph.add_edge(
                            chunk.id,
                            related_id,
                            type='keyword',
                            weight=self.edge_types['keyword']
                        )

    def get_relevant_chunks(self, query_chunk: StructuredChunk, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get the most relevant chunks for a query chunk.
        
        Args:
            query_chunk (StructuredChunk): Query chunk
            k (int): Number of chunks to return
            
        Returns:
            List[Tuple[str, float]]: List of (chunk_id, relevance_score) tuples
        """
        relevance_scores = {}
        
        # Calculate relevance scores using multiple factors
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'chunk':
                # Combine multiple relevance factors
                score = self._calculate_relevance_score(query_chunk, node)
                if score > 0:
                    relevance_scores[node] = score
        
        # Sort by score and return top k
        sorted_chunks = sorted(relevance_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        return sorted_chunks[:k]

    def _calculate_relevance_score(self, query_chunk: StructuredChunk, 
                                 target_id: str) -> float:
        """
        Calculate relevance score between query chunk and target chunk.
        
        Args:
            query_chunk (StructuredChunk): Query chunk
            target_id (str): ID of target chunk
            
        Returns:
            float: Relevance score
        """
        if not self.graph.has_node(target_id):
            return 0.0
        
        target_data = self.graph.nodes[target_id]
        
        # Initialize score components
        semantic_score = 0.0
        structural_score = 0.0
        path_score = 0.0
        
        # Semantic similarity (cached)
        cache_key = (query_chunk.id, target_id)
        if cache_key in self.similarity_cache:
            semantic_score = self.similarity_cache[cache_key]
        else:
            semantic_score = self._calculate_semantic_similarity(
                query_chunk.text,
                target_data.get('text', '')
            )
            self.similarity_cache[cache_key] = semantic_score
        
        # Structural similarity
        structural_score = self._calculate_structural_similarity(
            query_chunk,
            target_id
        )
        
        # Path-based similarity
        path_score = self._calculate_path_score(query_chunk.id, target_id)
        
        # Combine scores with weights
        final_score = (
            0.5 * semantic_score +
            0.3 * structural_score +
            0.2 * path_score
        )
        
        return final_score

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score
        """
        # Simple word overlap similarity for demonstration
        # In practice, you might want to use more sophisticated methods
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

    def _calculate_structural_similarity(self, chunk: StructuredChunk, 
                                      target_id: str) -> float:
        """
        Calculate structural similarity based on document hierarchy.
        
        Args:
            chunk (StructuredChunk): Source chunk
            target_id (str): ID of target chunk
            
        Returns:
            float: Structural similarity score
        """
        target_data = self.graph.nodes[target_id]
        
        # Same section bonus
        if chunk.section_title == target_data.get('section_title'):
            return 1.0
        
        # Calculate path overlap in hierarchy
        chunk_path = chunk.hierarchical_path
        target_path = self._get_hierarchical_path(target_id)
        
        if not chunk_path or not target_path:
            return 0.0
        
        # Find common prefix length
        common_length = 0
        for a, b in zip(chunk_path, target_path):
            if a != b:
                break
            common_length += 1
        
        max_length = max(len(chunk_path), len(target_path))
        return common_length / max_length if max_length > 0 else 0.0

    def _calculate_path_score(self, source_id: str, target_id: str) -> float:
        """
        Calculate similarity based on graph path properties.
        
        Args:
            source_id (str): Source node ID
            target_id (str): Target node ID
            
        Returns:
            float: Path-based similarity score
        """
        try:
            # Get shortest path length
            path_length = nx.shortest_path_length(
                self.graph,
                source_id,
                target_id,
                weight='weight'
            )
            
            # Convert path length to similarity score
            return 1.0 / (1.0 + path_length)
            
        except nx.NetworkXNoPath:
            return 0.0

    def _get_hierarchical_path(self, chunk_id: str) -> List[str]:
        """
        Get hierarchical path for a chunk from the graph.
        
        Args:
            chunk_id (str): Chunk ID
            
        Returns:
            List[str]: Hierarchical path
        """
        path = []
        current = chunk_id
        
        while True:
            # Get incoming hierarchical edges
            predecessors = [
                (pred, self.graph.edges[pred, current])
                for pred in self.graph.predecessors(current)
                if self.graph.edges[pred, current].get('type') == 'hierarchical'
            ]
            
            if not predecessors:
                break
                
            # Move up the hierarchy
            current = predecessors[0][0]
            if self.graph.nodes[current].get('type') == 'section':
                path.append(self.graph.nodes[current].get('title'))
            
        return list(reversed(path))

    def save_graph(self, filepath: Optional[Path] = None) -> None:
        """
        Save the knowledge graph to disk.
        
        Args:
            filepath (Optional[Path]): Path to save the graph
        """
        save_path = filepath or self.storage_path
        if not save_path:
            raise ValueError("No storage path specified")
            
        try:
            # Convert graph to dictionary format
            graph_data = {
                'nodes': dict(self.graph.nodes(data=True)),
                'edges': dict(self.graph.edges(data=True)),
                'stats': dict(self.stats),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'num_nodes': self.graph.number_of_nodes(),
                    'num_edges': self.graph.number_of_edges()
                }
            }
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
            self.logger.info(f"Successfully saved graph to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving graph: {str(e)}")
            raise

    def load_graph(self, filepath: Optional[Path] = None) -> None:
        """
        Load the knowledge graph from disk.
        
        Args:
            filepath (Optional[Path]): Path to load the graph from
        """
        load_path = filepath or self.storage_path
        if not load_path:
            raise ValueError("No storage path specified")
            
        try:
            # Load from file
            with open(load_path, 'r') as f:
                graph_data = json.load(f)
            
            # Create new graph
            new_graph = nx.DiGraph()
            
            # Add nodes
            for node, data in graph_data['nodes'].items():
                new_graph.add_node(node, **data)
            
            # Add edges
            for source, target, data in graph_data['edges'].items():
                new_graph.add_edge(source, target, **data)
            
            # Update instance variables
            self.graph = new_graph
            self.stats.update(graph_data['stats'])
            
            self.logger.info(f"Successfully loaded graph from {load_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading graph: {str(e)}")
            raise

    def get_graph_statistics(self) -> Dict[str, Any]:
            """
            Get statistics about the knowledge graph.
            
            Returns:
                Dict[str, Any]: Dictionary of graph statistics
            """
            stats = {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'node_types': defaultdict(int),
                'edge_types': defaultdict(int),
                'avg_degree': float(self.graph.number_of_edges()) / max(1, self.graph.number_of_nodes()),
                'density': nx.density(self.graph),
                'strongly_connected_components': nx.number_strongly_connected_components(self.graph),
                'documents': self.stats['documents'],
                'chunks': self.stats['chunks']
            }
            
            # Count node types
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                stats['node_types'][node_type] += 1
            
            # Count edge types
            for _, _, data in self.graph.edges(data=True):
                edge_type = data.get('type', 'unknown')
                stats['edge_types'][edge_type] += 1
            
            # Calculate average path length (if graph is connected)
            try:
                stats['avg_path_length'] = nx.average_shortest_path_length(self.graph)
            except nx.NetworkXError:
                stats['avg_path_length'] = None
            
            return dict(stats)  # Convert defaultdict to regular dict

    def get_subgraph_for_chunks(self, chunk_ids: List[str], 
                            radius: int = 2) -> nx.DiGraph:
        """
        Get a subgraph containing the specified chunks and their neighborhood.
        
        Args:
            chunk_ids (List[str]): List of chunk IDs
            radius (int): Number of hops to include around each chunk
            
        Returns:
            nx.DiGraph: Subgraph containing specified chunks and their context
        """
        # Get nodes within radius of each chunk
        nodes_to_include = set()
        for chunk_id in chunk_ids:
            if not self.graph.has_node(chunk_id):
                continue
                
            # Get nodes within radius
            for dist in range(radius + 1):
                # Forward neighbors
                forward = nx.single_source_shortest_path_length(
                    self.graph, chunk_id, cutoff=dist)
                nodes_to_include.update(forward.keys())
                
                # Backward neighbors
                backward = nx.single_source_shortest_path_length(
                    self.graph.reverse(), chunk_id, cutoff=dist)
                nodes_to_include.update(backward.keys())
        
        # Create subgraph
        return self.graph.subgraph(nodes_to_include).copy()

    def get_chunk_context(self, chunk_id: str) -> Dict[str, Any]:
        """
        Get contextual information about a chunk.
        
        Args:
            chunk_id (str): ID of the chunk
            
        Returns:
            Dict[str, Any]: Dictionary containing chunk context
        """
        if not self.graph.has_node(chunk_id):
            raise ValueError(f"Chunk {chunk_id} not found in graph")
        
        context = {
            'chunk_data': dict(self.graph.nodes[chunk_id]),
            'document_id': None,
            'section_path': self._get_hierarchical_path(chunk_id),
            'references': [],
            'referenced_by': [],
            'related_chunks': [],
            'previous_chunk': None,
            'next_chunk': None
        }
        
        # Get document ID
        for pred in self.graph.predecessors(chunk_id):
            if self.graph.nodes[pred].get('type') == 'document':
                context['document_id'] = pred
                break
        
        # Get references
        for _, target, data in self.graph.out_edges(chunk_id, data=True):
            if data.get('type') == 'reference':
                context['references'].append({
                    'chunk_id': target,
                    'text': self.graph.nodes[target].get('text', '')[:100]  # Preview
                })
        
        # Get referenced by
        for source, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get('type') == 'reference':
                context['referenced_by'].append({
                    'chunk_id': source,
                    'text': self.graph.nodes[source].get('text', '')[:100]
                })
        
        # Get related chunks
        for _, target, data in self.graph.out_edges(chunk_id, data=True):
            if data.get('type') in ['semantic', 'keyword']:
                context['related_chunks'].append({
                    'chunk_id': target,
                    'text': self.graph.nodes[target].get('text', '')[:100],
                    'relation_type': data.get('type'),
                    'weight': data.get('weight', 0.0)
                })
        
        # Get adjacent chunks
        for _, target, data in self.graph.out_edges(chunk_id, data=True):
            if data.get('type') == 'adjacent':
                context['next_chunk'] = {
                    'chunk_id': target,
                    'text': self.graph.nodes[target].get('text', '')[:100]
                }
                break
        
        for source, _, data in self.graph.in_edges(chunk_id, data=True):
            if data.get('type') == 'adjacent':
                context['previous_chunk'] = {
                    'chunk_id': source,
                    'text': self.graph.nodes[source].get('text', '')[:100]
                }
                break
        
        return context

    def find_paths_between_chunks(self, source_id: str, target_id: str, 
                                max_paths: int = 3) -> List[List[str]]:
        """
        Find multiple paths between two chunks.
        
        Args:
            source_id (str): Source chunk ID
            target_id (str): Target chunk ID
            max_paths (int): Maximum number of paths to return
            
        Returns:
            List[List[str]]: List of paths (each path is a list of node IDs)
        """
        try:
            # Get all simple paths between chunks
            all_paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=5  # Limit path length
            ))
            
            # Sort paths by length and weight
            weighted_paths = []
            for path in all_paths:
                # Calculate path weight
                path_weight = sum(
                    self.graph.edges[path[i], path[i+1]].get('weight', 0.0)
                    for i in range(len(path)-1)
                )
                weighted_paths.append((path, path_weight))
            
            # Sort by path weight (higher is better) and length (shorter is better)
            weighted_paths.sort(key=lambda x: (-x[1], len(x[0])))
            
            return [path for path, _ in weighted_paths[:max_paths]]
            
        except nx.NetworkXNoPath:
            return []

    def cleanup(self) -> None:
        """
        Perform cleanup operations on the graph.
        """
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
        
        # Remove duplicate edges
        duplicate_edges = []
        seen_edges = set()
        
        for source, target, data in self.graph.edges(data=True):
            edge_key = (source, target, data.get('type'))
            if edge_key in seen_edges:
                duplicate_edges.append((source, target))
            else:
                seen_edges.add(edge_key)
        
        self.graph.remove_edges_from(duplicate_edges)
        
        # Clear similarity cache
        self.similarity_cache.clear()
        
        self.logger.info(f"Cleaned up {len(isolated_nodes)} isolated nodes and {len(duplicate_edges)} duplicate edges")