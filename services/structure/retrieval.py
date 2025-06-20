import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict

from ..models.structured_chunk import StructuredChunk
from .knowledge_graph import DocumentKnowledgeGraph
from .semantic_types import SemanticTypeDetector, SemanticType

@dataclass
class RetrievalResult:
    """Data class for retrieval results."""
    chunk_id: str
    text: str
    score: float
    semantic_type: SemanticType
    section_path: List[str]
    document_id: str
    metadata: Dict
    reference_chunks: List[str] = None
    context_chunks: List[str] = None

class RetrievalStrategy(Enum):
    """Enumeration of retrieval strategies."""
    SEMANTIC = "semantic"
    GRAPH = "graph"
    HYBRID = "hybrid"
    STRUCTURE = "structure"

class StructureAwareRetrieval:
    """
    Implements structure-aware retrieval combining semantic search, 
    knowledge graph traversal, and structural context.
    """
    
    def __init__(self, knowledge_graph: DocumentKnowledgeGraph,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize components
        self.knowledge_graph = knowledge_graph
        self.semantic_detector = SemanticTypeDetector()
        self.encoder = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Retrieval parameters
        self.params = {
            'semantic_weight': 0.4,
            'graph_weight': 0.3,
            'structure_weight': 0.3,
            'context_window': 2,
            'min_score': 0.3
        }
        
        # Cache for embeddings
        self.embedding_cache = {}

    def retrieve_chunks(self, query: str, k: int = 5, 
                       strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
                       filters: Optional[Dict] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using specified strategy.
        
        Args:
            query (str): Query text
            k (int): Number of chunks to retrieve
            strategy (RetrievalStrategy): Retrieval strategy to use
            filters (Optional[Dict]): Filters to apply to results
            
        Returns:
            List[RetrievalResult]: Ranked list of retrieval results
        """
        try:
            self.logger.info(f"Starting retrieval for query: {query[:100]}...")
            
            # Detect query type and characteristics
            query_type = self._analyze_query(query)
            
            # Select retrieval method based on strategy and query type
            if strategy == RetrievalStrategy.SEMANTIC:
                results = self._semantic_search(query, k * 2)
            elif strategy == RetrievalStrategy.GRAPH:
                results = self._graph_based_retrieval(query, k * 2)
            elif strategy == RetrievalStrategy.STRUCTURE:
                results = self._structure_based_retrieval(query, k * 2)
            else:  # HYBRID
                results = self._hybrid_retrieval(query, k * 2, query_type)
            
            # Apply filters if specified
            if filters:
                results = self._apply_filters(results, filters)
            
            # Add context and post-process
            final_results = self._post_process_results(results, query_type)
            
            # Sort by score and return top k
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in chunk retrieval: {str(e)}")
            raise

    def _analyze_query(self, query: str) -> Dict:
        """
        Analyze query to determine type and characteristics.
        
        Args:
            query (str): Query text
            
        Returns:
            Dict: Query analysis results
        """
        # Parse query
        doc = self.nlp(query)
        
        # Detect semantic type
        semantic_type, confidence = self.semantic_detector.detect_type(query)
        
        # Extract key entities and concepts
        entities = [ent.text for ent in doc.ents]
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Analyze query structure
        query_words = [token.text.lower() for token in doc]
        is_question = any(word in {'what', 'who', 'when', 'where', 'why', 'how'} 
                         for word in query_words)
        
        return {
            'semantic_type': semantic_type,
            'confidence': confidence,
            'entities': entities,
            'key_phrases': key_phrases,
            'is_question': is_question,
            'length': len(query_words)
        }

    def _semantic_search(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform semantic similarity search.
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            List[RetrievalResult]: Search results
        """
        # Encode query
        query_embedding = self.encoder.encode(query)
        
        # Get all chunks from knowledge graph
        chunks = []
        chunk_embeddings = []
        
        for node, data in self.knowledge_graph.graph.nodes(data=True):
            if data.get('type') == 'chunk':
                text = data.get('text', '')
                
                # Get cached embedding or compute new one
                if text in self.embedding_cache:
                    embedding = self.embedding_cache[text]
                else:
                    embedding = self.encoder.encode(text)
                    self.embedding_cache[text] = embedding
                
                chunks.append((node, data))
                chunk_embeddings.append(embedding)
        
        if not chunks:
            return []
        
        # Calculate similarities
        similarities = cosine_similarity(
            [query_embedding],
            chunk_embeddings
        )[0]
        
        # Create results
        results = []
        for (chunk_id, data), score in zip(chunks, similarities):
            if score >= self.params['min_score']:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    text=data.get('text', ''),
                    score=float(score),
                    semantic_type=data.get('semantic_type', SemanticType.GENERAL),
                    section_path=data.get('hierarchical_path', []),
                    document_id=data.get('document_id', ''),
                    metadata=data.get('metadata', {})
                )
                results.append(result)
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _graph_based_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform graph-based retrieval using knowledge graph.
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            List[RetrievalResult]: Retrieved results
        """
        # First get some seed chunks using semantic search
        seed_results = self._semantic_search(query, k // 2)
        
        # Expand through graph
        expanded_chunks = set()
        for result in seed_results:
            # Get connected chunks within 2 hops
            neighbors = nx.single_source_shortest_path_length(
                self.knowledge_graph.graph,
                result.chunk_id,
                cutoff=2
            )
            expanded_chunks.update(neighbors.keys())
        
        # Score expanded chunks
        scored_chunks = []
        for chunk_id in expanded_chunks:
            node_data = self.knowledge_graph.graph.nodes[chunk_id]
            if node_data.get('type') != 'chunk':
                continue
            
            # Calculate graph-based score
            graph_score = self._calculate_graph_score(chunk_id, seed_results)
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                text=node_data.get('text', ''),
                score=graph_score,
                semantic_type=node_data.get('semantic_type', SemanticType.GENERAL),
                section_path=node_data.get('hierarchical_path', []),
                document_id=node_data.get('document_id', ''),
                metadata=node_data.get('metadata', {})
            )
            scored_chunks.append(result)
        
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:k]

    def _structure_based_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Perform structure-aware retrieval focusing on document hierarchy.
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            
        Returns:
            List[RetrievalResult]: Retrieved results
        """
        # Get initial semantic matches
        seed_results = self._semantic_search(query, k)
        
        # Group by sections
        section_chunks = defaultdict(list)
        for result in seed_results:
            if result.section_path:
                section_key = tuple(result.section_path)
                section_chunks[section_key].append(result)
        
        # Get additional chunks from relevant sections
        expanded_results = []
        for section_path, chunks in section_chunks.items():
            # Get all chunks from the section
            section_nodes = [
                (node, data) for node, data in self.knowledge_graph.graph.nodes(data=True)
                if (data.get('type') == 'chunk' and 
                    tuple(data.get('hierarchical_path', [])) == section_path)
            ]
            
            # Score section chunks
            for node_id, data in section_nodes:
                # Calculate combined score
                semantic_score = np.mean([r.score for r in chunks])
                structure_score = self._calculate_structure_score(
                    data.get('hierarchical_path', []),
                    chunks[0].section_path if chunks else []
                )
                
                final_score = (0.7 * semantic_score + 0.3 * structure_score)
                
                if final_score >= self.params['min_score']:
                    result = RetrievalResult(
                        chunk_id=node_id,
                        text=data.get('text', ''),
                        score=final_score,
                        semantic_type=data.get('semantic_type', SemanticType.GENERAL),
                        section_path=data.get('hierarchical_path', []),
                        document_id=data.get('document_id', ''),
                        metadata=data.get('metadata', {})
                    )
                    expanded_results.append(result)
        
        expanded_results.sort(key=lambda x: x.score, reverse=True)
        return expanded_results[:k]

    def _hybrid_retrieval(self, query: str, k: int, 
                         query_analysis: Dict) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval combining multiple strategies.
        
        Args:
            query (str): Query text
            k (int): Number of results to return
            query_analysis (Dict): Query analysis results
            
        Returns:
            List[RetrievalResult]: Retrieved results
        """
        # Get results from each strategy
        semantic_results = self._semantic_search(query, k)
        graph_results = self._graph_based_retrieval(query, k)
        structure_results = self._structure_based_retrieval(query, k)
        
        # Combine results with weights based on query type
        combined_scores = defaultdict(float)
        
        # Adjust weights based on query characteristics
        weights = self._adjust_weights(query_analysis)
        
        # Combine scores
        for result in semantic_results:
            combined_scores[result.chunk_id] += weights['semantic'] * result.score
            
        for result in graph_results:
            combined_scores[result.chunk_id] += weights['graph'] * result.score
            
        for result in structure_results:
            combined_scores[result.chunk_id] += weights['structure'] * result.score
        
        # Create final results
        final_results = []
        for chunk_id, score in combined_scores.items():
            node_data = self.knowledge_graph.graph.nodes[chunk_id]
            if node_data.get('type') != 'chunk':
                continue
                
            result = RetrievalResult(
                chunk_id=chunk_id,
                text=node_data.get('text', ''),
                score=score,
                semantic_type=node_data.get('semantic_type', SemanticType.GENERAL),
                section_path=node_data.get('hierarchical_path', []),
                document_id=node_data.get('document_id', ''),
                metadata=node_data.get('metadata', {})
            )
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]

    def _calculate_graph_score(self, chunk_id: str, 
                             seed_results: List[RetrievalResult]) -> float:
        """
        Calculate graph-based relevance score.
        
        Args:
            chunk_id (str): Chunk ID to score
            seed_results (List[RetrievalResult]): Seed results for context
            
        Returns:
            float: Graph-based score
        """
        score = 0.0
        
        for seed in seed_results:
            try:
                # Calculate shortest path length
                path_length = nx.shortest_path_length(
                    self.knowledge_graph.graph,
                    seed.chunk_id,
                    chunk_id,
                    weight='weight'
                )
                # Convert path length to score (closer is better)
                path_score = 1.0 / (1.0 + path_length)
                score = max(score, path_score * seed.score)
                
            except nx.NetworkXNoPath:
                continue
        
        return score

    def _calculate_structure_score(self, path1: List[str], 
                                 path2: List[str]) -> float:
        """
        Calculate structural similarity between hierarchical paths.
        
        Args:
            path1 (List[str]): First hierarchical path
            path2 (List[str]): Second hierarchical path
            
        Returns:
            float: Structural similarity score
        """
        if not path1 or not path2:
            return 0.0
        
        # Find common prefix length
        common_length = 0
        for a, b in zip(path1, path2):
            if a != b:
                break
            common_length += 1
        
        max_length = max(len(path1), len(path2))
        return