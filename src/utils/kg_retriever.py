import networkx as nx
from sentence_transformers import SentenceTransformer
import torch
import os

def load_knowledge_graph(graph_path):
    """
    Load the knowledge graph from a GraphML file.
    
    Args:
        graph_path: Path to GraphML file
        
    Returns:
        networkx.Graph: Loaded graph
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Knowledge graph file not found: {graph_path}")
    
    try:
        G = nx.read_graphml(graph_path)
        print(f"Knowledge graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return G
    except Exception as e:
        print(f"Error loading knowledge graph: {e}")
        # Return an empty graph as fallback
        return nx.Graph()

def setup_retrieval_model(graph):
    """
    Set up the retrieval model and encode graph nodes.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        tuple: (retrieval_model, entity_embeddings)
    """
    # Load a retrieval model for RAG
    try:
        retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        # Create a simple fallback
        from sentence_transformers import models, SentenceTransformer
        word_embedding_model = models.Transformer('distilbert-base-uncased')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        retrieval_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Convert KG nodes to embeddings
    entity_embeddings = {}
    for node in graph.nodes:
        entity_embeddings[node] = retrieval_model.encode(node)
    
    return retrieval_model, entity_embeddings

def get_related_entities(graph, entity):
    """
    Get entities related to the given entity in the knowledge graph.
    
    Args:
        graph: NetworkX graph
        entity: Entity to find relations for
        
    Returns:
        list: Related entities with their relations
    """
    if entity not in graph:
        return []
    
    return [(target, data.get('relation', 'related_to')) 
            for target, data in graph[entity].items()]

def find_best_matching_entity(query, retrieval_model, entity_embeddings):
    """
    Find the best matching entity in the knowledge graph for a query.
    
    Args:
        query: Query string
        retrieval_model: SentenceTransformer model
        entity_embeddings: Dict of entity embeddings
        
    Returns:
        str: Best matching entity
    """
    if not entity_embeddings:
        return None
    
    # Encode the query
    query_embedding = retrieval_model.encode(query)
    
    # Find the best matching entity
    best_match = max(
        entity_embeddings.items(),
        key=lambda item: torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(item[1]).unsqueeze(0)
        ).item()
    )
    
    return best_match[0]
