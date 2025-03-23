import networkx as nx 
import torch
import numpy as np
from torcheeg.models.gnn.dgcnn import GraphConvolution
from utils.cka import HookManager


# Networkx utils, Graph utils
def make_graph(adjacency_matrix):
    """
    Creates a NetworkX graph from an adjacency matrix.
    
    Parameters:
        adjacency_matrix : torch.Tensor/np.ndarray
          adjacency matrix representation
    
    Returns:
        networkx.Graph: graph created from adjacency matrix
    
    Raises:
        TypeError: if input is neither torch.Tensor nor np.ndarray
    """
    if isinstance(adjacency_matrix, torch.Tensor):
        return nx.from_numpy_array(adjacency_matrix.numpy())
    elif isinstance(adjacency_matrix, np.ndarray):
        return nx.from_numpy_array(adjacency_matrix)
    else:
        raise TypeError("adjacency_matrix must be a torch.Tensor or np.ndarray.")

def get_barycenter(adj):
    """
    Calculates barycenter of graph - the node which minimizes its distance from all other nodes
    AKA the center of the graph
    If multiple nodes tie for the smallest distance, then all of them are returned (as a list)
    
     Parameters
    -----------
    adj: torch matrix
        Adjacency matrix of graph
        
    Returns
    -----
    bar: list of ints
        The node id(s) that minimize the distance to all other nodes
    """
    
    G = nx.from_numpy_array(adj.numpy())
    bar = nx.barycenter(G)
    return bar

def check_isomorphism(G1, G2):
    """
    Check if two graphs are isomorphic ie structurally the same
    
    Parameters
    -----------
    adj1: torch matrix
        Adjacency matrix of first graph
    adj2: torch matrix
        Adjacency matrix of second graph
        
    Returns
    -----
    bool
        Whether the two graphs are isomorphic
    """
    return nx.vf2pp_is_isomorphic(G1, G2, node_label=None)

def get_graph_edit_dist(G1, G2):
    """
    Get graph edit distance of two graphs
    
    Parameters
    -----------
    adj1: networkx graph
        The one graph bassed on an adjacency matrix 
    adj2: networkx graph
        The other graph bassed on an adjacency matrix 
        
    Returns
    -----
    g_dist: float
        Graph edit distance of the two graphs

    """
    g_dist = nx.graph_edit_distance(G1, G2)
    return g_dist

def get_simrank_similarity(G):
    """
    Get simrank similarity of nodes in a graph
    
    Parameters
    -----------
    adj: Networkx graph
        Graph made from an adjacency matrix 
        
    Returns
    -----
    dict
        Simrank similarity of each pair of two nodes
    """
    
    return nx.simrank_similarity(G)


def get_activations(model,data, layer_type=[GraphConvolution]):
    """
    Get model activations from specified types of layers
    
    Parameters
    -----------
    model: torch.nn model
    layer_type: list of torch.nn layers
    
    Returns
    -----
    activations: torch FloatTensor
        Extracted activations from specified layers
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    hook_manager = HookManager(model, layer_type)
    model(data) # apparently a forward pass needs to be done for hooks to run
                   # this is just a placeholder so hooks get registered
    activations = hook_manager.get_activations()
    return activations