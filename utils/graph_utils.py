import networkx as nx 
import torch
import numpy as np
from torcheeg.models.gnn.dgcnn import GraphConvolution
from utils.cka import HookManager
import utils.model_utils as mu 


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
    
def make_binary_graph(adjacency_matrix,thresh=0.5):
    """
    Creates a binary networkX graph from an adjacency matrix where all edge under the threshold are 0 and the rest are 1
        
    Parameters:
        adjacency_matrix : torch.Tensor/
          adjacency matrix representation
    
    Returns:
        networkx.Graph: graph created from adjacency matrix
    """
    tmp = adjacency_matrix.clone()
    tmp[tmp < thresh] = 0.0
    tmp[tmp >= thresh] =1.0
    return make_graph(tmp), tmp

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

def check_not_isomorphism(G1, G2):
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
    return not nx.vf2pp_is_isomorphic(G1, G2, node_label=None)

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


def simrank_to_matrix(sim):
    """
    Makes the simrank martricx from the simrank lists 

    
    Parameters
    --------
    sim : list of lists [][]
        simrank returned by the simrank networksX
    Returns
    --------
    mat : numpy((n,n))
    
    """
    n = len(sim)
    mat = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            
            mat[i, j] = sim[i][j]
    return mat

def get_graph_metrics_internal(mod_list, prints=False):

    barycenters = []; simrank_similarities = []
    
    # Barycenters and simrank similarity
    for i in range(len(mod_list)):
        
        curr_adj = mu.get_adj_mat(mod_list[i])
        if nx.is_connected(make_graph(curr_adj)):
            curr_barycenter = get_barycenter(curr_adj)
            barycenters.append(curr_barycenter)
        else:
            print("graph not connected")
        G = make_graph(curr_adj)
        sim = get_simrank_similarity(G) # not printing because it's a huge dict of all node pair simiarities
        simrank_similarities.append(sim)
        
        if prints:
            print(f"---For model idx {i}---")
            print(f"Barycenter: {curr_barycenter}")
    
    return barycenters, simrank_similarities

def get_graph_metrics_external(mod_list, prints=False):
    
    isomorphism_checks = []; geds = []
    graphs = [make_graph(mu.get_adj_mat(mod_list[i])) for i in range(len(mod_list))]
    
    # Isomorphism check and graph edit distance
    for i in range(len(mod_list)):
        for j in range(i+1, len(mod_list)):
            G1 = graphs[i]
            G2 = graphs[j]
            
            is_isomorphic = not check_not_isomorphism(G1, G2)
            isomorphism_checks.append(is_isomorphic)
            if prints:
                print(f"---Graphs for model {i} and model {j}---")
                print(f"Is isomorphic: {is_isomorphic}")
            
            # if graphs are not isomorphic, get approximation of their edit distance
            if is_isomorphic == False:
                approx_ged = next(nx.optimize_graph_edit_distance(G1, G2))
                geds.append(approx_ged)
                if prints:
                    print(f"GED (approx): {approx_ged}")
    return isomorphism_checks, geds