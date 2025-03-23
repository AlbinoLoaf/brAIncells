import networkx as nx 
from torcheeg.models.gnn.dgcnn import GraphConvolution
from cka import HookManager

# Networkx utils
def check_isomorphism(adj1, adj2):
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

    
    G1 = nx.from_numpy_array(adj1.numpy())
    G2 = nx.from_numpy_array(adj2.numpy())
    
    return nx.vf2pp_is_isomorphic(G1, G2, node_label=None)

def get_graph_edit_dist(adj1, adj2):
    """
    Get graph edit distance of two graphs
    
    Parameters
    -----------
    adj1: torch matrix
        Adjacency matrix of first graph
    adj2: torch matrix
        Adjacency matrix of second graph
        
    Returns
    -----
    g_dist: float
        Graph edit distance of the two graphs

    """
    G1 = nx.from_numpy_array(adj1.numpy())
    G2 = nx.from_numpy_array(adj2.numpy())
    g_dist = nx.graph_edit_distance(G1, G2)
    return g_dist

def get_simrank_similarity(adj):
    """
    Get simrank similarity of nodes in a graph
    
    Parameters
    -----------
    adj: torch matrix
        Adjacency matrix of graph
        
    Returns
    -----
    dict
        Simrank similarity of each pair of two nodes
    """
    
    G = nx.from_numpy_array(adj.numpy())
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