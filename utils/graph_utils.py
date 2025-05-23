import networkx as nx 
import torch
import numpy as np
from torcheeg.models.gnn.dgcnn import GraphConvolution
from utils.cka import HookManager
import utils.model_utils as mu
import utils.visual_utils as vu
from collections import Counter


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


def get_max_betweeness_nodes(model_dict):
    """
    Get nodes with highest betweeness centrality for each model
    
    Parameters
    -----------
    model_dict: dict
        Dictionary of the form dict[seed][channels] containing 
        the trained model objects (torcheeg.models DGCNN objects)
        
    Returns
    -----
    betweeness_nodes: dict
        The node(s) that has the highest betweeness centrality for each model
        Dictionary of the form dict[channels]
    """
    betweeness_nodes = dict()
    for n_chans in model_dict.keys():
        
        curr_models = model_dict[n_chans]   
        chan_betweeness = []
        
        for model in curr_models:
            
            curr_adj_mat = mu.get_adj_mat(model)
            G = make_graph(curr_adj_mat)
            betweeness_vals = sorted(nx.betweenness_centrality(G).items(), key=lambda x:x[1], reverse=True)
            highest = betweeness_vals[0][1]
            highest_nodes = [x[0] for x in betweeness_vals if x[1] == highest]
            chan_betweeness.append(highest_nodes)
            
        betweeness_nodes[n_chans] = chan_betweeness
        
    return betweeness_nodes


def get_max_current_flow_betweeness_nodes(model_dict):
    """
    Get nodes with highest current flow betweeness centrality for each model
    
    Parameters
    -----------
    model_dict: dict
        Dictionary of the form dict[seed][channels] containing 
        the trained model objects (torcheeg.models DGCNN objects)
        
    Returns
    -----
    current_flow_betweeness_nodes: dict
        The node(s) that has the highest current flow betweeness centrality for each model
        Dictionary of the form dict[channels]
    """
    current_flow_betweeness_nodes = dict()
    for n_chans in model_dict.keys():
        
        curr_models = model_dict[n_chans]   
        chan_curr_flow_betweeness = []
        
        for model in curr_models:
            curr_adj_mat = mu.get_adj_mat(model)
            if nx.is_connected(make_graph(curr_adj_mat)):
                G = make_graph(curr_adj_mat)
                curr_flow_betweeness_vals = sorted(nx.current_flow_betweenness_centrality(G).items(), key=lambda x:x[1], reverse=True)
                highest = curr_flow_betweeness_vals[0][1]
                highest_nodes = [x[0] for x in curr_flow_betweeness_vals if x[1] == highest]
                chan_curr_flow_betweeness.append(highest_nodes)
            else:
                print("graph not connected")
              
        current_flow_betweeness_nodes[n_chans] = chan_curr_flow_betweeness
        
    return current_flow_betweeness_nodes

def get_max_laplacian_centrality_nodes(model_dict):
    """
    Get nodes with highest Laplacian centrality for each model
    
    Parameters
    -----------
    model_dict: dict
        Dictionary of the form dict[seed][channels] containing 
        the trained model objects (torcheeg.models DGCNN objects)
        
    Returns
    -----
    laplacian_centrality_nodes: dict
        The node(s) that has the highest Laplacian centrality for each model
        Dictionary of the form dict[channels]
    """
    
    laplacian_centrality_nodes = dict()
    for n_chans in model_dict.keys():
        
        curr_models = model_dict[n_chans]   
        chan_laplacian_centrality = []
        
        for model in curr_models:
            
            curr_adj_mat = mu.get_adj_mat(model)
            G = make_graph(curr_adj_mat)
            laplacian_centrality_vals = sorted(nx.laplacian_centrality(G).items(), key=lambda x:x[1], reverse=True)
            highest = laplacian_centrality_vals[0][1]
            highest_nodes = [x[0] for x in laplacian_centrality_vals if x[1] == highest]
            chan_laplacian_centrality.append(highest_nodes)
            
        laplacian_centrality_nodes[n_chans] = chan_laplacian_centrality
        
    return laplacian_centrality_nodes


def get_bary_counts(bary_dict):
    """
    Get barycenter counts by channel, overall, and the raw barycenter values
    
    Parameters
    -----------
    bary_dict: dict
        Dictionary of the form dict[channels] containing the
        barycenters for the model adj matrices
        
    Returns
    -----
    freqs_by_chan: dict
        The node(s) that are the barycenter for each model, gruped by channels
        Dictionary of the form dict[channels]
    node_counts_all: list
        Counts for each electrode being a barycenter (for all the models)
    all_bary: list
        Raw barycenter values (electrode labels) for all models as a list
    """
    freqs_by_chan = dict()
    all_bary = []
    for n_chans in bary_dict.keys():
        bary_list = []
        [bary_list.extend(x) for x in bary_dict[n_chans]]
        all_bary.extend(bary_list)
        node_counts = dict(sorted(Counter(bary_list).items()))
        node_counts = list(node_counts.values())
        freqs_by_chan[n_chans] = node_counts
        
    node_counts_all = dict(sorted(Counter(all_bary).items()))
    node_counts_all = list(node_counts_all.values())
    
    return freqs_by_chan, node_counts_all, all_bary


def shrink(lst, target):
    """
    Takes a list and shrinks it uniformly to hit the target, if it cannot shrink uniformly it will suptract to the most populated arrays first

    Parameters
    --------
    lst : list 
        A list of synthetic data counts
    target : int
        The target for how much it is shrinking 

    Returns
    ---------
    adjusted : list 
         new list to be used instead of the old list    
    """
    reduction = (sum(lst) - target) // len(lst)
    lst = [x - reduction for x in lst]

    excess = sum(lst) - target
    adjusted = lst.copy()

    indices_to_reduce = sorted(
        range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)[:excess]

    for idx in indices_to_reduce:
        adjusted[idx] -= 1

    return adjusted


def grow(lst, target):
    """
    Takes a list and gows it uniformly to hit the target, if it cannot grow uniformly it will add to the most populated arrays

    Parameters
    --------
    lst : list 
        A list of synthetic data counts
    target : int
        The target for how much it is growing 

    Returns
    ---------
    adjusted : list 
         new list to be used instead of the old list    
    """
    growth = (target - sum(lst)) // len(lst)
    lst = [x + growth for x in lst]

    shortage = target - sum(lst)
    adjusted = lst.copy()

    indices_to_increase = sorted(range(len(adjusted)), key=lambda i: adjusted[i], reverse=True)[:shortage]
    for idx in indices_to_increase:
        adjusted[idx] += 1

    return adjusted


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
    """
    Get one graph metrics (barycenter, simrank)
    
    
    Parameters
    --------
    mod_list: list
        List of model objects (torcheeg.models DGCNN)
    prints: bool
        Whether to print the barycenters for each model
        
    Returns
    --------
    barycenters: list
        List of barycenters for models in mod_list
    simrank_similarities: list
        List of simrank similarities (as dicts) for models in mod_list
    
    """

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
    """
    Get two graph metrics (isomorphism check, graph edit distance)
    
    
    Parameters
    --------
    mod_list: list
        List of model objects (torcheeg.models DGCNN)
    prints: bool
        Whether to print the metrics
        
    Returns
    --------
    isomorphism_checks: list
        List of isomorphism check results for all model pairs (bool)
    geds: list
        List of graph edit distance (approximation) for all model pairs
    
    """
    
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


def get_sorted_metrics(metric_dict, node_labels, ascending=True):
    
    node_counts = vu.dict_to_counts(metric_dict)
    
    sorted_data = sorted(zip(node_counts, node_labels))
    node_counts_sorted, node_labels_sorted = zip(*sorted_data)
    
    assert set(node_labels_sorted) == set(node_labels)
        
    return list(node_counts_sorted), list(node_labels_sorted)

def get_centers(lst):
    bary_list = []
    [bary_list.extend(x) for x in lst]
    return bary_list

def group_barycenters(lst,mapper):
    return list(map(mapper,lst))

def FCCPP_group(lst):
    return group_barycenters(lst,lambda a: 1 if a<6 else 2 if a<13 else 3 if a <18 else 4)