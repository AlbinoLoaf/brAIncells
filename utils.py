import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#torch imports 
import torch
import torch.nn.functional as F
from torcheeg import transforms
from torcheeg.models.gnn.dgcnn import GraphConvolution
from sklearn.model_selection import train_test_split
from torcheeg.models.gnn.dgcnn import GraphConvolution
from cka import HookManager

#Preprocessing
def band_preprocess(X, preprocessed_data_path):
    """
    Apply band differential entropy preprocessing
    
    ...
    
    Parameters
    -----
    X : torch.Tensor
        Features
    preprocessed_data_path : string
        Path to load preprocessed data from / to save preprocessed data to

    Returns
    -----
    X_bde : torch.FloatTensor
        Data preprocessed using band differential entropy
    
    """
    
    bands = {"delta": [1, 4],"theta": [4, 8],"alpha": [8, 14],"beta": [14, 31],"gamma": [31, 49]}
    if os.path.exists(preprocessed_data_path):

        with open(preprocessed_data_path, "rb") as f:
            X_bde = np.load(f)

    else:
        t = transforms.BandDifferentialEntropy(band_dict=bands)

        X_bde = []
        for i in range(X.shape[0]):

            bde_tmp = t(eeg=X[i])
            X_bde.append(bde_tmp)

        X_bde = [x["eeg"] for x in X_bde]

        with open(preprocessed_data_path, "wb") as f:
            np.save(f, X_bde)

    X_bde = torch.FloatTensor(X_bde)     
    return X_bde


#Matrix 
def threshold(mat, thresh=0.2):
    """ 
    Helper function for get_adj_mat, Sets all entries in a matrix below the threshold to 0

    ...

    Parameters
    -----
    mat : two numpy arrays as a matrix
        Torch.tensor as an matrix for numpy to cut off all low values
    thresh : float
        Threshold for the cut off

    Returns
    -----
    mat : two numpy arrays as a matrix
        returns the matrix it recieved but with all  values below the thresh set to zero
    """
    mat[mat < thresh] = 0.0
    return mat

def get_adj_mat(model, thresh=0.2):
    """ 
    Extracts the adjacency matrix from a DGCNN model and normalise it, cutting noise

    ...

    Parameters
    -----
    model : DGCNN   
        nural network model with a learned adjecency matrix
    thresh : float
        Threshold for the cut off

    Returns
    ----- 
     A : two numpy arrays as a matrix
        The adjacency matrix from the network
    """
    A=F.relu(model.A)
    N=A.shape[0]
    A=A*(torch.ones(N,N)-torch.eye(N,N))
    A=A+A.T
    A=threshold(A.detach(),thresh)
    return A

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


#visualisation tools
def plot_matrix(title,matrix_data,xlabel,ylabel,cbarlabel="",cellvalues=True):
    """
    Used for plotting our matrixis to unsure they are all the same format.

    ...

    Parameters
    ------------
    title : string
        The title on the plot
    matrix_data : np.ndarray
        The data for creating the heatmap
    xlabel : list
        A list of strings for the label marks
    ylabel : list
        A list of strings for the label marks
    cbarlabel : string
        default = "", set if cbar name is needed
    cellvalues : bool
        default = true, if fase will stop showing the valies for each matrix cell
    
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix_data,cmap='inferno',vmin=0,vmax=1)
    plt.xticks(ticks=np.arange(len(xlabel)), labels=xlabel)
    plt.yticks(ticks=np.arange(len(ylabel)), labels=ylabel)
    # Colourbar, with fixed ticks to enable comparison
        # plt.colorbar(label="Edge Strength")

    cbar = plt.colorbar(label=cbarlabel)
    cbar.set_ticks([x/20 for x in range(0,21)])
    if cellvalues:
        for i in range(matrix_data.shape[0]):  
            for j in range(matrix_data.shape[1]):
                plt.text(j, i, f"{matrix_data[i, j]:.2f}", ha='center', va='center', color='white' if matrix_data[i, j] < 0.5 else 'black')

    plt.title(title)
    ax = plt.gca()  
    ax.set_xticks(np.arange(len(xlabel)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ylabel)) - 0.5, minor=True)
    ax.grid(which="minor", color="Black", linewidth=0.5)
    ax.tick_params(which="minor", size=0) 
    plt.show()


def visualize_adj_mat(adj_mat):
    """"
    Implements plot matrix sepcifically to make an adjacency matrix

    ...

    Parameters
    -----
        adj_mat : torch.Tensor
            Tensor matrix representing an adjacency matrix
    """
    num_nodes = adj_mat.shape[0]
    node_labels = np.arange(1, num_nodes + 1)
    plot_matrix("Adjecency matrix",adj_mat,node_labels,node_labels,cbarlabel="Edge strength",cellvalues=False)

    
def graph_plot(mods,plot_func,row,col):
    """
    Plot multiple graphs (using a given plotting function) in one figure using subplots
    
    ...
    
    Parameters
    -----
        mods: list of torch.nn models
        plot_func: function
            Function to plot graphs with
        row: int
            Number of rows in the subplot
        col: int
            Number of columns in the subplot
    """
    plt.figure(figsize=(10, 5))
    pos= nx.spring_layout(nx.from_numpy_array(get_adj_mat(mods[0][0]).numpy()),seed=7)

    for i in range(len(mods)):
        G=nx.from_numpy_array(get_adj_mat(mods[i][0]).numpy())
        plot_func(f"G{i+1}",G,row,col,i+1,pos)
    plt.show()


def graph_visual(title,G,row,col,idx,pos):
    """
    Visualize networkx graph
    
    ...
    
    Parameters
    -----
        title: string
            Title of plot
        G: nx.Graph
            Graph object constructed from adjacency matrix
        row: int
            Number of rows in the subplot
        col: int
            Number of columns in the subplot
        idx: int
            Index of current plot in the figure (which subplot is it in the figure)
        pos: int
            Graph layout (nx.spring_layout) - workaround so we can have the same 
            layout for all graphs
    
    Returns
    -----
        fig - plt.subplot containing plot for given graph
    """
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.7]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.7]

    fig = plt.subplot(row, col, idx)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3, edge_color= "red")
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=2, alpha=0.5, edge_color="black")
    plt.title(title)
    return fig

