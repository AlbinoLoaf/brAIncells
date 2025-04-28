import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils.model_utils import get_adj_mat
from collections import Counter
import pandas as pd
import mne
#visualisation tools
def plot_matrix(title,matrix_data,xlabel,ylabel,cbarlabel="",cellvalues=True, figsize=(10,10)):
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
    cbarlabel : stringvu.visualize_adj_mat
        default = "", set if cbar name is needed
    cellvalues : bool
        default = true, if fase will stop showing the valies for each matrix cell
    
    """
    plt.figure(figsize=figsize)
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

    
def graph_plot(adj,plot_func,row,col, bary_list):
    """
    Plot multiple graphs (using a given plotting function) in one figure using subplots
    
    ...
    
    Parameters
    -----
        mods: list of adjacency matrix
        plot_func: function
            Function to plot graphs with
        row: int
            Number of rows in the subplot
        col: int
            Number of columns in the subplot
    """
    plt.figure(figsize=(10,10))
    pos= nx.spring_layout(nx.from_numpy_array(adj[0].numpy()),seed=7)

    for i in range(len(adj)):
        assert np.array_equal(adj[i], adj[i].T), "The adjacency is not symetric"
        G=nx.from_numpy_array(adj[i].numpy())
        plot_func(f"G{i+1}",G,row,col,i+1,pos,bary_list[i])
    plt.show()


def graph_visual(title,G,row,col,idx,pos,bary_list):
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
    # get node labels 
    node_labels = pd.read_csv("node_names.tsv", sep="\t")
    node_labels = list(node_labels['name'])
    label_dict = {node: label for node, label in zip(G.nodes(), node_labels)}

    # MNE EEG montage
    montage = mne.channels.make_standard_montage('standard_1020')
    channels = node_labels

    # get the positions of the electrodes through MNE
    pos = montage.get_positions()['ch_pos']
    pos = np.array([pos[i][:2] for i in channels])

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.7]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.7]

    fig = plt.subplot(row, col, idx)
    nx.draw_networkx_nodes(G, pos,nodelist=bary_list, node_color="plum")
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes() if n not in bary_list], node_color="lightblue")
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3, edge_color= "red")
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=1, alpha=0.5, edge_color="black")
    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=10, font_color="black")
    plt.title(title)
    return fig

def dict_to_counts(d):
    
    nodes = []   
    for key in d.keys():
        
        curr_list = []
        [curr_list.extend(x) for x in d[key]]
        nodes.extend(curr_list)
        
    counts = Counter(nodes)
    
    # check all nodes have at least a count of 1 (appear in the counter)
    if len(counts.keys()) != 22:
        
        # add count of 0 for all nodes that don't appear in the counter
        for i in range(22):
            if i not in counts.keys():
                counts[i] = 0
        
    return [counts[i] for i in range(22)]

def dict_to_histogram(metric_dict,chan,node_labs): 

        node_counts = dict_to_counts(metric_dict)
        plt.figure(figsize=(10,5))
        plt.axhline(5)
        plt.bar(node_labs, node_counts, color="plum", edgecolor="black")
        plt.title(f"Barycenter bar chart for n_chans={chan}")
        plt.xlabel("Barycenter")
        plt.ylabel("Frequency")
        plt.show()

def bary_hist(bary_list):

    counter = Counter(bary_list)
    sorted_labels = sorted(counter.items())  
    labels, values = zip(*sorted_labels)

    node_labels = pd.read_csv("node_names.tsv", sep="\t")
    node_labels = list(node_labels['name'])

    fig = plt.figure(figsize=(8, 6))
    indexes = np.arange(len(labels))
    plt.bar(indexes, values)
    plt.xticks(indexes, node_labels)
    plt.show()
    
def simrank_to_matrix(sim):
    
    n = len(sim)
    mat = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            
            mat[i, j] = sim[i][j]
    return mat

def plot_simrank(sims, n_nodes=22, figsize=(5,5)):
    
    for k in sims.keys():
        curr_sims = sims[k]
        for i in range(len(curr_sims)):
            curr_mat = simrank_to_matrix(curr_sims[i])
            vu.plot_matrix(f"Simrank for k = {k} model idx {i}", curr_mat, list(range(n_nodes)), list(range(n_nodes)), 
                           cbarlabel="", cellvalues=False, figsize=figsize)
            
            
def plot_loss_curves(path, mods):
    
    for i  in range(len(mods)):
        filepath=f"{path}/Training_validation_loss{i}.npy"
        if not new_models:
            try: 
                with open(filepath, "rb") as f:
                    data = np.load(f)
            except:
                print(f"File with the data could not be found looking at address: {filepath}")
        else:
            with open(filepath, "wb") as f:
                np.save(f, mods[0][1])
                data = mods[0][1]
        plt.plot(data[0])
        plt.plot(data[1])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["training loss","validation loss"])
        plt.title(f"Model{i} Training vs validation loss")
        plt.show()
        
def mne_plot(node_labels_path):

    node_labels = pd.read_csv(node_labels_path, sep="\t")
    node_labels = list(node_labels['name'])

    info = mne.create_info(ch_names=node_labels,sfreq=1000,ch_types='eeg')

    n_channels = len(node_labels)
    data = np.zeros((n_channels, 1000))

    raw = mne.io.RawArray(data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    raw.plot_sensors(kind='topomap', show_names=True)