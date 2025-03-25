import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils.model_utils import get_adj_mat
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
    cbarlabel : stringvu.visualize_adj_mat
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
    plt.figure(figsize=(10,10))
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

