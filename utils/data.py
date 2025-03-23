import os
import numpy as np
import torch
import torch.nn.functional as F
from torcheeg import transforms

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
