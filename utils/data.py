import os
import numpy as np
import torch
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


