import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from torcheeg.io.eeg_signal import EEGSignalIO
from torcheeg import transforms
from torcheeg.models import DGCNN
from util.model_utils import TrainNN

model_arch = {
    "in_channels" : 5,
    "num_electrodes" : 22,
    "num_classes" : 4,
    "num_layers" : 2,
    "hid_channels" : 16
    }

# Training parameters
train_dict = {
    "lr": 1e-4,
    "epochs": 40,
    "w_decay": 1e-3,
    }

#Models 
model_meta ={
    "amount":1,
    "plot":True,
    "new_models":True,
    "name": "dgcnn_mod"
    }

path="artifacts"
data_path = "data/"
metadata_path = data_path + "sample_metadata.tsv"
preprocessed_data_path = f"{path}/preprocessed_data.npy"
node_labels_path = "node_names.tsv"
model_obj_path = "model_runs/"

# Data preprocessing constants
bands = {"delta": [1, 4],"theta": [4, 8],"alpha": [8, 14],"beta": [14, 31],"gamma": [31, 49]}

def seed_all(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
    
def load_data(data_path, metadata_path, preprocessed_data_path):

    IO = EEGSignalIO(io_path=str(data_path), io_mode='lmdb')

    ## Read metadata dataframeimports
    metadata = pd.read_csv(metadata_path, sep='\t')

    # Verifying connection to data
    idxs = np.arange(len(metadata))

    # Read features and labels as torch tensors
    X = torch.FloatTensor(np.array([IO.read_eeg(str(i)) for i in idxs]))
    y = torch.tensor(metadata["value"].values, dtype=torch.long)

    X_bde = band_preprocess(X, preprocessed_data_path)     

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    nsamples_train, nchannels_train, bands = X_train.shape
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    return train_loader

train_loader = load_data(data_path, metadata_path, preprocessed_data_path)