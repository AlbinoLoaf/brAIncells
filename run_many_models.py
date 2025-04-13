import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader,TensorDataset
from torcheeg.io.eeg_signal import EEGSignalIO
from torcheeg.models import DGCNN
import utils.data_utils as du 
from utils.model_utils import TrainNN

# Path constants
path="artifacts"
data_path = "../data/"
metadata_path = data_path + "sample_metadata.tsv"
preprocessed_data_path = f"{path}/preprocessed_data.npy"
node_labels_path = "node_names.tsv"
model_obj_path = "model_runs/"

# Data preprocessing constants
bands = {"delta": [1, 4],"theta": [4, 8],"alpha": [8, 14],"beta": [14, 31],"gamma": [31, 49]}
has_val_set = False

# Model architecture
model_arch = {
    "model":DGCNN,
    "chans" : 5,
    "electrodes" : 22,
    "outputs" : 5,
    "layers" : 2,
    "hid_chans" : 16
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

## Establish connection to datafile
IO = EEGSignalIO(io_path=str(data_path), io_mode='lmdb')

## Read metadata dataframeimports
metadata = pd.read_csv(metadata_path, sep='\t')

# Verifying connection to data
idxs = np.arange(len(metadata))

# Read features and labels as torch tensors
X = torch.FloatTensor(np.array([IO.read_eeg(str(i)) for i in idxs]))
y = torch.tensor(metadata["value"].values, dtype=torch.long)

X_bde = du.band_preprocess(X, preprocessed_data_path)     

if has_val_set:
    X_train, y_train, X_val, y_val, X_test, y_test = du.split_data(X_bde, y, has_val_set=has_val_set, seed=42)
    assert (X_train.shape[0]+X_val.shape[0]+X_test.shape[0])==X.shape[0], "Data samples lost in preprossesing"
    assert (y_train.shape[0]+y_val.shape[0]+y_test.shape[0])==X.shape[0], "Data samples lost in preprossesing"
    
    nsamples_val, nchannels_val, bands = X_val.shape
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
else:
    X_train, y_train, X_test, y_test = du.split_data(X_bde, y, has_val_set=has_val_set, seed=42)
    assert (X_train.shape[0]+X_test.shape[0])==X.shape[0], "Data samples lost in preprossesing"
    assert (y_train.shape[0]+y_test.shape[0])==X.shape[0], "Data samples lost in preprossesing"

assert X_train.shape[1]==X.shape[1],"Preprossed data lost channels"
assert X_train.shape[2]==len(bands),"Preprossed data does have incorrect amount of bands"

nsamples_train, nchannels_train, bands = X_train.shape
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def train_models(modeltrainer,dict_model_arch, dict_training,dict_model_meta, path=path):
    """
    Training a model with random initialisation but consitent parameters. 
    
    Hyper parameters should be set outside of this function 
    
    path and model_names are both needed parameters that need to be defined outside of this function. 
    
    Path is to where your artifacts are located and model_name is what the model is called 
    
    ...
    
    Parameters
    -----------
    model : nn.modules
        The model being trained
    modeltrainer : training class
        a class for training the model provided should return a trained model
    num_models : int 
        Default 1, how many models it trains
    new : bool
        Weather it should attempt to use saved models
    """
    mods = []
    for i in range(dict_model_meta["amount"]):   
            tmp_mod = dict_model_arch["model"](in_channels=dict_model_arch["chans"], 
                                            num_electrodes=dict_model_arch["electrodes"], 
                                            hid_channels=dict_model_arch["hid_chans"], 
                                            num_layers=dict_model_arch["layers"],
                                            num_classes=dict_model_arch["outputs"])
            
            mod_name=dict_model_meta["name"]
            
            model_path=f"{path}/{mod_name}_{i}.pth"

            if dict_model_meta["amount"]>1:
                print(f"Model {i+1}")
                
            if not os.path.exists(model_path):
                    print(f"[train_models]: Could not resolve path: {model_path}")
                    dict_model_meta["new_models"]=True
                
            if dict_model_meta["new_models"] or not os.path.exists(model_path):
                
                print(f"[train_models]: Training new models")
                trainer = modeltrainer()
                
                mods.append(trainer.train_model(tmp_mod, train_loader, 
                                                    path=model_path,
                                                    name=mod_name,
                                                    has_val_set=False,
                                                    val_loader=None,
                                                    learning_rate=dict_training["lr"],
                                                    w_decay=dict_training["w_decay"],
                                                    epochs=dict_training["epochs"], 
                                                    prints=dict_model_meta["plot"],
                                                    modrun=i))   
            else:
                print(f"[train_models]: Loading models from {model_path}")
                tmp_mod.load_state_dict(torch.load(model_path))
                tmp_mod.eval()
                mods.append([tmp_mod,[]])
    return mods


def seed_all(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_models_hpc(param_list, n_runs, dict_model_arch, dict_training, dict_model_meta):

    run_idx = 1
    while run_idx < n_runs:
        random_seed = random.randint(0, 999999)
        seed_all(random_seed)
        path_name = model_obj_path + f"run_{run_idx}_seed_{random_seed}"
        print(f"[run_models_hpc] : Run idx: {run_idx}  | Curr seed: {random_seed}")
        print(f"[run_models_hpc] : Model dir: {path_name}")
        os.makedirs(path_name)
        models_dict = dict([(x, []) for x in param_list])
        
        for n_chans in param_list:
            model_name = f"model_chans_{n_chans}_seed_{random_seed}"
            
            new_dict_arch = dict_model_arch
            new_dict_arch["hid_chans"] = n_chans
            
            new_dict_meta = dict_model_meta
            new_dict_meta["name"] = model_name
            
            curr_model = [x[0] for x in train_models(TrainNN, new_dict_arch, dict_training,
                                                     new_dict_meta, path=path_name)]
            models_dict[n_chans].extend(curr_model)
        
        run_idx += 1

PARAM_LIST = [8, 16, 24]
MODELS_PER_PARAM = 9

run_models_hpc(PARAM_LIST, MODELS_PER_PARAM, model_arch, train_dict, model_meta)

