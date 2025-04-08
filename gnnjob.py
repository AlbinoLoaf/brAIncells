import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

#torch imports 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torcheeg.io.eeg_signal import EEGSignalIO
from torcheeg.models import DGCNN
from torcheeg.models.gnn.dgcnn import GraphConvolution
from sklearn.metrics import accuracy_score
# helper sctipts 
import utils.graph_utils as gu
import utils.data_utils as du
import utils.model_utils as mu 
import utils.visual_utils as vu
from utils.model_utils import TrainNN
from utils.cka import CKACalculator
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

import wandb


## Path constants
path="artifacts"
modelname="dgcnn_mod"
data_path = "../data/"
preprocessed_data_path = f"{path}/preprocessed_data.npy"
has_val_set = False
seed = 42 #should not be set herer right ??

## Establish connection to datafile
IO = EEGSignalIO(io_path=str(data_path), io_mode='lmdb')
bands = {"delta": [1, 4],"theta": [4, 8],"alpha": [8, 14],"beta": [14, 31],"gamma": [31, 49]}

## Read metadata dataframeimports
metadata = pd.read_csv(data_path + 'sample_metadata.tsv', sep='\t')


# Verifying connection to data
idxs = np.arange(len(metadata))

# Read features and labels as torch tensors
X = torch.FloatTensor(np.array([IO.read_eeg(str(i)) for i in idxs]))
y = torch.tensor(metadata["value"].values, dtype=torch.long)

X_bde = du.band_preprocess(X, preprocessed_data_path)     

if has_val_set:
    X_train, y_train, X_val, y_val, X_test, y_test = du.split_data(X_bde, y, has_val_set=has_val_set, seed=seed)
    assert (X_train.shape[0]+X_val.shape[0]+X_test.shape[0])==X.shape[0], "Data samples lost in preprossesing"
    assert (y_train.shape[0]+y_val.shape[0]+y_test.shape[0])==X.shape[0], "Data samples lost in preprossesing"
    
    nsamples_val, nchannels_val, bands = X_val.shape
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
else:
    X_train, y_train, X_test, y_test = du.split_data(X_bde, y, has_val_set=has_val_set, seed=seed)
    assert (X_train.shape[0]+X_test.shape[0])==X.shape[0], "Data samples lost in preprossesing"
    assert (y_train.shape[0]+y_test.shape[0])==X.shape[0], "Data samples lost in preprossesing"

assert X_train.shape[1]==X.shape[1],"Preprossed data lost channels"
assert X_train.shape[2]==len(bands),"Preprossed data does have incorrect amount of bands"

nsamples_train, nchannels_train, bands = X_train.shape
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



def model_metrics(model, X_train, y_train, X_test, y_test, X_val=None, y_val=None,plots=True):
    """
    Display model metrics accuracy, f1 score and confusion matrix for the training, validation
    (if applicable) and test sets
    
    ...
    
    Parameters
    -----
    model: torcheeg.models DGCNN
    X_train, X_val, X_test: torch.FloatTensor
        Input features for the train, validation (if applicable) and test sets
    y_train, y_val, y_test: torch.FloatTensor
        Output labels for the train, validation (if applicable) and test sets
    """
    preds_train = mu.get_preds(model, X_train)
    preds_test = mu.get_preds(model, X_test)    
            
    y_train_npy = y_train.numpy()
    y_test_npy = y_test.numpy()
    
    acc_train = accuracy_score(y_train_npy, preds_train)
    acc_test = accuracy_score(y_test_npy, preds_test)
    
    f1_train = f1_score(y_train_npy, preds_train, average="macro")
    f1_test = f1_score(y_test_npy, preds_test, average="macro")
    
    conf_mat_train = confusion_matrix(y_train, preds_train)        
    conf_mat_test = confusion_matrix(y_test, preds_test)
    return acc_train,f1_train,conf_mat_train, acc_test, f1_test,conf_mat_test

def generate_model_dict(models, X_train, y_train, X_test, y_test, node_labels=None, compute_adj=True):
    """
    Generate a dictionary with relevant metrics and artifacts for each model.

    Parameters
    ----------
    models : list
        List of tuples, each containing (model, training_logs or [])
    X_train, y_train, X_test, y_test : torch.Tensor
        Dataset splits
    node_labels : list, optional
        If provided, will be used to label adjacency matrices
    compute_adj : bool
        Whether to extract adjacency matrices

    Returns
    -------
    model_metrics_dict : dict
        Dictionary containing metrics and artifacts per model index
    """
    model_metrics_dict = {}

    for i, (model, _) in enumerate(models):
        model = model.eval().cpu()
        
        # Metrics
        acc_train, f1_train, conf_train, acc_test, f1_test, conf_test = model_metrics(
            model, X_train, y_train, X_test, y_test, plots=False
        )

        # Adjacency matrix
        adj = None
        if compute_adj:
            adj = mu.get_adj_mat(model)
            if node_labels is not None:
                adj = pd.DataFrame(adj, index=node_labels, columns=node_labels)

        model_metrics_dict[i] = {
            "model": model,
            "adjacency_matrix": adj,
            "acc_train": acc_train,
            "f1_train": f1_train,
            "confusion_train": conf_train,
            "acc_test": acc_test,
            "f1_test": f1_test,
            "confusion_test": conf_test
        }

    return model_metrics_dict

wandb.login()
# DICTIONARY PARAMETERS
def train_models(model,modeltrainer,hid_chans,seed_list,num_models=1,new=False,prints=False,path=path,modelname=modelname):
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
    for i in range(num_models):
        
        tmp_mod = model(in_channels=num_chans, num_electrodes=num_electrodes, 
                              hid_channels=hid_chans, num_layers=num_layers, num_classes=num_outputs)
        model_path=f"{path}/{modelname}_chan_{hid_chans}_{i}.pth"
        print(f"Model {i+1}")
        if new or not os.path.exists(model_path):    
            if not os.path.exists(model_path) and not new:
                print(f"Could not resolve path: {model_path}")
                new_models=True
            trainer = modeltrainer()
            
            if has_val_set:
                mods.append(trainer.train_model(tmp_mod, train_loader, learning_rate=lr,path=path,name=f"{modelname}_chan{hid_chans}",
                            has_val_set=has_val_set,val_loader=val_loader,w_decay=w_decay,epochs=epochs, 
                            prints=prints, modrun=i, seed=seed_list[i]))
            else:
                mods.append(trainer.train_model(tmp_mod, train_loader, learning_rate=lr,path=path,name=f"{modelname}_chan{hid_chans}",
                                                has_val_set=has_val_set,val_loader=None,w_decay=w_decay,epochs=epochs, 
                                                prints=prints, modrun=i, seed=seed_list[i]))   
        else: 
            tmp_mod.load_state_dict(torch.load(model_path))
            tmp_mod.eval()
            mods.append([tmp_mod,[]])
    return mods
# Model parameters
num_chans = 5
num_electrodes = 22
num_outputs = y.max().item() + 1
num_layers = 2
hid_chans = 16

# Training parameters
lr = 1e-4
epochs = 40
w_decay = 1e-3

seed_list = [42, 30, 66, 89]

#Models 
modruns = 4
plot=False
new_models=True

param_list = [8, 16, 24]
seed_list = [42, 30, 66, 89]

n_models = 4




def lst_to_dict(lst):
    return dict([(x, []) for x in lst])
def internal_dict(lst):
    models_dict     = lst_to_dict(lst)  #Dict with all models
    bary_dict       = lst_to_dict(lst)  #Graph metric dict n for barycenter
    sim_dict        = lst_to_dict(lst)  #Graph metric dict n for simrank
    edit_dists      = lst_to_dict(lst)  #Graph metric dict n for GED between similar param models
    return models_dict,bary_dict,sim_dict,edit_dists


def run_models_hpc(param_list, n_runs):
    
    run_idx = 1
    
    while run_idx < n_runs:
        random_seed = random.randint(0, 999999)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        print(run_idx)
        path_name = f"models/run_{run_idx}_seed_{random_seed}"
        os.makedirs(path_name)
        print(run_idx)
        
        models_dict, bary_dict, sim_dict, _ = internal_dict(param_list)
        
        for n_chans in param_list:
            model_name = f"model_seed_{random_seed}"
            curr_model = [x[0] for x in train_models(DGCNN, TrainNN, n_chans, seed_list, num_models = 1,
                                                     prints=False, new=True, path=path_name, modelname=model_name)]
            models_dict[n_chans].extend(curr_model)
        
        for n_chans in param_list:
            models = models_dict[n_chans]
            bary, sim, _, _ = gu.get_graph_metrics(models, prints=False)
            bary_dict[n_chans].extend(bary)
            sim_dict[n_chans].extend(sim)
        
        file_suffix = f"_seed_{random_seed}"

        with open(f'{path_name}/barycenters{file_suffix}.pkl', 'wb') as fp:
            pickle.dump(bary_dict, fp)
            
        with open(f'{path_name}/simrank{file_suffix}.pkl', 'wb') as fp:
            pickle.dump(sim_dict, fp)
        
        run_idx += 1


run_models_hpc([8,16,24,32,64,128],5000)

def multi_parameter_mod(param_list, seed_list, n_models):
    # TO DO: take into account seeds. Right now the seed_list doesn't do anything
    # all combinations of model indexes between two parameter sets
    # ex all model combinations between models with 8 hidden neurons and models with 16 hidden neurons
    combs_external =[
        (i, j)
        for i in range(n_models)
        for j in range(n_models, 2 * n_models)
        ]
    #combs_external = list(itertools.product([x for x in range(n_models)], [x+n_models for x in range(n_models)]))
    # all combinations of parameter values, in this case number of hidden neurons
    param_combs = [
        (param_list[i], param_list[j]) 
        for i in range(0,   len(param_list)) 
        for j in range(i+1, len(param_list))]
    
    models_dict, bary_dict, sim_dict, edit_dists_internal = internal_dict(param_list)
    edit_dists_external = lst_to_dict(param_combs)   #Graph metric dict n for GED between different param models
    
    # train n_models models with each number of hidden neurons specified in the param_list
    for n_chans in param_list:
        curr_model = [x[0] for x in train_models(DGCNN, TrainNN, n_chans, seed_list, num_models = n_models,
                                                 prints=plot, new=False)]
        models_dict[n_chans].extend(curr_model)
    
    # calculate all metrics for models with the same number of hidden neurons
    for n_chans in param_list:
        models = models_dict[n_chans]
        bary, sim, _, ed = gu.get_graph_metrics(models, prints=plot)
        bary_dict[n_chans].extend(bary)
        sim_dict[n_chans].extend(sim)
        edit_dists_internal[n_chans].extend(ed)
    
    # calculate edit distance between models with different number of hidden neurons
    for param_comb in param_combs:
        for ext_comb in combs_external:
            model1_idx = ext_comb[0]; model2_idx = ext_comb[1]
            model1 = models_dict[param_comb[0]][model1_idx]
            model2 = models_dict[param_comb[1]][model2_idx-n_models]
            G1 = gu.make_graph(mu.get_adj_mat(model1))
            G2 = gu.make_graph(mu.get_adj_mat(model2))
            ed_external = next(nx.optimize_graph_edit_distance(G1, G2))
            edit_dists_external[param_comb].append(ed_external)

    return models_dict, bary_dict, sim_dict, edit_dists_internal, edit_dists_external

models, barycenters, sims, edit_dists_internal, edit_dists_external = multi_parameter_mod(param_list, seed_list, n_models)