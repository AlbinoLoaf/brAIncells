import os
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
# helper sctipts 
import utils.graph_utils as gu
import utils.data_utils as du
import utils.model_utils as mu 
import utils.visual_utils as vu
from utils.model_utils import TrainNN
from utils.cka import CKACalculator

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

def train_models(model,modeltrainer,hid_chans,seed_list, num_models=1,new =False, prints=False):
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
        model_path=f"{path}/{modelname}_chan{hid_chans}_{i}.pth"
        #print(f"Model {i+1}")
        if new or not os.path.exists(model_path):    
            if not os.path.exists(model_path) and not new:
                #print(f"Could not resolve path: {model_path}")
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
mods = train_models(DGCNN, TrainNN,hid_chans,seed_list, num_models=modruns, prints=plot, new=new_models)

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

for k in edit_dists_internal.keys():
    
    curr_list = edit_dists_internal[k]
    #print(f"{k}: {sum(curr_list) / len(curr_list)}")


for k in edit_dists_external.keys():
    
    curr_list = edit_dists_external[k]
    #print(f"{k}: {sum(curr_list) / len(curr_list)}")


node_labels = pd.read_csv("node_names.tsv", sep="\t")
node_labels = list(node_labels['name'])

for k in barycenters.keys():
    a=0
    #print(f"For n_neurons = {k}")
    #print([[node_labels[x] for x in s] for s in barycenters[k]])


for i in range(modruns):
    #print(f"Model {i+1} test")
    test_model = mods[i][0].to("cpu").eval()
    mu.model_metrics(test_model, X_train, y_train, X_test, y_test, None, None,plots=plot)


    graphs=[]
adj_mats=[]
for i in range(modruns):
    adj_mats.append(mu.get_adj_mat(mods[i][0]))
    graphs.append(gu.make_graph(adj_mats[i]))
if plot:
    for adj in adj_mats:
        vu.visualize_adj_mat(adj)
#throw error if isomophic
for G1 in range(len(mods)):
    for G2 in range(G1+1,len(mods)):
        assert gu.check_not_isomorphism(graphs[G1],graphs[G2]), f"Graph G{G1} and G{G2} are isomophic"

if plot:
    for i in param_list:
        bary_list = barycenters[i]
        vu.graph_plot(adj_mats,vu.graph_visual,2,2,bary_list)


