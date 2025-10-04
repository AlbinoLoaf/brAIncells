import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheeg.models import DGCNN
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,  ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_all_models(model_dict):
    """
    Extract model objects from seed channel dict to list
    
    ...
    
    Parameters
    -----
    model_dict: dict
        Dictionary of the form dict[seed][channels] containing 
        the trained model objects (torcheeg.models DGCNN objects)
    
    Returns
    -----
    models: list
        List of all the model objects from the dictionary
    """
    
    models = []
    
    for seed_val in model_dict.keys():
        
        curr_models_dict = model_dict[seed_val]
        
        for chan_val in curr_models_dict.keys():
            
            curr_models = curr_models_dict[chan_val]
            models.append(curr_models)
        
    return models

def read_saved_models(saved_path):
    model_objects = {}

    for folder in os.listdir(saved_path):
        if "gpuerror" in folder:
            continue

        folder_path = os.path.join(saved_path, folder)
        folder_seed = int(folder.split("_")[-1])
        model_objects[folder_seed] = {}

        for file_name in os.listdir(folder_path):
            if not file_name.startswith("model"):
                continue

            full_model_path = os.path.join(folder_path, file_name)
            checkpoint = torch.load(full_model_path, map_location='cpu')

            params = checkpoint['params']
            model = DGCNN(**params)
            model.load_state_dict(checkpoint['model_state'])

            chans = params.get('num_electrodes', 22)  # or parse from filename if needed
            model_objects[folder_seed][chans] = model

    return model_objects

# def read_saved_models(saved_path):
#     """
#     Read saved model objects from specified path 
    
#     ...
    
#     Parameters
#     -----
#     saved_path: str
#         Path where model objects are saved
    
#     Returns
#     -----
#     model_objects : dict
#         Dictionary containing read model objects in the form dict[seed][n_chans]
#         (where each dict[seed] contains a dictionary organised by number of channels)
#     """
#     main_folder_files = []
#     for folder in os.listdir(saved_path):
#         if "gpuerror" not in folder:
#             file_names = os.listdir(saved_path + folder)
#             folder_seed = int(folder.split("_")[-1])
#             main_folder_files.append([folder, file_names, folder_seed])

#     model_objects = dict()
#     for folder_name, folder_files, seed in main_folder_files:
#         model_objects[seed] = dict()
#         for file_name in folder_files:

#             if file_name[:5] == "model":
                
#                 chans = int(file_name.split("_")[2])
#                 full_model_path = saved_path + "/" + folder_name + "/" + file_name
                
#                 def findmod(in_channels=5, num_electrodes=22,hid_channels=8, num_layers=2, num_classes=5):
#                     return DGCNN(in_channels=in_channels, num_electrodes=num_electrodes, 
#                               hid_channels=hid_channels, num_layers=num_layers, num_classes=num_classes)
#                 param_dict = {chans:"hid_channels=8"}
#                 curr_mod = findmod(param_dict[param])
                
#                 model_weights = torch.load(full_model_path, map_location=torch.device('cpu'))
#                 curr_mod.load_state_dict(model_weights)
#                 model_objects[seed][chans] = curr_mod
                
#     return model_objects




def model_by_param(model_dict, param):
    """
    Destributes each seed of a given model into channels
    ...
    
    Parameters
    -----
    model_dict: dict
        Dictionary of the form dict[seed][channels] containing 
        the trained model objects (torcheeg.models DGCNN objects)
    param: string
        A string that describes the metric that are in file name
    
    Returns
    -----
    model_by_chans, barycenters_by_chans, sims_by_chans: dict
        Dictionary containing graph metrics of the form dict[n_chans]
    isomorphism_by_chans, geds_by_chans: dict
        Dictionary containing GED and isomorphism check results of the form dict[n_chans]
        Only returned when get_external is set to True
    """
    
    model_by_param = dict()
    for curr_seed in model_dict.keys():
        for curr_param in model_dict[curr_seed].keys():
            if curr_param not in model_by_param.keys():
                model_by_param[curr_param] = []
            model_by_param[curr_param].append(model_dict[curr_seed][curr_param])
    return model_by_param