import os
import torch
import pickle
import numpy as np

from torcheeg import transforms
from torcheeg.models import DGCNN
from sklearn.model_selection import train_test_split

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



def read_saved_models(saved_path):    
    main_folder_files = []
    for folder in os.listdir(saved_path):
        if "gpuerror" not in folder:
            file_names = os.listdir(saved_path + "/" + folder)
            folder_seed = int(folder.split("_")[-1])
            main_folder_files.append([folder, file_names, folder_seed])

    model_objects = dict()
    bary_dict = dict()
    simrank_dict = dict()
    for folder_name, folder_files, seed in main_folder_files:
        #print(f"for seed = {seed}")
        model_objects[seed] = dict()
        bary_dict[seed] = dict()
        simrank_dict[seed] = dict()
        for file_name in folder_files:

            if file_name[:5] == "model":
                #print(f"file_name: {file_name} - MODEL file")
                chans = int(file_name.split("_")[3][4:])
                full_model_path = saved_path + "/" + folder_name + "/" + file_name
                
                curr_mod = DGCNN(in_channels=5, num_electrodes=22, 
                              hid_channels=chans, num_layers=2, num_classes=5)
                
                model_weights = torch.load(full_model_path)
                curr_mod.load_state_dict(model_weights)
                model_objects[seed][chans] = curr_mod
            if file_name[:10] == "barycenter":
                full_dict_path = saved_path + "/" + folder_name + "/" + file_name
                with open(full_dict_path,'rb') as fb:
                    bary_dict[seed] = pickle.load(fb)
            if file_name[:7] == "simrank":
                full_dict_path = saved_path + "/" + folder_name + "/" + file_name
                with open(full_dict_path,'rb') as fb:
                    simrank_dict[seed] = pickle.load(fb)
                
    return bary_dict, simrank_dict, model_objects


def split_data(X, y, has_val_set=False, seed=42):
    
    if has_val_set:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
        
        return (X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

        return (X_train, y_train, X_test, y_test)
